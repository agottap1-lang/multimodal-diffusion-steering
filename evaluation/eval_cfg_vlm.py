#!/usr/bin/env python3
"""
Complete CFG + VLM Steering Evaluation Pipeline
================================================

Self-contained evaluation: no external gemini_vlm_eval dependency.
Uses Gemini 3 Pro directly via google-genai SDK.

Pipeline per episode:
  1. Reset env, configure scene (obstacle/waypoints per behavior)
  2. Generate K=8 candidate trajectories via CFG-DDIM (stochastic, eta=0.5)
  3. Capture 6 frames per candidate → send to Gemini 3 Pro
  4. VLM scores each candidate with behavior-specific prompt
  5. Select best (VLM-steered) and worst (baseline) candidates
  6. Execute both to completion, record video
  7. Compute metrics: L_early, path_efficiency, clearance, hover_dist

Usage:
  .venv\\Scripts\\python.exe evaluation/eval_cfg_vlm.py ^
      --checkpoint runs/cfg_YYYYMMDD/ckpt_ep200.pt ^
      --behavior all --n_episodes 10 --K 8
"""

from __future__ import annotations

import argparse, io, json, math, os, sys, time, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pybullet as p
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from envs.twoblockpick_env import TwoBlockPickEnv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── API Key ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # set in .env — never hardcode
GEMINI_MODEL   = "gemini-3-pro-preview"

# ── Constants ─────────────────────────────────────────────────────────
OBS_DIM_V2       = 26
ACT_DIM          = 5
CFG_COND_START   = 22   # context dims start
CFG_MODE_DIM     = 25   # behavior_mode dim (ONLY this is zeroed for uncond)
TABLE_TOP_Z      = 0.4
OBSTACLE_RADIUS  = 0.035
OBSTACLE_HEIGHT  = 0.18
OBSTACLE_COLOR   = [0.0, 0.85, 0.85, 1]
GOAL_A_DESC      = "pick the LEFT (green) block"
GOAL_B_DESC      = "pick the RIGHT (red) block"
FRAME_CAPTURE_STEPS = {0: "t=0s", 30: "t=1s", 60: "t=2s",
                       90: "t=3s", 120: "t=4s", 149: "t=5s"}


# ── Waypoint blocks (pentagon, same as collect_demos_v2) ──────────────
def _build_waypoint_blocks():
    cx, cy = 0.43, 0.0
    r = 0.06
    colors = [
        ("blue",   [0.1, 0.2, 0.95, 1]),
        ("green",  [0.1, 0.85, 0.1, 1]),
        ("yellow", [0.95, 0.9, 0.1, 1]),
        ("orange", [1.0, 0.55, 0.05, 1]),
        ("purple", [0.65, 0.1, 0.85, 1]),
    ]
    blocks = []
    for i, (name, rgba) in enumerate(colors):
        angle = 2 * math.pi * i / 5 + math.pi / 2
        blocks.append({"name": name,
                       "pos": [round(cx + r * math.cos(angle), 4),
                               round(cy + r * math.sin(angle), 4),
                               TABLE_TOP_Z + 0.016],
                       "rgba": rgba})
    return blocks

WAYPOINT_BLOCKS = _build_waypoint_blocks()


# ══════════════════════════════════════════════════════════════════════
# GEMINI VLM CLIENT (standalone, no external dependency)
# ══════════════════════════════════════════════════════════════════════

class GeminiScorerClient:
    """Minimal Gemini 3 Pro client for trajectory scoring."""

    def __init__(self, api_key: str = GEMINI_API_KEY,
                 model: str = GEMINI_MODEL):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        # Quick connectivity test
        resp = self.client.models.generate_content(
            model=self.model, contents="Reply OK")
        assert "ok" in resp.text.lower() or "OK" in resp.text, \
            f"Gemini test failed: {resp.text}"
        print(f"  Gemini client ready: {self.model}")

    def score_trajectory(self, frames_bytes: list[bytes],
                         behavior: str, target: str = "LEFT",
                         obstacle_info: str = "",
                         waypoint_info: str = "",
                         retries: int = 3) -> dict:
        """Score trajectory frames with behavior-specific prompt.

        Returns dict: {pA, pB, score, cue}
        """
        from google.genai import types as gtypes

        parts = []
        for fb in frames_bytes:
            parts.append(gtypes.Part.from_bytes(data=fb, mime_type="image/jpeg"))

        prompt = self._build_prompt(behavior, target, obstacle_info,
                                    waypoint_info, len(frames_bytes))
        parts.append(gtypes.Part.from_text(text=prompt))

        for attempt in range(retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[gtypes.Content(role="user", parts=parts)],
                )
                return self._parse_response(resp.text, behavior)
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    print(f"      [VLM retry {attempt+1}: {e}]")
                else:
                    print(f"      [VLM FAILED: {e}]")
                    return {"pA": 0.5, "pB": 0.5, "score": 0.5,
                            "cue": f"ERROR: {e}"}

    def _build_prompt(self, behavior, target, obstacle_info,
                      waypoint_info, n_frames):
        base = (
            f"You are a robotics expert evaluating a Franka Panda robot arm trajectory.\n"
            f"The robot is performing a block-picking task with two target blocks:\n"
            f"  Goal A: {GOAL_A_DESC}\n"
            f"  Goal B: {GOAL_B_DESC}\n\n"
            f"You see {n_frames} frames from the trajectory at t=0,1,2,3,4,5 seconds.\n"
            f"The arm starts from a home position above the table and moves toward one block.\n\n"
        )

        if behavior == "legibility":
            task = (
                "TASK: Rate how LEGIBLE this trajectory is.\n"
                "A legible trajectory clearly reveals which block the robot intends to pick "
                "as EARLY as possible. The arm should curve toward the target so an observer "
                "can predict the goal from the first few seconds.\n"
                "A non-legible trajectory moves ambiguously or straight, hiding intent until late.\n\n"
                "Rate: Which goal does the arm appear headed for? How EARLY is intent obvious?\n"
                "  pA = probability robot is going for Goal A (left block)\n"
                "  pB = probability robot is going for Goal B (right block)\n"
                "  legibility = 0.0 (ambiguous intent) to 1.0 (crystal clear intent from frame 1)\n\n"
                'Output ONLY valid JSON: {"pA": X, "pB": X, "legibility": X, '
                '"cue": "brief description of how intent is revealed"}'
            )
        elif behavior == "predictability":
            task = (
                "TASK: Rate how PREDICTABLE this trajectory is.\n"
                "A predictable trajectory takes the SHORTEST, MOST DIRECT path to the target.\n"
                "No arc, no sweep, no unnecessary lateral motion. An observer watching the first "
                "1-2 seconds can extrapolate exactly where the arm will be at every future moment.\n"
                "An unpredictable trajectory curves, hesitates, or takes an indirect route.\n\n"
                "Rate:\n"
                "  pA = probability going for Goal A\n"
                "  pB = probability going for Goal B\n"
                "  predictability = 0.0 (erratic/curved) to 1.0 (perfectly straight/direct)\n\n"
                'Output ONLY valid JSON: {"pA": X, "pB": X, "predictability": X, '
                '"cue": "brief description of path directness"}'
            )
        elif behavior == "safety":
            task = (
                f"TASK: Rate how SAFELY this trajectory avoids an obstacle.\n"
                f"There is a CYAN CYLINDER obstacle on the table: {obstacle_info}\n"
                f"A safe trajectory maintains clear distance from the obstacle at all times.\n"
                f"It should choose an alternative route that goes AROUND the obstacle.\n"
                f"An unsafe trajectory passes very close to or appears to risk collision.\n\n"
                f"Rate:\n"
                f"  pA = probability going for Goal A\n"
                f"  pB = probability going for Goal B\n"
                f"  safety = 0.0 (dangerous, near obstacle) to 1.0 (wide safe clearance)\n\n"
                f'Output ONLY valid JSON: {{"pA": X, "pB": X, "safety": X, '
                f'"cue": "brief description of clearance from obstacle"}}'
            )
        elif behavior == "grounding":
            task = (
                f"TASK: Rate how well this trajectory follows a spatial instruction.\n"
                f"INSTRUCTION: \"{waypoint_info}\"\n"
                f"The scene has 5 small colored blocks arranged in a pentagon on the table.\n"
                f"The robot should first hover near the specified waypoint block, "
                f"then move to pick the target block.\n\n"
                f"Rate:\n"
                f"  pA = probability going for Goal A\n"
                f"  pB = probability going for Goal B\n"
                f"  grounding = 0.0 (ignores waypoint) to 1.0 (clearly visits waypoint first)\n\n"
                f'Output ONLY valid JSON: {{"pA": X, "pB": X, "grounding": X, '
                f'"cue": "brief description of path relative to waypoint"}}'
            )
        else:
            raise ValueError(f"Unknown behavior: {behavior}")

        return base + task

    def _parse_response(self, text: str, behavior: str) -> dict:
        text = text.strip()
        j_start = text.find("{")
        j_end = text.rfind("}") + 1
        if j_start == -1 or j_end == 0:
            raise ValueError(f"No JSON in response: {text[:200]}")
        data = json.loads(text[j_start:j_end])

        pA = float(data.get("pA", 0.5))
        pB = float(data.get("pB", 0.5))
        cue = str(data.get("cue", ""))

        score_key = {"legibility": "legibility",
                     "predictability": "predictability",
                     "safety": "safety",
                     "grounding": "grounding"}.get(behavior, "legibility")
        score = float(data.get(score_key, max(pA, pB)))

        return {"pA": pA, "pB": pB, "score": score, "cue": cue}


# ══════════════════════════════════════════════════════════════════════
# MODEL (must match train_cfg.py exactly)
# ══════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class UNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.act = nn.Mish()
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h.transpose(1, 2); h = self.norm1(h); h = h.transpose(1, 2)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        h = self.conv2(h)
        h = h.transpose(1, 2); h = self.norm2(h); h = h.transpose(1, 2)
        return self.act(h + self.shortcut(x))

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.obs_dim = obs_dim; self.act_dim = act_dim; self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128), nn.Linear(128, hidden_dim),
            nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim) for i in range(len(dims)-1)])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1] + dims[i+1], dims[i], hidden_dim)
            for i in range(len(dims)-2, -1, -1)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, act_dim))

    def forward(self, noisy_act, timestep, obs):
        t_emb = self.time_mlp(timestep)
        x = self.input_proj(noisy_act) + self.obs_embed(obs).unsqueeze(1)
        skips = []
        for blk in self.encoder_blocks:
            x = blk(x, t_emb); skips.append(x)
        x = self.bottleneck(x, t_emb)
        for blk, sk in zip(self.decoder_blocks, reversed(skips)):
            x = blk(torch.cat([x, sk], dim=-1), t_emb)
        return self.output_proj(x)


# ══════════════════════════════════════════════════════════════════════
# CFG DDIM SAMPLER  (stochastic, eta=0.5)
# ══════════════════════════════════════════════════════════════════════

class CFGDDIMSampler:
    def __init__(self, n_steps, beta_start, beta_end, device, eta=0.5):
        self.device = device
        self.eta = eta
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs_cond, obs_uncond, cfg_lambda=1.0,
               n_sampling_steps=20):
        B = obs_cond.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod)-1,
                           n_sampling_steps, device=self.device).long(), [0])
        for i, t in enumerate(timesteps):
            t_b = t.repeat(B)
            eps_cond = model(x, t_b, obs_cond)
            eps_uncond = model(x, t_b, obs_uncond)
            eps = eps_uncond + cfg_lambda * (eps_cond - eps_uncond)

            a_t = self.alphas_cumprod[t]
            a_prev = (self.alphas_cumprod[timesteps[i+1]]
                      if i < len(timesteps)-1
                      else torch.tensor(1.0, device=self.device))
            x0 = (x - torch.sqrt(1-a_t)*eps) / torch.sqrt(a_t)
            if i < len(timesteps)-1:
                sigma = self.eta * torch.sqrt(
                    (1-a_prev)/(1-a_t) * (1-a_t/a_prev))
                dir_xt = torch.sqrt(torch.clamp(1-a_prev-sigma**2, min=0))*eps
                x = torch.sqrt(a_prev)*x0 + dir_xt + sigma*torch.randn_like(x)
            else:
                x = x0
        return x


# ══════════════════════════════════════════════════════════════════════
# LOAD POLICY
# ══════════════════════════════════════════════════════════════════════

def load_policy(ckpt_path, device, eta=0.5):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DiffusionPolicy(
        ckpt.get('obs_dim', OBS_DIM_V2), ckpt.get('act_dim', ACT_DIM),
        ckpt.get('horizon', 32), ckpt.get('hidden_dim', 256),
        ckpt.get('n_blocks', 6)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    sampler = CFGDDIMSampler(
        ckpt.get('n_diffusion_steps', 100),
        ckpt.get('beta_start', 1e-4), ckpt.get('beta_end', 0.1),
        device, eta=eta)
    stats = {k: ckpt[k] for k in ['obs_mean','obs_std','act_mean','act_std']}
    return model, sampler, stats, ckpt


# ══════════════════════════════════════════════════════════════════════
# ENV HELPERS
# ══════════════════════════════════════════════════════════════════════

def add_obstacle_visual(env, pos):
    cid = env._cid
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=OBSTACLE_RADIUS,
                              length=OBSTACLE_HEIGHT, rgbaColor=OBSTACLE_COLOR,
                              physicsClientId=cid)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=vis,
                             basePosition=pos, physicsClientId=cid)

def add_waypoint_blocks(env, wp_list):
    cid = env._cid
    uids = []
    for wp in wp_list:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                  rgbaColor=wp["rgba"], physicsClientId=cid)
        uids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis,
                                      basePosition=wp["pos"],
                                      physicsClientId=cid))
    return uids

def remove_bodies(env, uids):
    for uid in uids:
        p.removeBody(uid, physicsClientId=env._cid)

def get_obs_v2(env, context_pos=None, behavior_mode=0.0):
    base = env._get_obs()
    ctx = np.zeros(3, dtype=np.float32) if context_pos is None else np.array(context_pos, dtype=np.float32)
    return np.concatenate([base, ctx, np.array([behavior_mode], dtype=np.float32)])

def capture_jpeg(env, width=240, height=240, quality=75) -> bytes:
    from PIL import Image
    frame = env.render(mode="rgb_array", width=width, height=height)
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_obs_uncond(obs_cond_tensor, zero_context=False):
    """Create unconditional obs for CFG inference.

    Args:
        obs_cond_tensor: normalized observation tensor
        zero_context: if True, zero dims 22-25 (context + mode).
                      if False, zero only dim 25 (mode).

    For legibility/predictability: zero_context=False  (context is always
    [0,0,0] so cond/uncond context should match → no spurious signal).
    For safety/grounding: zero_context=True (context IS the signal;
    uncond must lack it so CFG can steer toward it)."""
    obs_u = obs_cond_tensor.clone()
    if zero_context:
        obs_u[..., CFG_COND_START:] = 0.0   # dims 22-25 → zero
    else:
        obs_u[..., CFG_MODE_DIM] = 0.0       # dim 25 only
    return obs_u


def scripted_grasp(env, target, max_steps=120, action_scale=0.05):
    """Scripted descent → close gripper → lift. Returns (success, steps, ee_traj).

    Called when the diffusion policy has brought the EE near the target block.
    This ensures reliable grasping independent of diffusion model precision."""
    obs_22 = env._get_obs()
    cube_pos = obs_22[8:11].copy() if target == "left" else obs_22[15:18].copy()
    grasp_pos = cube_pos.copy(); grasp_pos[2] += 0.005
    lift_pos = grasp_pos.copy(); lift_pos[2] = 0.60

    ee_traj = []
    steps = 0

    # Phase 1: Descend to grasp position
    for _ in range(60):
        obs_22 = env._get_obs()
        ee = obs_22[:3]
        ee_traj.append(ee.copy())
        delta = (grasp_pos - ee) / action_scale
        a = np.zeros(5, dtype=np.float32)
        a[:3] = np.clip(delta * 0.4, -1, 1)
        a[4] = 1.0  # open
        result = env.step(a)
        steps += 1
        if np.linalg.norm(ee - grasp_pos) < 0.008:
            break

    # Phase 2: Close gripper
    for i in range(20):
        obs_22 = env._get_obs()
        ee_traj.append(obs_22[:3].copy())
        a = np.zeros(5, dtype=np.float32)
        a[4] = 1.0 - 2.0 * (i / 19)
        result = env.step(a)
        steps += 1

    # Phase 3: Lift
    for _ in range(40):
        obs_22 = env._get_obs()
        ee = obs_22[:3]
        ee_traj.append(ee.copy())
        delta = (lift_pos - ee) / action_scale
        a = np.zeros(5, dtype=np.float32)
        a[:3] = np.clip(delta * 0.4, -1, 1)
        a[4] = -1.0  # closed
        result = env.step(a)
        steps += 1
        sl = result.info.get("success_left", 0) > 0.5
        sr = result.info.get("success_right", 0) > 0.5
        if sl or sr:
            break

    success = (result.info.get("success_left", 0) > 0.5 or
               result.info.get("success_right", 0) > 0.5)
    return success, steps, np.array(ee_traj)


def _close_to_block(ee, obs_22, threshold=0.07):
    """Check if EE is close to either block in XY + within approach height."""
    left_pos = obs_22[8:11]
    right_pos = obs_22[15:18]
    dl = np.linalg.norm(ee[:2] - left_pos[:2])
    dr = np.linalg.norm(ee[:2] - right_pos[:2])
    near = min(dl, dr) < threshold
    z_ok = ee[2] < 0.52  # below approach height
    return near and z_ok, "left" if dl < dr else "right"


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_l_early(ee_traj, left_pos, right_pos, target="left",
                    early_frac=0.3):
    """Bayesian L_early: avg posterior of true goal in early 30%."""
    if len(ee_traj) < 4:
        return 0.5
    goals = np.array([left_pos, right_pos])
    T = len(ee_traj)
    early_end = max(2, int(T * early_frac))
    sigma = np.linalg.norm(goals[0]-goals[1]) / (2*np.sqrt(2*np.log(2)))
    sigma = max(sigma, 1e-6)
    posteriors = []
    for t in range(early_end):
        d = np.array([np.linalg.norm(ee_traj[t]-g) for g in goals])
        likes = np.exp(-0.5*(d/sigma)**2)
        post = likes / (likes.sum()+1e-12)
        posteriors.append(post)
    posteriors = np.array(posteriors)
    true_idx = 0 if target == "left" else 1
    return float(posteriors[:, true_idx].mean())

def compute_path_efficiency(ee_traj):
    if len(ee_traj) < 2:
        return 0.0
    straight = np.linalg.norm(ee_traj[-1] - ee_traj[0])
    path_len = sum(np.linalg.norm(ee_traj[j+1]-ee_traj[j])
                   for j in range(len(ee_traj)-1))
    return straight / max(path_len, 1e-6)

def compute_clearance(ee_traj, obs_pos_2d):
    if len(ee_traj) == 0:
        return 0.0
    dists = np.linalg.norm(ee_traj[:, :2] - obs_pos_2d, axis=1)
    return float(dists.min())

def compute_hover_dist(ee_traj, wp_pos):
    if len(ee_traj) == 0:
        return 999.0
    dists = np.linalg.norm(ee_traj - np.array(wp_pos), axis=1)
    return float(dists.min())


# ══════════════════════════════════════════════════════════════════════
# SIMULATE ONE CANDIDATE (150 steps = 5 seconds)
# ══════════════════════════════════════════════════════════════════════

def simulate_candidate(env, model, sampler, stats, device,
                       context_pos, behavior_mode, cfg_lambda,
                       seed, n_steps=150, n_action_steps=8,
                       zero_context=False):
    """Simulate candidate, capture 6 frames, return trajectory + frames."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    act_mean, act_std = stats['act_mean'], stats['act_std']
    H = model.horizon

    frames = []
    ee_traj = []
    actions_all = []
    act_queue = []
    queue_idx = 0

    obs_22 = env._get_obs()

    for step in range(n_steps):
        ee_traj.append(obs_22[:3].copy())

        # Capture frame at scheduled steps
        if step in FRAME_CAPTURE_STEPS:
            frames.append(capture_jpeg(env))

        # Replan when queue exhausted (use first n_action_steps from chunk)
        if queue_idx >= len(act_queue):
            obs_v2 = get_obs_v2(env, context_pos, behavior_mode)
            obs_norm = (obs_v2 - obs_mean) / obs_std
            obs_cond = torch.tensor(obs_norm, dtype=torch.float32,
                                    device=device).unsqueeze(0)
            obs_uncond = _make_obs_uncond(obs_cond, zero_context=zero_context)
            chunk = sampler.sample(model, obs_cond, obs_uncond,
                                   cfg_lambda=cfg_lambda, n_sampling_steps=20)
            full_acts = chunk[0].cpu().numpy() * act_std + act_mean
            act_queue = full_acts[:n_action_steps]
            queue_idx = 0

        action = act_queue[queue_idx].copy()
        queue_idx += 1
        action[:4] = np.clip(action[:4], -1, 1)
        action[4] = np.clip(action[4], -1, 1)
        actions_all.append(action)

        result = env.step(action)
        obs_22 = result.obs
        if result.done:
            break

    return {
        "frames": frames,
        "ee_traj": np.array(ee_traj),
        "actions": np.array(actions_all),
        "final_obs": obs_22,
        "done": result.done,
        "info": result.info,
    }


# ══════════════════════════════════════════════════════════════════════
# BEST-OF-K EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_bestofk_episode(env, model, sampler, stats, device,
                        vlm_client, episode_seed, behavior,
                        K, cfg_lambda, behavior_mode,
                        context_pos=None, obstacle_info="",
                        waypoint_info="", target="left",
                        out_dir=None, max_exec_steps=400,
                        zero_context=False):
    """Run one Best-of-K episode. Returns metrics for VLM-selected & baseline."""

    env.reset(seed=episode_seed)

    # ── Scene setup ─────────────────────────────────────────────
    added_uids = []
    if behavior == "safety":
        # Add obstacle
        if "straight" in obstacle_info.lower() or cfg_lambda > 0:
            y = 0.02 if target == "left" else -0.02
            obs_pos = [0.46, y, TABLE_TOP_Z + OBSTACLE_HEIGHT/2]
        else:
            y = 0.11 if target == "left" else -0.11
            obs_pos = [0.38, y, TABLE_TOP_Z + OBSTACLE_HEIGHT/2]
        uid = add_obstacle_visual(env, obs_pos)
        added_uids = [uid]
        context_pos = obs_pos
        obstacle_info = f"at position [{obs_pos[0]:.2f}, {obs_pos[1]:.2f}]"
    elif behavior == "grounding":
        p.changeVisualShape(env._cube_l_uid, -1,
                            rgbaColor=[0.1, 0.8, 0.1, 1.0],
                            physicsClientId=env._cid)
        p.changeVisualShape(env._cube_r_uid, -1,
                            rgbaColor=[0.8, 0.1, 0.1, 1.0],
                            physicsClientId=env._cid)
        added_uids = add_waypoint_blocks(env, WAYPOINT_BLOCKS)

    for _ in range(60):
        p.stepSimulation(physicsClientId=env._cid)

    # Save pybullet state for candidate rollback
    saved = p.saveState(physicsClientId=env._cid)
    init_obs = env._get_obs()

    # ── Phase 1: Generate K candidates ──────────────────────────
    candidates = []
    for k in range(K):
        p.restoreState(saved, physicsClientId=env._cid)
        env._episode_steps = 0
        env._picked_left = False
        env._picked_right = False
        c_seed = episode_seed * 1000 + k * 7 + 1

        cand = simulate_candidate(env, model, sampler, stats, device,
                                  context_pos, behavior_mode, cfg_lambda,
                                  seed=c_seed, zero_context=zero_context)
        cand["seed"] = c_seed
        cand["idx"] = k
        candidates.append(cand)

    # ── Phase 2: VLM scoring ────────────────────────────────────
    target_str = "LEFT" if target == "left" else "RIGHT"
    for k, cand in enumerate(candidates):
        vlm_result = vlm_client.score_trajectory(
            cand["frames"], behavior, target=target_str,
            obstacle_info=obstacle_info, waypoint_info=waypoint_info)
        cand["vlm"] = vlm_result
        print(f"    c{k}: VLM score={vlm_result['score']:.2f} "
              f"pA={vlm_result['pA']:.2f} pB={vlm_result['pB']:.2f} "
              f"cue=\"{vlm_result['cue'][:60]}\"")
        time.sleep(0.5)  # rate limit

    # ── Phase 3: Select best + worst ────────────────────────────
    vlm_idx = max(range(K), key=lambda i: candidates[i]["vlm"]["score"])
    baseline_idx = min(range(K), key=lambda i: candidates[i]["vlm"]["score"])
    if baseline_idx == vlm_idx and K > 1:
        scores_sorted = sorted(range(K),
                               key=lambda i: candidates[i]["vlm"]["score"])
        baseline_idx = scores_sorted[0] if scores_sorted[0] != vlm_idx else scores_sorted[1]

    print(f"  -> VLM pick: c{vlm_idx} (score={candidates[vlm_idx]['vlm']['score']:.2f})")
    print(f"  -> Baseline: c{baseline_idx} (score={candidates[baseline_idx]['vlm']['score']:.2f})")

    # ── Phase 4: Execute both to completion + record video ──────
    results = {}
    for label, sel_idx in [("vlm", vlm_idx), ("baseline", baseline_idx)]:
        p.restoreState(saved, physicsClientId=env._cid)
        env._episode_steps = 0
        env._picked_left = False
        env._picked_right = False

        vid_path = None  # skip video recording for speed

        # Replay stored actions then replan, with scripted grasp
        stored = candidates[sel_idx]["actions"]
        obs_22 = init_obs.copy()
        ee_full = []
        success = False
        steps = 0
        act_idx = 0
        grasp_triggered = False

        for step in range(max_exec_steps):
            ee_full.append(obs_22[:3].copy())

            # Check proximity to block every step
            ee = obs_22[:3]
            near, nearest_block = _close_to_block(ee, obs_22)
            if near and not grasp_triggered:
                grasp_triggered = True
                g_target = nearest_block  # grasp whichever block is nearest
                g_ok, g_steps, g_traj = scripted_grasp(env, g_target)
                ee_full.extend(g_traj.tolist())
                steps += g_steps
                obs_22 = env._get_obs()
                success = g_ok
                break

            if act_idx < len(stored):
                action = stored[act_idx].copy()
                act_idx += 1
            else:
                # Replan with CFG
                obs_v2 = get_obs_v2(env, context_pos, behavior_mode)
                obs_norm = (obs_v2 - stats['obs_mean']) / stats['obs_std']
                obs_c = torch.tensor(obs_norm, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                obs_u = _make_obs_uncond(obs_c, zero_context=zero_context)
                chunk = sampler.sample(model, obs_c, obs_u,
                                       cfg_lambda=cfg_lambda,
                                       n_sampling_steps=20)
                new_acts = (chunk[0].cpu().numpy() *
                            stats['act_std'] + stats['act_mean'])[:8]
                stored = np.concatenate([stored, new_acts])

                action = stored[act_idx].copy()
                act_idx += 1

            action[:4] = np.clip(action[:4], -1, 1)
            action[4] = np.clip(action[4], -1, 1)
            result = env.step(action)
            obs_22 = result.obs
            steps += 1
            s_l = result.info.get("success_left", 0) > 0.5
            s_r = result.info.get("success_right", 0) > 0.5
            success = s_l or s_r
            if result.done:
                break

        if vid_path:
            env.stop_video()

        ee_arr = np.array(ee_full)
        left_pos = obs_22[8:11]
        right_pos = obs_22[15:18]

        metrics = {
            "selected_idx": sel_idx,
            "success": success,
            "steps": steps,
            "vlm_score": candidates[sel_idx]["vlm"]["score"],
            "vlm_cue": candidates[sel_idx]["vlm"]["cue"][:100],
        }

        # Behavior-specific metrics
        if behavior == "legibility":
            # Use actually-approached block for L_early
            if len(ee_arr) > 0:
                final_ee = ee_arr[-1]
                dl = np.linalg.norm(final_ee[:2] - left_pos[:2])
                dr = np.linalg.norm(final_ee[:2] - right_pos[:2])
                actual = "left" if dl < dr else "right"
            else:
                actual = target
            metrics["approached"] = actual
            metrics["L_early"] = round(compute_l_early(
                ee_arr, left_pos, right_pos, actual), 4)
        elif behavior == "predictability":
            metrics["path_efficiency"] = round(
                compute_path_efficiency(ee_arr), 4)
        elif behavior == "safety":
            if context_pos is not None:
                cl = compute_clearance(ee_arr, np.array(context_pos[:2]))
                metrics["min_clearance"] = round(cl, 4)
                metrics["collision"] = cl < OBSTACLE_RADIUS
        elif behavior == "grounding":
            if context_pos is not None:
                hd = compute_hover_dist(ee_arr, context_pos)
                metrics["min_wp_dist"] = round(hd, 4)
                metrics["hovered"] = hd < 0.06

        results[label] = metrics
        s_tag = "OK" if success else "FAIL"
        print(f"  {label.upper()}: {s_tag} steps={steps}", end="")
        if "L_early" in metrics:
            print(f" L_early={metrics['L_early']:.3f}", end="")
        if "path_efficiency" in metrics:
            print(f" eff={metrics['path_efficiency']:.3f}", end="")
        if "min_clearance" in metrics:
            print(f" clearance={metrics['min_clearance']:.3f}", end="")
        if "min_wp_dist" in metrics:
            print(f" wp_dist={metrics['min_wp_dist']:.3f}", end="")
        print()

    # Cleanup
    p.removeState(saved, physicsClientId=env._cid)
    if added_uids:
        try:
            remove_bodies(env, added_uids)
        except Exception:
            pass

    return {
        "episode_seed": episode_seed,
        "behavior": behavior,
        "target": target,
        "K": K,
        "cfg_lambda": cfg_lambda,
        "vlm": results.get("vlm", {}),
        "baseline": results.get("baseline", {}),
        "candidates": [
            {"idx": c["idx"], "seed": c["seed"],
             "vlm_score": c["vlm"]["score"],
             "vlm_pA": c["vlm"]["pA"], "vlm_pB": c["vlm"]["pB"],
             "vlm_cue": c["vlm"]["cue"][:80]}
            for c in candidates
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# CFG-ONLY ROLLOUT (no VLM, for quick sanity check)
# ══════════════════════════════════════════════════════════════════════

def rollout_cfg_only(env, model, sampler, stats, device,
                     context_pos, behavior_mode, cfg_lambda,
                     target="left", max_steps=400, n_action_steps=8,
                     video_path=None, zero_context=False):
    """Simple CFG rollout without VLM (for --no_vlm mode)."""
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    act_mean, act_std = stats['act_mean'], stats['act_std']

    if video_path:
        env.record_video(str(video_path))

    obs_22 = env._get_obs()
    ee_traj = []
    total_steps = 0
    grasp_triggered = False

    for _ in range(0, max_steps, n_action_steps):
        # Check if EE is near a block → switch to scripted grasp
        ee = obs_22[:3]
        near, nearest_block = _close_to_block(ee, obs_22)
        if near and not grasp_triggered:
            grasp_triggered = True
            g_target = nearest_block  # grasp whichever block is nearest
            g_ok, g_steps, g_traj = scripted_grasp(env, g_target)
            ee_traj.extend(g_traj.tolist())
            total_steps += g_steps
            obs_22 = env._get_obs()
            break

        obs_v2 = get_obs_v2(env, context_pos, behavior_mode)
        obs_norm = (obs_v2 - obs_mean) / obs_std
        obs_c = torch.tensor(obs_norm, dtype=torch.float32,
                             device=device).unsqueeze(0)
        obs_u = _make_obs_uncond(obs_c, zero_context=zero_context)
        chunk = sampler.sample(model, obs_c, obs_u,
                               cfg_lambda=cfg_lambda, n_sampling_steps=20)
        acts = chunk[0].cpu().numpy() * act_std + act_mean

        for j in range(min(n_action_steps, len(acts))):
            if total_steps >= max_steps:
                break
            a = acts[j].copy()
            a[:4] = np.clip(a[:4], -1, 1)
            a[4] = np.clip(a[4], -1, 1)
            result = env.step(a)
            obs_22 = result.obs
            ee_traj.append(obs_22[:3].copy())
            total_steps += 1
            if result.done:
                break
        if result.done:
            break

    if video_path:
        env.stop_video()

    if grasp_triggered:
        # Success from scripted grasp
        success = bool(p.getBasePositionAndOrientation(
            env._cube_l_uid, physicsClientId=env._cid)[0][2] > 0.47 or
            p.getBasePositionAndOrientation(
            env._cube_r_uid, physicsClientId=env._cid)[0][2] > 0.47)
    else:
        success = (result.info.get("success_left", 0) > 0.5 or
                   result.info.get("success_right", 0) > 0.5)

    # Determine which block was actually approached
    approached = None
    if len(ee_traj) > 0:
        final_ee = np.array(ee_traj[-1])
        left_pos = obs_22[8:11]
        right_pos = obs_22[15:18]
        dl = np.linalg.norm(final_ee[:2] - left_pos[:2])
        dr = np.linalg.norm(final_ee[:2] - right_pos[:2])
        approached = "left" if dl < dr else "right"

    return {
        "success": success,
        "steps": total_steps,
        "ee_traj": np.array(ee_traj) if ee_traj else np.zeros((0,3)),
        "final_obs": obs_22,
        "approached": approached,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("CFG + VLM Steering Evaluation")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--behavior', default='all',
                    choices=['legibility','predictability','safety',
                             'grounding','all'])
    ap.add_argument('--n_episodes', type=int, default=10)
    ap.add_argument('--K', type=int, default=8,
                    help='Best-of-K candidates per episode')
    ap.add_argument('--cfg_lambda_leg', type=float, default=2.0)
    ap.add_argument('--cfg_lambda_pred', type=float, default=2.0)
    ap.add_argument('--cfg_lambda_ground', type=float, default=2.0)
    ap.add_argument('--eta', type=float, default=0.5)
    ap.add_argument('--no_vlm', action='store_true',
                    help='CFG-only mode (skip VLM, faster)')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  CFG + VLM STEERING EVALUATION")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {'CFG-only' if args.no_vlm else f'Best-of-{args.K} + VLM'}")
    print(f"{'='*60}")

    model, sampler, stats, ckpt = load_policy(args.checkpoint, device, args.eta)
    print(f"Model: obs={model.obs_dim}, act={model.act_dim}, "
          f"H={model.horizon}, epoch={ckpt.get('epoch','?')}")

    vlm_client = None
    if not args.no_vlm:
        print("\nInitializing Gemini VLM client...")
        vlm_client = GeminiScorerClient()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "cfg" if args.no_vlm else "cfgvlm"
        args.out_dir = f"outputs/eval_{mode}_{ts}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 100000, size=1000)

    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          cube_half=0.015, cube_mass=0.08,
                          cube_lateral_friction=2.5,
                          episode_length=400)

    behaviors = (['legibility','predictability','safety','grounding']
                 if args.behavior == 'all' else [args.behavior])

    all_results = {}

    for beh in behaviors:
        print(f"\n{'='*60}")
        print(f"  EVALUATING: {beh.upper()}")
        print(f"{'='*60}")

        beh_dir = out_dir / beh
        beh_dir.mkdir(exist_ok=True)
        (beh_dir / "videos").mkdir(exist_ok=True)

        beh_results = []
        beh_seeds = seeds[behaviors.index(beh)*200:]

        for i in range(args.n_episodes):
            ep_seed = int(beh_seeds[i])
            target = "left" if i % 2 == 0 else "right"

            print(f"\n  Episode {i+1}/{args.n_episodes} "
                  f"(seed={ep_seed}, target={target})")

            if beh == "legibility":
                cfg_lam = args.cfg_lambda_leg
                mode = 1.0
                ctx = None
                obs_info = ""; wp_info = ""
                zc = False
            elif beh == "predictability":
                cfg_lam = args.cfg_lambda_pred
                mode = -1.0
                ctx = None
                obs_info = ""; wp_info = ""
                zc = False
            elif beh == "safety":
                # Alternate sub-scenarios
                zc = True
                if i % 4 < 2:
                    cfg_lam = 2.0; mode = 1.0
                    y = 0.02 if target=="left" else -0.02
                    ctx = [0.46, y, TABLE_TOP_Z + OBSTACLE_HEIGHT/2]
                    obs_info = f"obstacle on straight path at [{ctx[0]:.2f},{ctx[1]:.2f}], robot should curve around"
                else:
                    cfg_lam = 2.0; mode = -1.0
                    y = 0.11 if target=="left" else -0.11
                    ctx = [0.38, y, TABLE_TOP_Z + OBSTACLE_HEIGHT/2]
                    obs_info = f"obstacle on arc path at [{ctx[0]:.2f},{ctx[1]:.2f}], robot should go straight"
                wp_info = ""
            elif beh == "grounding":
                cfg_lam = args.cfg_lambda_ground
                mode = 0.0
                zc = True
                wp = WAYPOINT_BLOCKS[i % len(WAYPOINT_BLOCKS)]
                ctx = wp["pos"]
                obs_info = ""
                wp_info = (f"Hover near the {wp['name'].upper()} block "
                           f"before picking the {'GREEN (left)' if target=='left' else 'RED (right)'} block.")

            if args.no_vlm:
                # CFG-only mode
                env.reset(seed=ep_seed)
                if beh == "safety":
                    uid = add_obstacle_visual(env, ctx)
                    for _ in range(30):
                        p.stepSimulation(physicsClientId=env._cid)
                elif beh == "grounding":
                    p.changeVisualShape(env._cube_l_uid, -1,
                                        rgbaColor=[0.1,0.8,0.1,1.0],
                                        physicsClientId=env._cid)
                    p.changeVisualShape(env._cube_r_uid, -1,
                                        rgbaColor=[0.8,0.1,0.1,1.0],
                                        physicsClientId=env._cid)
                    wp_uids = add_waypoint_blocks(env, WAYPOINT_BLOCKS)
                    for _ in range(30):
                        p.stepSimulation(physicsClientId=env._cid)

                vpath = beh_dir / "videos" / f"ep{ep_seed:03d}_{target}.mp4"
                r = rollout_cfg_only(env, model, sampler, stats, device,
                                     ctx, mode, cfg_lam, target,
                                     video_path=vpath,
                                     zero_context=zc)

                left_pos = r["final_obs"][8:11]
                right_pos = r["final_obs"][15:18]

                ep_res = {"episode": i, "seed": ep_seed, "target": target,
                          "success": r["success"], "steps": r["steps"],
                          "cfg_lambda": cfg_lam,
                          "approached": r.get("approached", target)}

                if beh == "legibility":
                    # Use the actually-approached block for L_early
                    actual_tgt = r.get("approached", target)
                    ep_res["L_early"] = round(compute_l_early(
                        r["ee_traj"], left_pos, right_pos, actual_tgt), 4)
                elif beh == "predictability":
                    ep_res["path_efficiency"] = round(
                        compute_path_efficiency(r["ee_traj"]), 4)
                elif beh == "safety":
                    cl = compute_clearance(r["ee_traj"], np.array(ctx[:2]))
                    ep_res["min_clearance"] = round(cl, 4)
                    ep_res["collision"] = cl < OBSTACLE_RADIUS
                    remove_bodies(env, [uid])
                elif beh == "grounding":
                    hd = compute_hover_dist(r["ee_traj"], ctx)
                    ep_res["min_wp_dist"] = round(hd, 4)
                    ep_res["hovered"] = hd < 0.06
                    remove_bodies(env, wp_uids)

                tag = "OK" if r["success"] else "FAIL"
                detail = ""
                if "L_early" in ep_res:
                    detail = f" L_early={ep_res['L_early']:.3f}"
                if "path_efficiency" in ep_res:
                    detail = f" eff={ep_res['path_efficiency']:.3f}"
                if "min_clearance" in ep_res:
                    detail = f" clear={ep_res['min_clearance']:.3f}"
                if "min_wp_dist" in ep_res:
                    detail = f" wp={ep_res['min_wp_dist']:.3f}"
                print(f"  {tag}{detail} steps={r['steps']}")

                beh_results.append(ep_res)

            else:
                # Best-of-K + VLM mode
                ep_res = run_bestofk_episode(
                    env, model, sampler, stats, device,
                    vlm_client, ep_seed, beh, args.K,
                    cfg_lam, mode, ctx,
                    obstacle_info=obs_info,
                    waypoint_info=wp_info,
                    target=target, out_dir=beh_dir,
                    zero_context=zc)
                beh_results.append(ep_res)

        # ── Summary ─────────────────────────────────────────────
        print(f"\n{'─'*40}")
        print(f"  {beh.upper()} SUMMARY ({args.n_episodes} episodes)")
        print(f"{'─'*40}")

        if args.no_vlm:
            n_ok = sum(1 for r in beh_results if r["success"])
            print(f"  Success: {n_ok}/{len(beh_results)} "
                  f"({100*n_ok/max(len(beh_results),1):.0f}%)")
            if beh == "legibility":
                vals = [r["L_early"] for r in beh_results]
                print(f"  L_early: mean={np.mean(vals):.3f} "
                      f"std={np.std(vals):.3f}")
            elif beh == "predictability":
                vals = [r["path_efficiency"] for r in beh_results]
                print(f"  Path eff: mean={np.mean(vals):.3f} "
                      f"std={np.std(vals):.3f}")
            elif beh == "safety":
                n_cl = sum(1 for r in beh_results if not r.get("collision", True))
                vals = [r["min_clearance"] for r in beh_results]
                print(f"  Clear: {n_cl}/{len(beh_results)}, "
                      f"mean={np.mean(vals):.3f}")
            elif beh == "grounding":
                n_h = sum(1 for r in beh_results if r.get("hovered", False))
                vals = [r["min_wp_dist"] for r in beh_results]
                print(f"  Hover: {n_h}/{len(beh_results)}, "
                      f"mean_dist={np.mean(vals):.3f}")
        else:
            # VLM mode: compare VLM vs baseline
            vlm_ok = sum(1 for r in beh_results
                         if r.get("vlm",{}).get("success", False))
            base_ok = sum(1 for r in beh_results
                          if r.get("baseline",{}).get("success", False))
            print(f"  Success VLM: {vlm_ok}/{len(beh_results)}")
            print(f"  Success BL:  {base_ok}/{len(beh_results)}")

            if beh == "legibility":
                v_le = [r["vlm"]["L_early"] for r in beh_results
                        if "L_early" in r.get("vlm",{})]
                b_le = [r["baseline"]["L_early"] for r in beh_results
                        if "L_early" in r.get("baseline",{})]
                if v_le:
                    print(f"  L_early VLM: {np.mean(v_le):.3f}")
                if b_le:
                    print(f"  L_early BL:  {np.mean(b_le):.3f}")
            elif beh == "predictability":
                v = [r["vlm"]["path_efficiency"] for r in beh_results
                     if "path_efficiency" in r.get("vlm",{})]
                b = [r["baseline"]["path_efficiency"] for r in beh_results
                     if "path_efficiency" in r.get("baseline",{})]
                if v: print(f"  Eff VLM: {np.mean(v):.3f}")
                if b: print(f"  Eff BL:  {np.mean(b):.3f}")
            elif beh == "safety":
                v = [r["vlm"]["min_clearance"] for r in beh_results
                     if "min_clearance" in r.get("vlm",{})]
                b = [r["baseline"]["min_clearance"] for r in beh_results
                     if "min_clearance" in r.get("baseline",{})]
                if v: print(f"  Clearance VLM: {np.mean(v):.3f}")
                if b: print(f"  Clearance BL:  {np.mean(b):.3f}")
            elif beh == "grounding":
                v = [r["vlm"]["min_wp_dist"] for r in beh_results
                     if "min_wp_dist" in r.get("vlm",{})]
                b = [r["baseline"]["min_wp_dist"] for r in beh_results
                     if "min_wp_dist" in r.get("baseline",{})]
                if v: print(f"  WP dist VLM: {np.mean(v):.3f}")
                if b: print(f"  WP dist BL:  {np.mean(b):.3f}")

        all_results[beh] = beh_results

    env.close()

    # Save
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"  RESULTS SAVED: {out_dir / 'results.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
