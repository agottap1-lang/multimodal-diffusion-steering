#!/usr/bin/env python3
"""
Best-of-K VLM Behavior Evaluation v2 — Self-contained, no external deps.
=========================================================================

Fixes over v1:
  1. Stochastic DDIM (η>0) for truly diverse K candidates
  2. No gemini_vlm_eval dependency — VLM calls inlined via google.genai
  3. --render_only mode for visual verification (no VLM needed)
  4. Candidate diversity tracking
  5. Cleaner error handling

Usage:
  # Step 1: Render-only to verify safety/grounding visuals
  .venv\\Scripts\\python.exe evaluation/eval_behaviors_v2.py ^
      --checkpoint runs/diffusion_20260402_072747/ckpt_ep100.pt ^
      --behavior safety --render_only --n_episodes 3

  # Step 2: Full VLM eval (after verifying videos look right)
  .venv\\Scripts\\python.exe evaluation/eval_behaviors_v2.py ^
      --checkpoint runs/diffusion_20260402_072747/ckpt_ep100.pt ^
      --behavior legibility --n_episodes 20 --K 8 ^
      --api_key YOUR_KEY --vlm_model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import torch.nn as nn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.twoblockpick_env import TwoBlockPickEnv

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOAL_A_DESC = "pick the LEFT block"
GOAL_B_DESC = "pick the RIGHT block"
FRAME_STEPS = [0, 30, 60, 90, 120, 149]  # capture at ~0,1,2,3,4,5 sec


# ═══════════════════════════════════════════════════════════════════════════
# MODEL (identical architecture to training — do not modify)
# ═══════════════════════════════════════════════════════════════════════════

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
        self.shortcut = (nn.Linear(in_dim, out_dim)
                         if in_dim != out_dim else nn.Identity())
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.act = nn.Mish()

    def forward(self, x, t_emb):
        h = self.act(
            self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2)
            + self.time_proj(t_emb).unsqueeze(1))
        h = self.act(
            self.norm2(self.conv2(h).transpose(1, 2)).transpose(1, 2)
            + self.shortcut(x))
        return h


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128), nn.Linear(128, hidden_dim),
            nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i + 1], hidden_dim)
            for i in range(len(dims) - 1)])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i + 1] + dims[i + 1], dims[i], hidden_dim)
            for i in range(len(dims) - 2, -1, -1)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, act_dim))

    def forward(self, noisy_act, timestep, obs):
        t_emb = self.time_mlp(timestep)
        x = self.input_proj(noisy_act) + self.obs_embed(obs).unsqueeze(1)
        skips = []
        for blk in self.encoder_blocks:
            x = blk(x, t_emb)
            skips.append(x)
        x = self.bottleneck(x, t_emb)
        for blk, sk in zip(self.decoder_blocks, reversed(skips)):
            x = blk(torch.cat([x, sk], dim=-1), t_emb)
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════════
# STOCHASTIC DDIM SAMPLER  (η > 0 → diverse candidates)
# ═══════════════════════════════════════════════════════════════════════════

class StochasticDDIMSampler:
    """DDIM with tunable stochasticity via η.

    η = 0.0  →  deterministic DDIM (all candidates near-identical)
    η = 1.0  →  equivalent to DDPM  (maximum diversity)
    η ∈ [0.5, 0.8]  →  good balance of quality + diversity
    """

    def __init__(self, n_steps, beta_start, beta_end, device, eta=0.8):
        self.device = device
        self.eta = eta
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs, n_sampling_steps=15):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)

        timesteps = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod) - 1,
                           n_sampling_steps, device=self.device).long(), [0])

        for i, t in enumerate(timesteps):
            t_b = t.repeat(B)
            eps_pred = model(x, t_b, obs)

            a_t = self.alphas_cumprod[t]
            a_prev = (self.alphas_cumprod[timesteps[i + 1]]
                      if i < len(timesteps) - 1
                      else torch.tensor(1.0, device=self.device))

            # Predict clean sample
            x0_pred = (x - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)

            if i < len(timesteps) - 1:
                # Stochastic DDIM noise injection
                sigma = self.eta * torch.sqrt(
                    (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
                dir_xt = torch.sqrt(
                    torch.clamp(1 - a_prev - sigma ** 2, min=0)) * eps_pred
                noise = torch.randn_like(x) * sigma
                x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
            else:
                x = x0_pred

        return x


# ═══════════════════════════════════════════════════════════════════════════
# LOAD POLICY
# ═══════════════════════════════════════════════════════════════════════════

def load_policy(ckpt_path: str, device: torch.device, eta: float = 0.8):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sampler = StochasticDDIMSampler(
        n_steps=cfg.get("n_diffusion_steps", 100),
        beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02),
        device=device,
        eta=eta,
    )

    obs_mean = torch.tensor(ckpt["obs_mean"], dtype=torch.float32).numpy()
    obs_std = torch.tensor(ckpt["obs_std"], dtype=torch.float32).numpy()
    act_mean = torch.tensor(ckpt["act_mean"], dtype=torch.float32).numpy()
    act_std = torch.tensor(ckpt["act_std"], dtype=torch.float32).numpy()

    return model, sampler, obs_mean, obs_std, act_mean, act_std, cfg


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP PER BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════

def setup_safety_env(env) -> dict:
    """Add a prominent red cylinder obstacle between EE home and blocks."""
    cid = env._cid
    obs_pos = [0.45, 0.0, 0.42]
    obs_radius = 0.03
    obs_height = 0.10
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obs_radius,
                                 height=obs_height, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obs_radius,
                              length=obs_height,
                              rgbaColor=[0.9, 0.1, 0.1, 1.0],
                              physicsClientId=cid)
    uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis,
                            basePosition=obs_pos, physicsClientId=cid)
    return {"obstacle_uid": uid,
            "obstacle_pos": np.array(obs_pos),
            "obstacle_radius": obs_radius}


def setup_grounding_env(env) -> dict:
    """Color blocks: GREEN (left), RED (right). Add BLUE waypoint."""
    cid = env._cid
    p.changeVisualShape(env._cube_l_uid, -1,
                        rgbaColor=[0.1, 0.8, 0.1, 1.0],
                        physicsClientId=cid)
    p.changeVisualShape(env._cube_r_uid, -1,
                        rgbaColor=[0.8, 0.1, 0.1, 1.0],
                        physicsClientId=cid)
    wp_pos = [0.48, 0.03, 0.421]
    col = p.createCollisionShape(p.GEOM_BOX,
                                 halfExtents=[0.015, 0.015, 0.015],
                                 physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_BOX,
                              halfExtents=[0.015, 0.015, 0.015],
                              rgbaColor=[0.1, 0.1, 0.9, 1.0],
                              physicsClientId=cid)
    wp_uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=wp_pos, physicsClientId=cid)
    instruction = ("Pass near the BLUE block first, "
                   "then pick up the GREEN block (left).")
    return {"waypoint_uid": wp_uid, "waypoint_pos": np.array(wp_pos),
            "instruction": instruction}


def setup_env_for_behavior(env, behavior: str) -> dict:
    if behavior == "safety":
        return setup_safety_env(env)
    elif behavior == "grounding":
        return setup_grounding_env(env)
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def capture_jpeg(env, quality=85) -> bytes:
    from PIL import Image
    rgb = env.render(width=640, height=480)
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def compute_l_early(ee_traj: np.ndarray, obs_final: np.ndarray,
                    early_frac: float = 0.3) -> dict:
    if len(ee_traj) < 4:
        return {"L_early": 0.5, "true_goal": "unknown"}
    goals = np.array([obs_final[8:11], obs_final[15:18]])
    T = len(ee_traj)
    early_end = max(2, int(T * early_frac))
    sigma = np.linalg.norm(goals[0] - goals[1]) / (2 * np.sqrt(2 * np.log(2)))
    posteriors = []
    for t in range(early_end):
        d0 = np.linalg.norm(ee_traj[t] - goals[0])
        d1 = np.linalg.norm(ee_traj[t] - goals[1])
        l0 = np.exp(-d0 ** 2 / (2 * sigma ** 2))
        l1 = np.exp(-d1 ** 2 / (2 * sigma ** 2))
        total = l0 + l1 + 1e-12
        posteriors.append([l0 / total, l1 / total])
    posteriors = np.array(posteriors)
    mean_p0 = posteriors[:, 0].mean()
    mean_p1 = posteriors[:, 1].mean()
    if mean_p0 >= mean_p1:
        return {"L_early": float(mean_p0), "true_goal": "left"}
    else:
        return {"L_early": float(mean_p1), "true_goal": "right"}


def compute_diversity(candidates: list) -> dict:
    """Pairwise L2 distance between early action sequences."""
    K = len(candidates)
    if K < 2:
        return {"mean_l2": 0.0, "max_l2": 0.0, "min_l2": 0.0}
    n = min(50, min(len(c["actions"]) for c in candidates))
    trajs = [c["actions"][:n].flatten() for c in candidates]
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(float(np.linalg.norm(trajs[i] - trajs[j])))
    return {"mean_l2": float(np.mean(dists)),
            "max_l2": float(np.max(dists)),
            "min_l2": float(np.min(dists))}


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATE ONE CANDIDATE (150 steps = 5 sec at 30 Hz)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_candidate(env, model, sampler, obs, obs_mean, obs_std,
                       act_mean, act_std, device, seed: int,
                       n_steps: int = 150):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    queue = deque()
    frames_bytes = []
    ee_traj = []
    actions_all = []
    current_obs = obs.copy()

    for step in range(n_steps):
        ee_traj.append(current_obs[:3].copy())

        if len(queue) == 0:
            obs_norm = (current_obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            act_seq = sampler.sample(model, obs_t, n_sampling_steps=15)
            act_np = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in act_np:
                queue.append(a)

        action = queue.popleft()
        actions_all.append(action.copy())

        if step in FRAME_STEPS:
            frames_bytes.append(capture_jpeg(env))

        result = env.step(action)
        current_obs = result.obs
        if result.done:
            while len(actions_all) < n_steps:
                actions_all.append(action.copy())
                ee_traj.append(current_obs[:3].copy())
            break

    return {"actions": np.array(actions_all),
            "frames": frames_bytes,
            "ee_trajectory": np.array(ee_traj),
            "seed": seed}


# ═══════════════════════════════════════════════════════════════════════════
# VLM SCORING (inline — no external library)
# ═══════════════════════════════════════════════════════════════════════════

def _vlm_call(client, model_name: str, frames: list, prompt: str,
              retries: int = 3) -> str | None:
    """Send frames + prompt to Gemini, return raw text or None."""
    from google.genai import types
    parts = []
    for fb in frames:
        parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
    parts.append(prompt)

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model_name, contents=parts)
            return resp.text.strip()
        except Exception as e:
            print(f"      VLM attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _parse_json(text: str | None) -> dict:
    """Parse VLM JSON response, handling markdown fences."""
    if text is None:
        return {}
    clean = text
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


def vlm_score_legibility(client, model_name, frames, video_id) -> dict:
    prompt = (
        "You are evaluating a robot arm's trajectory in a tabletop pick task.\n"
        "There are exactly two blocks on the table:\n"
        "  Goal A: pick the LEFT block\n"
        "  Goal B: pick the RIGHT block\n\n"
        "You are shown 6 frames (at t=0,1,2,3,4,5 seconds) of the robot's motion.\n"
        "Based on the arm's path shape, lateral curvature, and approach direction,\n"
        "judge which goal the robot is heading towards.\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"pA": <float 0-1>, "pB": <float 0-1>, '
        '"choice": "A" or "B", "cue": "<brief reason>"}\n\n'
        "pA + pB must sum to 1.0. Higher = more confident."
    )
    data = _parse_json(_vlm_call(client, model_name, frames, prompt))
    pA = float(data.get("pA", 0.5))
    pB = float(data.get("pB", 0.5))
    return {"pA": pA, "pB": pB,
            "choice": data.get("choice", "?"),
            "cue": data.get("cue", "VLM_FAIL"),
            "score": max(pA, pB)}


def vlm_score_predictability(client, model_name, frames, video_id) -> dict:
    prompt = (
        "You are evaluating a robot arm's trajectory for PREDICTABILITY.\n"
        "The robot is performing a tabletop pick task with two blocks.\n\n"
        "You see 6 frames (t=0,1,2,3,4,5s).\n"
        "Rate the trajectory's predictability from 0 to 1:\n"
        "  1.0 = Very direct, efficient, easy to predict next position\n"
        "  0.5 = Somewhat predictable with minor detours\n"
        "  0.0 = Highly erratic, unpredictable\n\n"
        "Consider: straightness of path, smoothness, consistency.\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"score": <float 0-1>, "cue": "<brief reason>"}'
    )
    data = _parse_json(_vlm_call(client, model_name, frames, prompt))
    return {"pA": 0.5, "pB": 0.5, "choice": "?",
            "cue": data.get("cue", "VLM_FAIL"),
            "score": float(data.get("score", 0.5))}


def vlm_score_safety(client, model_name, frames, video_id,
                     obstacle_pos=None) -> dict:
    prompt = (
        "You are evaluating a robot arm's trajectory for SAFETY.\n"
        "The robot picks a block from a table. There is a RED CYLINDER\n"
        "obstacle on the table that the robot must avoid.\n\n"
        "You see 6 frames (t=0,1,2,3,4,5s).\n"
        "Rate obstacle-avoidance safety from 0 to 1:\n"
        "  1.0 = Clearly avoids the red obstacle, maintains distance\n"
        "  0.5 = Gets somewhat close but no collision\n"
        "  0.0 = Passes through or very close to the red obstacle\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"score": <float 0-1>, "cue": "<brief reason>"}'
    )
    data = _parse_json(_vlm_call(client, model_name, frames, prompt))
    return {"pA": 0.5, "pB": 0.5, "choice": "?",
            "cue": data.get("cue", "VLM_FAIL"),
            "score": float(data.get("score", 0.5))}


def vlm_score_grounding(client, model_name, frames, video_id,
                        instruction: str = "") -> dict:
    prompt = (
        "You are evaluating a robot arm for SPATIAL INSTRUCTION FOLLOWING.\n"
        "The scene has: GREEN block (left), RED block (right), "
        "BLUE block (center waypoint).\n\n"
        f'Instruction: "{instruction}"\n\n'
        "You see 6 frames (t=0,1,2,3,4,5s).\n"
        "Rate how well the trajectory follows the instruction:\n"
        "  1.0 = Clearly approaches blue waypoint first, then picks green block\n"
        "  0.5 = Picks correct block but ignores waypoint\n"
        "  0.0 = Wrong block or completely ignores instruction\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"score": <float 0-1>, "cue": "<brief reason>"}'
    )
    data = _parse_json(_vlm_call(client, model_name, frames, prompt))
    return {"pA": 0.5, "pB": 0.5, "choice": "?",
            "cue": data.get("cue", "VLM_FAIL"),
            "score": float(data.get("score", 0.5))}


def score_candidate(client, model_name, frames, video_id, behavior,
                    env_meta=None) -> dict:
    if behavior == "legibility":
        return vlm_score_legibility(client, model_name, frames, video_id)
    elif behavior == "predictability":
        return vlm_score_predictability(client, model_name, frames, video_id)
    elif behavior == "safety":
        obs_pos = env_meta.get("obstacle_pos") if env_meta else None
        return vlm_score_safety(client, model_name, frames, video_id, obs_pos)
    elif behavior == "grounding":
        inst = env_meta.get("instruction", "") if env_meta else ""
        return vlm_score_grounding(client, model_name, frames, video_id, inst)
    raise ValueError(f"Unknown behavior: {behavior}")


# ═══════════════════════════════════════════════════════════════════════════
# RUN ONE EPISODE
# ═══════════════════════════════════════════════════════════════════════════

def _execute_trajectory(env, obs_init, actions, model, sampler,
                        obs_mean, obs_std, act_mean, act_std, device,
                        max_steps, video_path=None):
    """Execute candidate actions → replan when exhausted → record video."""
    if video_path:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        env.record_video(video_path, width=640, height=480, fps=30)

    queue = deque(actions)
    obs = obs_init.copy()
    ee_full = []
    success = False
    steps = 0
    result = None

    while steps < max_steps:
        ee_full.append(obs[:3].copy())
        if len(queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            act_seq = sampler.sample(model, obs_t, n_sampling_steps=15)
            act_np = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in act_np:
                queue.append(a)
        action = queue.popleft()
        result = env.step(action)
        obs = result.obs
        steps += 1
        s_l = result.info.get("success_left", 0) > 0.5
        s_r = result.info.get("success_right", 0) > 0.5
        success = s_l or s_r
        if result.done:
            break

    if video_path:
        env.stop_video()

    picked = "none"
    if result is not None:
        if result.info.get("success_left", 0) > 0.5:
            picked = "left"
        elif result.info.get("success_right", 0) > 0.5:
            picked = "right"

    return {"success": success, "picked": picked, "steps": steps,
            "ee_trajectory": np.array(ee_full), "final_obs": obs}


def run_episode(model, sampler, obs_mean, obs_std, act_mean, act_std,
                device, episode_seed: int, behavior: str, K: int,
                max_steps: int = 600, out_dir: Path | None = None,
                client=None, model_name: str = None,
                render_only: bool = False, sleep: float = 1.0) -> dict:
    """Run one Best-of-K episode."""
    env = TwoBlockPickEnv(render=False, episode_length=max_steps,
                          cube_jitter=0.0)
    obs_init = env.reset(seed=episode_seed)

    env_meta = setup_env_for_behavior(env, behavior)

    # Settle physics after env mods
    for _ in range(60):
        p.stepSimulation(physicsClientId=env._cid)
    obs_init = env._get_obs()

    saved_state = p.saveState(physicsClientId=env._cid)

    # ── Helper to restore env state ──
    def restore():
        p.restoreState(saved_state, physicsClientId=env._cid)
        env._target_pos = obs_init[:3].copy()
        env._target_yaw = 0.0
        env._grip_cmd = 1.0
        env._episode_steps = 0

    # ═══ RENDER-ONLY MODE ═══════════════════════════════════
    if render_only:
        restore()
        vpath = None
        if out_dir:
            vpath = str(out_dir / "videos"
                        / f"ep{episode_seed:03d}_{behavior}.mp4")
        torch.manual_seed(episode_seed)
        exec_result = _execute_trajectory(
            env, obs_init, [], model, sampler,
            obs_mean, obs_std, act_mean, act_std, device,
            max_steps, video_path=vpath)

        l_early = compute_l_early(exec_result["ee_trajectory"],
                                  exec_result["final_obs"])
        status = "OK" if exec_result["success"] else "FAIL"
        print(f"  {status}  picked={exec_result['picked']}  "
              f"L_early={l_early['L_early']:.4f}  steps={exec_result['steps']}")

        p.removeState(saved_state, physicsClientId=env._cid)
        env.close()
        return {"episode_seed": episode_seed, "behavior": behavior,
                "render_only": True, "success": exec_result["success"],
                "picked": exec_result["picked"],
                "steps": exec_result["steps"],
                "L_early": l_early["L_early"],
                "video_path": vpath}

    # ═══ FULL BEST-OF-K ════════════════════════════════════
    # Phase 1: Generate K candidates
    candidates = []
    for k in range(K):
        restore()
        c_seed = episode_seed * 1000 + k * 137 + 1
        cand = simulate_candidate(
            env, model, sampler, obs_init, obs_mean, obs_std,
            act_mean, act_std, device, seed=c_seed)
        cand["idx"] = k
        candidates.append(cand)
        print(f"    Candidate {k + 1}/{K} simulated", end="", flush=True)
    print()

    # Diversity check
    diversity = compute_diversity(candidates)
    print(f"    Diversity: mean_L2={diversity['mean_l2']:.2f}  "
          f"min={diversity['min_l2']:.2f}  max={diversity['max_l2']:.2f}")

    # Phase 2: VLM scoring
    for k, cand in enumerate(candidates):
        vid_id = f"ep{episode_seed}_c{k}_{behavior}"
        vlm_result = score_candidate(
            client, model_name, cand["frames"], vid_id, behavior, env_meta)
        cand["vlm"] = vlm_result
        print(f"    VLM c{k}: score={vlm_result['score']:.3f}  "
              f"cue=\"{vlm_result['cue'][:60]}\"")
        time.sleep(sleep)

    # Feasibility bonus
    block_l = obs_init[8:11]
    block_r = obs_init[15:18]
    for cand in candidates:
        final_ee = cand["ee_trajectory"][-1]
        d_l = np.linalg.norm(final_ee[:2] - block_l[:2])
        d_r = np.linalg.norm(final_ee[:2] - block_r[:2])
        closest = min(d_l, d_r)
        feasibility = max(0, 0.05 * (1.0 - closest / 0.2))
        cand["vlm"]["combined_score"] = cand["vlm"]["score"] + feasibility

    # Phase 3: Select best + worst
    vlm_selected = max(range(K),
                       key=lambda i: candidates[i]["vlm"]["combined_score"])
    sorted_idx = sorted(range(K),
                        key=lambda i: candidates[i]["vlm"]["combined_score"])
    baseline_idx = sorted_idx[0]
    if baseline_idx == vlm_selected and K > 1:
        baseline_idx = sorted_idx[1]

    print(f"  -> VLM pick: c{vlm_selected} "
          f"(score={candidates[vlm_selected]['vlm']['score']:.3f})")
    print(f"  -> Baseline: c{baseline_idx} "
          f"(score={candidates[baseline_idx]['vlm']['score']:.3f})")

    # Phase 4: Execute both and record video
    results = {}
    for label, sel_idx in [("vlm", vlm_selected), ("baseline", baseline_idx)]:
        restore()
        vpath = None
        if out_dir:
            vpath = str(out_dir / "videos"
                        / f"ep{episode_seed:03d}_{label}_{behavior}.mp4")
        exec_result = _execute_trajectory(
            env, obs_init, candidates[sel_idx]["actions"],
            model, sampler, obs_mean, obs_std, act_mean, act_std,
            device, max_steps, video_path=vpath)

        l_early = compute_l_early(exec_result["ee_trajectory"],
                                  exec_result["final_obs"])
        results[label] = {
            "selected_idx": sel_idx,
            "success": exec_result["success"],
            "picked": exec_result["picked"],
            "steps": exec_result["steps"],
            "L_early": l_early["L_early"],
            "true_goal": l_early["true_goal"],
            "vlm_score": candidates[sel_idx]["vlm"]["score"],
            "vlm_cue": candidates[sel_idx]["vlm"]["cue"],
            "video_path": vpath,
        }
        status = "OK" if exec_result["success"] else "FAIL"
        print(f"  {label.upper()}: {status}  picked={exec_result['picked']}  "
              f"L_early={l_early['L_early']:.4f}  steps={exec_result['steps']}")

    p.removeState(saved_state, physicsClientId=env._cid)
    env.close()

    return {
        "episode_seed": episode_seed,
        "behavior": behavior,
        "K": K,
        "diversity": diversity,
        "vlm": results["vlm"],
        "baseline": results["baseline"],
        "candidates": [
            {"idx": c["idx"], "seed": c["seed"],
             "vlm_score": c["vlm"]["score"],
             "vlm_cue": c["vlm"]["cue"][:100]}
            for c in candidates],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("Best-of-K VLM Behavior Eval v2")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--behavior", required=True,
                    choices=["legibility", "predictability",
                             "safety", "grounding"])
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.8,
                    help="Stochastic DDIM η (0=deterministic, 1=DDPM)")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--render_only", action="store_true",
                    help="Just render videos, skip VLM scoring")
    ap.add_argument("--vlm_model", type=str, default="gemini-2.5-pro",
                    help="Gemini model name for VLM scoring")
    ap.add_argument("--api_key", type=str, default=None,
                    help="Gemini API key (overrides GEMINI_API_KEY env)")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds between VLM calls (rate-limit)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, sampler, obs_mean, obs_std, act_mean, act_std, cfg = \
        load_policy(args.checkpoint, device, eta=args.eta)
    print(f"Policy: {args.checkpoint}")
    print(f"  horizon={cfg['horizon']}, act_dim={cfg['act_dim']}, "
          f"eta={args.eta}")

    # VLM client (only if not render_only)
    client, model_name = None, None
    if not args.render_only:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: Provide --api_key or set GEMINI_API_KEY env var.")
            sys.exit(1)
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = args.vlm_model
        print(f"VLM: {model_name}")

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        suffix = "render" if args.render_only else "eval"
        out_dir = Path(f"outputs/behaviors/{args.behavior}_{suffix}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)

    mode_str = "RENDER-ONLY" if args.render_only else f"BEST-of-{args.K}"
    print(f"\n{'=' * 60}")
    print(f"  {args.behavior.upper()} -- {mode_str}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}\n")

    all_results = []
    for ep in range(args.n_episodes):
        ep_seed = ep * 7 + 42
        print(f"\n-- Episode {ep + 1}/{args.n_episodes} (seed={ep_seed}) --")
        r = run_episode(
            model, sampler, obs_mean, obs_std, act_mean, act_std,
            device, ep_seed, args.behavior,
            K=1 if args.render_only else args.K,
            out_dir=out_dir, client=client, model_name=model_name,
            render_only=args.render_only, sleep=args.sleep)
        all_results.append(r)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    if args.render_only:
        n_ok = sum(1 for r in all_results if r["success"])
        print(f"  RENDER DONE: {n_ok}/{len(all_results)} success")
        print(f"  Videos in: {out_dir / 'videos'}")
        summary = {"behavior": args.behavior, "render_only": True,
                   "n_success": n_ok, "n_episodes": len(all_results)}
    else:
        vlm_ok = [r["vlm"]["success"] for r in all_results]
        bl_ok = [r["baseline"]["success"] for r in all_results]
        vlm_le = [r["vlm"]["L_early"] for r in all_results]
        bl_le = [r["baseline"]["L_early"] for r in all_results]
        vlm_sc = [r["vlm"]["vlm_score"] for r in all_results]
        bl_sc = [r["baseline"]["vlm_score"] for r in all_results]

        from scipy import stats
        diffs = np.array(vlm_le) - np.array(bl_le)
        if np.std(diffs) > 1e-8:
            t_stat, p_val = stats.ttest_rel(vlm_le, bl_le)
        else:
            t_stat, p_val = 0.0, 1.0

        summary = {
            "behavior": args.behavior,
            "n_episodes": args.n_episodes,
            "K": args.K, "eta": args.eta,
            "vlm_model": args.vlm_model,
            "vlm_success_rate": float(np.mean(vlm_ok)),
            "baseline_success_rate": float(np.mean(bl_ok)),
            "vlm_L_early_mean": float(np.mean(vlm_le)),
            "vlm_L_early_std": float(np.std(vlm_le)),
            "baseline_L_early_mean": float(np.mean(bl_le)),
            "baseline_L_early_std": float(np.std(bl_le)),
            "vlm_score_mean": float(np.mean(vlm_sc)),
            "baseline_score_mean": float(np.mean(bl_sc)),
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(p_val),
        }
        print(f"  RESULTS: {args.behavior.upper()}")
        print(f"  VLM:      success={summary['vlm_success_rate']:.0%}  "
              f"L_early={summary['vlm_L_early_mean']:.4f}")
        print(f"  Baseline: success={summary['baseline_success_rate']:.0%}  "
              f"L_early={summary['baseline_L_early_mean']:.4f}")
        print(f"  VLM score: {summary['vlm_score_mean']:.4f} vs "
              f"{summary['baseline_score_mean']:.4f}  "
              f"(p={summary['paired_ttest_p']:.4f})")
    print(f"{'=' * 60}")

    output = {"summary": summary, "episodes": all_results}
    rpath = out_dir / "results.json"
    with open(rpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {rpath}")


if __name__ == "__main__":
    main()
