#!/usr/bin/env python3
"""
vlm_pairwise_eval.py  —  Correct VLM behavioral evaluation via pairwise comparison

Design:
  For each episode, generate TWO trajectories from the SAME start state:
    CONDITIONED : CFG-guided  (cfg_lambda=2.0, behavior mode ON)
    BASELINE    : No guidance (cfg_lambda=0.0, behavior mode zeroed)

  Send BOTH to the VLM (side-by-side, labeled A/B with order randomized).
  Ask a FORCED BINARY CHOICE: "Which trajectory better demonstrates [behavior]?"

  Ground truth: conditioned trajectory should always win.
  Metric: accuracy over N episodes (chance = 50%).

Improvements over cross_model_comparison.py:
  - Real behavioral contrast (conditioned vs baseline), not noise-seed variants
  - Forced binary choice eliminates grade inflation
  - Frames start at step 15 (skip identical step-0 frame)
  - 320×320 resolution (was 240×240)
  - Randomized A/B assignment eliminates position bias
  - 40 API calls total (was 320)
  - Clean binary accuracy metric with interpretable chance baseline

Usage:
  .venv\\Scripts\\python.exe evaluation/vlm_pairwise_eval.py ^
      --checkpoint runs/cfg_20260406_005407/ckpt_ep200.pt ^
      --openai_key sk-proj-... ^
      [--anthropic_key sk-ant-...] ^
      [--behavior all|legibility|predictability|safety|grounding] ^
      [--n_episodes 10] ^
      [--sleep 1.0]
"""

import argparse, json, math, io, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pybullet as p
import base64

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# ── Constants ──────────────────────────────────────────────────────────
OBS_DIM_V2    = 26
ACT_DIM       = 5
CFG_COND_START = 22
CFG_MODE_DIM  = 25
TABLE_TOP_Z   = 0.4
OBSTACLE_RADIUS = 0.035
OBSTACLE_HEIGHT = 0.18
OBSTACLE_COLOR  = [0.0, 0.85, 0.85, 1]

# Frames at these steps — step 0 is SKIPPED (identical for all trajectories)
FRAME_STEPS = [15, 40, 70, 100, 135]
FRAME_RES   = 320    # pixel width/height
FRAME_QUAL  = 85     # JPEG quality

BEHAVIORS = ["legibility", "predictability", "safety", "grounding"]

# Episode seeds per behavior — same 10 used in Gemini eval
EPISODE_SEEDS = {
    "legibility":    [8925, 77395, 65457, 43887, 43301, 85859, 8594, 69736, 20146, 9417],
    "predictability":[36626,90858, 49506, 69970, 45741, 26586, 76422,96917, 26352, 77875],
    "safety":        [99802,77740, 32041, 97182, 49930, 50074, 43369,14389, 93437, 1393],
    "grounding":     [61628,77899, 73904, 13455, 82605, 53606, 32308,51422, 96698, 85757],
}
EPISODE_TARGETS = {
    "legibility":    ["left","right","left","right","left","right","left","right","left","right"],
    "predictability":["left","right","left","right","left","right","left","right","left","right"],
    "safety":        ["left","right","left","right","left","right","left","right","left","right"],
    "grounding":     ["left","right","left","right","left","right","left","right","left","right"],
}


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
# EPISODE CONTEXT
# ══════════════════════════════════════════════════════════════════════

def get_episode_context(behavior: str, ep_index: int, target: str) -> dict:
    if behavior == "legibility":
        return dict(cfg_lambda=2.0, behavior_mode=1.0,
                    context_pos=None, zero_context=False,
                    obstacle_info=None, waypoint_info=None)

    elif behavior == "predictability":
        return dict(cfg_lambda=2.0, behavior_mode=-1.0,
                    context_pos=None, zero_context=False,
                    obstacle_info=None, waypoint_info=None)

    elif behavior == "safety":
        if ep_index % 4 < 2:
            y = 0.02 if target == "left" else -0.02
            ctx = [0.46, y, TABLE_TOP_Z + OBSTACLE_HEIGHT / 2]
            cfg_lam = 2.0
            mode = 1.0
            obs_info = (f"cyan cylinder at table position "
                        f"x={ctx[0]:.2f} y={ctx[1]:.2f} "
                        f"(on the straight-line path to the block)")
        else:
            y = 0.11 if target == "left" else -0.11
            ctx = [0.38, y, TABLE_TOP_Z + OBSTACLE_HEIGHT / 2]
            cfg_lam = 2.0
            mode = -1.0
            obs_info = (f"cyan cylinder at table position "
                        f"x={ctx[0]:.2f} y={ctx[1]:.2f} "
                        f"(slightly off the direct path)")
        return dict(cfg_lambda=cfg_lam, behavior_mode=mode,
                    context_pos=ctx, zero_context=True,
                    obstacle_info=obs_info, waypoint_info=None)

    elif behavior == "grounding":
        wp = WAYPOINT_BLOCKS[ep_index % len(WAYPOINT_BLOCKS)]
        ctx = wp["pos"]
        wp_info = (f"Hover near the {wp['name'].upper()} block "
                   f"(small colored block at approx x={ctx[0]:.2f} y={ctx[1]:.2f}) "
                   f"BEFORE picking the "
                   f"{'GREEN left' if target == 'left' else 'RED right'} block.")
        return dict(cfg_lambda=2.0, behavior_mode=0.0,
                    context_pos=ctx, zero_context=True,
                    obstacle_info=None, waypoint_info=wp_info)

    raise ValueError(f"Unknown behavior: {behavior}")


# ══════════════════════════════════════════════════════════════════════
# PROMPT BUILDER — pairwise forced-choice
# ══════════════════════════════════════════════════════════════════════

SCENE_DESC = (
    "A Franka Panda robot arm operates on a table with two target blocks:\n"
    "  - LEFT block  (green)\n"
    "  - RIGHT block (red)\n"
    "The arm starts from a raised home position and moves to pick one block.\n"
    "Each trajectory shows {n} frames at simulation steps "
    "{steps} out of 150 total steps (earlier frames = earlier in time).\n\n"
)


def build_pairwise_prompt(behavior: str, target: str,
                           n_frames: int, obstacle_info=None,
                           waypoint_info=None) -> str:
    steps_str = ", ".join(str(s) for s in FRAME_STEPS[:n_frames])
    scene = SCENE_DESC.format(n=n_frames, steps=steps_str)
    target_str = "LEFT (green)" if target == "left" else "RIGHT (red)"

    task_header = (
        f"The robot is picking the {target_str} block.\n"
        f"You are shown TRAJECTORY A ({n_frames} frames) followed by "
        f"TRAJECTORY B ({n_frames} frames).\n\n"
        f"TASK: Which trajectory better demonstrates {behavior.upper()}?\n\n"
    )

    if behavior == "legibility":
        criteria = (
            "LEGIBILITY = how clearly and EARLY the robot reveals its intention "
            "to pick the target block.\n"
            "- A LEGIBLE trajectory curves deliberately toward the target from the "
            "very first frames. An observer can predict the goal from step 15 alone.\n"
            "- A NON-LEGIBLE trajectory moves straight down or ambiguously at first, "
            "revealing which block it wants only late.\n"
            "Judge purely on: does the early motion clearly commit to the target direction?\n"
        )
    elif behavior == "predictability":
        criteria = (
            "PREDICTABILITY = how direct and straight the path to the target is.\n"
            "- A PREDICTABLE trajectory takes the shortest-path approach: arm drops "
            "straight toward the block with minimal lateral deviation. You can mentally "
            "extrapolate its endpoint from the first few frames.\n"
            "- An UNPREDICTABLE trajectory arcs, sweeps, or takes an indirect route.\n"
            "Judge purely on: directness and linearity of the approach path.\n"
        )
    elif behavior == "safety":
        obs = obstacle_info or "a cyan cylinder on the table"
        criteria = (
            f"SAFETY = how well the trajectory avoids the obstacle ({obs}).\n"
            "- A SAFE trajectory routes clearly around the obstacle, "
            "maintaining visible clearance throughout.\n"
            "- An UNSAFE trajectory passes close to or through the obstacle region.\n"
            "Judge purely on: does the arm visibly avoid the obstacle?\n"
        )
    elif behavior == "grounding":
        wp = waypoint_info or "a specific colored waypoint block"
        criteria = (
            f"SPATIAL GROUNDING: {wp}\n"
            "- A GROUNDED trajectory visibly moves toward the waypoint block "
            "FIRST before redirecting to pick the target.\n"
            "- An UNGROUNDED trajectory goes directly to the target, ignoring "
            "the waypoint entirely.\n"
            "Judge purely on: does the arm visit the waypoint before the target?\n"
        )
    else:
        criteria = f"Rate which trajectory better demonstrates {behavior}.\n"

    choice_instruction = (
        "\nChoose the trajectory that BETTER demonstrates the property above.\n"
        "If they look identical, pick your best guess — you MUST choose one.\n\n"
        "Respond with ONLY valid JSON (no extra text):\n"
        '{"choice": "A", "confidence": 0.0-1.0, "reason": "one sentence"}\n'
        '  OR\n'
        '{"choice": "B", "confidence": 0.0-1.0, "reason": "one sentence"}'
    )

    return scene + task_header + criteria + choice_instruction


def parse_pairwise_response(text: str) -> dict:
    """Extract choice (A/B), confidence, reason from VLM response."""
    if not text:
        return {"choice": None, "confidence": 0.5, "reason": "EMPTY"}
    text = text.strip()
    j0 = text.find("{")
    j1 = text.rfind("}") + 1
    if j0 == -1 or j1 == 0:
        # Fallback: look for bare A or B
        upper = text.upper()
        if '"A"' in upper or upper.strip().startswith("A"):
            return {"choice": "A", "confidence": 0.6, "reason": text[:100]}
        if '"B"' in upper or upper.strip().startswith("B"):
            return {"choice": "B", "confidence": 0.6, "reason": text[:100]}
        return {"choice": None, "confidence": 0.5, "reason": f"PARSE_FAIL:{text[:80]}"}
    try:
        data = json.loads(text[j0:j1])
        choice = str(data.get("choice", "")).upper().strip()
        if choice not in ("A", "B"):
            choice = None
        return {
            "choice": choice,
            "confidence": float(data.get("confidence", 0.5)),
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as e:
        return {"choice": None, "confidence": 0.5, "reason": f"JSON_ERR:{e}:{text[:60]}"}


# ══════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
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
        h = h.transpose(1,2); h = self.norm1(h); h = h.transpose(1,2)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        h = self.conv2(h)
        h = h.transpose(1,2); h = self.norm2(h); h = h.transpose(1,2)
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
        dims = [hidden_dim, hidden_dim*2, hidden_dim*4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim) for i in range(len(dims)-1)])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1]+dims[i+1], dims[i], hidden_dim)
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


class CFGDDIMSampler:
    def __init__(self, n_steps, beta_start, beta_end, device, eta=0.5):
        self.device = device; self.eta = eta
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
            eps_c = model(x, t_b, obs_cond)
            eps_u = model(x, t_b, obs_uncond)
            eps = eps_u + cfg_lambda * (eps_c - eps_u)
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


def load_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DiffusionPolicy(
        ckpt.get("obs_dim", OBS_DIM_V2), ckpt.get("act_dim", ACT_DIM),
        ckpt.get("horizon", 32), ckpt.get("hidden_dim", 256),
        ckpt.get("n_blocks", 6)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sampler = CFGDDIMSampler(
        ckpt.get("n_diffusion_steps", 100),
        ckpt.get("beta_start", 1e-4), ckpt.get("beta_end", 0.1),
        device, eta=0.5)
    stats = {k: ckpt[k] for k in ["obs_mean","obs_std","act_mean","act_std"]}
    return model, sampler, stats


# ══════════════════════════════════════════════════════════════════════
# ENV HELPERS
# ══════════════════════════════════════════════════════════════════════

def get_obs_v2(env, context_pos=None, behavior_mode=0.0):
    base = env._get_obs()
    ctx = (np.zeros(3, dtype=np.float32) if context_pos is None
           else np.array(context_pos, dtype=np.float32))
    return np.concatenate([base, ctx, np.array([behavior_mode], dtype=np.float32)])

def _make_obs_uncond(obs_cond_tensor, zero_context=False):
    obs_u = obs_cond_tensor.clone()
    if zero_context:
        obs_u[..., CFG_COND_START:] = 0.0
    else:
        obs_u[..., CFG_MODE_DIM] = 0.0
    return obs_u

def capture_jpeg(env, width=FRAME_RES, height=FRAME_RES, quality=FRAME_QUAL) -> bytes:
    from PIL import Image
    frame = env.render(mode="rgb_array", width=width, height=height)
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

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
        try:
            p.removeBody(uid, physicsClientId=env._cid)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# SIMULATE ONE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════

def simulate_trajectory(env, model, sampler, stats, device,
                        context_pos, behavior_mode, cfg_lambda,
                        seed, n_steps=150, n_action_steps=8,
                        zero_context=False):
    """
    Run one rollout. Returns captured JPEG frames at FRAME_STEPS.
    cfg_lambda=0.0 → baseline (no guidance, pure conditional policy).
    cfg_lambda=2.0 → behavior-conditioned.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
    act_mean, act_std = stats["act_mean"], stats["act_std"]

    frames = []
    act_queue = []
    queue_idx = 0
    frame_steps_set = set(FRAME_STEPS)

    for step in range(n_steps):
        if step in frame_steps_set:
            frames.append(capture_jpeg(env))

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
        action[4]  = np.clip(action[4],  -1, 1)
        result = env.step(action)
        if result.done:
            break

    return frames


# ══════════════════════════════════════════════════════════════════════
# VLM SCORERS
# ══════════════════════════════════════════════════════════════════════

def _b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("utf-8")


class GPTPairwiseScorer:
    def __init__(self, api_key: str, model: str = "gpt-5.4"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model  = model
        # Connectivity test
        r = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "Reply with the word OK."}]}])
        text = r.output_text.strip()
        assert "ok" in text.lower(), f"GPT connectivity failed: {text}"
        print(f"  [GPT] Connected — model={self.model}")

    def score(self, frames_A, frames_B, prompt_text) -> dict:
        content = []
        content.append({"type": "input_text",
                         "text": prompt_text + "\n\nTRAJECTORY A:"})
        for i, f in enumerate(frames_A):
            content.append({"type": "input_text",
                             "text": f"  A-frame {i+1}/{len(frames_A)} (step {FRAME_STEPS[i]})"})
            content.append({"type": "input_image",
                             "image_url": f"data:image/jpeg;base64,{_b64(f)}"})
        content.append({"type": "input_text", "text": "\nTRAJECTORY B:"})
        for i, f in enumerate(frames_B):
            content.append({"type": "input_text",
                             "text": f"  B-frame {i+1}/{len(frames_B)} (step {FRAME_STEPS[i]})"})
            content.append({"type": "input_image",
                             "image_url": f"data:image/jpeg;base64,{_b64(f)}"})
        content.append({"type": "input_text",
                         "text": "\nNow give your JSON answer (choice A or B):"})

        for attempt in range(3):
            try:
                r = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": content}])
                raw = r.output_text.strip()
                result = parse_pairwise_response(raw)
                result["raw"] = raw[:300]
                return result
            except Exception as e:
                print(f"    [GPT attempt {attempt+1}/3]: {e}")
                time.sleep(5)
        return {"choice": None, "confidence": 0.5, "reason": "ALL_RETRIES_FAILED", "raw": ""}


# ══════════════════════════════════════════════════════════════════════
# GEMINI PAIRWISE SCORER
# ══════════════════════════════════════════════════════════════════════

class GeminiPairwiseScorer:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model  = model
        # Connectivity test
        resp = self.client.models.generate_content(
            model=self.model, contents="Reply OK")
        assert "ok" in resp.text.lower(), f"Gemini connectivity failed: {resp.text}"
        print(f"  [Gemini] Connected — model={self.model}")

    def score(self, frames_A, frames_B, prompt_text) -> dict:
        from google.genai import types as gtypes
        parts = []
        parts.append(gtypes.Part.from_text(text=prompt_text + "\n\nTRAJECTORY A:"))
        for i, f in enumerate(frames_A):
            parts.append(gtypes.Part.from_text(
                text=f"  A-frame {i+1}/{len(frames_A)} (step {FRAME_STEPS[i]})"))
            parts.append(gtypes.Part.from_bytes(data=f, mime_type="image/jpeg"))
        parts.append(gtypes.Part.from_text(text="\nTRAJECTORY B:"))
        for i, f in enumerate(frames_B):
            parts.append(gtypes.Part.from_text(
                text=f"  B-frame {i+1}/{len(frames_B)} (step {FRAME_STEPS[i]})"))
            parts.append(gtypes.Part.from_bytes(data=f, mime_type="image/jpeg"))
        parts.append(gtypes.Part.from_text(
            text="\nNow give your JSON answer (choice A or B):"))

        contents = [gtypes.Content(role="user", parts=parts)]
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents)
                raw = resp.text.strip()
                result = parse_pairwise_response(raw)
                result["raw"] = raw[:300]
                return result
            except Exception as e:
                print(f"    [Gemini attempt {attempt+1}/3]: {e}")
                time.sleep(5)
        return {"choice": None, "confidence": 0.5,
                "reason": "ALL_RETRIES_FAILED", "raw": ""}


class ClaudePairwiseScorer:
    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model
        # Connectivity test
        r = self.client.messages.create(
            model=self.model, max_tokens=32,
            messages=[{"role": "user", "content": "Reply with the word OK."}])
        text = r.content[0].text.strip()
        assert "ok" in text.lower(), f"Claude connectivity failed: {text}"
        print(f"  [Claude] Connected — model={self.model}")

    def score(self, frames_A, frames_B, prompt_text) -> dict:
        content = []
        content.append({"type": "text",
                         "text": prompt_text + "\n\nTRAJECTORY A:"})
        for i, f in enumerate(frames_A):
            content.append({"type": "text",
                             "text": f"  A-frame {i+1}/{len(frames_A)} (step {FRAME_STEPS[i]})"})
            content.append({"type": "image",
                             "source": {"type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": _b64(f)}})
        content.append({"type": "text", "text": "\nTRAJECTORY B:"})
        for i, f in enumerate(frames_B):
            content.append({"type": "text",
                             "text": f"  B-frame {i+1}/{len(frames_B)} (step {FRAME_STEPS[i]})"})
            content.append({"type": "image",
                             "source": {"type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": _b64(f)}})
        content.append({"type": "text",
                         "text": "\nNow give your JSON answer (choice A or B):"})

        for attempt in range(3):
            try:
                r = self.client.messages.create(
                    model=self.model, max_tokens=256,
                    messages=[{"role": "user", "content": content}])
                raw = r.content[0].text.strip()
                result = parse_pairwise_response(raw)
                result["raw"] = raw[:300]
                return result
            except Exception as e:
                print(f"    [Claude attempt {attempt+1}/3]: {e}")
                time.sleep(5)
        return {"choice": None, "confidence": 0.5, "reason": "ALL_RETRIES_FAILED", "raw": ""}


# ══════════════════════════════════════════════════════════════════════
# SINGLE BEHAVIOR EVALUATION
# ══════════════════════════════════════════════════════════════════════

def run_behavior(behavior, env, model, sampler, stats, device,
                 gpt_scorer, claude_scorer, gemini_scorer,
                 n_episodes, sleep=1.0, rng_seed=42):
    rng = random.Random(rng_seed)
    seeds   = EPISODE_SEEDS[behavior][:n_episodes]
    targets = EPISODE_TARGETS[behavior][:n_episodes]

    ep_results = []
    gpt_correct = 0
    claude_correct = 0
    gemini_correct = 0
    gpt_n = 0
    claude_n = 0
    gemini_n = 0

    for ep_i, (ep_seed, target) in enumerate(zip(seeds, targets)):
        ctx = get_episode_context(behavior, ep_i, target)
        print(f"\n  Episode {ep_i+1}/{n_episodes}  seed={ep_seed}  "
              f"target={target}  lambda={ctx['cfg_lambda']}  mode={ctx['behavior_mode']:.1f}")

        # Randomize which label the conditioned traj gets
        cond_is_A = rng.choice([True, False])
        label_cond = "A" if cond_is_A else "B"
        label_base = "B" if cond_is_A else "A"
        print(f"    Assignment: conditioned={label_cond}  baseline={label_base}")

        # Reset env and add scene objects
        env.reset(seed=ep_seed)
        added_uids = []
        if behavior == "safety" and ctx["context_pos"] is not None:
            added_uids.append(add_obstacle_visual(env, ctx["context_pos"]))
        elif behavior == "grounding":
            p.changeVisualShape(env._cube_l_uid, -1,
                                rgbaColor=[0.1, 0.8, 0.1, 1.0],
                                physicsClientId=env._cid)
            p.changeVisualShape(env._cube_r_uid, -1,
                                rgbaColor=[0.8, 0.1, 0.1, 1.0],
                                physicsClientId=env._cid)
            added_uids.extend(add_waypoint_blocks(env, WAYPOINT_BLOCKS))

        for _ in range(60):
            p.stepSimulation(physicsClientId=env._cid)

        saved_state = p.saveState(physicsClientId=env._cid)

        # ── Simulate CONDITIONED trajectory ──────────────────
        p.restoreState(saved_state, physicsClientId=env._cid)
        env._episode_steps = 0
        traj_seed = ep_seed * 1000 + 1   # same seed for both → isolates CFG effect
        frames_cond = simulate_trajectory(
            env, model, sampler, stats, device,
            ctx["context_pos"], ctx["behavior_mode"], ctx["cfg_lambda"],
            seed=traj_seed, zero_context=ctx["zero_context"])
        print(f"    Conditioned: {len(frames_cond)} frames")

        # ── Simulate BASELINE trajectory (no CFG) ────────────
        p.restoreState(saved_state, physicsClientId=env._cid)
        env._episode_steps = 0
        frames_base = simulate_trajectory(
            env, model, sampler, stats, device,
            ctx["context_pos"], ctx["behavior_mode"], cfg_lambda=0.0,
            seed=traj_seed, zero_context=ctx["zero_context"])
        print(f"    Baseline:    {len(frames_base)} frames")

        # Build A/B assignment
        frames_A = frames_cond if cond_is_A else frames_base
        frames_B = frames_base if cond_is_A else frames_cond
        n_frames = min(len(frames_A), len(frames_B), len(FRAME_STEPS))
        frames_A = frames_A[:n_frames]
        frames_B = frames_B[:n_frames]

        prompt = build_pairwise_prompt(
            behavior, target, n_frames,
            obstacle_info=ctx["obstacle_info"],
            waypoint_info=ctx["waypoint_info"])

        ep_rec = {
            "ep_index": ep_i,
            "episode_seed": ep_seed,
            "target": target,
            "traj_seed": traj_seed,
            "cond_is_A": cond_is_A,
            "correct_choice": label_cond,
            "n_frames": n_frames,
            "gpt": None,
            "claude": None,
            "gemini": None,
        }

        # ── GPT scoring ───────────────────────────────────────
        if gpt_scorer is not None:
            gpt_r = gpt_scorer.score(frames_A, frames_B, prompt)
            correct = (gpt_r["choice"] == label_cond)
            gpt_r["correct"] = correct
            ep_rec["gpt"] = gpt_r
            gpt_n += 1
            if correct:
                gpt_correct += 1
            result_str = "CORRECT" if correct else "wrong"
            print(f"    GPT   -> choice={gpt_r['choice']}  "
                  f"conf={gpt_r['confidence']:.2f}  "
                  f"{result_str}  "
                  f"reason=\"{gpt_r['reason'][:80]}\"")
            time.sleep(sleep)

        # ── Claude scoring ────────────────────────────────────
        if claude_scorer is not None:
            cla_r = claude_scorer.score(frames_A, frames_B, prompt)
            correct = (cla_r["choice"] == label_cond)
            cla_r["correct"] = correct
            ep_rec["claude"] = cla_r
            claude_n += 1
            if correct:
                claude_correct += 1
            result_str = "correct" if correct else "WRONG"
            print(f"    Claude-> choice={cla_r['choice']}  "
                  f"conf={cla_r['confidence']:.2f}  "
                  f"{result_str}  "
                  f"reason=\"{cla_r['reason'][:80]}\"")
            time.sleep(sleep)

        # ── Gemini scoring ────────────────────────────────────
        if gemini_scorer is not None:
            gem_r = gemini_scorer.score(frames_A, frames_B, prompt)
            correct = (gem_r["choice"] == label_cond)
            gem_r["correct"] = correct
            ep_rec["gemini"] = gem_r
            gemini_n += 1
            if correct:
                gemini_correct += 1
            result_str = "correct" if correct else "WRONG"
            print(f"    Gemini-> choice={gem_r['choice']}  "
                  f"conf={gem_r['confidence']:.2f}  "
                  f"{result_str}  "
                  f"reason=\"{gem_r['reason'][:80]}\"")
            time.sleep(sleep)

        ep_results.append(ep_rec)

        # Clean up scene objects for next episode
        remove_bodies(env, added_uids)
        added_uids = []

    gpt_acc    = gpt_correct    / gpt_n    if gpt_n    > 0 else None
    claude_acc = claude_correct  / claude_n if claude_n > 0 else None
    gemini_acc = gemini_correct  / gemini_n if gemini_n > 0 else None

    print(f"\n  [{behavior}] Results (chance=50%):")
    if gpt_acc    is not None:
        print(f"    GPT    accuracy = {gpt_correct}/{gpt_n} = {gpt_acc:.1%}")
    if claude_acc is not None:
        print(f"    Claude accuracy = {claude_correct}/{claude_n} = {claude_acc:.1%}")
    if gemini_acc is not None:
        print(f"    Gemini accuracy = {gemini_correct}/{gemini_n} = {gemini_acc:.1%}")

    return {
        "behavior": behavior,
        "n_episodes": len(ep_results),
        "gpt_accuracy": gpt_acc,
        "claude_accuracy": claude_acc,
        "gemini_accuracy": gemini_acc,
        "gpt_correct": gpt_correct,
        "gpt_n": gpt_n,
        "claude_correct": claude_correct,
        "claude_n": claude_n,
        "gemini_correct": gemini_correct,
        "gemini_n": gemini_n,
        "episodes": ep_results,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Pairwise VLM evaluation: conditioned vs baseline trajectory")
    ap.add_argument("--checkpoint",  required=True,
                    help="Path to policy checkpoint (.pt)")
    ap.add_argument("--openai_key",  default=None)
    ap.add_argument("--anthropic_key", default=None)
    ap.add_argument("--gpt_model",   default="gpt-5.4")
    ap.add_argument("--claude_model",default="claude-opus-4-5")
    ap.add_argument("--behavior",    default="all",
                    choices=["all"] + BEHAVIORS)
    ap.add_argument("--n_episodes",  type=int, default=10)
    ap.add_argument("--sleep",       type=float, default=1.0,
                    help="Seconds to sleep between API calls")
    ap.add_argument("--out_dir",     default="outputs/pairwise_eval")
    ap.add_argument("--gemini_key",  default=os.environ.get("GEMINI_API_KEY", ""))
    ap.add_argument("--gemini_model", default="gemini-2.5-flash")
    ap.add_argument("--skip_gpt",    action="store_true")
    ap.add_argument("--skip_claude", action="store_true")
    ap.add_argument("--skip_gemini", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  PAIRWISE VLM EVALUATION")
    print(f"  Design: conditioned (lambda=2.0) vs baseline (lambda=0.0)")
    print(f"  Metric: accuracy (chance = 50%)")
    print(f"  Frames: {FRAME_STEPS} at {FRAME_RES}x{FRAME_RES}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load policy
    print(f"Loading policy: {args.checkpoint}")
    model, sampler, stats = load_policy(args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  obs={ckpt.get('obs_dim')} act={ckpt.get('act_dim')} "
          f"H={ckpt.get('horizon')}")

    # Build scorers
    gpt_scorer    = None
    claude_scorer = None
    gemini_scorer = None

    if not args.skip_gemini:
        if not args.gemini_key:
            print("WARNING: no --gemini_key, skipping Gemini")
        else:
            print("\nConnecting to Gemini...")
            try:
                gemini_scorer = GeminiPairwiseScorer(args.gemini_key,
                                                      model=args.gemini_model)
            except Exception as e:
                print(f"  Gemini connection failed: {e} — skipping")
                gemini_scorer = None

    if not args.skip_gpt:
        if not args.openai_key:
            print("WARNING: no --openai_key, skipping GPT")
        else:
            print("\nConnecting to GPT...")
            gpt_scorer = GPTPairwiseScorer(args.openai_key, model=args.gpt_model)

    if not args.skip_claude:
        if not args.anthropic_key:
            print("WARNING: no --anthropic_key, skipping Claude")
        else:
            print("\nConnecting to Claude...")
            try:
                claude_scorer = ClaudePairwiseScorer(args.anthropic_key,
                                                      model=args.claude_model)
            except Exception as e:
                print(f"  Claude connection failed: {e} — skipping")
                claude_scorer = None

    if gpt_scorer is None and claude_scorer is None and gemini_scorer is None:
        print("ERROR: No VLM scorers available. Provide at least one API key.")
        sys.exit(1)

    behaviors_to_run = BEHAVIORS if args.behavior == "all" else [args.behavior]

    # Create env once (reused across behaviors)
    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          cube_half=0.015, cube_mass=0.08,
                          cube_lateral_friction=2.5,
                          episode_length=400)

    all_results = {
        "checkpoint": args.checkpoint,
        "design": "pairwise: conditioned (cfg_lambda=2.0) vs baseline (cfg_lambda=0.0)",
        "frame_steps": FRAME_STEPS,
        "frame_res": FRAME_RES,
        "models": {
            "gpt": args.gpt_model if gpt_scorer else "SKIPPED",
            "claude": args.claude_model if claude_scorer else "SKIPPED",
            "gemini": args.gemini_model if gemini_scorer else "SKIPPED",
        },
        "behaviors": {}
    }

    print()
    for beh in behaviors_to_run:
        print(f"\n{'='*60}")
        print(f"  BEHAVIOR: {beh.upper()}")
        print(f"  Episodes: {args.n_episodes}   API calls: "
              f"{args.n_episodes * sum([gpt_scorer is not None, claude_scorer is not None])}")
        print(f"{'='*60}")

        result = run_behavior(
            beh, env, model, sampler, stats, device,
            gpt_scorer, claude_scorer, gemini_scorer,
            n_episodes=args.n_episodes,
            sleep=args.sleep)

        all_results["behaviors"][beh] = result

        # Save checkpoint after each behavior
        beh_dir = out_dir / beh
        beh_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = beh_dir / "results.json"
        with open(ckpt_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Save combined results
    combined_path = out_dir / "pairwise_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved: {combined_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (chance = 50%)")
    print(f"{'='*60}")
    print(f"  {'BEHAVIOR':<16} {'GPT':<16} {'Claude':<16} {'Gemini'}")
    print(f"  {'-'*62}")
    for beh, res in all_results["behaviors"].items():
        gpt_str = (f"{res['gpt_correct']}/{res['gpt_n']}={res['gpt_accuracy']:.0%}"
                   if res['gpt_accuracy'] is not None else "SKIP")
        cla_str = (f"{res['claude_correct']}/{res['claude_n']}={res['claude_accuracy']:.0%}"
                   if res['claude_accuracy'] is not None else "SKIP")
        gem_str = (f"{res['gemini_correct']}/{res['gemini_n']}={res['gemini_accuracy']:.0%}"
                   if res['gemini_accuracy'] is not None else "SKIP")
        print(f"  {beh:<16} {gpt_str:<16} {cla_str:<16} {gem_str}")
    print()


if __name__ == "__main__":
    main()
