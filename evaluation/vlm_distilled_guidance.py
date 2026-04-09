#!/usr/bin/env python3
"""
VLM-Distilled Classifier Guidance (VDCG)
=========================================

Three-phase pipeline:
  Phase 1: Generate trajectories → render → VLM scores (visual grounding)
  Phase 2: Train differentiable proxy f_θ on (trajectory, VLM_score) pairs
  Phase 3: Use f_θ as classifier guidance in DDIM denoising

This is real VLM steering: the VLM provides visually-grounded scores,
distilled into a differentiable proxy that injects gradients at every
denoising step. Not filtering (Best-of-K), not text synthesis.

Usage:
  # Phase 1+2: Build proxy from VLM scores
  python evaluation/vlm_distilled_guidance.py build-proxy \
      --checkpoint runs/diffusion_20260402_072747/ckpt_ep100.pt \
      --behavior legibility --n_trajectories 200

  # Phase 3: Evaluate with VLM-distilled guidance
  python evaluation/vlm_distilled_guidance.py evaluate \
      --checkpoint runs/diffusion_20260402_072747/ckpt_ep100.pt \
      --proxy_path outputs/vdcg/legibility/proxy.pt \
      --guidance_scale 5.0 --n_episodes 20
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# ── constants ────────────────────────────────────────────────────────────
ACTION_SCALE = 0.05
OBS_EE_POS = slice(0, 3)
OBS_LEFT_POS = slice(8, 11)
OBS_RIGHT_POS = slice(15, 18)

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ═════════════════════════════════════════════════════════════════════════
# MODEL (same architecture as all other eval scripts — DO NOT CHANGE)
# ═════════════════════════════════════════════════════════════════════════

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
        h = self.conv1(x)
        h = h.transpose(1, 2)
        h = self.norm1(h)
        h = h.transpose(1, 2)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        return self.act(h + self.shortcut(x))

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
            x = blk(x, t_emb); skips.append(x)
        x = self.bottleneck(x, t_emb)
        for blk, sk in zip(self.decoder_blocks, reversed(skips)):
            x = blk(torch.cat([x, sk], dim=-1), t_emb)
        return self.output_proj(x)


# ═════════════════════════════════════════════════════════════════════════
# DDIM SAMPLER (baseline, no guidance)
# ═════════════════════════════════════════════════════════════════════════

class DDIMSampler:
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs, n_sampling_steps=10):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod) - 1,
                           n_sampling_steps, device=self.device).long(), [0])
        for i, t in enumerate(timesteps):
            t_b = t.repeat(B)
            eps = model(x, t_b, obs)
            a_t = self.alphas_cumprod[t]
            a_prev = (self.alphas_cumprod[timesteps[i + 1]]
                      if i < len(timesteps) - 1
                      else torch.tensor(1.0, device=self.device))
            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x = (torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * eps
                 if i < len(timesteps) - 1 else x0)
        return x


# ═════════════════════════════════════════════════════════════════════════
# VLM-DISTILLED PROXY NETWORK
# ═════════════════════════════════════════════════════════════════════════

class TrajectoryScoreProxy(nn.Module):
    """Differentiable proxy: (ee_trajectory, goals) → VLM score ∈ [0,1].

    Input features (auto-computed from trajectory + goals):
      - Per-timestep Bayesian posterior P(g*|x_t) for early window (10-d)
      - Velocity alignment with goal direction (9-d)
      - Lateral displacement from midline (10-d)
      - Min/mean distance to each goal (4-d)
      - Path efficiency ratio (1-d)
    Total: 34-d feature vector → MLP → scalar score
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(34, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def extract_features(
        self,
        ee_traj: torch.Tensor,   # (H, 3)
        goals: torch.Tensor,     # (K, 3)
        true_goal_idx: int = 0,
        early_frac: float = 0.3,
    ) -> torch.Tensor:
        """Extract differentiable feature vector from trajectory.

        All computations use only torch ops — gradients flow through ee_traj.
        """
        H = ee_traj.shape[0]
        early_end = max(2, int(H * early_frac))

        # Auto-calibrate sigma from inter-goal distance
        d_goals = torch.cdist(goals, goals)
        mask = d_goals > 1e-6
        d_min = d_goals[mask].min() if mask.any() else torch.tensor(
            0.14, device=ee_traj.device)
        sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))

        true_goal = goals[true_goal_idx]
        other_idx = 1 - true_goal_idx
        other_goal = goals[other_idx]

        # ── Feature 1: Bayesian posterior at 10 evenly-spaced early points ──
        indices = torch.linspace(0, early_end - 1, 10,
                                 device=ee_traj.device).long()
        sample_pts = ee_traj[indices]  # (10, 3)
        diff = sample_pts.unsqueeze(1) - goals.unsqueeze(0)  # (10, K, 3)
        sq_dist = (diff ** 2).sum(-1)  # (10, K)
        log_like = -sq_dist / (2.0 * sigma ** 2 + 1e-8)
        posteriors = torch.softmax(log_like, dim=-1)  # (10, K)
        feat_posterior = posteriors[:, true_goal_idx]  # (10,)

        # ── Feature 2: Velocity alignment at 9 intervals ──
        vel_indices = torch.linspace(0, early_end - 2, 9,
                                     device=ee_traj.device).long()
        velocities = ee_traj[vel_indices + 1] - ee_traj[vel_indices]  # (9, 3)
        to_goal = true_goal.unsqueeze(0) - ee_traj[vel_indices]  # (9, 3)
        vel_norm = F.normalize(velocities, dim=-1, eps=1e-6)
        goal_norm = F.normalize(to_goal, dim=-1, eps=1e-6)
        feat_alignment = (vel_norm * goal_norm).sum(-1)  # (9,) cosine sim

        # ── Feature 3: Signed lateral displacement (10 points) ──
        midline_y = (goals[0, 1] + goals[1, 1]) / 2.0
        target_side = torch.sign(true_goal[1] - midline_y + 1e-6)
        lateral = (sample_pts[:, 1] - midline_y) * target_side  # (10,)
        # Normalize by inter-goal distance
        feat_lateral = lateral / (d_min + 1e-6)  # (10,)

        # ── Feature 4: Summary distance stats (4-d) ──
        d_true = torch.norm(sample_pts - true_goal.unsqueeze(0), dim=-1)
        d_other = torch.norm(sample_pts - other_goal.unsqueeze(0), dim=-1)
        feat_dist = torch.stack([
            d_true.min(), d_true.mean(),
            d_other.min(), d_other.mean(),
        ]) / (d_min + 1e-6)  # (4,) normalized

        # ── Feature 5: Path efficiency (1-d) ──
        segments = ee_traj[1:early_end] - ee_traj[:early_end - 1]
        path_len = segments.norm(dim=-1).sum()
        straight = torch.norm(ee_traj[early_end - 1] - ee_traj[0])
        feat_efficiency = (straight / (path_len + 1e-6)).unsqueeze(0)  # (1,)

        # Concatenate all features: 10 + 9 + 10 + 4 + 1 = 34
        features = torch.cat([
            feat_posterior, feat_alignment, feat_lateral,
            feat_dist, feat_efficiency,
        ])  # (34,)
        return features

    def forward(
        self,
        ee_traj: torch.Tensor,
        goals: torch.Tensor,
        true_goal_idx: int = 0,
        early_frac: float = 0.3,
    ) -> torch.Tensor:
        """Score trajectory. Returns scalar in [0,1], differentiable w.r.t. ee_traj."""
        features = self.extract_features(ee_traj, goals, true_goal_idx,
                                         early_frac)
        return self.net(features.unsqueeze(0)).squeeze()


# ═════════════════════════════════════════════════════════════════════════
# L_EARLY (for comparison / goal inference)
# ═════════════════════════════════════════════════════════════════════════

def l_early_intent_torch(
    ee_traj: torch.Tensor, goals: torch.Tensor,
    true_goal_idx: int = 0, early_frac: float = 0.30,
) -> torch.Tensor:
    H = ee_traj.shape[0]
    early_end = max(1, int(H * early_frac))
    early_traj = ee_traj[:early_end]
    dists = torch.cdist(goals, goals)
    mask = dists > 1e-6
    d_min = dists[mask].min() if mask.any() else torch.tensor(
        0.14, device=goals.device)
    sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))
    diff = early_traj.unsqueeze(1) - goals.unsqueeze(0)
    sq_dist = (diff ** 2).sum(-1)
    log_like = -sq_dist / (2.0 * sigma ** 2)
    posteriors = torch.softmax(log_like, dim=-1)
    return posteriors[:, true_goal_idx].mean()


# ═════════════════════════════════════════════════════════════════════════
# GUIDED DDIM SAMPLER (pluggable score function)
# ═════════════════════════════════════════════════════════════════════════

class GuidedDDIMSampler:
    """DDIM with classifier guidance using ANY differentiable score function.

    At step t:
        ε̃ = ε_θ(x_t) − w · √(1−ᾱ_t) · ∇_{x_t} score_fn(ee_traj(x̂₀))
    """

    def __init__(self, n_steps, beta_start, beta_end, device,
                 guidance_scale: float = 5.0, grad_clip: float = 1.0):
        self.device = device
        self.guidance_scale = guidance_scale
        self.grad_clip = grad_clip
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def sample(
        self,
        model: DiffusionPolicy,
        obs: torch.Tensor,
        ee_pos_start: torch.Tensor,
        goals: torch.Tensor,
        score_fn,   # callable(ee_traj, goals, true_goal_idx) → scalar tensor
        n_sampling_steps: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod) - 1,
                           n_sampling_steps, device=self.device).long(), [0])

        final_score = 0.0

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (self.alphas_cumprod[timesteps[i + 1]]
                          if i < len(timesteps) - 1
                          else torch.tensor(1.0, device=self.device))
            sqrt_a = torch.sqrt(alpha_t)
            sqrt_1ma = torch.sqrt(1.0 - alpha_t)

            # Forward pass with gradient tracking
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                eps_pred = model(x_in, t_batch, obs)
                x0_pred = (x_in - sqrt_1ma * eps_pred) / sqrt_a

                # Convert predicted actions → EE trajectory
                delta_pos = x0_pred[0, :, :3] * ACTION_SCALE
                ee_traj = torch.cumsum(delta_pos, dim=0) + ee_pos_start

                # Infer committed goal (detached, no extra gradient path)
                with torch.no_grad():
                    l0 = l_early_intent_torch(
                        ee_traj.detach(), goals, 0).item()
                    l1 = l_early_intent_torch(
                        ee_traj.detach(), goals, 1).item()
                true_goal = 0 if l0 >= l1 else 1

                # Score with the pluggable function
                score = score_fn(ee_traj, goals, true_goal)
                grad = torch.autograd.grad(score, x_in)[0]

            final_score = float(score.item())

            with torch.no_grad():
                g = grad.detach()
                gn = g.norm()
                if gn > self.grad_clip:
                    g = g * (self.grad_clip / (gn + 1e-8))

                guided_eps = (eps_pred.detach()
                              - self.guidance_scale * sqrt_1ma * g)
                x0_guided = (x - sqrt_1ma * guided_eps) / sqrt_a

                if i < len(timesteps) - 1:
                    x = (torch.sqrt(alpha_prev) * x0_guided
                         + torch.sqrt(1.0 - alpha_prev) * guided_eps)
                else:
                    x = x0_guided

        return x, final_score


# ═════════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADING
# ═════════════════════════════════════════════════════════════════════════

def load_policy(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg.get("hidden_dim", 256),
        n_blocks=cfg.get("n_blocks", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    obs_mean = np.array(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.array(ckpt["obs_std"], dtype=np.float32)
    act_mean = np.array(ckpt["act_mean"], dtype=np.float32)
    act_std = np.array(ckpt["act_std"], dtype=np.float32)
    return model, obs_mean, obs_std, act_mean, act_std, cfg


# ═════════════════════════════════════════════════════════════════════════
# PHASE 1: GENERATE TRAJECTORIES + VLM SCORING
# ═════════════════════════════════════════════════════════════════════════

def generate_trajectory(
    model, sampler, obs, obs_mean, obs_std, act_mean, act_std,
    device, n_steps: int = 150,
) -> dict:
    """Run policy for n_steps, return EE trajectory + obs trajectory."""
    queue = deque()
    ee_traj = []
    current_obs = obs.copy()

    for step in range(n_steps):
        ee_traj.append(current_obs[:3].copy())
        if len(queue) == 0:
            obs_norm = (current_obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            act_seq = sampler.sample(model, obs_t, n_sampling_steps=10)
            act_np = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in act_np:
                queue.append(a)
        action = queue.popleft()
        # We need the env for this — handled in the driver
        yield action, current_obs


def capture_jpeg(env, width=480, height=480, quality=90) -> bytes:
    from PIL import Image
    frame = env.render(mode="rgb_array", width=width, height=height)
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


FRAME_STEPS = {0: 0, 30: 1, 60: 2, 90: 3, 120: 4, 149: 5}

BEHAVIOR_PROMPTS = {
    "legibility": (
        "You are evaluating a robot arm trajectory for LEGIBILITY.\n"
        "Goal A: pick the left block. Goal B: pick the right block.\n"
        "These 6 frames show t=0,1,2,3,4,5 seconds.\n\n"
        "Rate how LEGIBLY the arm reveals WHICH block it intends to pick.\n"
        "A legible trajectory curves TOWARD the target early, making intent obvious.\n"
        "An illegible trajectory stays ambiguous near the centerline.\n\n"
        "Output ONLY JSON: {\"pA\": X, \"pB\": X, \"score\": X, \"cue\": \"...\"}\n"
        "score: 0.0 (can't tell which block) to 1.0 (perfectly obvious from frame 1)"
    ),
    "predictability": (
        "You are evaluating a robot arm trajectory for PREDICTABILITY.\n"
        "Goal A: pick the left block. Goal B: pick the right block.\n"
        "These 6 frames show t=0,1,2,3,4,5 seconds.\n\n"
        "Rate how PREDICTABLE and DIRECT this path is.\n"
        "A predictable trajectory is a straight beeline — no curves or arcs.\n"
        "An unpredictable trajectory has unnecessary lateral sweeps.\n\n"
        "Output ONLY JSON: {\"pA\": X, \"pB\": X, \"score\": X, \"cue\": \"...\"}\n"
        "score: 0.0 (erratic/indirect) to 1.0 (perfectly straight line)"
    ),
    "safety": (
        "You are evaluating a robot arm trajectory for SAFETY.\n"
        "Goal A: pick the left block. Goal B: pick the right block.\n"
        "A RED CYLINDRICAL OBSTACLE is between the start position and blocks.\n"
        "These 6 frames show t=0,1,2,3,4,5 seconds.\n\n"
        "Rate how SAFELY the arm avoids the RED obstacle.\n"
        "Safe = wide clearance at all times. Unsafe = passes very close.\n\n"
        "Output ONLY JSON: {\"pA\": X, \"pB\": X, \"score\": X, \"cue\": \"...\"}\n"
        "score: 0.0 (dangerous) to 1.0 (wide clearance)"
    ),
    "sequential": (
        "You are evaluating a robot arm trajectory for INSTRUCTION FOLLOWING.\n"
        "Goal A: pick the left (GREEN) block. Goal B: pick the right (RED) block.\n"
        "A BLUE waypoint block is between them.\n"
        "INSTRUCTION: Pass near the BLUE block BEFORE picking the GREEN block.\n"
        "These 6 frames show t=0,1,2,3,4,5 seconds.\n\n"
        "Rate how well the trajectory follows the instruction.\n\n"
        "Output ONLY JSON: {\"pA\": X, \"pB\": X, \"score\": X, \"cue\": \"...\"}\n"
        "score: 0.0 (ignores waypoint) to 1.0 (clearly visits blue then green)"
    ),
}


def score_with_vlm(frames_bytes: list, behavior: str,
                   vlm_model: str = "gemini-2.5-flash-preview-05-20",
                   sleep: float = 1.0) -> dict:
    """Send frames to Gemini, get behavior-specific score."""
    from google import genai
    from google.genai import types as gtypes

    client_g = genai.Client()

    parts = []
    for fb in frames_bytes:
        parts.append(gtypes.Part.from_bytes(data=fb, mime_type="image/jpeg"))
    parts.append(gtypes.Part.from_text(BEHAVIOR_PROMPTS[behavior]))

    for attempt in range(3):
        try:
            resp = client_g.models.generate_content(
                model=vlm_model,
                contents=[gtypes.Content(role="user", parts=parts)],
            )
            text = resp.text.strip()
            j_start = text.find("{")
            j_end = text.rfind("}") + 1
            if j_start == -1 or j_end == 0:
                raise ValueError(f"No JSON: {text[:200]}")
            data = json.loads(text[j_start:j_end])
            return dict(
                pA=float(data.get("pA", 0.5)),
                pB=float(data.get("pB", 0.5)),
                score=float(data.get("score", 0.5)),
                cue=str(data.get("cue", "")),
            )
        except Exception as e:
            if attempt < 2:
                print(f"    [VLM retry {attempt + 1}: {e}]")
                time.sleep(2 ** attempt)
            else:
                print(f"    [VLM FAILED: {e}]")
                return dict(pA=0.5, pB=0.5, score=0.5, cue=f"ERROR: {e}")

    time.sleep(sleep)
    return dict(pA=0.5, pB=0.5, score=0.5, cue="TIMEOUT")


def build_proxy_dataset(
    model, obs_mean, obs_std, act_mean, act_std, cfg, device,
    behavior: str, n_trajectories: int = 200,
    guidance_scales: list = None, out_dir: Path = None,
    vlm_model: str = "gemini-2.5-flash-preview-05-20",
) -> Tuple[list, list]:
    """Phase 1: Generate diverse trajectories + VLM scores.

    Returns:
        ee_trajectories: list of (H, 3) numpy arrays
        vlm_scores: list of float
    """
    if guidance_scales is None:
        guidance_scales = [0.0, 0.0, 2.0, 5.0, 10.0]  # mix of guided/unguided

    n_diff = cfg.get("n_diffusion_steps", 100)
    beta_s = cfg.get("beta_start", 1e-4)
    beta_e = cfg.get("beta_end", 0.1)

    baseline_sampler = DDIMSampler(n_diff, beta_s, beta_e, device)

    ee_trajectories = []
    goal_positions = []
    true_goal_indices = []
    vlm_scores = []
    metadata = []

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Generating {n_trajectories} trajectories + VLM scoring")
    print(f"  Behavior: {behavior}")
    print(f"  Guidance scales: {guidance_scales}")
    print(f"{'=' * 60}\n")

    for traj_idx in range(n_trajectories):
        # Vary seeds and guidance scales for diversity
        ep_seed = traj_idx * 13 + 7
        w = guidance_scales[traj_idx % len(guidance_scales)]

        env = TwoBlockPickEnv(render=False, episode_length=200,
                              cube_jitter=0.0)
        obs = env.reset(seed=ep_seed)

        # If using guidance, build guided sampler
        if w > 0:
            sampler = GuidedDDIMSampler(
                n_diff, beta_s, beta_e, device,
                guidance_scale=w, grad_clip=1.0)
        else:
            sampler = baseline_sampler

        # Run trajectory, capture frames
        queue = deque()
        ee_traj = []
        frames = []
        current_obs = obs.copy()

        for step in range(150):
            ee_traj.append(current_obs[:3].copy())

            if len(queue) == 0:
                obs_norm = (current_obs - obs_mean) / obs_std
                obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                     device=device).unsqueeze(0)

                if w > 0:
                    ee_start = torch.tensor(
                        current_obs[:3], dtype=torch.float32, device=device)
                    left_g = torch.tensor(
                        current_obs[8:11], dtype=torch.float32, device=device)
                    right_g = torch.tensor(
                        current_obs[15:18], dtype=torch.float32, device=device)
                    goals_t = torch.stack([left_g, right_g])
                    act_seq, _ = sampler.sample(
                        model, obs_t, ee_start, goals_t,
                        score_fn=l_early_intent_torch,
                        n_sampling_steps=10)
                else:
                    act_seq = sampler.sample(
                        model, obs_t, n_sampling_steps=10)

                act_np = act_seq[0].cpu().numpy() * act_std + act_mean
                for a in act_np:
                    queue.append(a)

            action = queue.popleft()

            if step in FRAME_STEPS:
                frames.append(capture_jpeg(env))

            result = env.step(action)
            current_obs = result.obs
            if result.done:
                while len(ee_traj) < 150:
                    ee_traj.append(current_obs[:3].copy())
                break

        env.close()
        ee_arr = np.array(ee_traj)

        # Infer true goal from trajectory
        goals_np = np.array([obs[8:11], obs[15:18]])
        d0 = np.linalg.norm(ee_arr[-1] - goals_np[0])
        d1 = np.linalg.norm(ee_arr[-1] - goals_np[1])
        true_goal_idx = 0 if d0 < d1 else 1

        # VLM scoring
        vlm_result = score_with_vlm(frames, behavior, vlm_model=vlm_model)

        ee_trajectories.append(ee_arr)
        goal_positions.append(goals_np)
        true_goal_indices.append(true_goal_idx)
        vlm_scores.append(vlm_result["score"])
        metadata.append(dict(
            seed=ep_seed, w=w,
            true_goal=true_goal_idx,
            vlm_score=vlm_result["score"],
            vlm_cue=vlm_result["cue"],
            vlm_pA=vlm_result["pA"],
            vlm_pB=vlm_result["pB"],
        ))

        print(f"  [{traj_idx + 1:3d}/{n_trajectories}] "
              f"w={w:5.1f} goal={'L' if true_goal_idx == 0 else 'R'} "
              f"VLM={vlm_result['score']:.2f} "
              f"cue=\"{vlm_result['cue'][:50]}\"")

    # Save dataset
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset = dict(
            ee_trajectories=[t.tolist() for t in ee_trajectories],
            goal_positions=[g.tolist() for g in goal_positions],
            true_goal_indices=true_goal_indices,
            vlm_scores=vlm_scores,
            metadata=metadata,
            behavior=behavior,
        )
        with open(out_dir / "proxy_dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"\n  Dataset saved: {out_dir / 'proxy_dataset.json'}")

    return ee_trajectories, goal_positions, true_goal_indices, vlm_scores


# ═════════════════════════════════════════════════════════════════════════
# PHASE 2: TRAIN PROXY
# ═════════════════════════════════════════════════════════════════════════

def train_proxy(
    ee_trajectories: list,
    goal_positions: list,
    true_goal_indices: list,
    vlm_scores: list,
    device: torch.device,
    n_epochs: int = 500,
    lr: float = 1e-3,
    out_path: Path = None,
) -> TrajectoryScoreProxy:
    """Phase 2: Train differentiable proxy on VLM scores."""

    proxy = TrajectoryScoreProxy(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(proxy.parameters(), lr=lr)

    # Pre-extract features (faster than doing it inside training loop)
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Training proxy on {len(vlm_scores)} samples")
    print(f"{'=' * 60}\n")

    features_list = []
    targets = []
    for i in range(len(ee_trajectories)):
        ee_t = torch.tensor(ee_trajectories[i], dtype=torch.float32,
                            device=device)
        goals_t = torch.tensor(goal_positions[i], dtype=torch.float32,
                               device=device)
        tg_idx = true_goal_indices[i]

        with torch.no_grad():
            feat = proxy.extract_features(ee_t, goals_t, tg_idx)
        features_list.append(feat)
        targets.append(vlm_scores[i])

    X = torch.stack(features_list)  # (N, 34)
    Y = torch.tensor(targets, dtype=torch.float32, device=device)  # (N,)

    # Standardize features for stable training
    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-6)
    X_norm = (X - feat_mean) / feat_std

    # Train
    best_loss = float("inf")
    for epoch in range(n_epochs):
        proxy.train()
        pred = proxy.net(X_norm).squeeze()  # (N,)
        loss = F.mse_loss(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in proxy.state_dict().items()}

        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                pred_np = pred.cpu().numpy()
                target_np = Y.cpu().numpy()
                from scipy import stats
                rho, p = stats.spearmanr(pred_np, target_np)
            print(f"  Epoch {epoch + 1:4d}/{n_epochs}  "
                  f"loss={loss.item():.6f}  ρ={rho:.3f}  p={p:.4f}")

    # Restore best weights
    proxy.load_state_dict(best_state)
    proxy.eval()

    # Save proxy + normalization stats
    if out_path:
        torch.save(dict(
            proxy_state=proxy.state_dict(),
            feat_mean=feat_mean.cpu(),
            feat_std=feat_std.cpu(),
        ), out_path)
        print(f"\n  Proxy saved: {out_path}")
        print(f"  Best MSE loss: {best_loss:.6f}")

    # Final Spearman correlation
    with torch.no_grad():
        pred = proxy.net(X_norm).squeeze().cpu().numpy()
        rho, p = stats.spearmanr(pred, Y.cpu().numpy())
    print(f"  Final Spearman ρ = {rho:.3f} (p={p:.4f})")
    if rho < 0.3:
        print("  ⚠ WARNING: Low correlation — VLM scores may not be consistent")
        print("  → Consider: increasing n_trajectories, checking VLM prompt quality")

    return proxy, feat_mean, feat_std


# ═════════════════════════════════════════════════════════════════════════
# PHASE 3: GUIDED EVALUATION
# ═════════════════════════════════════════════════════════════════════════

def make_proxy_score_fn(proxy, feat_mean, feat_std, device):
    """Create a closure matching the score_fn interface for GuidedDDIMSampler.

    The returned function takes (ee_traj, goals, true_goal_idx) and returns
    a differentiable scalar. Gradients flow through ee_traj.
    """
    feat_mean_d = feat_mean.to(device)
    feat_std_d = feat_std.to(device)

    def score_fn(ee_traj, goals, true_goal_idx, early_frac=0.3):
        features = proxy.extract_features(ee_traj, goals, true_goal_idx,
                                          early_frac)
        features_norm = (features - feat_mean_d) / feat_std_d
        return proxy.net(features_norm.unsqueeze(0)).squeeze()

    return score_fn


def run_eval_episode(
    model, sampler, obs_mean, obs_std, act_mean, act_std,
    device, guided: bool, score_fn=None,
    n_sampling_steps: int = 10, max_steps: int = 400,
) -> dict:
    """Run one episode (baseline or guided). Returns metrics."""
    env = TwoBlockPickEnv(render=False, episode_length=max_steps,
                          cube_jitter=0.0)
    obs = env.reset()
    queue = deque()
    ee_traj = []
    guided_scores = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_traj.append(obs[:3].copy())

        if len(queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)

            if guided and score_fn is not None:
                ee_start = torch.tensor(
                    obs[:3], dtype=torch.float32, device=device)
                left_g = torch.tensor(
                    obs[8:11], dtype=torch.float32, device=device)
                right_g = torch.tensor(
                    obs[15:18], dtype=torch.float32, device=device)
                goals_t = torch.stack([left_g, right_g])
                act_seq, sc = sampler.sample(
                    model, obs_t, ee_start, goals_t,
                    score_fn=score_fn,
                    n_sampling_steps=n_sampling_steps)
                guided_scores.append(sc)
            else:
                act_seq = sampler.sample(
                    model, obs_t, n_sampling_steps=n_sampling_steps)

            chunk = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in chunk:
                queue.append(a)

        action = queue.popleft()
        result = env.step(action)
        obs = result.obs
        last_obs = obs
        success = (result.info.get("success_left", 0) > 0.5 or
                   result.info.get("success_right", 0) > 0.5)
        if result.done:
            break

    env.close()

    # Compute L_early from executed trajectory
    ee_arr = np.array(ee_traj)
    goals_np = np.stack([last_obs[8:11], last_obs[15:18]])
    T = len(ee_arr)
    early_end = max(2, int(T * 0.3))
    sigma = np.linalg.norm(goals_np[0] - goals_np[1]) / (
        2 * np.sqrt(2 * np.log(2)))

    posteriors = []
    for t in range(early_end):
        d0 = np.linalg.norm(ee_arr[t] - goals_np[0])
        d1 = np.linalg.norm(ee_arr[t] - goals_np[1])
        l0 = np.exp(-d0 ** 2 / (2 * sigma ** 2))
        l1 = np.exp(-d1 ** 2 / (2 * sigma ** 2))
        total = l0 + l1 + 1e-12
        posteriors.append([l0 / total, l1 / total])
    posteriors = np.array(posteriors)
    l_early = max(posteriors[:, 0].mean(), posteriors[:, 1].mean())
    true_goal = "left" if posteriors[:, 0].mean() >= posteriors[:, 1].mean() else "right"

    picked = "left" if result.info.get("success_left", 0) > 0.5 else (
             "right" if result.info.get("success_right", 0) > 0.5 else "none")

    return dict(
        success=success,
        picked=picked,
        steps=step + 1,
        L_early=float(l_early),
        true_goal=true_goal,
        guided_score_mean=float(np.mean(guided_scores)) if guided_scores else 0.0,
    )


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM-Distilled Classifier Guidance")
    sub = parser.add_subparsers(dest="command")

    # ── build-proxy ──
    bp = sub.add_parser("build-proxy",
                        help="Phase 1+2: Generate data + train proxy")
    bp.add_argument("--checkpoint", required=True)
    bp.add_argument("--behavior", required=True,
                    choices=list(BEHAVIOR_PROMPTS.keys()))
    bp.add_argument("--n_trajectories", type=int, default=200)
    bp.add_argument("--out_dir", type=str, default=None)
    bp.add_argument("--vlm_model", type=str,
                    default="gemini-2.5-flash-preview-05-20")

    # ── evaluate ──
    ev = sub.add_parser("evaluate",
                        help="Phase 3: Evaluate with VLM-distilled guidance")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--proxy_path", required=True)
    ev.add_argument("--guidance_scale", type=float, default=5.0)
    ev.add_argument("--n_episodes", type=int, default=20)
    ev.add_argument("--out_dir", type=str, default=None)

    # ── ablation ── (compare VLM-proxy vs L_early vs baseline)
    ab = sub.add_parser("ablation",
                        help="Compare: baseline vs L_early vs VLM-proxy")
    ab.add_argument("--checkpoint", required=True)
    ab.add_argument("--proxy_path", required=True)
    ab.add_argument("--guidance_scale", type=float, default=5.0)
    ab.add_argument("--n_episodes", type=int, default=20)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, obs_mean, obs_std, act_mean, act_std, cfg = load_policy(
        args.checkpoint, device)
    n_diff = cfg.get("n_diffusion_steps", 100)
    beta_s = cfg.get("beta_start", 1e-4)
    beta_e = cfg.get("beta_end", 0.1)

    if args.command == "build-proxy":
        out_dir = Path(args.out_dir) if args.out_dir else Path(
            f"outputs/vdcg/{args.behavior}")

        ee_trajs, goal_pos, tg_idxs, scores = build_proxy_dataset(
            model, obs_mean, obs_std, act_mean, act_std, cfg, device,
            args.behavior, args.n_trajectories, out_dir=out_dir,
            vlm_model=args.vlm_model)

        proxy, feat_mean, feat_std = train_proxy(
            ee_trajs, goal_pos, tg_idxs, scores, device,
            out_path=out_dir / "proxy.pt")

    elif args.command == "evaluate":
        # Load proxy
        proxy_data = torch.load(args.proxy_path, map_location=device,
                                weights_only=False)
        proxy = TrajectoryScoreProxy().to(device)
        proxy.load_state_dict(proxy_data["proxy_state"])
        proxy.eval()
        feat_mean = proxy_data["feat_mean"]
        feat_std = proxy_data["feat_std"]

        score_fn = make_proxy_score_fn(proxy, feat_mean, feat_std, device)

        guided_sampler = GuidedDDIMSampler(
            n_diff, beta_s, beta_e, device,
            guidance_scale=args.guidance_scale, grad_clip=1.0)
        baseline_sampler = DDIMSampler(n_diff, beta_s, beta_e, device)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir) if args.out_dir else Path(
            f"outputs/vdcg/eval_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  PHASE 3: VLM-Distilled Guidance Evaluation")
        print(f"  w={args.guidance_scale}, {args.n_episodes} episodes")
        print(f"{'=' * 60}\n")

        baseline_results = []
        guided_results = []

        for ep in range(args.n_episodes):
            # Baseline
            r_bl = run_eval_episode(
                model, baseline_sampler, obs_mean, obs_std,
                act_mean, act_std, device, guided=False)
            baseline_results.append(r_bl)

            # Guided
            r_gd = run_eval_episode(
                model, guided_sampler, obs_mean, obs_std,
                act_mean, act_std, device, guided=True,
                score_fn=score_fn)
            guided_results.append(r_gd)

            bl_ok = "OK" if r_bl["success"] else "FAIL"
            gd_ok = "OK" if r_gd["success"] else "FAIL"
            print(f"  Ep {ep + 1:2d}/{args.n_episodes}  "
                  f"BL: {bl_ok} L={r_bl['L_early']:.4f}  "
                  f"GD: {gd_ok} L={r_gd['L_early']:.4f} "
                  f"proxy={r_gd['guided_score_mean']:.4f}")

        # Summary
        bl_le = [r["L_early"] for r in baseline_results]
        gd_le = [r["L_early"] for r in guided_results]
        bl_s = [r["success"] for r in baseline_results]
        gd_s = [r["success"] for r in guided_results]

        from scipy import stats
        diffs = np.array(gd_le) - np.array(bl_le)
        if np.std(diffs) > 1e-8:
            t_stat, p_val = stats.ttest_rel(gd_le, bl_le)
        else:
            t_stat, p_val = 0.0, 1.0

        print(f"\n{'=' * 60}")
        print(f"  RESULTS (VLM-Distilled Guidance, w={args.guidance_scale})")
        print(f"{'=' * 60}")
        print(f"  Baseline:  success={np.mean(bl_s):.0%}  "
              f"L_early={np.mean(bl_le):.4f}±{np.std(bl_le):.4f}")
        print(f"  VLM-VDCG:  success={np.mean(gd_s):.0%}  "
              f"L_early={np.mean(gd_le):.4f}±{np.std(gd_le):.4f}")
        print(f"  Δ L_early: {np.mean(diffs):+.4f} (p={p_val:.4f})")
        print(f"{'=' * 60}")

        output = dict(
            guidance_scale=args.guidance_scale,
            n_episodes=args.n_episodes,
            proxy_path=str(args.proxy_path),
            baseline=[{k: (bool(v) if isinstance(v, (bool, np.bool_))
                           else float(v) if isinstance(v, (float, np.floating))
                           else v)
                       for k, v in r.items()} for r in baseline_results],
            guided=[{k: (bool(v) if isinstance(v, (bool, np.bool_))
                         else float(v) if isinstance(v, (float, np.floating))
                         else v)
                     for k, v in r.items()} for r in guided_results],
            summary=dict(
                baseline_success=float(np.mean(bl_s)),
                guided_success=float(np.mean(gd_s)),
                baseline_L_early=float(np.mean(bl_le)),
                guided_L_early=float(np.mean(gd_le)),
                delta=float(np.mean(diffs)),
                p_value=float(p_val),
            ),
        )
        with open(out_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results: {out_dir / 'results.json'}")

    elif args.command == "ablation":
        proxy_data = torch.load(args.proxy_path, map_location=device,
                                weights_only=False)
        proxy = TrajectoryScoreProxy().to(device)
        proxy.load_state_dict(proxy_data["proxy_state"])
        proxy.eval()
        feat_mean = proxy_data["feat_mean"]
        feat_std = proxy_data["feat_std"]

        proxy_fn = make_proxy_score_fn(proxy, feat_mean, feat_std, device)

        baseline_sampler = DDIMSampler(n_diff, beta_s, beta_e, device)
        learly_sampler = GuidedDDIMSampler(
            n_diff, beta_s, beta_e, device,
            guidance_scale=args.guidance_scale, grad_clip=1.0)
        proxy_sampler = GuidedDDIMSampler(
            n_diff, beta_s, beta_e, device,
            guidance_scale=args.guidance_scale, grad_clip=1.0)

        print(f"\n{'=' * 60}")
        print(f"  ABLATION: Baseline vs L_early vs VLM-Proxy")
        print(f"  w={args.guidance_scale}, {args.n_episodes} episodes")
        print(f"{'=' * 60}\n")

        results = {"baseline": [], "l_early": [], "vlm_proxy": []}

        for ep in range(args.n_episodes):
            r_bl = run_eval_episode(
                model, baseline_sampler, obs_mean, obs_std,
                act_mean, act_std, device, guided=False)
            r_le = run_eval_episode(
                model, learly_sampler, obs_mean, obs_std,
                act_mean, act_std, device, guided=True,
                score_fn=l_early_intent_torch)
            r_vp = run_eval_episode(
                model, proxy_sampler, obs_mean, obs_std,
                act_mean, act_std, device, guided=True,
                score_fn=proxy_fn)

            results["baseline"].append(r_bl)
            results["l_early"].append(r_le)
            results["vlm_proxy"].append(r_vp)

            print(f"  Ep {ep + 1:2d}  "
                  f"BL: L={r_bl['L_early']:.4f}  "
                  f"LE: L={r_le['L_early']:.4f}  "
                  f"VP: L={r_vp['L_early']:.4f}")

        for name, rs in results.items():
            les = [r["L_early"] for r in rs]
            suc = [r["success"] for r in rs]
            print(f"\n  {name:12s}: success={np.mean(suc):.0%}  "
                  f"L_early={np.mean(les):.4f}±{np.std(les):.4f}")


if __name__ == "__main__":
    main()
