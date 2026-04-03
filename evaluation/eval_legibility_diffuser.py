#!/usr/bin/env python3
"""Evaluate Legibility Diffuser (Bronars et al. RA-L 2024) vs LPS vs Baseline.

Loads:
  --checkpoint        : trained GoalCondDiffusionPolicy (from train_legibility_diffuser.py)
  --baseline_checkpoint: unconditional DiffusionPolicy (88-92% baseline)

Three conditions compared:
  1. Baseline   — unconditioned UNet DDIM  (cube_jitter=0.0, 10 steps)
  2. LPS        — DPS gradient guidance on baseline model (from eval_legibility_guided.py)
  3. LegDiff    — Legibility Diffuser CFG (ε̂ = ε_uncond + w*(ε_cond − ε_uncond))

Goal detection (LegDiff):
  First planning step: sample unconditioned trajectory, check which block EE approaches.
  Lock in that goal_id for all subsequent replans in the episode.

CFG inference:
  ε̂ = ε_uncond + cfg_scale * (ε_cond − ε_uncond)   (exact Bronars 2024 formula)

Usage:
  python scripts/eval_legibility_diffuser.py \\
      --checkpoint runs/legdiff_*/ckpt.pt \\
      --baseline_checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt \\
      --n_episodes 20 --cfg_scale 3.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# ── Goal constants ────────────────────────────────────────────────────
GOAL_LEFT  = 0
GOAL_RIGHT = 1
NULL_GOAL  = 2
ACTION_SCALE = 0.05   # TwoBlockPickEnv delta-pos scale

DEFAULT_BASELINE = 'runs/diffusion_20260222_195530/ckpt_ep100.pt'


# ══════════════════════════════════════════════════════════════════════
# BASELINE MODEL — identical to eval_with_videos.py / eval_legibility_guided.py
# ══════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class UNetBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, time_dim: int) -> None:
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.conv1     = nn.Linear(in_dim, out_dim)
        self.conv2     = nn.Linear(out_dim, out_dim)
        self.shortcut  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm1     = nn.GroupNorm(8, out_dim)
        self.norm2     = nn.GroupNorm(8, out_dim)
        self.act       = nn.Mish()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h.transpose(1, 2); h = self.norm1(h); h = h.transpose(1, 2)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        h = self.conv2(h)
        h = h.transpose(1, 2); h = self.norm2(h); h = h.transpose(1, 2)
        return self.act(h + self.shortcut(x))


# ══════════════════════════════════════════════════════════════════
# TRUE 1D CONV BUILDING BLOCKS (replaces MLP inside UNetBlock)
# ══════════════════════════════════════════════════════════════════

class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish.  Tensor shape: (B, C, T) channels-first."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1dResBlock(nn.Module):
    """Residual block with two Conv1dBlocks and FiLM conditioning.

    x    : (B, C, T) channels-first
    cond : (B, cond_dim) global condition (time + obs + optional goal)

    FiLM injects cond as additive bias broadcast over T:
        h = conv1(x) + cond_proj(cond).unsqueeze(-1)
        h = conv2(h)
        return h + shortcut(x)
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 kernel_size: int = 5) -> None:
        super().__init__()
        self.block1    = Conv1dBlock(in_ch,  out_ch, kernel_size)
        self.block2    = Conv1dBlock(out_ch, out_ch, kernel_size)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.shortcut  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.cond_proj(cond).unsqueeze(-1)   # FiLM bias, broadcast over T
        h = self.block2(h)
        return h + self.shortcut(x)


class DiffusionPolicy(nn.Module):
    """Unconditional UNet diffusion policy (baseline)."""

    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128), nn.Linear(128, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim) for i in range(len(dims)-1)])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1]+dims[i+1], dims[i], hidden_dim)
            for i in range(len(dims)-2, -1, -1)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, act_dim),
        )

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
# LEGIBILITY DIFFUSER MODEL — goal-conditioned version
# ══════════════════════════════════════════════════════════════════════

class GoalCondDiffusionPolicy(nn.Module):
    """Legibility Diffuser with TRUE 1D Conv temporal UNet (Bronars RA-L 2024).

    Architecture (channels-first throughout):
      Encoder  : Conv1d(act_dim→256) → ResBlock(256→256) → ResBlock(256→512)
      Bottleneck: ResBlock(512→512)
      Decoder  : cat(512 skip) → ResBlock(1024→512) → ResBlock(512→256)
      Output   : GN+Mish+Conv1d(256→act_dim)

    Key: Conv1d(k=5) lets each output timestep see its ±2 temporal
    neighbours in the prediction horizon — the MLP version could not.
    """

    NULL_GOAL = NULL_GOAL

    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        cond_dim     = hidden_dim

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128), nn.Linear(128, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.goal_embed = nn.Embedding(3, cond_dim)   # 0=left, 1=right, 2=null
        self.input_proj = nn.Conv1d(act_dim, hidden_dim, 1)

        dims = [hidden_dim, hidden_dim * 2]   # [256, 512]
        self.enc_block1 = Conv1dResBlock(dims[0], dims[0], cond_dim)
        self.enc_block2 = Conv1dResBlock(dims[0], dims[1], cond_dim)
        self.bottleneck = Conv1dResBlock(dims[1], dims[1], cond_dim)
        self.dec_block1 = Conv1dResBlock(dims[1] * 2, dims[1], cond_dim)
        self.dec_block2 = Conv1dResBlock(dims[1],     dims[0], cond_dim)
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, dims[0]), nn.Mish(),
            nn.Conv1d(dims[0], act_dim, 1),
        )

    def forward(self, noisy_act, timestep, obs, goal_id):
        x    = noisy_act.permute(0, 2, 1)                    # (B, act_dim, H)
        x    = self.input_proj(x)                             # (B, 256, H)
        cond = (self.time_mlp(timestep)
                + self.obs_embed(obs)
                + self.goal_embed(goal_id))                   # (B, 256)
        x    = self.enc_block1(x, cond)                      # (B, 256, H)
        x    = self.enc_block2(x, cond)                      # (B, 512, H)
        skip = x
        x    = self.bottleneck(x, cond)                      # (B, 512, H)
        x    = torch.cat([x, skip], dim=1)                   # (B, 1024, H)
        x    = self.dec_block1(x, cond)                      # (B, 512, H)
        x    = self.dec_block2(x, cond)                      # (B, 256, H)
        return self.output_proj(x).permute(0, 2, 1)          # (B, H, act_dim)


class GoalCondDiffusionPolicyMLP(nn.Module):
    """LEGACY MLP version — kept only to load old checkpoints (arch='mlp')."""

    NULL_GOAL = NULL_GOAL

    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128), nn.Linear(128, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.goal_embed = nn.Embedding(3, hidden_dim)
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim) for i in range(len(dims)-1)])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1]+dims[i+1], dims[i], hidden_dim)
            for i in range(len(dims)-2, -1, -1)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, noisy_act, timestep, obs, goal_id):
        t_emb    = self.time_mlp(timestep)
        obs_emb  = self.obs_embed(obs)
        goal_emb = self.goal_embed(goal_id)
        x = self.input_proj(noisy_act) + (obs_emb + goal_emb).unsqueeze(1)
        skips = []
        for blk in self.encoder_blocks:
            x = blk(x, t_emb); skips.append(x)
        x = self.bottleneck(x, t_emb)
        for blk, sk in zip(self.decoder_blocks, reversed(skips)):
            x = blk(torch.cat([x, sk], dim=-1), t_emb)
        return self.output_proj(x)


# ══════════════════════════════════════════════════════════════════════
# DDIM SAMPLERS
# ══════════════════════════════════════════════════════════════════════

def _make_alphas(n_steps, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
    return torch.cumprod(1.0 - betas, dim=0)   # alphas_cumprod


class BaselineDDIMSampler:
    """Standard DDIM — matches eval_with_videos.py exactly."""

    def __init__(self, n_steps, beta_start, beta_end, device):
        self.device = device
        self.alphas_cumprod = _make_alphas(n_steps, beta_start, beta_end, device)

    @torch.no_grad()
    def sample(self, model: DiffusionPolicy, obs: torch.Tensor,
               n_sampling_steps=10) -> torch.Tensor:
        B  = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x  = torch.randn(B, H, A, device=self.device)
        ts = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod)-1,
                           n_sampling_steps, device=self.device).long(), [0])
        for i, t in enumerate(ts):
            t_b = t.repeat(B)
            eps = model(x, t_b, obs)
            a   = self.alphas_cumprod[t]
            a_p = self.alphas_cumprod[ts[i+1]] if i < len(ts)-1 else torch.tensor(1.0, device=self.device)
            x0  = (x - torch.sqrt(1-a) * eps) / torch.sqrt(a)
            x   = torch.sqrt(a_p) * x0 + torch.sqrt(1-a_p) * eps if i < len(ts)-1 else x0
        return x


class LegDiffDDIMSampler:
    """Legibility Diffuser DDIM sampler with CFG.

    CFG formula (Bronars 2024):
        ε̂ = ε_uncond + w * (ε_cond − ε_uncond)

    Goal detection: at first plan, unconditioned sample → check which block
    the end-effector trajectory approaches → lock in goal for entire episode.
    """

    NULL_ID = NULL_GOAL

    def __init__(self, n_steps, beta_start, beta_end, device, cfg_scale=3.0):
        self.device          = device
        self.cfg_scale       = cfg_scale
        self.alphas_cumprod  = _make_alphas(n_steps, beta_start, beta_end, device)

    @torch.no_grad()
    def detect_goal(
        self,
        model: GoalCondDiffusionPolicy,
        obs_t: torch.Tensor,   # (1, obs_dim) — already normalised
        obs_raw: np.ndarray,   # (obs_dim,) — raw, for goal positions
        n_sampling_steps: int = 10,
    ) -> int:
        """Sample unconditionally and see which block the EE trajectory approaches."""
        null_id = torch.full((1,), self.NULL_ID, dtype=torch.long, device=self.device)
        chunk   = self.sample_cfg(model, obs_t, null_id,
                                  cfg_scale=0.0,
                                  n_sampling_steps=n_sampling_steps)
        ee_start   = obs_raw[0:3]
        delta_pos  = chunk[0, :, :3].cpu().numpy() * ACTION_SCALE
        ee_traj    = np.cumsum(delta_pos, axis=0) + ee_start   # (H, 3)
        final_ee   = ee_traj[-1]
        left_goal  = obs_raw[8:11]
        right_goal = obs_raw[15:18]
        d_left  = np.linalg.norm(final_ee - left_goal)
        d_right = np.linalg.norm(final_ee - right_goal)
        return GOAL_LEFT if d_left <= d_right else GOAL_RIGHT

    @torch.no_grad()
    def sample_cfg(
        self,
        model: GoalCondDiffusionPolicy,
        obs: torch.Tensor,      # (1, obs_dim)
        goal_id: torch.Tensor,  # (1,) — GOAL_LEFT / GOAL_RIGHT
        cfg_scale: Optional[float] = None,
        n_sampling_steps: int = 10,
    ) -> torch.Tensor:
        """CFG-guided DDIM sample.

        When cfg_scale=0 the goal_id is still passed but uncond==cond so
        output equals a plain conditional sample (used for goal detection).
        """
        w  = self.cfg_scale if cfg_scale is None else cfg_scale
        B  = obs.shape[0]
        H  = model.horizon
        A  = model.act_dim
        x  = torch.randn(B, H, A, device=self.device)
        ts = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod)-1,
                           n_sampling_steps, device=self.device).long(), [0])

        null_id = torch.full_like(goal_id, self.NULL_ID)

        for i, t in enumerate(ts):
            t_b = t.repeat(B)
            a   = self.alphas_cumprod[t]
            a_p = self.alphas_cumprod[ts[i+1]] if i < len(ts)-1 else torch.tensor(1.0, device=self.device)

            eps_cond   = model(x, t_b, obs, goal_id)
            if w == 0.0:
                eps_cfg = eps_cond
            else:
                eps_uncond = model(x, t_b, obs, null_id)
                eps_cfg    = eps_uncond + w * (eps_cond - eps_uncond)

            x0 = (x - torch.sqrt(1-a) * eps_cfg) / torch.sqrt(a)
            x  = torch.sqrt(a_p) * x0 + torch.sqrt(1-a_p) * eps_cfg if i < len(ts)-1 else x0

        return x

    @property
    def Optional(self):
        from typing import Optional as Opt
        return Opt


# Optional import — we fix the reference above
Optional = None   # will be resolved at runtime (used in type hint only)


# ══════════════════════════════════════════════════════════════════════
# L_EARLY_INTENT  (differentiable Bayesian posterior, Dragan 2013)
# ══════════════════════════════════════════════════════════════════════

def l_early_intent(
    ee_traj: np.ndarray,    # (T, 3) actual EE positions
    goals:   np.ndarray,    # (2, 3) goal positions
    true_goal_idx: int = 0,
    early_frac: float = 0.30,
) -> float:
    """Numpy version of L_early_intent for episode reporting."""
    T      = len(ee_traj)
    K      = goals.shape[0]
    end    = max(1, int(T * early_frac))
    early  = ee_traj[:end]       # (end, 3)
    dists  = np.linalg.norm(goals[None] - goals[:, None], axis=-1)
    d_min  = dists[dists > 1e-6].min() if (dists > 1e-6).any() else 0.14
    sigma  = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))
    diff   = early[:, None] - goals[None]          # (end, K, 3)
    sq_dist = (diff**2).sum(-1)                    # (end, K)
    log_p  = -sq_dist / (2.0 * sigma**2)
    log_p  -= log_p.max(-1, keepdims=True)[0]
    p      = np.exp(log_p); p /= p.sum(-1, keepdims=True)
    return float(p[:, true_goal_idx].mean())


# ══════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_baseline_episode(
    model: DiffusionPolicy,
    sampler: BaselineDDIMSampler,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_mean: np.ndarray,
    act_std: np.ndarray,
    device,
    n_sampling_steps: int = 10,
    cube_jitter: float = 0.0,
    max_steps: int = 400,
) -> dict:
    """Baseline unconditional episode — matches eval_with_videos.py exactly."""
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=cube_jitter)
    obs = env.reset()
    queue: deque = deque(maxlen=model.horizon)
    ee_traj: List[np.ndarray] = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_traj.append(obs[0:3].copy())
        if len(queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t    = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            chunk    = sampler.sample(model, obs_t, n_sampling_steps)
            for a in (chunk[0].cpu().numpy() * act_std + act_mean):
                queue.append(a)
        action  = queue.popleft()
        result  = env.step(action)
        obs     = result.obs; last_obs = obs
        success = result.info.get('success_left', 0) > 0.5 or result.info.get('success_right', 0) > 0.5
        if result.done:
            break

    env.close()
    le, goal = _measure_legibility(np.array(ee_traj), last_obs)
    return dict(success=success, steps=step+1, l_early=le, true_goal=goal)


def run_legdiff_episode(
    model: GoalCondDiffusionPolicy,
    sampler: LegDiffDDIMSampler,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_mean: np.ndarray,
    act_std: np.ndarray,
    device,
    cfg_scale: float = 3.0,
    n_sampling_steps: int = 10,
    cube_jitter: float = 0.0,
    max_steps: int = 400,
) -> dict:
    """Legibility Diffuser episode with CFG goal conditioning."""
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=cube_jitter)
    obs = env.reset()
    queue: deque = deque(maxlen=model.horizon)
    ee_traj: List[np.ndarray] = []
    success = False
    last_obs = obs
    committed_goal: Optional[torch.Tensor] = None

    for step in range(max_steps):
        ee_traj.append(obs[0:3].copy())
        if len(queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t    = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            # Detect goal on first plan and lock in for rest of episode
            if committed_goal is None:
                gid = sampler.detect_goal(model, obs_t, obs, n_sampling_steps)
                committed_goal = torch.tensor([gid], dtype=torch.long, device=device)

            chunk = sampler.sample_cfg(model, obs_t, committed_goal,
                                       cfg_scale=cfg_scale,
                                       n_sampling_steps=n_sampling_steps)
            for a in (chunk[0].cpu().numpy() * act_std + act_mean):
                queue.append(a)

        action  = queue.popleft()
        result  = env.step(action)
        obs     = result.obs; last_obs = obs
        success = result.info.get('success_left', 0) > 0.5 or result.info.get('success_right', 0) > 0.5
        if result.done:
            break

    env.close()
    le, goal = _measure_legibility(np.array(ee_traj), last_obs)
    return dict(success=success, steps=step+1, l_early=le, true_goal=goal,
                committed_goal='left' if committed_goal is not None and committed_goal.item() == GOAL_LEFT else 'right')


def _measure_legibility(ee_traj: np.ndarray, last_obs: np.ndarray) -> Tuple[float, str]:
    if len(ee_traj) < 4:
        return 0.0, 'unknown'
    goals = np.stack([last_obs[8:11], last_obs[15:18]])   # (2, 3)
    l0 = l_early_intent(ee_traj, goals, 0)
    l1 = l_early_intent(ee_traj, goals, 1)
    if l0 >= l1:
        return l0, 'left'
    return l1, 'right'


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def _load_baseline(path: str, device) -> Tuple[DiffusionPolicy, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt['config']
    m    = DiffusionPolicy(
        obs_dim    = cfg['obs_dim'],
        act_dim    = cfg['act_dim'],
        horizon    = cfg['horizon'],
        hidden_dim = cfg.get('hidden_dim', 256),
        n_blocks   = cfg.get('n_blocks', 3),
    ).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m, ckpt


def _load_legdiff(path: str, device) -> Tuple[GoalCondDiffusionPolicy, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt['config']
    arch = cfg.get('arch', 'mlp')   # old checkpoints have no arch key → mlp

    if arch == 'conv1d':
        cls = GoalCondDiffusionPolicy
    else:
        cls = GoalCondDiffusionPolicyMLP   # backward compat for pre-conv1d runs

    m = cls(
        obs_dim    = cfg['obs_dim'],
        act_dim    = cfg['act_dim'],
        horizon    = cfg['horizon'],
        hidden_dim = cfg.get('hidden_dim', 256),
        n_blocks   = cfg.get('n_blocks', 3),
    ).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m, ckpt


def _summary(label: str, results: List[dict], cfg_scale=None) -> None:
    le = [r['l_early'] for r in results]
    sr = [r['success'] for r in results]
    tag = f"(w={cfg_scale})" if cfg_scale is not None else ""
    print(f"\n  {label} {tag}:")
    print(f"    L_early_intent : {np.mean(le):.4f} ± {np.std(le):.4f}")
    print(f"    Success rate   : {np.mean(sr):.1%}  ({sum(sr)}/{len(sr)})")
    print(f"    Steps (mean)   : {np.mean([r['steps'] for r in results]):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--checkpoint',
                    required=True,
                    help='Legibility Diffuser checkpoint (train_legibility_diffuser.py)')
    ap.add_argument('--baseline_checkpoint',
                    default=DEFAULT_BASELINE,
                    help='Unconditional baseline checkpoint (default: 88-92% model)')
    ap.add_argument('--n_episodes',     type=int,   default=20)
    ap.add_argument('--cfg_scale',      type=float, default=3.0,
                    help='CFG guidance strength w (Bronars 2024 uses 3–7.5)')
    ap.add_argument('--n_sampling_steps', type=int, default=10)
    ap.add_argument('--cube_jitter',    type=float, default=0.0)
    ap.add_argument('--skip_baseline',  action='store_true',
                    help='Skip baseline evaluation (faster)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print("  Legibility Diffuser (Bronars RA-L 2024) — Evaluation")
    print(f"{'='*65}")
    print(f"  Device          : {device}")
    print(f"  LegDiff ckpt    : {args.checkpoint}")
    print(f"  Baseline ckpt   : {args.baseline_checkpoint}")
    print(f"  CFG scale w     : {args.cfg_scale}")
    print(f"  DDIM steps      : {args.n_sampling_steps}")
    print(f"  Episodes        : {args.n_episodes} per condition")
    print(f"{'='*65}\n")

    # ── Load models ───────────────────────────────────────────────────
    ld_model, ld_ckpt = _load_legdiff(args.checkpoint, device)
    ld_cfg = ld_ckpt['config']
    print(f"  LegDiff model:  {sum(p.numel() for p in ld_model.parameters()):,} params, "
          f"epoch {ld_ckpt['epoch']}, loss {ld_ckpt['loss']:.6f}")

    ld_sampler = LegDiffDDIMSampler(
        n_steps    = ld_cfg['n_diffusion_steps'],
        beta_start = ld_cfg['beta_start'],
        beta_end   = ld_cfg['beta_end'],
        device     = device,
        cfg_scale  = args.cfg_scale,
    )
    ld_obs_mean = np.array(ld_ckpt['obs_mean'], dtype=np.float32)
    ld_obs_std  = np.array(ld_ckpt['obs_std'],  dtype=np.float32)
    ld_act_mean = np.array(ld_ckpt['act_mean'], dtype=np.float32)
    ld_act_std  = np.array(ld_ckpt['act_std'],  dtype=np.float32)

    # ── 1. Baseline ──────────────────────────────────────────────────
    baseline_results = []
    if not args.skip_baseline:
        bl_model, bl_ckpt = _load_baseline(args.baseline_checkpoint, device)
        bl_cfg = bl_ckpt['config']
        print(f"  Baseline model: {sum(p.numel() for p in bl_model.parameters()):,} params, "
              f"epoch {bl_ckpt['epoch']}, loss {bl_ckpt['loss']:.6f}")
        bl_sampler  = BaselineDDIMSampler(
            n_steps    = bl_cfg['n_diffusion_steps'],
            beta_start = bl_cfg['beta_start'],
            beta_end   = bl_cfg['beta_end'],
            device     = device,
        )
        bl_obs_mean = np.array(bl_ckpt['obs_mean'], dtype=np.float32)
        bl_obs_std  = np.array(bl_ckpt['obs_std'],  dtype=np.float32)
        bl_act_mean = np.array(bl_ckpt['act_mean'], dtype=np.float32)
        bl_act_std  = np.array(bl_ckpt['act_std'],  dtype=np.float32)

        print(f"\n── BASELINE (unconditioned) ── {args.n_episodes} episodes ──")
        for ep in range(args.n_episodes):
            r = run_baseline_episode(
                bl_model, bl_sampler,
                bl_obs_mean, bl_obs_std, bl_act_mean, bl_act_std,
                device, args.n_sampling_steps, args.cube_jitter)
            baseline_results.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
                  f"L_early={r['l_early']:.4f}  goal={r['true_goal']}  steps={r['steps']}")
        _summary("BASELINE", baseline_results)

    # ── 2. Legibility Diffuser (CFG) ─────────────────────────────────
    print(f"\n── LEGIBILITY DIFFUSER (CFG w={args.cfg_scale}) ── {args.n_episodes} episodes ──")
    legdiff_results = []
    for ep in range(args.n_episodes):
        r = run_legdiff_episode(
            ld_model, ld_sampler,
            ld_obs_mean, ld_obs_std, ld_act_mean, ld_act_std,
            device, args.cfg_scale, args.n_sampling_steps, args.cube_jitter)
        legdiff_results.append(r)
        tick = '✓' if r['success'] else '✗'
        print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
              f"L_early={r['l_early']:.4f}  "
              f"committed={r.get('committed_goal','?')}  "
              f"actual={r['true_goal']}  steps={r['steps']}")
    _summary("LEGIBILITY DIFFUSER", legdiff_results, cfg_scale=args.cfg_scale)

    # ── 3. Comparison ────────────────────────────────────────────────
    if baseline_results and legdiff_results:
        bl  = np.mean([r['l_early'] for r in baseline_results])
        ld  = np.mean([r['l_early'] for r in legdiff_results])
        delta = ld - bl
        print(f"\n{'='*65}")
        print("  LEGIBILITY IMPROVEMENT (LegDiff over baseline)")
        print(f"{'='*65}")
        print(f"  Baseline  L_early : {bl:.4f}")
        print(f"  LegDiff   L_early : {ld:.4f}  Δ={delta:+.4f}  ({delta/max(bl,1e-6)*100:+.1f}%)")
        print(f"  Success   Baseline: {np.mean([r['success'] for r in baseline_results]):.1%}  "
              f"LegDiff: {np.mean([r['success'] for r in legdiff_results]):.1%}")

    # ── 4. Save ──────────────────────────────────────────────────────
    out_dir  = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

    def _s(v):
        if isinstance(v, (bool, np.bool_)):    return bool(v)
        if isinstance(v, (float, np.floating)): return float(v)
        if isinstance(v, (int, np.integer)):   return int(v)
        return v

    out = dict(
        checkpoint          = str(args.checkpoint),
        baseline_checkpoint = str(args.baseline_checkpoint),
        cfg_scale           = args.cfg_scale,
        n_sampling_steps    = args.n_sampling_steps,
        n_episodes          = args.n_episodes,
        baseline            = [{k: _s(vv) for k, vv in r.items()} for r in baseline_results],
        legdiff             = [{k: _s(vv) for k, vv in r.items()} for r in legdiff_results],
    )
    out_path = out_dir / 'legdiff_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    main()
