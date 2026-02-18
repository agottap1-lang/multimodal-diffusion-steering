#!/usr/bin/env python
"""Train an unconditional DDPM diffusion policy for TwoBlockPick.

Usage:
    python scripts/train_diffusion_policy.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Sinusoidal timestep embedding ────────────────────────────────────

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


# ── Residual MLP block with FiLM time conditioning ──────────────────

class ResBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.time_fc = nn.Linear(time_dim, dim)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = h + self.time_fc(t_emb)          # FiLM-style conditioning
        h = self.act(self.fc2(h))
        return x + h                          # residual


# ── Noise prediction network ────────────────────────────────────────

class NoiseNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        n_blocks: int = 6,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon

        # time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # input projection
        input_dim = horizon * act_dim + obs_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # residual blocks
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, time_embed_dim) for _ in range(n_blocks)]
        )

        # output projection
        self.out_proj = nn.Linear(hidden_dim, horizon * act_dim)

    def forward(
        self,
        noisy_act: torch.Tensor,   # (B, H, act_dim)
        timestep: torch.Tensor,    # (B,)
        obs: torch.Tensor,         # (B, obs_dim)
    ) -> torch.Tensor:
        B = noisy_act.shape[0]
        t_emb = self.time_embed(timestep)                    # (B, t_dim)
        x = torch.cat([noisy_act.reshape(B, -1), obs], -1)  # (B, H*A+O)
        x = self.in_proj(x)                                  # (B, D)
        for blk in self.blocks:
            x = blk(x, t_emb)
        out = self.out_proj(x)                               # (B, H*A)
        return out.reshape(B, self.horizon, self.act_dim)


# ── DDPM schedule & helpers ──────────────────────────────────────────

class DDPMSchedule:
    def __init__(self, n_steps: int, beta_start: float, beta_end: float,
                 device: torch.device) -> None:
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps,
                               dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, 0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_ab = torch.sqrt(alpha_bar)
        self.sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise at level t."""
        s1 = self.sqrt_ab[t].reshape(-1, 1, 1)
        s2 = self.sqrt_1m_ab[t].reshape(-1, 1, 1)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model: NoiseNet, xt: torch.Tensor,
                 t_int: int, obs: torch.Tensor) -> torch.Tensor:
        """One reverse-diffusion step (DDPM)."""
        B = xt.shape[0]
        t_tensor = torch.full((B,), t_int, device=xt.device, dtype=torch.long)
        eps_pred = model(xt, t_tensor, obs)

        alpha = self.alphas[t_int]
        alpha_bar = self.alpha_bar[t_int]
        beta = self.betas[t_int]

        coef = beta / self.sqrt_1m_ab[t_int]
        mean = (1.0 / torch.sqrt(alpha)) * (xt - coef * eps_pred)

        if t_int > 0:
            noise = torch.randn_like(xt)
            sigma = torch.sqrt(beta)
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def p_sample_ddim(self, model: NoiseNet, xt: torch.Tensor,
                      t_int: int, t_prev_int: int, obs: torch.Tensor, eta: float = 0.0,
                      debug: bool = False, step_idx: int = -1) -> torch.Tensor:
        """One reverse-diffusion step (DDIM - deterministic when eta=0).
        
        Args:
            t_int: Current timestep index (noise level we're AT)
            t_prev_int: Next timestep index (noise level we're GOING TO)
                        For strided schedules, t_prev != t-1
            eta: Stochasticity control (0=deterministic, 1=DDPM-like)
            debug: Print internal values for first 5 steps
            step_idx: Which denoising step this is (0=first, for debug prints)
        
        Key fix: Explicitly pass t_prev instead of assuming t-1.
        This allows strided DDIM schedules (e.g., [99,80,60,40,20,0] for 6 steps).
        """
        B = xt.shape[0]
        t_tensor = torch.full((B,), t_int, device=xt.device, dtype=torch.long)
        eps_pred = model(xt, t_tensor, obs)

        # Assert finite prediction
        if not torch.isfinite(eps_pred).all():
            raise AssertionError(f"DDIM: eps_pred not finite at t={t_int}")

        # Use explicit t_prev_int (NOT t_int-1) to support strided schedules
        alpha_bar_t = self.alpha_bar[t_int]
        alpha_bar_t_prev = self.alpha_bar[t_prev_int] if t_prev_int >= 0 else torch.tensor(1.0, device=xt.device)
        
        # D) INSTRUMENT DDIM INTERNALS (first 5 steps only)
        if debug and step_idx >= 0 and step_idx < 5:
            print(f"\n  [DDIM step {step_idx}] t={t_int} → t_prev={t_prev_int}")
            print(f"    alpha_bar[{t_int}]:      {alpha_bar_t.item():.6f}")
            print(f"    alpha_bar[{t_prev_int}]: {alpha_bar_t_prev.item():.6f}")
            print(f"    eta:                     {eta:.6f}")
            print(f"    xt std:                  {xt.std().item():.6f}")
            print(f"    eps_pred std:            {eps_pred.std().item():.6f}")
        
        # Clamp to avoid numerical issues
        alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-8, max=1.0)
        alpha_bar_t_prev = torch.clamp(alpha_bar_t_prev, min=1e-8, max=1.0)
        
        # Compute sigma_t using DDPM posterior variance scaled by eta
        sigma_t = torch.zeros_like(alpha_bar_t)
        if eta > 0 and t_prev_int >= 0:
            # Posterior variance: beta_tilde_t
            beta_t = 1.0 - alpha_bar_t / alpha_bar_t_prev
            beta_t = torch.clamp(beta_t, min=0.0, max=0.999)
            variance = (1.0 - alpha_bar_t_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-8) * beta_t
            variance = torch.clamp(variance, min=0.0, max=1.0)
            sigma_t = eta * torch.sqrt(variance)
        
        if debug and step_idx >= 0 and step_idx < 5:
            print(f"    sigma_t:                 {sigma_t.item():.6f}")
        
        # DIRECT FORMULA (avoids x0 prediction):
        # x_{t-1} = a * x_t + b * eps_pred + sigma_t * z
        # where:
        #   a = sqrt(alpha_bar_t_prev / alpha_bar_t)
        #   b = sqrt(1 - alpha_bar_t_prev - sigma_t^2) - a * sqrt(1 - alpha_bar_t)
        
        a_coef = torch.sqrt(alpha_bar_t_prev / alpha_bar_t)
        b_coef = torch.sqrt(torch.clamp(1.0 - alpha_bar_t_prev - sigma_t**2, min=0.0)) - a_coef * torch.sqrt(1.0 - alpha_bar_t)
        
        if debug and step_idx >= 0 and step_idx < 5:
            print(f"    a_coef (xt scale):       {a_coef.item():.6f}")
            print(f"    b_coef (eps scale):      {b_coef.item():.6f}")
            print(f"    term1 (a*xt) std:        {(a_coef * xt).std().item():.6f}")
            print(f"    term2 (b*eps) std:       {(b_coef * eps_pred).std().item():.6f}")
        
        x_prev = a_coef * xt + b_coef * eps_pred
        
        if debug and step_idx >= 0 and step_idx < 5:
            print(f"    x_prev (before noise) std: {x_prev.std().item():.6f}")
        
        #Add stochastic noise if eta > 0
        if eta > 0 and t_prev_int >= 0:
            noise = torch.randn_like(xt)
            x_prev = x_prev + sigma_t * noise
            
            if debug and step_idx >= 0 and step_idx < 5:
                print(f"    noise std:               {noise.std().item():.6f}")
                print(f"    x_prev (after noise) std: {x_prev.std().item():.6f}")
        
        # Final assert for finite output
        if not torch.isfinite(x_prev).all():
            raise AssertionError(f"DDIM: x_prev not finite at t={t_int}→{t_prev_int}, eta={eta}")
        if torch.isnan(x_prev).any() or torch.isinf(x_prev).any():
            raise ValueError(f"DDIM produced NaN/inf at t={t_int}→{t_prev_int}, eta={eta}")
            
        return x_prev

    @torch.no_grad()
    def sample(self, model: NoiseNet, obs: torch.Tensor,
               horizon: int, act_dim: int, method: str = 'ddpm', eta: float = 0.0,
               ddim_steps: int | None = None, ddim_debug: bool = False) -> torch.Tensor:
        """Full diffusion sampling.
        
        method: 'ddpm' (stochastic, original) or 'ddim' (deterministic/stochastic)
        eta: DDIM stochasticity parameter
            - eta=0: fully deterministic (same obs → same actions, NO multimodality)
            - eta=1: equivalent to DDPM (stochastic)
            - 0 < eta < 1: controlled stochasticity
        ddim_steps: Number of inference steps for DDIM (default: use all n_steps)
                    Typical values: 20-50 for fast sampling
                    Using fewer steps is the KEY ADVANTAGE of DDIM!
        ddim_debug: Print timestep schedule and first 5 DDIM steps
        """
        B = obs.shape[0]
        x = torch.randn(B, horizon, act_dim, device=obs.device)
        
        if method == 'ddim':
            # Create strided DDIM schedule
            if ddim_steps is None or ddim_steps >= self.n_steps:
                # Default: use all timesteps (old behavior for compatibility)
                timesteps = list(reversed(range(self.n_steps)))
            else:
                # Strided schedule: evenly spaced timesteps from n_steps-1 to 0
                # Example: ddim_steps=20 on n_steps=100 → [99, 95, 90, ..., 5, 0]
                timesteps = list(reversed(torch.linspace(0, self.n_steps - 1, ddim_steps, dtype=torch.long).tolist()))
            
            if ddim_debug:
                print(f"\n[DDIM SCHEDULE] Using {len(timesteps)} steps (eta={eta})")
                print(f"  Timesteps: {timesteps[:10]}{'...' if len(timesteps) > 10 else ''}")
            
            for step_idx, t in enumerate(timesteps):
                t_prev = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else -1
                x = self.p_sample_ddim(model, x, t, t_prev, obs, eta=eta, 
                                       debug=ddim_debug, step_idx=step_idx)
        else:
            # DDPM: always use all timesteps
            for t in reversed(range(self.n_steps)):
                x = self.p_sample(model, x, t, obs)
        
        return x


# ── Dataset ──────────────────────────────────────────────────────────

class DemoChunkDataset(Dataset):
    """Creates (obs_t, action_chunk[t:t+H]) pairs from demo episodes.

    If mirror=True, each episode is duplicated with left↔right swapped:
      obs: negate ee_y (1), swap left_cube ↔ right_cube blocks
      act: negate dy (index 1)
    This doubles effective dataset size and enforces perfect symmetry.
    """

    # Obs indices for mirroring
    _MIRROR_NEGATE_OBS = [1]           # ee_y
    _LEFT_BLOCK  = list(range(8, 15))  # Lpos(3) + Lquat(4) at indices 8-14
    _RIGHT_BLOCK = list(range(15, 22)) # Rpos(3) + Rquat(4) at indices 15-21

    def __init__(self, path: str, horizon: int,
                 obs_mean: np.ndarray | None = None,
                 obs_std: np.ndarray | None = None,
                 act_mean: np.ndarray | None = None,
                 act_std: np.ndarray | None = None,
                 mirror: bool = True,
                 episode_indices: np.ndarray | None = None) -> None:
        data = np.load(path, allow_pickle=True)
        obs_all = data["obs"]                    # (N, T, obs_dim)
        act_all = data["actions"]                # (N, T, act_dim)
        ep_lens = data["episode_lengths"]        # (N,)

        # Optional: use only a subset of episodes (for train/val split)
        if episode_indices is not None:
            obs_all = obs_all[episode_indices]
            act_all = act_all[episode_indices]
            ep_lens = ep_lens[episode_indices]

        self.horizon = horizon
        samples_obs: list[np.ndarray] = []
        samples_act: list[np.ndarray] = []
        chunk_weights: list[float] = []  # priority weighting

        N, T, act_dim = act_all.shape
        for i in range(N):
            L = int(ep_lens[i])
            for t in range(L):
                chunk = np.zeros((horizon, act_dim), dtype=np.float32)
                end = min(t + horizon, L)
                chunk[:end - t] = act_all[i, t:end]
                # pad remainder with last valid action
                if end - t < horizon:
                    chunk[end - t:] = act_all[i, end - 1]
                samples_obs.append(obs_all[i, t])
                samples_act.append(chunk)
                
                # Weight by position in episode: early chunks (where left/right decision happens) get higher weight
                # Arc phase is roughly first 10% of trajectory; give it 2x weight
                # First 50% gets 1.5x, rest get 1.0
                frac = t / L
                if frac < 0.1:
                    weight = 2.0
                elif frac < 0.5:
                    weight = 1.5
                else:
                    weight = 1.0
                chunk_weights.append(weight)

        self.obs = np.stack(samples_obs)   # (M, obs_dim)
        self.act = np.stack(samples_act)   # (M, H, act_dim)
        
        # Store chunk weights for priority sampling
        self.chunk_weights = np.array(chunk_weights, dtype=np.float32)

        # ── Mirror augmentation ──────────────────────────────────────
        if mirror:
            m_obs = self.obs.copy()
            m_act = self.act.copy()
            m_weights = self.chunk_weights.copy()
            # negate ee_y
            m_obs[:, 1] *= -1
            # negate left/right cube y positions (indices 9 and 16)
            m_obs[:, 9] *= -1
            m_obs[:, 16] *= -1
            # swap left_cube ↔ right_cube blocks entirely
            left_save = m_obs[:, 8:15].copy()
            m_obs[:, 8:15] = m_obs[:, 15:22]
            m_obs[:, 15:22] = left_save
            # negate dy (action index 1)
            m_act[:, :, 1] *= -1
            self.obs = np.concatenate([self.obs, m_obs], axis=0)
            self.act = np.concatenate([self.act, m_act], axis=0)
            self.chunk_weights = np.concatenate([self.chunk_weights, m_weights], axis=0)
            print(f"  mirror augmentation: {len(samples_obs)} -> {self.obs.shape[0]} chunks")

        # compute / store normalisation stats
        if obs_mean is None:
            self.obs_mean = self.obs.mean(0).astype(np.float32)
            # Floor std at 0.01 so near-constant dims (e.g. EE quaternion)
            # don't explode to ±thousands after normalisation and drown
            # out cube-position information the policy needs to see.
            self.obs_std = np.maximum(
                self.obs.std(0).astype(np.float32), np.float32(0.01)
            )
        else:
            self.obs_mean = obs_mean.astype(np.float32)
            self.obs_std = obs_std.astype(np.float32)
        # Actions are already in [-1, 1] (the natural DDPM output range).
        # Mean/std normalization crushes dx/dy variance (expert has
        # saturated ±1 during approach but ~0 during grasp/lift →
        # tiny std ≈ 0.08).  Use identity normalisation instead.
        if act_mean is None:
            self.act_mean = np.zeros(act_dim, dtype=np.float32)
            self.act_std  = np.ones(act_dim, dtype=np.float32)
        else:
            self.act_mean = act_mean
            self.act_std = act_std

        # normalise obs only (actions stay in [-1, 1])
        self.obs = ((self.obs - self.obs_mean) / self.obs_std).astype(np.float32)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.obs[idx]),
                torch.from_numpy(self.act[idx]))


# ── EMA (Exponential Moving Average) ─────────────────────────────────

class EMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self.shadow)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Copy EMA weights into model; return original weights."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module,
                backup: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


# ── Training ─────────────────────────────────────────────────────────

def train(cfg: Dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n\n"
            "===============================================================\n"
            "  ERROR: CUDA not available!\n"
            "  Training requires GPU. Please activate venv with CUDA PyTorch:\n"
            "    .venv\\Scripts\\Activate.ps1\n"
            "===============================================================\n"
        )
    device = torch.device("cuda")
    print(f"device: {device}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── demo diagnostics ────────────────────────────────────────────
    raw = np.load(cfg["demo_path"], allow_pickle=True)
    n_demos = raw["obs"].shape[0]
    ep_lens = raw["episode_lengths"]
    labels = raw["labels"]
    n_left = int(np.sum(labels == "left"))
    n_right = int(np.sum(labels == "right"))
    print(f"\n  demos loaded     : {n_demos}")
    print(f"  left / right     : {n_left} / {n_right}")
    print(f"  mean ep length   : {ep_lens.mean():.1f}  "
          f"(min={ep_lens.min()}, max={ep_lens.max()})")
    # ── train / val split (episode-level, balanced L/R) ───────────
    val_frac = cfg.get("val_frac", 0.1)  # 10% held out
    left_idx = np.where(labels == "left")[0]
    right_idx = np.where(labels == "right")[0]
    rng_split = np.random.RandomState(cfg["seed"])
    rng_split.shuffle(left_idx)
    rng_split.shuffle(right_idx)
    n_val_per_side = max(1, int(len(left_idx) * val_frac))
    val_idx = np.sort(np.concatenate([
        left_idx[:n_val_per_side], right_idx[:n_val_per_side]]))
    train_idx = np.sort(np.concatenate([
        left_idx[n_val_per_side:], right_idx[n_val_per_side:]]))
    print(f"  train / val split: {len(train_idx)} train, {len(val_idx)} val "
          f"({n_val_per_side}L + {n_val_per_side}R held out)")
    del raw  # free memory

    # dataset (with mirror augmentation) — train split only
    ds = DemoChunkDataset(cfg["demo_path"], cfg["horizon"],
                          mirror=cfg.get("mirror_augment", True),
                          episode_indices=train_idx)

    # val dataset — uses train's normalization stats, NO mirror aug
    ds_val = DemoChunkDataset(cfg["demo_path"], cfg["horizon"],
                              obs_mean=ds.obs_mean, obs_std=ds.obs_std,
                              act_mean=ds.act_mean, act_std=ds.act_std,
                              mirror=False,
                              episode_indices=val_idx)

    # ── obs normalization diagnostics ─────────────────────────────
    floored = int(np.sum(ds.obs_std <= 0.0101))
    print(f"\n  obs std floored  : {floored}/{len(ds.obs_std)} dims hit 0.01 floor")
    dim_names = ["ee_x", "ee_y", "ee_z",
                 "ee_qx", "ee_qy", "ee_qz", "ee_qw",
                 "grip",
                 "Lx", "Ly", "Lz", "Lqx", "Lqy", "Lqz", "Lqw",
                 "Rx", "Ry", "Rz", "Rqx", "Rqy", "Rqz", "Rqw"]
    for i, (m, s) in enumerate(zip(ds.obs_mean, ds.obs_std)):
        tag = " [FLOORED]" if s <= 0.0101 else ""
        name = dim_names[i] if i < len(dim_names) else f"dim{i}"
        print(f"    [{i:2d}] {name:6s}  mean={m:+.4f}  std={s:.4f}{tag}")
    print()

    use_pin = torch.cuda.is_available()
    
    # Use weighted sampler for priority weighting (early chunks get higher weight)
    sampler = WeightedRandomSampler(
        weights=ds.chunk_weights,
        num_samples=len(ds),
        replacement=True
    )
    dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler,
                    drop_last=True, num_workers=0, pin_memory=use_pin)
    
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                        drop_last=False, num_workers=0, pin_memory=use_pin)
    print(f"train set: {len(ds)} chunks (priority weighted)  |  batches/epoch: {len(dl)}")
    print(f"val   set: {len(ds_val)} chunks  |  batches: {len(dl_val)}")

    # model
    model = NoiseNet(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg["hidden_dim"],
        n_blocks=cfg["n_blocks"],
        time_embed_dim=cfg["time_embed_dim"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    schedule = DDPMSchedule(cfg["n_diffusion_steps"],
                            cfg["beta_start"], cfg["beta_end"], device)

    # EMA
    ema = EMA(model, decay=cfg.get("ema_decay", 0.999))
    print(f"EMA decay: {ema.decay}")

    # checkpoint dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(cfg["ckpt_dir"]) / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # also create a "latest" symlink / copy
    latest = Path(cfg["ckpt_dir"]) / "latest"

    best_loss = float("inf")
    best_val_loss = float("inf")
    val_rising_count = 0          # consecutive epochs val loss increased
    log_lines: list[str] = []

    # ── sim-eval schedule ────────────────────────────────────────
    eval_epochs = set(cfg.get("eval_epochs", [100, 200, 300, 500, 750, 1000]))
    # Always eval at final epoch
    eval_epochs.add(cfg["epochs"])
    eval_epochs = sorted(eval_epochs)
    eval_results: list[dict] = []   # track metrics across checkpoints
    best_eval_metric = -1.0         # best (success_rate + bimodal_frac)
    best_eval_epoch = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train ────────────────────────────────────────────────
        model.train()
        losses: list[float] = []
        for obs_b, act_b in dl:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            B = obs_b.shape[0]
            t = torch.randint(0, schedule.n_steps, (B,), device=device)
            noise = torch.randn_like(act_b)
            noisy = schedule.q_sample(act_b, t, noise)

            pred = model(noisy, t, obs_b)
            ddpm_loss = nn.functional.mse_loss(pred, noise)
            
            # Trajectory smoothness regularization (penalize action velocity)
            # act_b shape: (B, H, act_dim) where H = horizon
            smooth_weight = cfg.get("smooth_weight", 0.0)
            if smooth_weight > 0:
                # Compute action differences along time dimension
                action_diff = act_b[:, 1:, :] - act_b[:, :-1, :]  # (B, H-1, act_dim)
                smooth_loss = torch.mean(action_diff.pow(2))
                loss = ddpm_loss + smooth_weight * smooth_loss
            else:
                smooth_loss = torch.tensor(0.0)
                loss = ddpm_loss

            optimizer.zero_grad()
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            ema.update(model)
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))

        # ── val (every 10 epochs) ────────────────────────────────
        val_loss_str = ""
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for obs_b, act_b in dl_val:
                    obs_b = obs_b.to(device)
                    act_b = act_b.to(device)
                    B = obs_b.shape[0]
                    t = torch.randint(0, schedule.n_steps, (B,), device=device)
                    noise = torch.randn_like(act_b)
                    noisy = schedule.q_sample(act_b, t, noise)
                    pred = model(noisy, t, obs_b)
                    vl = nn.functional.mse_loss(pred, noise)
                    val_losses.append(vl.item())
            mean_val = float(np.mean(val_losses))
            val_loss_str = f"  val={mean_val:.6f}"

            # overfit detection
            if mean_val < best_val_loss:
                best_val_loss = mean_val
                val_rising_count = 0
            else:
                val_rising_count += 1
            if val_rising_count >= 5:
                val_loss_str += "  ⚠ OVERFIT (val rising 5+ checks)"

        line = f"epoch {epoch:4d}/{cfg['epochs']}  loss={mean_loss:.6f}{val_loss_str}"
        if epoch % 10 == 0 or epoch == 1:
            print(line)
        log_lines.append(line)

        # save best + periodic (always save EMA weights for inference)
        if mean_loss < best_loss:
            best_loss = mean_loss
            _save_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                       ckpt_dir, ema=ema)
            _save_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                       latest, ema=ema)

        if epoch % 50 == 0:
            _save_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                       ckpt_dir, suffix=f"_ep{epoch}", ema=ema)

        # ── sim-eval at checkpoint epochs ───────────────────────
        if epoch in eval_epochs:
            print(f"\n{'='*55}")
            print(f"  SIM-EVAL at epoch {epoch}  (quick: K=5 M=10)")
            print(f"{'='*55}")
            _save_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                       ckpt_dir, suffix=f"_ep{epoch}", ema=ema)
            ckpt_path = ckpt_dir / f"ckpt_ep{epoch}.pt"
            
            # Create video directory for this eval in outputs/
            video_dir = Path("outputs") / f"train_eval_ep{epoch}_videos"
            
            try:
                metrics = _quick_sim_eval(str(ckpt_path), device, K=5, M=10,
                                          max_steps=cfg.get("eval_max_steps", 400),
                                          video_dir=str(video_dir),
                                          n_videos=5,
                                          execute_steps=cfg.get("eval_execute_steps", 8),
                                          sampling_method=cfg.get("eval_sampling_method", "ddim"),
                                          ddim_eta=cfg.get("eval_ddim_eta", 0.0),
                                          ddim_steps=cfg.get("eval_ddim_steps", None),
                                          temporal_ensemble=cfg.get("eval_temporal_ensemble", True))
            except Exception as e:
                print(f"  WARNING: Video recording failed: {e}")
                # Retry without videos
                metrics = _quick_sim_eval(str(ckpt_path), device, K=5, M=10,
                                          max_steps=cfg.get("eval_max_steps", 400),
                                          video_dir=None,
                                          n_videos=0,
                                          execute_steps=cfg.get("eval_execute_steps", 8),
                                          sampling_method=cfg.get("eval_sampling_method", "ddim"),
                                          ddim_eta=cfg.get("eval_ddim_eta", 0.0),
                                          ddim_steps=cfg.get("eval_ddim_steps", None),
                                          temporal_ensemble=cfg.get("eval_temporal_ensemble", True))
            metrics["epoch"] = epoch
            metrics["train_loss"] = mean_loss
            eval_results.append(metrics)

            sr = metrics["success_rate"]
            bm = metrics["bimodal_seeds"]
            pleft = metrics["p_left"]
            ent = metrics["mean_entropy"]
            combo = sr + bm / 5.0  # combined metric

            status = ""
            if sr > 0.3 and bm >= 2:
                status = "[PROMISING]"
            elif sr > 0.1:
                status = "[LOW SUCCESS]"
            else:
                status = "[POOR]"

            print(f"  success_rate   : {sr:.1%}")
            print(f"  bimodal_seeds  : {bm}/5")
            print(f"  p(left|success): {pleft:.2f}")
            print(f"  mean_entropy   : {ent:.3f}")
            print(f"  combined_score : {combo:.3f}  {status}")

            if combo > best_eval_metric:
                best_eval_metric = combo
                best_eval_epoch = epoch
                # save as best-eval checkpoint
                _save_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                           ckpt_dir, suffix="_best_eval", ema=ema)
                _save_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                           latest, ema=ema)
                print(f"  >> NEW BEST eval checkpoint (epoch {epoch})")

            # plateau detection: if last 2 evals didn't improve
            if len(eval_results) >= 3:
                last3 = [r["success_rate"] + r["bimodal_seeds"] / 5.0
                         for r in eval_results[-3:]]
                if all(s <= best_eval_metric * 0.95 for s in last3[-2:]):
                    print(f"  >> PLATEAU detected – best was epoch {best_eval_epoch}")

            print(f"{'='*55}\n")

            # save eval tracking
            eval_log_path = ckpt_dir / "eval_tracking.json"
            eval_log_path.write_text(json.dumps(eval_results, indent=2))

    # final save
    _save_ckpt(model, optimizer, ds, cfg, cfg["epochs"], mean_loss,
               ckpt_dir, ema=ema)
    _save_ckpt(model, optimizer, ds, cfg, cfg["epochs"], mean_loss,
               latest, ema=ema)

    # save log
    (ckpt_dir / "train_log.txt").write_text("\n".join(log_lines))
    print(f"\n✓ training done – checkpoints in {ckpt_dir}")
    print(f"  latest symlink: {latest}")
    if eval_results:
        print(f"  best eval: epoch {best_eval_epoch}  "
              f"(score={best_eval_metric:.3f})")
        # print eval summary table
        print(f"\n  {'epoch':>5s}  {'loss':>8s}  {'val':>8s}  "
              f"{'success':>7s}  {'bimod':>5s}  {'p(L)':>5s}  {'ent':>5s}")
        for r in eval_results:
            print(f"  {r['epoch']:5d}  {r['train_loss']:8.6f}  "
                  f"{'':>8s}  {r['success_rate']:6.1%}  "
                  f"{r['bimodal_seeds']:5d}  {r['p_left']:5.2f}  "
                  f"{r['mean_entropy']:5.3f}")


def _quick_sim_eval(ckpt_path: str, device: torch.device,
                    K: int = 5, M: int = 10,
                    max_steps: int = 400,
                    video_dir: str | None = None,
                    n_videos: int = 3,
                    execute_steps: int = 8,
                    sampling_method: str = 'ddim',
                    ddim_eta: float = 0.0,
                    ddim_steps: int | None = None,
                    temporal_ensemble: bool = True) -> dict:
    """Run a quick sim evaluation and return metrics dict.

    Uses the eval script's DiffusionPolicyRunner and rollout function.
    K env seeds × M sample seeds = K*M rollouts.
    
    Args:
        video_dir: If provided, save up to n_videos sample rollouts as MP4
        n_videos: Number of videos to save (default 3)
        sampling_method: 'ddpm' or 'ddim' (default: 'ddim')
        ddim_eta: DDIM stochasticity parameter (0.0=deterministic)
        ddim_steps: Number of DDIM inference steps (None=use all n_diffusion_steps)
    """
    from scripts.eval_multimodality import DiffusionPolicyRunner, rollout
    from envs.twoblockpick_env import TwoBlockPickEnv

    policy = DiffusionPolicyRunner(ckpt_path, device, 
                                    temporal_ensemble=temporal_ensemble,
                                    ensemble_decay=0.7,
                                    sampling_method=sampling_method,
                                    ddim_eta=ddim_eta,
                                    ddim_steps=ddim_steps)
    env = TwoBlockPickEnv(render=False, episode_length=max_steps)

    if video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    results = []
    env_seeds = list(range(100, 100 + K))
    video_count = 0
    
    for es in env_seeds:
        for mi in range(M):
            ss = 5000 + mi * 137
            
            # Record first n_videos rollouts
            vp = None
            if video_dir and video_count < n_videos:
                vp = str(Path(video_dir) / f"eval_env{es}_sample{ss}.mp4")
                video_count += 1
            
            res = rollout(env, policy, es, ss,
                          execute_steps=execute_steps, max_steps=max_steps,
                          video_path=vp)
            results.append(res)
            
            # Rename video to include outcome
            if vp and Path(vp).exists():
                final_vp = str(Path(video_dir) / f"eval_env{es}_sample{ss}_{res['outcome']}.mp4")
                try:
                    Path(vp).rename(final_vp)
                except:
                    pass

    env.close()

    n_left = sum(1 for r in results if r["outcome"] == "left_success")
    n_right = sum(1 for r in results if r["outcome"] == "right_success")
    n_success = n_left + n_right
    total = len(results)
    success_rate = n_success / total if total else 0.0
    p_left = n_left / n_success if n_success else 0.5

    # per-seed bimodality
    per_seed: dict[int, dict] = {}
    for r in results:
        es = r["env_seed"]
        per_seed.setdefault(es, {"L": 0, "R": 0, "F": 0})
        if r["outcome"] == "left_success":
            per_seed[es]["L"] += 1
        elif r["outcome"] == "right_success":
            per_seed[es]["R"] += 1
        else:
            per_seed[es]["F"] += 1

    bimodal_seeds = 0
    entropies = []
    for es, c in per_seed.items():
        s = c["L"] + c["R"]
        if s >= 3:
            probs = np.array([c["L"], c["R"]], dtype=float) / s
            ent = -sum(p * math.log2(p + 1e-12) for p in probs if p > 0)
            entropies.append(ent)
            if c["L"] >= 1 and c["R"] >= 1:
                bimodal_seeds += 1

    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    return {
        "success_rate": round(success_rate, 4),
        "bimodal_seeds": bimodal_seeds,
        "p_left": round(p_left, 4),
        "mean_entropy": round(mean_entropy, 4),
        "n_left": n_left,
        "n_right": n_right,
        "n_fail": total - n_success,
        "K": K,
        "M": M,
    }


def _save_ckpt(model, optimizer, ds, cfg, epoch, loss, directory,
               suffix: str = "", ema: EMA | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "loss": loss,
        "obs_mean": ds.obs_mean,
        "obs_std": ds.obs_std,
        "act_mean": ds.act_mean,
        "act_std": ds.act_std,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, directory / f"ckpt{suffix}.pt")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
