#!/usr/bin/env python
"""Train a STEERABLE DDPM diffusion policy for TwoBlockPick.

Key difference from train_diffusion_policy.py:
  - NoiseNet accepts a steering vector [lateral_bias, curvature_strength, target_side]
  - Steering vector is embedded via a small MLP and added to the observation conditioning
  - During training, steering vectors come from pre-computed demo analysis
  - During inference, steering vectors come from VLM trajectory analysis

Usage:
    python scripts/train_steered_diffusion.py --config configs/train.yaml
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse building blocks from original training script
from scripts.train_diffusion_policy import (
    SinusoidalEmbedding, ResBlock, DDPMSchedule, EMA
)


# ── Steerable Noise Net ─────────────────────────────────────────────

class SteerableNoiseNet(nn.Module):
    """Noise prediction network with steering vector conditioning.
    
    Architecture:
      input = [noisy_actions_flat, obs, steer_embed]
      -> Linear projection -> ResBlocks with FiLM time conditioning -> output
    
    The steering vector is embedded via a small MLP before concatenation,
    allowing the model to learn nonlinear mappings from steering space.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        steer_dim: int = 3,        # [lateral_bias, curvature_strength, target_side]
        steer_embed_dim: int = 64,  # Dimension of steering embedding
        hidden_dim: int = 256,
        n_blocks: int = 6,
        time_embed_dim: int = 128,
        steer_dropout: float = 0.1,  # Randomly zero steering for classifier-free guidance
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.steer_dim = steer_dim
        self.steer_dropout = steer_dropout

        # Time embedding (same as original)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Steering embedding MLP
        self.steer_embed = nn.Sequential(
            nn.Linear(steer_dim, steer_embed_dim),
            nn.Mish(),
            nn.Linear(steer_embed_dim, steer_embed_dim),
            nn.Mish(),
        )

        # Input projection: actions + obs + steering
        input_dim = horizon * act_dim + obs_dim + steer_embed_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks with time conditioning
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, time_embed_dim) for _ in range(n_blocks)]
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, horizon * act_dim)

    def forward(
        self,
        noisy_act: torch.Tensor,   # (B, H, act_dim)
        timestep: torch.Tensor,    # (B,)
        obs: torch.Tensor,         # (B, obs_dim)
        steer: torch.Tensor = None,# (B, steer_dim) - optional steering
    ) -> torch.Tensor:
        B = noisy_act.shape[0]
        t_emb = self.time_embed(timestep)  # (B, t_dim)

        # Steering embedding with optional dropout (classifier-free guidance)
        if steer is None:
            steer = torch.zeros(B, self.steer_dim, device=noisy_act.device)
        
        # During training: randomly drop steering for CFG
        if self.training and self.steer_dropout > 0:
            mask = (torch.rand(B, 1, device=steer.device) > self.steer_dropout).float()
            steer = steer * mask
        
        s_emb = self.steer_embed(steer)  # (B, steer_embed_dim)

        # Concatenate all inputs
        x = torch.cat([
            noisy_act.reshape(B, -1),  # (B, H*A)
            obs,                         # (B, O)
            s_emb,                       # (B, S)
        ], dim=-1)
        
        x = self.in_proj(x)  # (B, D)
        for blk in self.blocks:
            x = blk(x, t_emb)
        out = self.out_proj(x)  # (B, H*A)
        return out.reshape(B, self.horizon, self.act_dim)


# ── Dataset with steering ───────────────────────────────────────────

class SteeredDemoChunkDataset(Dataset):
    """Creates (obs_t, action_chunk[t:t+H], steer_vec) triples from demos.
    
    Each episode has a per-episode steering vector that is broadcast to
    all chunks from that episode.
    """

    _MIRROR_NEGATE_OBS = [1]
    _LEFT_BLOCK = list(range(8, 15))
    _RIGHT_BLOCK = list(range(15, 22))

    def __init__(self, path: str, horizon: int,
                 steering_vectors: np.ndarray,
                 obs_mean=None, obs_std=None,
                 act_mean=None, act_std=None,
                 mirror: bool = True,
                 episode_indices=None) -> None:
        data = np.load(path, allow_pickle=True)
        obs_all = data["obs"]
        act_all = data["actions"]
        ep_lens = data["episode_lengths"]

        if episode_indices is not None:
            obs_all = obs_all[episode_indices]
            act_all = act_all[episode_indices]
            ep_lens = ep_lens[episode_indices]
            steering_vectors = steering_vectors[episode_indices]

        self.horizon = horizon
        samples_obs = []
        samples_act = []
        samples_steer = []
        chunk_weights = []

        N, T, act_dim = act_all.shape
        for i in range(N):
            L = int(ep_lens[i])
            steer = steering_vectors[i]  # Per-episode steering
            for t in range(L):
                chunk = np.zeros((horizon, act_dim), dtype=np.float32)
                end = min(t + horizon, L)
                chunk[:end - t] = act_all[i, t:end]
                if end - t < horizon:
                    chunk[end - t:] = act_all[i, end - 1]
                samples_obs.append(obs_all[i, t])
                samples_act.append(chunk)
                samples_steer.append(steer)
                
                frac = t / L
                weight = 2.0 if frac < 0.1 else (1.5 if frac < 0.5 else 1.0)
                chunk_weights.append(weight)

        self.obs = np.stack(samples_obs)
        self.act = np.stack(samples_act)
        self.steer = np.stack(samples_steer)
        self.chunk_weights = np.array(chunk_weights, dtype=np.float32)

        # Mirror augmentation  
        if mirror:
            m_obs = self.obs.copy()
            m_act = self.act.copy()
            m_steer = self.steer.copy()
            m_weights = self.chunk_weights.copy()
            
            m_obs[:, 1] *= -1
            m_obs[:, 9] *= -1
            m_obs[:, 16] *= -1
            left_save = m_obs[:, 8:15].copy()
            m_obs[:, 8:15] = m_obs[:, 15:22]
            m_obs[:, 15:22] = left_save
            m_act[:, :, 1] *= -1
            
            # Mirror steering: negate lateral_bias, flip target_side
            m_steer[:, 0] *= -1       # Negate lateral bias
            m_steer[:, 2] = 1.0 - m_steer[:, 2]  # Flip target side 0<->1
            
            self.obs = np.concatenate([self.obs, m_obs], axis=0)
            self.act = np.concatenate([self.act, m_act], axis=0)
            self.steer = np.concatenate([self.steer, m_steer], axis=0)
            self.chunk_weights = np.concatenate([self.chunk_weights, m_weights], axis=0)
            print(f"  mirror augmentation: {len(samples_obs)} -> {self.obs.shape[0]} chunks")

        # Normalization
        if obs_mean is None:
            self.obs_mean = self.obs.mean(0).astype(np.float32)
            self.obs_std = np.maximum(self.obs.std(0).astype(np.float32), np.float32(0.01))
        else:
            self.obs_mean = obs_mean.astype(np.float32)
            self.obs_std = obs_std.astype(np.float32)

        if act_mean is None:
            self.act_mean = np.zeros(act_dim, dtype=np.float32)
            self.act_std = np.ones(act_dim, dtype=np.float32)
        else:
            self.act_mean = act_mean
            self.act_std = act_std

        self.obs = ((self.obs - self.obs_mean) / self.obs_std).astype(np.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.obs[idx]),
                torch.from_numpy(self.act[idx]),
                torch.from_numpy(self.steer[idx]))


# ── Extended DDPMSchedule for steered sampling ───────────────────────

class SteeredDDPMSchedule(DDPMSchedule):
    """DDPM schedule with steering-conditioned sampling."""
    
    @torch.no_grad()
    def p_sample_steered(self, model: SteerableNoiseNet, xt: torch.Tensor,
                         t_int: int, obs: torch.Tensor,
                         steer: torch.Tensor) -> torch.Tensor:
        B = xt.shape[0]
        t_tensor = torch.full((B,), t_int, device=xt.device, dtype=torch.long)
        eps_pred = model(xt, t_tensor, obs, steer)

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
    def p_sample_ddim_steered(self, model: SteerableNoiseNet, xt: torch.Tensor,
                               t_int: int, t_prev_int: int, obs: torch.Tensor,
                               steer: torch.Tensor, eta: float = 0.0,
                               cfg_scale: float = 0.0) -> torch.Tensor:
        """DDIM step with steering + optional classifier-free guidance."""
        B = xt.shape[0]
        t_tensor = torch.full((B,), t_int, device=xt.device, dtype=torch.long)
        
        # Conditional prediction
        eps_cond = model(xt, t_tensor, obs, steer)
        
        # Classifier-free guidance
        if cfg_scale > 0:
            zero_steer = torch.zeros_like(steer)
            eps_uncond = model(xt, t_tensor, obs, zero_steer)
            eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps_pred = eps_cond

        alpha_bar_t = self.alpha_bar[t_int]
        alpha_bar_t_prev = self.alpha_bar[t_prev_int] if t_prev_int >= 0 else torch.tensor(1.0, device=xt.device)
        
        alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-8, max=1.0)
        alpha_bar_t_prev = torch.clamp(alpha_bar_t_prev, min=1e-8, max=1.0)
        
        sigma_t = torch.zeros_like(alpha_bar_t)
        if eta > 0 and t_prev_int >= 0:
            beta_t = 1.0 - alpha_bar_t / alpha_bar_t_prev
            beta_t = torch.clamp(beta_t, min=0.0, max=0.999)
            variance = (1.0 - alpha_bar_t_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-8) * beta_t
            variance = torch.clamp(variance, min=0.0, max=1.0)
            sigma_t = eta * torch.sqrt(variance)
        
        a_coef = torch.sqrt(alpha_bar_t_prev / alpha_bar_t)
        b_coef = torch.sqrt(torch.clamp(1.0 - alpha_bar_t_prev - sigma_t**2, min=0.0)) - a_coef * torch.sqrt(1.0 - alpha_bar_t)
        
        x_prev = a_coef * xt + b_coef * eps_pred
        
        if eta > 0 and t_prev_int >= 0:
            noise = torch.randn_like(xt)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev

    @torch.no_grad()
    def sample_steered(self, model: SteerableNoiseNet, obs: torch.Tensor,
                        steer: torch.Tensor, horizon: int, act_dim: int,
                        method: str = 'ddim', eta: float = 0.0,
                        ddim_steps: int | None = None,
                        cfg_scale: float = 0.0) -> torch.Tensor:
        B = obs.shape[0]
        x = torch.randn(B, horizon, act_dim, device=obs.device)
        
        if method == 'ddim':
            if ddim_steps is None or ddim_steps >= self.n_steps:
                timesteps = list(reversed(range(self.n_steps)))
            else:
                timesteps = list(reversed(
                    torch.linspace(0, self.n_steps - 1, ddim_steps, dtype=torch.long).tolist()))
            
            for step_idx, t in enumerate(timesteps):
                t_prev = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else -1
                x = self.p_sample_ddim_steered(model, x, t, t_prev, obs, steer,
                                                eta=eta, cfg_scale=cfg_scale)
        else:
            for t in reversed(range(self.n_steps)):
                x = self.p_sample_steered(model, x, t, obs, steer)
        
        return x


# ── Training ─────────────────────────────────────────────────────────

def train(cfg: Dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training.")
    device = torch.device("cuda")
    print(f"device: {device}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Load steering vectors
    steer_path = Path("data/demos/steering_vectors.npy")
    if not steer_path.exists():
        raise FileNotFoundError(
            f"Steering vectors not found at {steer_path}.\n"
            "Run: python scripts/compute_steering_params.py"
        )
    steering_vectors = np.load(str(steer_path)).astype(np.float32)
    steer_dim = steering_vectors.shape[1]
    print(f"Steering vectors: {steering_vectors.shape} (dim={steer_dim})")
    print(f"  lateral_bias range: [{steering_vectors[:,0].min():.3f}, {steering_vectors[:,0].max():.3f}]")
    print(f"  curvature range:    [{steering_vectors[:,1].min():.3f}, {steering_vectors[:,1].max():.3f}]")
    print(f"  target_side:        {dict(zip(*np.unique(steering_vectors[:,2], return_counts=True)))}")

    # Demo diagnostics
    raw = np.load(cfg["demo_path"], allow_pickle=True)
    n_demos = raw["obs"].shape[0]
    labels = raw["labels"]
    ep_lens = raw["episode_lengths"]
    n_left = int(np.sum(labels == "left"))
    n_right = int(np.sum(labels == "right"))
    print(f"\n  demos: {n_demos} (left={n_left}, right={n_right})")
    print(f"  mean ep len: {ep_lens.mean():.1f}")

    # Train/val split
    val_frac = cfg.get("val_frac", 0.1)
    left_idx = np.where(labels == "left")[0]
    right_idx = np.where(labels == "right")[0]
    rng_split = np.random.RandomState(cfg["seed"])
    rng_split.shuffle(left_idx)
    rng_split.shuffle(right_idx)
    n_val = max(1, int(len(left_idx) * val_frac))
    val_idx = np.sort(np.concatenate([left_idx[:n_val], right_idx[:n_val]]))
    train_idx = np.sort(np.concatenate([left_idx[n_val:], right_idx[n_val:]]))
    print(f"  split: {len(train_idx)} train, {len(val_idx)} val")
    del raw

    # Datasets
    ds = SteeredDemoChunkDataset(
        cfg["demo_path"], cfg["horizon"], steering_vectors,
        mirror=cfg.get("mirror_augment", True),
        episode_indices=train_idx)
    
    ds_val = SteeredDemoChunkDataset(
        cfg["demo_path"], cfg["horizon"], steering_vectors,
        obs_mean=ds.obs_mean, obs_std=ds.obs_std,
        act_mean=ds.act_mean, act_std=ds.act_std,
        mirror=False,
        episode_indices=val_idx)

    sampler = WeightedRandomSampler(ds.chunk_weights, len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler,
                    drop_last=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                        drop_last=False, num_workers=0, pin_memory=True)
    print(f"train: {len(ds)} chunks | val: {len(ds_val)} chunks")

    # Steerable model
    steer_dropout = cfg.get("steer_dropout", 0.1)
    steer_embed_dim = cfg.get("steer_embed_dim", 64)
    model = SteerableNoiseNet(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        steer_dim=steer_dim,
        steer_embed_dim=steer_embed_dim,
        hidden_dim=cfg["hidden_dim"],
        n_blocks=cfg["n_blocks"],
        time_embed_dim=cfg["time_embed_dim"],
        steer_dropout=steer_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SteerableNoiseNet params: {n_params:,} (steer_dropout={steer_dropout})")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    schedule = SteeredDDPMSchedule(cfg["n_diffusion_steps"],
                                    cfg["beta_start"], cfg["beta_end"], device)
    ema = EMA(model, decay=cfg.get("ema_decay", 0.999))

    # Checkpoint dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(cfg["ckpt_dir"]) / f"steered_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = Path(cfg["ckpt_dir"]) / "steered_latest"

    best_loss = float("inf")
    best_val_loss = float("inf")
    log_lines = []

    epochs = cfg.get("steer_epochs", cfg["epochs"])
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for obs_b, act_b, steer_b in dl:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            steer_b = steer_b.to(device)

            B = obs_b.shape[0]
            t = torch.randint(0, schedule.n_steps, (B,), device=device)
            noise = torch.randn_like(act_b)
            noisy = schedule.q_sample(act_b, t, noise)

            pred = model(noisy, t, obs_b, steer_b)
            loss = nn.functional.mse_loss(pred, noise)

            # Smoothness regularization
            smooth_weight = cfg.get("smooth_weight", 0.0)
            if smooth_weight > 0:
                action_diff = act_b[:, 1:, :] - act_b[:, :-1, :]
                smooth_loss = torch.mean(action_diff.pow(2))
                loss = loss + smooth_weight * smooth_loss

            optimizer.zero_grad()
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            ema.update(model)
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))

        # Validation
        val_str = ""
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for obs_b, act_b, steer_b in dl_val:
                    obs_b = obs_b.to(device)
                    act_b = act_b.to(device)
                    steer_b = steer_b.to(device)
                    B = obs_b.shape[0]
                    t = torch.randint(0, schedule.n_steps, (B,), device=device)
                    noise = torch.randn_like(act_b)
                    noisy = schedule.q_sample(act_b, t, noise)
                    pred = model(noisy, t, obs_b, steer_b)
                    vl = nn.functional.mse_loss(pred, noise)
                    val_losses.append(vl.item())
            mean_val = float(np.mean(val_losses))
            val_str = f"  val={mean_val:.6f}"
            if mean_val < best_val_loss:
                best_val_loss = mean_val

        line = f"epoch {epoch:4d}/{epochs}  loss={mean_loss:.6f}{val_str}"
        if epoch % 10 == 0 or epoch == 1:
            print(line)
        log_lines.append(line)

        if mean_loss < best_loss:
            best_loss = mean_loss
            _save_steered_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                               ckpt_dir, steer_dim=steer_dim,
                               steer_embed_dim=steer_embed_dim,
                               steer_dropout=steer_dropout, ema=ema)
            _save_steered_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                               latest, steer_dim=steer_dim,
                               steer_embed_dim=steer_embed_dim,
                               steer_dropout=steer_dropout, ema=ema)

        if epoch % 100 == 0:
            _save_steered_ckpt(model, optimizer, ds, cfg, epoch, mean_loss,
                               ckpt_dir, suffix=f"_ep{epoch}",
                               steer_dim=steer_dim,
                               steer_embed_dim=steer_embed_dim,
                               steer_dropout=steer_dropout, ema=ema)

    # Final save
    _save_steered_ckpt(model, optimizer, ds, cfg, epochs, mean_loss,
                       ckpt_dir, ema=ema, steer_dim=steer_dim,
                       steer_embed_dim=steer_embed_dim,
                       steer_dropout=steer_dropout)
    _save_steered_ckpt(model, optimizer, ds, cfg, epochs, mean_loss,
                       latest, ema=ema, steer_dim=steer_dim,
                       steer_embed_dim=steer_embed_dim,
                       steer_dropout=steer_dropout)

    (ckpt_dir / "train_log.txt").write_text("\n".join(log_lines))
    print(f"\nDone. Checkpoints in {ckpt_dir}")
    print(f"  latest: {latest}")
    print(f"  best train loss: {best_loss:.6f}")
    print(f"  best val loss: {best_val_loss:.6f}")


def _save_steered_ckpt(model, optimizer, ds, cfg, epoch, loss, directory,
                        suffix="", ema=None, steer_dim=3,
                        steer_embed_dim=64, steer_dropout=0.1):
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
        "model_type": "steerable",
        "steer_dim": steer_dim,
        "steer_embed_dim": steer_embed_dim,
        "steer_dropout": steer_dropout,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, directory / f"ckpt{suffix}.pt")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    if args.epochs is not None:
        cfg["steer_epochs"] = args.epochs

    train(cfg)


if __name__ == "__main__":
    main()
