#!/usr/bin/env python3
"""Train a goal-conditioned Legibility Diffuser (Bronars et al. RA-L 2024).

Architecture: UNet DiffusionPolicy + goal embedding (same backbone as the
88-92% baseline from eval_with_videos.py / runs/diffusion_20260222_195530).

Training with Classifier-Free Guidance (CFG):
  - 15% of steps use null goal label (unconditional)
  - 85% use the true goal label: 0=left, 1=right, 2=null

Inference (CFG DDIM):
  ε̂ = ε_uncond + w * (ε_cond - ε_uncond)
  w > 1 amplifies goal commitment → earlier legibility signal

Reference:
  Bronars, Cheng, Xu. "Legibility Diffuser: Offline Imitation for
  Intent Expressive Motion." RA-L 2024.

Usage:
  python scripts/train_legibility_diffuser.py
  python scripts/train_legibility_diffuser.py --epochs 150 --p_uncond 0.15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Goal label constants ──────────────────────────────────────────────
GOAL_LEFT  = 0
GOAL_RIGHT = 1
NULL_GOAL  = 2   # unconditional / dropped label during CFG training
NUM_GOALS  = 2   # left + right


# ══════════════════════════════════════════════════════════════════════
# MODEL — identical UNet backbone + goal embedding
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


# ══════════════════════════════════════════════════════════════════════
# TRUE 1D CONV BUILDING BLOCKS
# Why Conv1d beats MLP: nn.Linear processes each horizon timestep
# independently. nn.Conv1d(kernel=5) lets each output step see its
# ±2 temporal neighbours — giving the model a sense of trajectory flow.
# ══════════════════════════════════════════════════════════════════════

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
    """Residual block with two Conv1dBlocks and FiLM time/goal conditioning.

    x    : (B, C, T)     — channels-first, T = prediction horizon
    cond : (B, cond_dim) — global condition (time + obs + goal summed)

    FiLM injects the condition as an additive bias broadcast over T:
        h = conv1(x)
        h = h + cond_proj(cond).unsqueeze(-1)   # (B, out_ch, 1) → broadcasts
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


class GoalCondDiffusionPolicy(nn.Module):
    """Legibility Diffuser with TRUE 1D Conv temporal UNet (Bronars RA-L 2024).

    Architecture (channels-first throughout):
      Input  : (B, H, act_dim) → permute → (B, act_dim, H)
      Encoder: Conv1d(act_dim→256) → ResBlock(256→256) → ResBlock(256→512)
      Bottleneck: ResBlock(512→512)
      Decoder: cat(512 skip) → ResBlock(1024→512) → ResBlock(512→256)
      Output : GroupNorm+Mish+Conv1d(256→act_dim) → permute → (B, H, act_dim)

    Key difference from MLP predecessor:
      MLP version: each of the H horizon steps processed INDEPENDENTLY.
      Conv1d (k=5): every output step sees its ±2 temporal neighbours,
      so the model learns smooth, physically coherent action trajectories.
    """

    NULL_GOAL = NULL_GOAL

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,   # kept for checkpoint compat; arch is fixed above
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        cond_dim     = hidden_dim

        # Global conditioning: time step embedding
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128),
            nn.Linear(128, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, cond_dim),
        )
        # Global conditioning: observation embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, cond_dim),
        )
        # Goal embedding: 3 tokens (0=left, 1=right, 2=null)
        self.goal_embed = nn.Embedding(NUM_GOALS + 1, cond_dim)

        # Pointwise channel expansion: act_dim → hidden_dim
        self.input_proj = nn.Conv1d(act_dim, hidden_dim, 1)

        dims = [hidden_dim, hidden_dim * 2]   # [256, 512]

        # Encoder: two residual blocks, channels grow 256 → 256 → 512
        self.enc_block1 = Conv1dResBlock(dims[0], dims[0], cond_dim)
        self.enc_block2 = Conv1dResBlock(dims[0], dims[1], cond_dim)

        # Bottleneck
        self.bottleneck = Conv1dResBlock(dims[1], dims[1], cond_dim)

        # Decoder: skip concat doubles channels, then project back down
        self.dec_block1 = Conv1dResBlock(dims[1] * 2, dims[1], cond_dim)  # 1024→512
        self.dec_block2 = Conv1dResBlock(dims[1],     dims[0], cond_dim)  # 512→256

        # Output: back to action space
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, dims[0]),
            nn.Mish(),
            nn.Conv1d(dims[0], act_dim, 1),
        )

    def forward(
        self,
        noisy_act: torch.Tensor,   # (B, H, act_dim)
        timestep:  torch.Tensor,   # (B,)
        obs:       torch.Tensor,   # (B, obs_dim)
        goal_id:   torch.Tensor,   # (B,) in {0,1,2}
    ) -> torch.Tensor:
        # --- (1) channels-first ---
        x = noisy_act.permute(0, 2, 1)               # (B, act_dim, H)
        x = self.input_proj(x)                        # (B, 256, H)

        # --- (2) global conditioning vector ---
        cond = (self.time_mlp(timestep)
                + self.obs_embed(obs)
                + self.goal_embed(goal_id))           # (B, 256)

        # --- (3) encoder ---
        x    = self.enc_block1(x, cond)              # (B, 256, H)
        x    = self.enc_block2(x, cond)              # (B, 512, H)
        skip = x                                      # save for skip connection

        # --- (4) bottleneck ---
        x = self.bottleneck(x, cond)                 # (B, 512, H)

        # --- (5) decoder (skip concat on channel dim) ---
        x = torch.cat([x, skip], dim=1)              # (B, 1024, H)
        x = self.dec_block1(x, cond)                 # (B, 512, H)
        x = self.dec_block2(x, cond)                 # (B, 256, H)

        # --- (6) back to (B, H, act_dim) ---
        return self.output_proj(x).permute(0, 2, 1)  # (B, H, act_dim)


# ══════════════════════════════════════════════════════════════════════
# DATASET — same windowing/augmentation as train_diffusion_policy.py
#           + goal_id per sample
# ══════════════════════════════════════════════════════════════════════

class GoalLabeledDataset(Dataset):
    """Demo chunks labeled with goal identity (left=0, right=1).

    Mirror augmentation: negates y-axis and swaps left/right blocks.
    When mirrored, goal labels are also flipped (left→right, right→left).
    """

    def __init__(
        self,
        path: str,
        horizon: int,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        mirror: bool = True,
    ) -> None:
        data    = np.load(path, allow_pickle=True)
        obs_all = data["obs"]               # (N, T, obs_dim)
        act_all = data["actions"]           # (N, T, act_dim)
        ep_lens = data["episode_lengths"]   # (N,)
        labels  = data["labels"]            # (N,) — 'left'/'right' strings

        N, T, act_dim = act_all.shape
        samples_obs:    list[np.ndarray] = []
        samples_act:    list[np.ndarray] = []
        samples_goal:   list[int]        = []
        chunk_weights:  list[float]      = []

        for i in range(N):
            L       = int(ep_lens[i])
            goal_id = GOAL_LEFT if labels[i] == 'left' else GOAL_RIGHT
            for t in range(L):
                chunk      = np.zeros((horizon, act_dim), dtype=np.float32)
                end        = min(t + horizon, L)
                chunk[:end - t] = act_all[i, t:end]
                if end - t < horizon:
                    chunk[end - t:] = act_all[i, end - 1]   # pad with last action
                samples_obs.append(obs_all[i, t])
                samples_act.append(chunk)
                samples_goal.append(goal_id)
                # Priority weight: early timesteps (decision point) matter more
                frac = t / max(L - 1, 1)
                chunk_weights.append(2.0 if frac < 0.1 else 1.5 if frac < 0.5 else 1.0)

        obs_arr    = np.stack(samples_obs)   # (M, obs_dim)
        act_arr    = np.stack(samples_act)   # (M, H, act_dim)
        goal_arr   = np.array(samples_goal, dtype=np.int64)
        weight_arr = np.array(chunk_weights, dtype=np.float32)

        # ── Mirror augmentation ──────────────────────────────────────
        if mirror:
            m_obs     = obs_arr.copy()
            m_act     = act_arr.copy()
            m_goal    = goal_arr.copy()
            m_weights = weight_arr.copy()

            m_obs[:, 1]  *= -1   # ee_y
            m_obs[:, 9]  *= -1   # left_cube y
            m_obs[:, 16] *= -1   # right_cube y
            left_save     = m_obs[:, 8:15].copy()
            m_obs[:, 8:15]  = m_obs[:, 15:22]
            m_obs[:, 15:22] = left_save
            m_act[:, :, 1] *= -1  # negate dy

            # Flip goal labels for mirrored samples (left demo → right after swap)
            m_goal = np.where(m_goal == GOAL_LEFT, GOAL_RIGHT, GOAL_LEFT)

            obs_arr    = np.concatenate([obs_arr,    m_obs],     axis=0)
            act_arr    = np.concatenate([act_arr,    m_act],     axis=0)
            goal_arr   = np.concatenate([goal_arr,   m_goal],    axis=0)
            weight_arr = np.concatenate([weight_arr, m_weights], axis=0)
            print(f"  mirror augment: {len(samples_obs)} → {obs_arr.shape[0]} chunks")

        # ── Normalisation ────────────────────────────────────────────
        if obs_mean is None:
            self.obs_mean = obs_arr.mean(0).astype(np.float32)
            self.obs_std  = np.maximum(obs_arr.std(0).astype(np.float32), 0.01)
        else:
            self.obs_mean = obs_mean.astype(np.float32)
            self.obs_std  = obs_std.astype(np.float32)

        # Actions: normalize by empirical mean/std so diffusion noise (σ≈1) is
        # comparable to the signal.  Actions are tiny in raw units (dx≈0.012 std),
        # so identity normalization buries the signal. Same approach as baseline.
        act_flat      = act_arr.reshape(-1, act_dim)
        self.act_mean = act_flat.mean(0).astype(np.float32)
        self.act_std  = np.maximum(act_flat.std(0).astype(np.float32), 0.01)

        self.obs = ((obs_arr - self.obs_mean) / self.obs_std).astype(np.float32)
        self.act = ((act_arr - self.act_mean) / self.act_std).astype(np.float32)
        self.goal    = goal_arr
        self.weights = weight_arr

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return (
            self.obs[idx],
            self.act[idx],
            self.goal[idx],
        )


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print("  Legibility Diffuser — Goal-Conditioned Diffusion Policy (CFG)")
    print(f"{'='*65}")
    print(f"  Device     : {device}")
    print(f"  Demo path  : {args.demo_path}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  p_uncond   : {args.p_uncond}  (CFG null label fraction)")
    print(f"  LR         : {args.lr}")
    print(f"  Batch size : {args.batch_size}")
    print(f"{'='*65}\n")

    # ── Dataset ──────────────────────────────────────────────────────
    print("Loading dataset...")
    ds = GoalLabeledDataset(args.demo_path, horizon=args.horizon, mirror=args.mirror)
    print(f"  Total chunks : {len(ds)}")
    print(f"  obs_dim      : {ds.obs.shape[1]}")
    print(f"  Left chunks  : {(ds.goal == GOAL_LEFT).sum()}")
    print(f"  Right chunks : {(ds.goal == GOAL_RIGHT).sum()}")

    sampler    = WeightedRandomSampler(ds.weights, num_samples=len(ds), replacement=True)
    dataloader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=True, drop_last=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = GoalCondDiffusionPolicy(
        obs_dim    = ds.obs.shape[1],
        act_dim    = ds.act.shape[-1],
        horizon    = args.horizon,
        hidden_dim = args.hidden_dim,
        n_blocks   = args.n_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters\n")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.1)

    # ── Diffusion schedule ───────────────────────────────────────────
    betas          = torch.linspace(args.beta_start, args.beta_end,
                                    args.n_diffusion_steps, device=device)
    alphas         = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_ab        = torch.sqrt(alphas_cumprod)
    sqrt_1m_ab     = torch.sqrt(1.0 - alphas_cumprod)

    # ── Run directory ────────────────────────────────────────────────
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / f'legdiff_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run dir    : {run_dir}\n")

    best_loss = float('inf')
    loss_hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for obs_b, act_b, goal_b in dataloader:
            obs_b  = obs_b.to(device)                   # (B, obs_dim)
            act_b  = act_b.to(device)                   # (B, H, act_dim)
            goal_b = goal_b.to(device, dtype=torch.long)  # (B,)

            B = obs_b.shape[0]

            # CFG dropout: replace p_uncond fraction with null goal
            drop_mask      = torch.rand(B, device=device) < args.p_uncond
            goal_train     = goal_b.clone()
            goal_train[drop_mask] = NULL_GOAL

            # Forward diffusion: q(x_t | x_0)
            t       = torch.randint(0, args.n_diffusion_steps, (B,), device=device)
            noise   = torch.randn_like(act_b)
            s1      = sqrt_ab[t].reshape(B, 1, 1)
            s2      = sqrt_1m_ab[t].reshape(B, 1, 1)
            xt      = s1 * act_b + s2 * noise

            # Predict noise
            eps_pred = model(xt, t, obs_b, goal_train)
            loss     = F.mse_loss(eps_pred, noise)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_hist.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:>4}/{args.epochs}  loss={avg_loss:.6f}"
                  f"  best={best_loss:.6f}  lr={lr_now:.2e}")

        # ── Save checkpoint ──────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            cfg_dict = dict(
                obs_dim           = int(ds.obs.shape[1]),
                act_dim           = int(ds.act.shape[-1]),
                horizon           = args.horizon,
                hidden_dim        = args.hidden_dim,
                n_blocks          = args.n_blocks,
                n_diffusion_steps = args.n_diffusion_steps,
                beta_start        = args.beta_start,
                beta_end          = args.beta_end,
                p_uncond          = args.p_uncond,
                goal_left         = GOAL_LEFT,
                goal_right        = GOAL_RIGHT,
                null_goal         = NULL_GOAL,
                arch              = 'conv1d',
            )
            ckpt_path = run_dir / f'ckpt_ep{epoch}.pt'
            torch.save(dict(
                model    = model.state_dict(),
                epoch    = epoch,
                loss     = avg_loss,
                config   = cfg_dict,
                obs_mean = ds.obs_mean.tolist(),
                obs_std  = ds.obs_std.tolist(),
                act_mean = ds.act_mean.tolist(),
                act_std  = ds.act_std.tolist(),
            ), ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path.name}")

    # Save final + symlink
    final_path = run_dir / 'ckpt.pt'
    torch.save(dict(
        model    = model.state_dict(),
        epoch    = args.epochs,
        loss     = loss_hist[-1],
        config   = cfg_dict,
        obs_mean = ds.obs_mean.tolist(),
        obs_std  = ds.obs_std.tolist(),
        act_mean = ds.act_mean.tolist(),
        act_std  = ds.act_std.tolist(),
    ), final_path)

    # Write loss history
    with open(run_dir / 'loss.json', 'w') as f:
        json.dump({'epochs': list(range(1, args.epochs + 1)), 'loss': loss_hist}, f)

    print(f"\n{'='*65}")
    print(f"  Training complete.  Best loss: {best_loss:.6f}")
    print(f"  Final checkpoint:  {final_path}")
    print(f"{'='*65}\n")
    print("  Next step: eval")
    print(f"  python scripts/eval_legibility_diffuser.py --checkpoint {final_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--demo_path',         default='data/demos/demos.npz')
    ap.add_argument('--output_dir',        default='runs')
    ap.add_argument('--epochs',            type=int,   default=150)
    ap.add_argument('--batch_size',        type=int,   default=256)
    ap.add_argument('--lr',                type=float, default=1e-4)
    ap.add_argument('--weight_decay',      type=float, default=1e-6)
    ap.add_argument('--grad_clip',         type=float, default=1.0)
    ap.add_argument('--horizon',           type=int,   default=32)
    ap.add_argument('--hidden_dim',        type=int,   default=256)
    ap.add_argument('--n_blocks',          type=int,   default=3)
    ap.add_argument('--n_diffusion_steps', type=int,   default=100)
    ap.add_argument('--beta_start',        type=float, default=1e-4)
    ap.add_argument('--beta_end',          type=float, default=0.1)
    ap.add_argument('--p_uncond',          type=float, default=0.15,
                    help='CFG null-label dropout probability (Bronars 2024 uses 0.15)')
    ap.add_argument('--mirror',            action='store_true', default=True,
                    help='Mirror augmentation (doubles dataset, enforces symmetry)')
    ap.add_argument('--no_mirror',         dest='mirror', action='store_false')
    ap.add_argument('--save_every',        type=int,   default=50)
    ap.add_argument('--seed',              type=int,   default=42)
    args = ap.parse_args()

    train(args)


if __name__ == '__main__':
    main()
