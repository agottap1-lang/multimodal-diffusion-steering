#!/usr/bin/env python
"""
CFG-Conditioned Diffusion Policy Training (26-d obs)
=====================================================

Trains on V2 demos (demos_v2.npz) with Classifier-Free Guidance support:
  - 26-d observation: 22 base + 3 context_pos + 1 behavior_mode
  - 15% random dropout of conditioning dims (context + mode) during training
  - At inference: eps = eps_uncond + lambda * (eps_cond - eps_uncond)

Usage:
  .venv\\Scripts\\python.exe scripts/train_cfg.py --epochs 200
  .venv\\Scripts\\python.exe scripts/train_cfg.py --epochs 200 --demo_path data/demos/demos_v2.npz
"""

import argparse, math, sys, time, copy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# MODEL ARCHITECTURE (same U-Net as train.py)
# ==============================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
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
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128),
            nn.Linear(128, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(act_dim, hidden_dim)

        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim)
            for i in range(len(dims) - 1)
        ])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1] + dims[i+1], dims[i], hidden_dim)
            for i in range(len(dims) - 2, -1, -1)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, noisy_act, timestep, obs):
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_embed(obs)
        x = self.input_proj(noisy_act) + obs_emb.unsqueeze(1)

        skips = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            skips.append(x)
        x = self.bottleneck(x, t_emb)
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = torch.cat([x, skip], dim=-1)
            x = block(x, t_emb)
        return self.output_proj(x)


# ==============================================================================
# DDPM SCHEDULER
# ==============================================================================

class DDPMScheduler:
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.n_steps = n_steps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, x0, t, noise):
        a = self.alphas_cumprod[t][:, None, None]
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


# ==============================================================================
# DATASET with CFG dropout
# ==============================================================================

OBS_DIM_V2 = 26        # 22 base + 3 context + 1 mode
CFG_COND_START = 22    # indices [22:26] = context_xyz + behavior_mode
CFG_DROPOUT_PROB = 0.15


class V2DemoDataset(Dataset):
    def __init__(self, demo_path, horizon, cfg_dropout=CFG_DROPOUT_PROB):
        data = np.load(demo_path, allow_pickle=True)
        self.obs = data['obs']            # (N, 400, 26)
        self.acts = data['actions']       # (N, 400, 5)
        self.lengths = data['episode_lengths']
        self.horizon = horizon
        self.cfg_dropout = cfg_dropout

        assert self.obs.shape[-1] == OBS_DIM_V2, \
            f"Expected obs_dim={OBS_DIM_V2}, got {self.obs.shape[-1]}"

        # Create chunks
        self.chunks = []
        for ep in range(len(self.obs)):
            ep_len = int(self.lengths[ep])
            for start in range(0, max(1, ep_len - horizon + 1), horizon // 2):
                obs = self.obs[ep][start]
                acts = self.acts[ep][start:start + horizon]
                if len(acts) == horizon:
                    self.chunks.append((obs.copy(), acts.copy()))

        # Global statistics
        all_obs, all_acts = [], []
        for ep in range(len(self.obs)):
            el = int(self.lengths[ep])
            all_obs.append(self.obs[ep][:el])
            all_acts.append(self.acts[ep][:el])
        all_obs = np.concatenate(all_obs)
        all_acts = np.concatenate(all_acts)

        self.obs_mean = all_obs.mean(0).astype(np.float32)
        self.obs_std  = np.maximum(all_obs.std(0), 0.01).astype(np.float32)
        self.act_mean = all_acts.mean(0).astype(np.float32)
        self.act_std  = np.maximum(all_acts.std(0), 0.01).astype(np.float32)

        # Store unconditional values (mean of conditioning dims, normalized)
        self.uncond_obs_norm = np.zeros(OBS_DIM_V2, dtype=np.float32)
        # After z-score, unconditional = 0 (the mean)

        print(f"Dataset: {len(self.chunks)} chunks from {len(self.obs)} episodes")
        print(f"  obs_dim={OBS_DIM_V2}, act_dim={self.acts.shape[-1]}")
        print(f"  CFG dropout prob: {self.cfg_dropout}")
        print(f"  Action mean: {self.act_mean}")
        print(f"  Action std:  {self.act_std}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        obs, acts = self.chunks[idx]
        obs = obs.copy()

        # CFG dropout: zero out conditioning dims (context + mode) with prob p
        # After z-score normalization, 0 = mean → "no conditioning info"
        if np.random.random() < self.cfg_dropout:
            obs[CFG_COND_START:] = self.obs_mean[CFG_COND_START:]  # replace with mean (→ 0 after norm)

        obs_norm = (obs - self.obs_mean) / self.obs_std
        acts_norm = (acts - self.act_mean) / self.act_std

        return {
            'obs': torch.tensor(obs_norm, dtype=torch.float32),
            'actions': torch.tensor(acts_norm, dtype=torch.float32),
        }


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_epoch(model, ema_model, loader, scheduler, optimizer, scaler, device,
                ema_decay=0.999):
    model.train()
    total_loss = 0.0
    for batch in loader:
        obs = batch['obs'].to(device)
        acts = batch['actions'].to(device)
        B = obs.shape[0]

        t = torch.randint(0, scheduler.n_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(acts)
        noisy = scheduler.add_noise(acts, t, noise)

        with torch.amp.autocast('cuda'):
            pred = model(noisy, t, obs)
            loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), model.parameters()):
                ep.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    ap = argparse.ArgumentParser("CFG Diffusion Policy Training (26-d)")
    ap.add_argument('--demo_path', default='data/demos/demos_v2.npz')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--n_blocks', type=int, default=6)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--n_diffusion_steps', type=int, default=100)
    ap.add_argument('--beta_start', type=float, default=0.0001)
    ap.add_argument('--beta_end', type=float, default=0.1)
    ap.add_argument('--cfg_dropout', type=float, default=0.15)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required!")
    device = torch.device('cuda')
    print(f"\nDevice: {device}")

    # Dataset
    dataset = V2DemoDataset(args.demo_path, args.horizon, args.cfg_dropout)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    # Model
    obs_dim = OBS_DIM_V2   # 26
    act_dim = 5
    model = DiffusionPolicy(obs_dim, act_dim, args.horizon,
                            args.hidden_dim, args.n_blocks).to(device)
    ema_model = DiffusionPolicy(obs_dim, act_dim, args.horizon,
                                args.hidden_dim, args.n_blocks).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  (hidden={args.hidden_dim}, blocks={args.n_blocks})")
    print(f"Horizon: {args.horizon}, Diffusion steps: {args.n_diffusion_steps}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = DDPMScheduler(args.n_diffusion_steps, args.beta_start, args.beta_end, device)
    scaler = torch.amp.GradScaler('cuda')

    run_dir = Path('runs') / f'cfg_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    save_epochs = {50, 100, 150, 200, args.epochs}
    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, ema_model, loader, sched, optimizer, scaler,
                           device, args.ema_decay)

        if loss < best_loss:
            best_loss = loss

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {loss:.6f} "
                  f"| Best: {best_loss:.6f} | {elapsed:.0f}s")

        if epoch in save_epochs:
            ckpt = {
                'model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'best_loss': best_loss,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'horizon': args.horizon,
                'hidden_dim': args.hidden_dim,
                'n_blocks': args.n_blocks,
                'n_diffusion_steps': args.n_diffusion_steps,
                'beta_start': args.beta_start,
                'beta_end': args.beta_end,
                'cfg_dropout': args.cfg_dropout,
                'obs_mean': dataset.obs_mean,
                'obs_std': dataset.obs_std,
                'act_mean': dataset.act_mean,
                'act_std': dataset.act_std,
            }
            torch.save(ckpt, run_dir / f'ckpt_ep{epoch}.pt')
            torch.save(ckpt, run_dir / 'ckpt_best.pt')
            print(f"  -> Saved {run_dir / f'ckpt_ep{epoch}.pt'}")

    print(f"\nDone! {args.epochs} epochs in {time.time()-t0:.0f}s")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Run: {run_dir}")


if __name__ == "__main__":
    main()
