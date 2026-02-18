#!/usr/bin/env python
"""
🤖 OPTIMIZED Diffusion Policy Training for TwoBlockPick
========================================================
Goal: Train policy to pick EITHER left OR right cube with high success rate
Target: 30-40% @ epoch 100, 50-60% @ epoch 300 (realistic with GPU)

Key Improvements:
- ✅ GPU acceleration (RTX 4060)
- ✅ 1D Convolutions for temporal modeling
- ✅ Clean minimal output
- ✅ Comprehensive comments
- ✅ Learning rate scheduling
- ✅ Data augmentation

Usage:
    python scripts/train_optimized.py --config configs/train.yaml
"""

import argparse, json, math, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ═══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE: 1D Convolutional Diffusion Model
# ═══════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps"""
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


class Conv1dBlock(nn.Module):
    """1D Convolution + GroupNorm + Mish activation"""
    def __init__(self, in_ch, out_ch, kernel_size=5, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.Mish()
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UNet1D(nn.Module):
    """
    1D U-Net for temporal action sequences
    Processes actions as (batch, channels, time) for better temporal modeling
    """
    def __init__(self, obs_dim, act_dim, horizon, dim=128, time_dim=64):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Observation encoder (shared across all action timesteps)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, dim)
        )
        
        # Action encoder: transform (batch, horizon, act_dim) -> (batch, dim, horizon)
        self.act_in = nn.Conv1d(act_dim, dim, kernel_size=1)
        
        # U-Net encoder (downsample time dimension)
        self.down1 = Conv1dBlock(dim, dim * 2)        # H -> H
        self.down2 = Conv1dBlock(dim * 2, dim * 4)    # H -> H
        
        # Bottleneck with time conditioning
        self.mid = Conv1dBlock(dim * 4, dim * 4)
        self.time_proj = nn.Linear(time_dim, dim * 4)
        
        # U-Net decoder (upsample time dimension)
        self.up1 = Conv1dBlock(dim * 8, dim * 2)      # concat skip: 4+4 -> 2
        self.up2 = Conv1dBlock(dim * 4, dim)          # concat skip: 2+2 -> 1
        
        # Output projection
        self.out = nn.Conv1d(dim, act_dim, kernel_size=1)
    
    def forward(self, noisy_act, timestep, obs):
        """
        Args:
            noisy_act: (B, H, A) - noisy action sequence
            timestep: (B,) - diffusion timestep [0, n_steps-1]
            obs: (B, O) - current observation
        Returns:
            pred_noise: (B, H, A) - predicted noise to remove
        """
        B = noisy_act.shape[0]
        
        # Encode timestep and observation
        t_emb = self.time_mlp(timestep)  # (B, time_dim)
        obs_emb = self.obs_encoder(obs)   # (B, dim)
        
        # Transform actions to (B, channels, time) for conv1d
        x = noisy_act.permute(0, 2, 1)    # (B, A, H)
        x = self.act_in(x)                # (B, dim, H)
        
        # Add observation globally to all timesteps
        x = x + obs_emb[:, :, None]       # broadcast obs across time
        
        # U-Net forward pass
        d1 = self.down1(x)                # (B, dim*2, H)
        d2 = self.down2(d1)               # (B, dim*4, H)
        
        # Bottleneck with time conditioning
        m = self.mid(d2)                  # (B, dim*4, H)
        t_proj = self.time_proj(t_emb)    # (B, dim*4)
        m = m + t_proj[:, :, None]        # add time info
        
        # U-Net decoder with skip connections
        u1 = self.up1(torch.cat([m, d2], dim=1))     # (B, dim*2, H)
        u2 = self.up2(torch.cat([u1, d1], dim=1))    # (B, dim, H)
        
        # Output projection
        out = self.out(u2)                # (B, A, H)
        return out.permute(0, 2, 1)       # (B, H, A)


# ═══════════════════════════════════════════════════════════════════
# DDPM DIFFUSION SCHEDULER
# ═══════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """DDPM forward/reverse diffusion process"""
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.n_steps = n_steps
        self.device = device
        
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    def add_noise(self, x0, t, noise):
        """Forward diffusion: q(x_t | x_0)"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    @torch.no_grad()
    def denoise_step(self, model, xt, t_idx, obs, eta=0.0):
        """
        Single reverse step: p(x_{t-1} | x_t)
        eta=0: DDIM (deterministic), eta=1: DDPM (stochastic)
        """
        B = xt.shape[0]
        t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
        
        # Predict noise
        pred_noise = model(xt, t, obs)
        
        #Compute x_{t-1}
        alpha_t = self.alphas[t_idx]
        alpha_bar_t = self.alphas_cumprod[t_idx]
        beta_t = self.betas[t_idx]
        
        # Mean of p(x_{t-1} | x_t)
        coef = beta_t / self.sqrt_one_minus_alphas_cumprod[t_idx]
        mean = (xt - coef * pred_noise) / torch.sqrt(alpha_t)
        
        if t_idx == 0:
            return mean
        
        # Add noise (DDPM) or reduced noise (DDIM)
        sigma = eta * torch.sqrt(beta_t)
        noise = torch.randn_like(xt)
        return mean + sigma * noise


# ═══════════════════════════════════════════════════════════════════
# DATASET & DATA LOADING
# ═══════════════════════════════════════════════════════════════════

class DemoDataset(Dataset):
    """Load demonstration dataset with data augmentation"""
    def __init__(self, obs, acts, horizon, mirror_aug=True, noise_std=0.01):
        self.horizon = horizon
        self.mirror_aug = mirror_aug
        self.noise_std = noise_std
        
        # Convert to chunks
        self.chunks = []
        for o, a in zip(obs, acts):
            T = len(o)
            for i in range(T - horizon + 1):
                self.chunks.append((o[i], a[i:i+horizon]))
        
        self.chunks = [(torch.tensor(o, dtype=torch.float32),
                        torch.tensor(a, dtype=torch.float32))
                       for o, a in self.chunks]
    
    def __len__(self):
        return len(self.chunks) * (2 if self.mirror_aug else 1)
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.chunks)
        obs, act = self.chunks[real_idx]
        
        # Mirror augmentation (flip y-axis for left/right symmetry)
        if self.mirror_aug and idx >= len(self.chunks):
            obs = obs.clone()
            act = act.clone()
            # Flip y positions and y velocities
            obs[1] *= -1   # ee_y
            obs[9] *= -1   # left_cube_y
            obs[16] *= -1  # right_cube_y
            act[:, 1] *= -1  # dy
        
        # Add small observation noise for regularization
        if self.noise_std > 0:
            obs = obs + torch.randn_like(obs) * self.noise_std
        
        return obs, act


def create_dataloader(demo_path, horizon, batch_size, mirror_aug=True):
    """Load demos and create dataloader"""
    data = np.load(demo_path, allow_pickle=True)
    obs = data['obs']
    acts = data['actions']  # Fixed: 'actions' not 'acts'
    
    # Compute normalization stats
    all_obs = np.concatenate(obs, axis=0)
    all_acts = np.concatenate(acts, axis=0)
    
    obs_mean = all_obs.mean(axis=0)
    obs_std = np.maximum(all_obs.std(axis=0), 0.01)  # floor at 0.01
    act_mean = all_acts.mean(axis=0)
    act_std = np.maximum(all_acts.std(axis=0), 1.0)   # keep at 1.0 for actions
    
    # Normalize
    obs = [(o - obs_mean) / obs_std for o in obs]
    acts = [(a - act_mean) / act_std for a in acts]
    
    # Create dataset
    dataset = DemoDataset(obs, acts, horizon, mirror_aug=mirror_aug)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=0, pin_memory=True)
    
    stats = {
        'obs_mean': obs_mean, 'obs_std': obs_std,
        'act_mean': act_mean, 'act_std': act_std
    }
    
    print(f"📊 Loaded {len(obs)} demos, {len(dataset)} chunks "
          f"({'with' if mirror_aug else 'without'} augmentation)")
    
    return loader, stats


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, scheduler, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for obs, act in loader:
        obs, act = obs.to(device), act.to(device)
        B = obs.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, scheduler.n_steps, (B,), device=device)
        
        # Add noise to actions
        noise = torch.randn_like(act)
        noisy_act = scheduler.add_noise(act, t, noise)
        
        # Predict noise
        pred_noise = model(noisy_act, t, obs)
        
        # Compute loss
        loss = F.mse_loss(pred_noise, noise)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    model = UNet1D(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        dim=cfg.get('hidden_dim', 128),
        time_dim=cfg.get('time_embed_dim', 64)
    ).to(device)
    
    print(f"📐 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create scheduler
    scheduler = DDPMScheduler(
        n_steps=cfg['n_diffusion_steps'],
        beta_start=cfg['beta_start'],
        beta_end=cfg['beta_end'],
        device=device
    )
    
    # Load data
    loader, stats = create_dataloader(
        cfg['demo_path'],
        cfg['horizon'],
        cfg['batch_size'],
        mirror_aug=cfg.get('mirror_augment', True)
    )
    
    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], 
                                   weight_decay=cfg.get('weight_decay', 1e-6))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.1
    )
    
    # Training loop
    run_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Training for {cfg['epochs']} epochs...")
    print(f"💾 Checkpoints: {run_dir}\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, cfg['epochs'] + 1):
        start_time = time.time()
        
        # Train
        loss = train_epoch(model, loader, scheduler, optimizer, device, epoch)
        lr_scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Print progress (every 10 epochs or milestones)
        if epoch % 10 == 0 or epoch in [1, 50, 100, 200, 300, 400, 500]:
            print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Loss: {loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        if epoch % 50 == 0 or epoch == cfg['epochs']:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'config': cfg,
                **stats
            }
            torch.save(ckpt, run_dir / f'ckpt_ep{epoch}.pt')
            torch.save(ckpt, run_dir / 'ckpt.pt')  # latest
            
            if loss < best_loss:
                best_loss = loss
                torch.save(ckpt, run_dir / 'ckpt_best.pt')
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    print(f"💾 Checkpoints saved to: {run_dir}")


if __name__ == '__main__':
    main()
