#!/usr/bin/env python
"""
CLEAN DIFFUSION POLICY - Minimal working implementation
Based on: Diffusion Policy (Chi et al., 2023)
Architecture: Simple MLP with ResNet blocks
"""

import argparse, math, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# ==============================================================================
# MODEL ARCHITECTURE
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
    """U-Net residual block with time conditioning"""
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
        # x: (B, seq_len, in_dim)
        h = self.conv1(x)
        h = h.transpose(1, 2)  # (B, out_dim, seq_len)
        h = self.norm1(h)
        h = h.transpose(1, 2)  # (B, seq_len, out_dim)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        
        return self.act(h + self.shortcut(x))


class DiffusionPolicy(nn.Module):
    """U-Net based diffusion policy"""
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128),
            nn.Linear(128, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Observation embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Input projection for noisy actions
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        
        # U-Net encoder (downsampling path)
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim) 
            for i in range(len(dims) - 1)
        ])
        
        # Bottleneck
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        
        # U-Net decoder (upsampling path with skip connections)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1] + dims[i+1], dims[i], hidden_dim)  # +skip connection
            for i in range(len(dims) - 2, -1, -1)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, noisy_act, timestep, obs):
        """
        noisy_act: (B, H, A)
        timestep: (B,)
        obs: (B, O)
        """
        B = noisy_act.shape[0]
        t_emb = self.time_mlp(timestep)  # (B, hidden_dim)
        obs_emb = self.obs_embed(obs)  # (B, hidden_dim)
        
        # Project actions to hidden dim: (B, H, A) -> (B, H, hidden_dim)
        x = self.input_proj(noisy_act)
        
        # Add observation conditioning via broadcast
        x = x + obs_emb.unsqueeze(1)
        
        # Encoder with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x, t_emb)
        
        # Decoder with skip connections
        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=-1)  # Concatenate skip connection
            x = block(x, t_emb)
        
        # Output: (B, H, hidden_dim) -> (B, H, A)
        # CRITICAL: Predict UNBOUNDED noise, not clipped actions!
        out = self.output_proj(x)
        return out  # NO TANH - noise is Gaussian, not bounded


# ==============================================================================
# DDPM SCHEDULER
# ==============================================================================

class DDPMScheduler:
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.n_steps = n_steps
        self.device = device
        
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def add_noise(self, x0, t, noise):
        """Add noise to clean data"""
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None]
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise


# ==============================================================================
# DATASET
# ==============================================================================

class DemoDataset(Dataset):
    def __init__(self, demo_path, horizon):
        data = np.load(demo_path, allow_pickle=True)
        
        self.obs = data['obs']
        self.acts = data['actions']
        self.lengths = data['episode_lengths']
        self.horizon = horizon
        
        # Create chunks
        self.chunks = []
        for ep_idx in range(len(self.obs)):
            ep_len = int(self.lengths[ep_idx])
            for start in range(0, max(1, ep_len - horizon + 1), horizon // 2):
                obs = self.obs[ep_idx][start]
                acts = self.acts[ep_idx][start:start + horizon]
                if len(acts) == horizon:
                    self.chunks.append((obs, acts))
        
        # Compute GLOBAL statistics
        all_obs = []
        all_acts = []
        for ep_idx in range(len(self.obs)):
            ep_len = int(self.lengths[ep_idx])
            all_obs.append(self.obs[ep_idx][:ep_len])
            all_acts.append(self.acts[ep_idx][:ep_len])
        
        all_obs = np.concatenate(all_obs, axis=0)
        all_acts = np.concatenate(all_acts, axis=0)
        
        self.obs_mean = all_obs.mean(axis=0).astype(np.float32)
        self.obs_std = np.maximum(all_obs.std(axis=0), 0.01).astype(np.float32)
        
        # Use mean/std normalization (z-score) for actions - CRITICAL for diffusion
        self.act_mean = all_acts.mean(axis=0).astype(np.float32)
        self.act_std = np.maximum(all_acts.std(axis=0), 0.01).astype(np.float32)
        
        print(f"Dataset: {len(self.chunks)} chunks, {len(self.obs)} episodes")
        print(f"Action stats: mean={self.act_mean}, std={self.act_std}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        obs, acts = self.chunks[idx]
        
        # Normalize with mean/std (z-score) - CRITICAL for diffusion
        obs = (obs - self.obs_mean) / self.obs_std
        acts = (acts - self.act_mean) / self.act_std
        
        return {
            'obs': torch.tensor(obs, dtype=torch.float32),
            'actions': torch.tensor(acts, dtype=torch.float32),
        }


# ==============================================================================
# TRAINING
# ==============================================================================

def train_epoch(model, ema_model, loader, scheduler, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        obs = batch['obs'].to(device)
        acts = batch['actions'].to(device)
        B = obs.shape[0]
        
        # Random timesteps
        t = torch.randint(0, scheduler.n_steps, (B,), device=device, dtype=torch.long)
        
        # Add noise
        noise = torch.randn_like(acts)
        noisy_acts = scheduler.add_noise(acts, t, noise)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            pred_noise = model(noisy_acts, t, obs)
            loss = F.mse_loss(pred_noise, noise)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA model - CRITICAL for stable evaluation
        with torch.no_grad():
            ema_decay = 0.999
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available!")
    device = torch.device('cuda')
    print(f"\nDevice: {device} (GPU-only mode)")
    
    # Dataset
    dataset = DemoDataset(cfg['demo_path'], cfg['horizon'])
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Model
    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3)
    ).to(device)
    
    # EMA model - CRITICAL for stable evaluation
    ema_model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3)
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler_ddpm = DDPMScheduler(cfg['n_diffusion_steps'], cfg['beta_start'], cfg['beta_end'], device)
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop
    run_dir = Path('runs') / f'diffusion_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, ema_model, loader, scheduler_ddpm, optimizer, scaler, device)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.6f}")
        
        # Save checkpoint with EMA model - CRITICAL
        if epoch in [50, 100] or epoch == args.epochs:
            ckpt = {
                'model': ema_model.state_dict(),  # Use EMA for evaluation!
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'config': cfg,
                'obs_mean': dataset.obs_mean,
                'obs_std': dataset.obs_std,
                'act_mean': dataset.act_mean,
                'act_std': dataset.act_std,
            }
            torch.save(ckpt, run_dir / f'ckpt_ep{epoch}.pt')
            print(f"  Saved checkpoint: {run_dir / f'ckpt_ep{epoch}.pt'}")
    
    print(f"\nTraining complete! Run: {run_dir}")


if __name__ == "__main__":
    main()
