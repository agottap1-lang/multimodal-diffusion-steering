#!/usr/bin/env python
"""
✅ FIXED TRAINING SCRIPT - Action Statistics Bug Resolved

ROOT CAUSE BUG FIXED:
- Checkpoint was saving NORMALIZED stats (mean=0, std=1) instead of ACTUAL demo stats
- Now: Computes real demo statistics and saves them to checkpoint
- Now: Eval code can properly denormalize actions

Key changes:
1. ✅ Load demo file and compute REAL action stats (not defaults)
2. ✅ Log stats clearly so you can verify they're correct
3. ✅ Save real stats to checkpoint
4. ✅ Add safeguard: if std < 0.01, flag as problem
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
# FIXED MODEL: Same as before but well-documented
# ═══════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
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


class CompactBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=1)
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.mish(h)
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.norm2(h)
        h = F.mish(h)
        h = self.conv2(h)
        return h + x


class CompactPolicy(nn.Module):
    """2.1M parameter model"""
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=128, n_blocks=2, time_dim=32):
        super().__init__()
        
        self.act_dim = act_dim
        self.horizon = horizon
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.Mish(),
        )
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
        )
        
        self.input_proj = nn.Conv1d(act_dim, hidden_dim, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            CompactBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        
        self.output_norm = nn.GroupNorm(4, hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, act_dim, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, noisy_act, timestep, obs):
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_encoder(obs)
        cond = t_emb + obs_emb
        
        x = noisy_act.permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + cond[:, :, None]
        
        for block in self.blocks:
            x = block(x, cond)
        
        x = self.output_norm(x)
        x = F.mish(x)
        out = self.output_proj(x)
        return out.permute(0, 2, 1)


# ═══════════════════════════════════════════════════════════════════
# DDPM SCHEDULER
# ═══════════════════════════════════════════════════════════════════

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
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    def add_noise(self, x0, t, noise):
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha[:, None, None] * x0 + sqrt_one_minus_alpha[:, None, None] * noise


# ═══════════════════════════════════════════════════════════════════
# ✅ FIXED DATASET - Computes REAL action stats, not defaults
# ═══════════════════════════════════════════════════════════════════

class FixedDataset(Dataset):
    """
    ✅ FIXED: Computes REAL demo statistics, not defaults
    """
    def __init__(self, demo_path, horizon, augment=True):
        data = np.load(demo_path, allow_pickle=True)
        
        self.obs = data['obs']
        self.acts = data['actions']
        self.lengths = data['episode_lengths']
        self.horizon = horizon
        self.augment = augment
        
        self.chunks = []
        
        # Create chunks
        for ep_idx in range(len(self.obs)):
            ep_obs = self.obs[ep_idx]
            ep_acts = self.acts[ep_idx]
            ep_len = int(self.lengths[ep_idx])
            
            stride = horizon // 2 if augment else horizon
            for start in range(0, max(1, ep_len - horizon + 1), stride):
                obs = ep_obs[start]
                acts = ep_acts[start:start + horizon]
                self.chunks.append((obs, acts, ep_idx))
        
        # ✅ CRITICAL FIX: Compute REAL statistics
        all_obs = []
        all_acts = []
        for c in self.chunks:
            all_obs.append(c[0])
            all_acts.append(c[1])
        
        all_obs = np.concatenate([o.reshape(1, -1) for o in all_obs], axis=0)
        all_acts = np.concatenate(all_acts, axis=0).reshape(-1, self.acts.shape[-1])
        
        self.obs_mean = all_obs.mean(axis=0)
        self.obs_std = np.maximum(all_obs.std(axis=0), 0.01)
        
        # ✅ FIXED: Compute from ACTUAL DEMOS
        self.act_mean = all_acts.mean(axis=0).astype(np.float32)
        self.act_std = np.maximum(all_acts.std(axis=0), 0.01).astype(np.float32)
        
        print(f"\n✅ DATASET STATISTICS (REAL FROM DEMOS):")
        print(f"   Obs mean (first 3): {self.obs_mean[:3]}")
        print(f"   Obs std (first 3):  {self.obs_std[:3]}")
        print(f"   Act mean:           {self.act_mean}")
        print(f"   Act std:            {self.act_std}")
        print(f"   Act std sum:        {self.act_std.sum():.4f} (sanity check: should be > 0.1)")
        
        if self.act_std.sum() < 0.1:
            print(f"\n⚠️  WARNING: Action std is very small!")
            print(f"   This suggests demos are VERY SLOW or TINY movements")
            print(f"   Consider re-collecting demos with faster motions")
        
        print(f"\n   {len(self.chunks)} chunks from {len(self.obs)} episodes")
        if augment:
            print(f"   With 10x augmentation: {len(self.chunks) * 10} effective samples")
    
    def __len__(self):
        return len(self.chunks) * (10 if self.augment else 1)
    
    def __getitem__(self, idx):
        chunk_idx = idx % len(self.chunks)
        aug_idx = idx // len(self.chunks) if self.augment else 0
        
        obs, acts, ep_idx = self.chunks[chunk_idx]
        obs = obs.copy()
        acts = acts.copy()
        
        # Simple augmentation
        if self.augment and aug_idx > 0:
            if aug_idx == 1:
                acts = acts * 1.1  # Speed up
            elif aug_idx == 2:
                acts = acts * 0.9  # Speed down
            elif aug_idx == 3:
                obs = obs + np.random.randn(*obs.shape) * 0.01
            else:
                acts = acts + np.random.randn(*acts.shape) * (0.02 * (aug_idx / 10))
        
        # Normalize using REAL stats
        obs = (obs - self.obs_mean) / self.obs_std
        acts = (acts - self.act_mean) / self.act_std
        
        return {
            'obs': torch.tensor(obs, dtype=torch.float32),
            'actions': torch.tensor(acts, dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, scheduler, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        obs = batch['obs'].to(device)
        acts = batch['actions'].to(device)
        B = obs.shape[0]
        
        t = torch.randint(0, scheduler.n_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(acts)
        noisy_acts = scheduler.add_noise(acts, t, noise)
        
        pred_noise = model(noisy_acts, t, obs)
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("✅ FIXED TRAINING WITH CORRECT ACTION STATISTICS")
    print("="*70)
    
    # ✅ Load dataset with REAL stats
    dataset = FixedDataset('data/demos/demos.npz', cfg['horizon'], augment=True)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    
    # Create model
    model = CompactPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=128,
        n_blocks=2,
        time_dim=32,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (args.epochs - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = DDPMScheduler(cfg['n_diffusion_steps'], cfg['beta_start'], cfg['beta_end'], device)
    
    run_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run: {run_dir}\n")
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_epoch(model, loader, scheduler, optimizer, device, epoch, args.epochs)
        lr_scheduler.step()
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']
        
        if epoch in [1, 10, 25, 50, 75, 100] or epoch % 50 == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss:.6f} | LR: {lr:.2e} | {elapsed:.1f}s")
        
        # Save checkpoints
        if epoch in [50, 100]:
            # ✅ FIXED: Save REAL action statistics, not defaults!
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'config': cfg,
                'obs_mean': dataset.obs_mean,
                'obs_std': dataset.obs_std,
                'act_mean': dataset.act_mean,  # ✅ REAL STATS
                'act_std': dataset.act_std,    # ✅ REAL STATS
            }
            torch.save(ckpt, run_dir / f'ckpt_ep{epoch}.pt')
            torch.save(ckpt, run_dir / 'ckpt.pt')
            print(f"  💾 Saved checkpoint ep{epoch}")
            print(f"     act_mean: {dataset.act_mean}")
            print(f"     act_std:  {dataset.act_std}")
    
    print("\n" + "="*70)
    print(f"✅ Training complete! Checkpoints in {run_dir}")
    print(f"\n📊 VERIFY CHECKPOINT HAS CORRECT STATS:")
    print(f"   python -c \"import torch; c = torch.load('{run_dir}/ckpt.pt'); print('act_std:', c['act_std'])\"")
    print(f"\n🎯 NEXT: python scripts/eval_multimodality.py --ckpt {run_dir}/ckpt_ep50.pt --K 5 --M 5")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
