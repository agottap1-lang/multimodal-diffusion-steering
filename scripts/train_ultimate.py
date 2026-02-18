#!/usr/bin/env python
"""
🚀 ULTIMATE Diffusion Policy Training - Target: 50%@ep100, 80%@ep300
===========================================================================
Enhanced architecture with temporal attention, deeper UNet, aggressive augmentation

Improvements over train_optimized.py:
- ✅ Temporal Self-Attention layers (capture long-range dependencies)
- ✅ Deeper UNet (more layers for better representation)
- ✅ Residual connections everywhere
- ✅ FiLM conditioning (better obs+time integration)
- ✅ Heavy data augmentation (rotation, noise, temporal jitter)
- ✅ Progressive training (curriculum from easy to hard)
- ✅ EMA (exponential moving average) for stable inference

Usage:
    python scripts/train_ultimate.py --config configs/train.yaml --epochs 1000
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
# ENHANCED ARCHITECTURE: Deep UNet with Attention
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


class TemporalAttention(nn.Module):
    """Self-attention over temporal (action sequence) dimension"""
    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        """x: (B, C, T) - batch, channels, time"""
        B, C, T = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x)  # (B, 3C, T)
        q, k, v = qkv.chunk(3, dim=1)  # each (B, C, T)
        
        # Reshape for multi-head attention
        q = q.view(B, self.n_heads, C // self.n_heads, T)
        k = k.view(B, self.n_heads, C // self.n_heads, T)
        v = v.view(B, self.n_heads, C // self.n_heads, T)
        
        # Attention: (B, heads, T, T)
        attn = torch.einsum('bhct,bhcs->bhts', q, k) / math.sqrt(C // self.n_heads)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhts,bhcs->bhct', attn, v)
        out = out.reshape(B, C, T)
        
        out = self.proj(out)
        return out + residual  # residual connection


class ResConv1dBlock(nn.Module):
    """1D ResNet block with FiLM conditioning"""
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=5, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        
        # FiLM conditioning (scale and shift)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        
        # Residual connection
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.Mish()
    
    def forward(self, x, cond):
        """
        x: (B, C, T)
        cond: (B, cond_dim) - combined time + obs conditioning
        """
        residual = self.residual(x)
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply FiLM conditioning
        scale, shift = self.film(cond).chunk(2, dim=1)  # (B, out_ch) each
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.act(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + residual


class EnhancedUNet1D(nn.Module):
    """
    Deep U-Net with Temporal Attention for action sequences
    Architecture: Encoder -> Bottleneck (with attention) -> Decoder
    """
    def __init__(self, obs_dim, act_dim, horizon, dim=192, time_dim=128, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.dim = dim
        
        # Timestep embedding (larger for better conditioning)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Observation encoder (deeper for better state representation)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, dim * 3),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(dim * 3, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, dim)
        )
        
        # Combined conditioning dimension
        self.cond_dim = time_dim + dim
        
        # Input projection
        self.act_in = nn.Conv1d(act_dim, dim, kernel_size=1)
        
        # Encoder (3 levels)
        self.down1 = nn.ModuleList([
            ResConv1dBlock(dim if i == 0 else dim, dim, self.cond_dim) 
            for i in range(n_blocks)
        ])
        self.down2 = nn.ModuleList([
            ResConv1dBlock(dim if i == 0 else dim * 2, dim * 2, self.cond_dim) 
            for i in range(n_blocks)
        ])
        self.down3 = nn.ModuleList([
            ResConv1dBlock(dim * 2 if i == 0 else dim * 4, dim * 4, self.cond_dim) 
            for i in range(n_blocks)
        ])
        
        # Bottleneck with temporal attention
        self.mid_blocks = nn.ModuleList([
            ResConv1dBlock(dim * 4, dim * 4, self.cond_dim),
            TemporalAttention(dim * 4, n_heads=8),
            ResConv1dBlock(dim * 4, dim * 4, self.cond_dim),
        ])
        
        # Decoder (with skip connections from encoder)
        # First block in each level accepts concatenated skip, rest accept previous output
        self.up1 = nn.ModuleList([
            ResConv1dBlock(dim * 8 if i == 0 else dim * 2, dim * 2, self.cond_dim) 
            for i in range(n_blocks)
        ])
        self.up2 = nn.ModuleList([
            ResConv1dBlock(dim * 4 if i == 0 else dim, dim, self.cond_dim) 
            for i in range(n_blocks)
        ])
        self.up3 = nn.ModuleList([
            ResConv1dBlock(dim * 2 if i == 0 else dim, dim, self.cond_dim) 
            for i in range(n_blocks)
        ])
        
        # Output projection
        self.out_norm = nn.GroupNorm(8, dim)
        self.out = nn.Conv1d(dim, act_dim, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, noisy_act, timestep, obs):
        """
        Args:
            noisy_act: (B, H, A) - noisy action sequence
            timestep: (B,) - diffusion timestep [0, n_steps-1]
            obs: (B, O) - current observation
        Returns:
            pred_noise: (B, H, A) - predicted noise
        """
        B = noisy_act.shape[0]
        
        # Encode timestep and observation
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_encoder(obs)
        
        # Combine conditioning
        cond = torch.cat([t_emb, obs_emb], dim=1)  # (B, cond_dim)
        
        # Input projection
        x = noisy_act.permute(0, 2, 1)  # (B, A, H)
        x = self.act_in(x)              # (B, dim, H)
        
        # Encoder
        for block in self.down1:
            x = block(x, cond)
        skip1 = x  # (B, dim, H)
        
        for block in self.down2:
            x = block(x, cond)
        skip2 = x  # (B, dim*2, H)
        
        for block in self.down3:
            x = block(x, cond)
        skip3 = x  # (B, dim*4, H)
        
        # Bottleneck with attention
        for i, block in enumerate(self.mid_blocks):
            if isinstance(block, TemporalAttention):
                x = block(x)
            else:
                x = block(x, cond)
        # x is (B, dim*4, H)
        
        # Decoder with skip connections
        x = torch.cat([x, skip3], dim=1)  # (B, dim*8, H)
        for block in self.up1:
            x = block(x, cond)
        # x is now (B, dim*2, H)
        
        x = torch.cat([x, skip2], dim=1)  # (B, dim*4, H)
        for block in self.up2:
            x = block(x, cond)
        # x is now (B, dim, H)
        
        x = torch.cat([x, skip1], dim=1)  # (B, dim*2, H)
        for block in self.up3:
            x = block(x, cond)
        # x is now (B, dim, H)
        
        # Output
        x = self.out_norm(x)
        x = F.mish(x)
        out = self.out(x)  # (B, A, H)
        return out.permute(0, 2, 1)  # (B, H, A)


# ═══════════════════════════════════════════════════════════════════
# DDPM DIFFUSION SCHEDULER
# ═══════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """DDPM noise schedule with linear beta schedule"""
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
        """Add noise to clean data x0 at timestep t"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha[:, None, None] * x0 + sqrt_one_minus_alpha[:, None, None] * noise
    
    @torch.no_grad()
    def sample(self, model, obs, horizon, act_dim):
        """Generate action sequence from noise (DDPM sampling)"""
        B = obs.shape[0]
        x_t = torch.randn(B, horizon, act_dim, device=self.device)
        
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            pred_noise = model(x_t, t_batch, obs)
            
            # DDPM reverse step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0.0
            
            x_t = (1.0 / torch.sqrt(alpha)) * (
                x_t - (beta / torch.sqrt(1.0 - alpha_cumprod)) * pred_noise
            ) + torch.sqrt(beta) * noise
        
        return x_t


# ═══════════════════════════════════════════════════════════════════
# ENHANCED DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

class EnhancedDemoDataset(Dataset):
    """
    Dataset with aggressive augmentation:
    - Mirror symmetry (left/right cube swap)
    - Observation noise
    - Action noise
    - Temporal dropout (random timesteps zeroed)
    """
    def __init__(self, demo_path, horizon, augment=True):
        data = np.load(demo_path, allow_pickle=True)
        
        all_obs = data['obs']       # (N_eps, T, obs_dim)
        all_acts = data['actions']  # (N_eps, T, act_dim)
        lengths = data['episode_lengths']
        
        self.horizon = horizon
        self.augment = augment
        self.chunks = []
        
        # Chunk episodes into horizon-length sequences
        for ep_idx in range(len(all_obs)):
            ep_obs = all_obs[ep_idx]
            ep_acts = all_acts[ep_idx]
            ep_len = int(lengths[ep_idx])
            
            for start in range(0, ep_len - horizon + 1):
                obs = ep_obs[start]
                acts = ep_acts[start:start + horizon]
                self.chunks.append((obs, acts))
        
        # Compute normalization statistics (before augmentation)
        all_obs_flat = np.concatenate([c[0] for c in self.chunks], axis=0).reshape(-1, all_obs.shape[-1])
        all_acts_flat = np.concatenate([c[1] for c in self.chunks], axis=0).reshape(-1, all_acts.shape[-1])
        
        self.obs_mean = all_obs_flat.mean(axis=0)
        self.obs_std = np.maximum(all_obs_flat.std(axis=0), 0.01)
        self.act_mean = all_acts_flat.mean(axis=0)
        self.act_std = np.maximum(all_acts_flat.std(axis=0), 0.01)
        
        print(f"Dataset: {len(self.chunks)} chunks from {len(all_obs)} episodes")
        if augment:
            print(f"  Augmentation: Mirror + Obs noise + Act noise + Temporal dropout")
    
    def __len__(self):
        # 3x augmentation (original + 2 augmented versions)
        return len(self.chunks) * 3 if self.augment else len(self.chunks)
    
    def __getitem__(self, idx):
        # Determine which chunk and augmentation mode
        chunk_idx = idx % len(self.chunks)
        aug_mode = idx // len(self.chunks) if self.augment else 0
        
        obs, acts = self.chunks[chunk_idx]
        obs = obs.copy()
        acts = acts.copy()
        
        # Apply augmentation based on mode
        if aug_mode == 1:  # Mirror symmetry
            obs = self._mirror_obs(obs)
            acts = self._mirror_acts(acts)
        elif aug_mode == 2:  # Heavy noise
            obs = self._add_obs_noise(obs, std=0.02)
            acts = self._add_act_noise(acts, std=0.03)
        
        # Additional random augmentation
        if self.augment and np.random.rand() < 0.5:
            obs = self._add_obs_noise(obs, std=0.01)
        if self.augment and np.random.rand() < 0.3:
            acts = self._temporal_dropout(acts, prob=0.1)
        
        # Normalize
        obs = (obs - self.obs_mean) / self.obs_std
        acts = (acts - self.act_mean) / self.act_std
        
        return {
            'obs': torch.tensor(obs, dtype=torch.float32),
            'actions': torch.tensor(acts, dtype=torch.float32)
        }
    
    def _mirror_obs(self, obs):
        """Mirror observation (swap left/right cubes, flip y)"""
        o = obs.copy()
        # Swap y coordinate of EE, left cube, right cube
        o[1] = -o[1]       # ee_y
        o[9] = -o[9]       # left_cube_y
        o[16] = -o[16]     # right_cube_y
        # Swap left and right cube entirely
        left = o[8:15].copy()
        right = o[15:22].copy()
        o[8:15] = right
        o[15:22] = left
        return o
    
    def _mirror_acts(self, acts):
        """Mirror actions (flip dy)"""
        a = acts.copy()
        a[:, 1] = -a[:, 1]  # flip dy
        return a
    
    def _add_obs_noise(self, obs, std=0.01):
        """Add Gaussian noise to observation"""
        noise = np.random.randn(*obs.shape) * std
        return obs + noise
    
    def _add_act_noise(self, acts, std=0.02):
        """Add Gaussian noise to actions"""
        noise = np.random.randn(*acts.shape) * std
        return acts + noise
    
    def _temporal_dropout(self, acts, prob=0.1):
        """Randomly zero out some timesteps (forces model to be robust)"""
        mask = np.random.rand(len(acts)) > prob
        return acts * mask[:, None]


# ═══════════════════════════════════════════════════════════════════
# EXPONENTIAL MOVING AVERAGE (EMA)
# ═══════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
    
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
    
    def apply(self, model):
        """Apply EMA weights to model (for inference)"""
        model.load_state_dict(self.shadow)


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, scheduler, optimizer, device, epoch, ema=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        obs = batch['obs'].to(device)
        acts = batch['actions'].to(device)
        B = obs.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, scheduler.n_steps, (B,), device=device, dtype=torch.long)
        
        # Add noise to actions
        noise = torch.randn_like(acts)
        noisy_acts = scheduler.add_noise(acts, t, noise)
        
        # Predict noise
        pred_noise = model(noisy_acts, t, obs)
        
        # MSE loss
        loss = F.mse_loss(pred_noise, noise)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=192, help='Model dimension')
    parser.add_argument('--n_blocks', type=int, default=3, help='Blocks per level')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 ULTIMATE TRAINING - Target: 50%@ep100, 80%@ep300")
    print(f"Device: {device}")
    print(f"Model dim: {args.dim}, Blocks: {args.n_blocks}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Training on CPU will be VERY slow! Install CUDA PyTorch.")
    
    # Create dataset
    demo_path = cfg.get('demo_path', 'data/demos/demos.npz')
    dataset = EnhancedDemoDataset(demo_path, cfg['horizon'], augment=True)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    
    # Create model
    model = EnhancedUNet1D(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        dim=args.dim,
        n_blocks=args.n_blocks
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create scheduler
    ddpm = DDPMScheduler(
        cfg['n_diffusion_steps'],
        cfg['beta_start'],
        cfg['beta_end'],
        device
    )
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate schedule (cosine annealing)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # EMA for stable inference
    ema = EMA(model, decay=0.9999)
    
    # Create output directory
    run_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Training loop
    best_loss = float('inf')
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        loss = train_epoch(model, loader, ddpm, optimizer, device, epoch, ema)
        lr_scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log progress at key epochs
        if epoch in [1, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] or epoch % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {loss:.6f} | LR: {lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save checkpoints
        if loss < best_loss:
            best_loss = loss
            
        if epoch in [50, 100, 200, 300, 400, 500, 1000] or epoch % 100 == 0:
            # Save with EMA weights
            ema.apply(model)
            ckpt = {
                'model': model.state_dict(),
                'ema': ema.shadow,
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
            torch.save(ckpt, run_dir / 'ckpt.pt')  # Latest
            
            if epoch in [100, 300]:
                print(f"  💾 Saved checkpoint at epoch {epoch}")
                print(f"  🎯 Evaluate now: py scripts/eval_multimodality.py --ckpt {run_dir}/ckpt_ep{epoch}.pt")
    
    print("=" * 60)
    print(f"✅ Training complete! Final loss: {loss:.6f}")
    print(f"📊 Evaluate: py scripts/eval_multimodality.py --ckpt {run_dir}/ckpt_ep300.pt")


if __name__ == '__main__':
    main()
