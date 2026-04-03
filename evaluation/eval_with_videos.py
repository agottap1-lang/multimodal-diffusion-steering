#!/usr/bin/env python
"""
Evaluation with video recording for visual confirmation
Records ALL episodes (both successful and failed) as videos
"""

import argparse, math, sys, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json
from collections import deque

# Import environment
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# ==============================================================================
# MODEL (Same as eval.py)
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
    """U-Net based diffusion policy"""
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
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
        B = noisy_act.shape[0]
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_embed(obs)
        
        x = self.input_proj(noisy_act)
        x = x + obs_emb.unsqueeze(1)
        
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            skip_connections.append(x)
        
        x = self.bottleneck(x, t_emb)
        
        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=-1)
            x = block(x, t_emb)
        
        out = self.output_proj(x)
        return out  # NO TANH - noise is Gaussian


# ==============================================================================
# DDIM SAMPLER
# ==============================================================================

class DDIMSampler:
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.device = device
        
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs, n_sampling_steps=10, temperature=1.0, initial_noise=None):
        """Sample with optional temperature scaling for diversity
        
        Args:
            temperature: Scale initial noise (higher = more diverse, default 1.0)
            initial_noise: Optional pre-generated noise for reproducibility
        """
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        
        if initial_noise is not None:
            x = initial_noise
        else:
            x = torch.randn(B, H, A, device=self.device) * temperature
        
        all_steps = torch.arange(len(self.alphas_cumprod), device=self.device)
        timesteps = torch.linspace(0, len(all_steps) - 1, n_sampling_steps, device=self.device).long()
        timesteps = torch.flip(timesteps, [0])
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            pred_noise = model(x, t_batch, obs)
            
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=self.device)
            
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            
            if i < len(timesteps) - 1:
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = x0_pred
        
        return x


# ==============================================================================
# EVALUATION WITH VIDEO RECORDING
# ==============================================================================

@torch.no_grad()
def evaluate_with_videos(model, sampler, ckpt, output_dir, n_episodes=50):
    """Evaluate policy and record videos of ALL episodes"""
    device = next(model.parameters()).device
    
    obs_mean = torch.tensor(ckpt['obs_mean'], device=device)
    obs_std = torch.tensor(ckpt['obs_std'], device=device)
    act_mean = ckpt['act_mean']
    act_std = ckpt['act_std']
    
    # Create video directories
    success_dir = output_dir / 'videos_success'
    failure_dir = output_dir / 'videos_failure'
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    successes = []
    lengths = []
    episodes = []  # per-episode metadata for legibility scoring
    
    print(f"\nRecording videos for {n_episodes} episodes...")
    print(f"Success videos: {success_dir}")
    print(f"Failure videos: {failure_dir}\n")
    
    for ep in range(n_episodes):
        env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
        
        obs = env.reset()
        
        # Setup video recording AFTER reset so frames aren't wiped
        video_path = output_dir / f"temp_ep{ep:03d}.mp4"
        env.record_video(str(video_path), width=640, height=480, fps=30)
        done = False
        steps = 0
        max_steps = 400
        success = False
        picked_side = None  # 'left' or 'right'
        last_info = {}
        
        action_queue = deque(maxlen=model.horizon)
        
        while not done and steps < max_steps:
            if len(action_queue) == 0:
                obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
                obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                
                act_seq = sampler.sample(model, obs_tensor, n_sampling_steps=10)
                
                # Denormalize from z-score to original scale
                act_seq = act_seq[0].cpu().numpy()
                act_seq = act_seq * act_std + act_mean
                
                for a in act_seq:
                    action_queue.append(a)
            
            action = action_queue.popleft()
            result = env.step(action)
            obs = result.obs
            done = result.done
            last_info = result.info
            success = last_info['success_left'] > 0.5 or last_info['success_right'] > 0.5
            steps += 1
        
        # Determine which block was picked
        if last_info.get('success_left', 0) > 0.5:
            picked_side = 'left'
        elif last_info.get('success_right', 0) > 0.5:
            picked_side = 'right'
        
        # Finalize video — must stop_video() BEFORE close() to flush frames
        env.stop_video()
        env.close()
        
        # Move video to appropriate directory
        final_path = None
        if Path(video_path).exists():
            if success:
                final_path = success_dir / f"ep{ep:03d}_{picked_side}_steps{steps}.mp4"
            else:
                final_path = failure_dir / f"ep{ep:03d}_fail_steps{steps}.mp4"
            shutil.move(str(video_path), str(final_path))
        
        successes.append(success)
        lengths.append(steps)
        ep_meta = {
            'episode': ep,
            'success': success,
            'picked_side': picked_side,
            'steps': steps,
            'video_path': str(final_path) if final_path else None,
        }
        episodes.append(ep_meta)
        
        side_str = f" → picked {picked_side}" if picked_side else ""
        status = "✓ SUCCESS" if success else "✗ FAILURE"
        print(f"  Episode {ep+1}/{n_episodes}: {status}{side_str} ({steps} steps)")
    
    return {
        'success_rate': np.mean(successes),
        'mean_length': np.mean(lengths),
        'successes': successes,
        'lengths': lengths,
        'episodes': episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: <checkpoint_dir>/video_eval)')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available!")
    device = torch.device('cuda')
    print(f"\nDevice: {device} (GPU-only mode)")
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.6f}")
    
    # Create model
    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3)
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Create sampler
    sampler = DDIMSampler(
        n_steps=cfg['n_diffusion_steps'],
        beta_start=cfg['beta_start'],
        beta_end=cfg['beta_end'],
        device=device
    )
    
    # Create output directory
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = Path(args.checkpoint).parent / 'video_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate with video recording
    results = evaluate_with_videos(model, sampler, ckpt, output_dir, args.n_episodes)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Mean Length: {results['mean_length']:.1f} steps")
    print(f"Successful episodes: {sum(results['successes'])}")
    print(f"Failed episodes: {len(results['successes']) - sum(results['successes'])}")
    print(f"{'='*60}")
    print(f"\nVideos saved to:")
    print(f"  Successes: {output_dir / 'videos_success'}")
    print(f"  Failures:  {output_dir / 'videos_failure'}\n")
    
    # Save full results + per-episode metadata
    save_results = {k: v for k, v in results.items() if k != 'episodes'}
    save_results['episodes'] = results['episodes']
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Results + episode metadata saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
