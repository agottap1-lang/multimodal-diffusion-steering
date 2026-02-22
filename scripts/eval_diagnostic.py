#!/usr/bin/env python
"""
🔍 EVALUATION WITH DIAGNOSTICS
Identifies exactly what's wrong with your policy:
- Action scaling issues
- Model collapse
- Evaluation bugs
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM
from scripts.train_diffusion_policy import DDPMSchedule, NoiseNet


class SimplePolicyEval:
    """Minimal evaluation to diagnose issues"""
    
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        
        self.device = device
        self.horizon = cfg['horizon']
        self.n_action_steps = cfg['n_action_steps']
        
        # Load demo stats
        self.obs_mean = torch.tensor(ckpt['obs_mean'], device=device, dtype=torch.float32)
        self.obs_std = torch.tensor(ckpt['obs_std'], device=device, dtype=torch.float32)
        self.act_mean = torch.tensor(ckpt['act_mean'], device=device, dtype=torch.float32)
        self.act_std = torch.tensor(ckpt['act_std'], device=device, dtype=torch.float32)
        
        # Load model (use old NoiseNet if that's what's saved)
        try:
            self.model = NoiseNet(
                obs_dim=cfg['obs_dim'],
                act_dim=cfg['act_dim'],
                horizon=cfg['horizon'],
                hidden_dim=cfg.get('hidden_dim', 256),
                n_blocks=cfg.get('n_blocks', 2),
                time_embed_dim=cfg.get('time_embed_dim', 64)
            ).to(device)
            self.model.load_state_dict(ckpt['model'])
        except:
            print("⚠️  Could not load model (may be using different architecture)")
            return False
        
        self.model.eval()
        
        # DDPM
        self.schedule = DDPMSchedule(
            cfg['n_diffusion_steps'],
            cfg['beta_start'],
            cfg['beta_end'],
            device
        )
        
        print(f"✓ Policy loaded from {ckpt_path}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
        print(f"  Loss: {ckpt.get('loss', '?'):.6f}")
        return True
    
    @torch.no_grad()
    def get_action(self, obs, verbose=False):
        """Get single action from policy"""
        obs_t = torch.tensor(obs, device=self.device, dtype=torch.float32)
        obs_t = (obs_t - self.obs_mean) / self.obs_std
        obs_t = obs_t.unsqueeze(0)
        
        # Sample from diffusion
        chunk = self.schedule.sample(
            self.model, obs_t,
            self.horizon, ACT_DIM,
            method='ddpm'
        )
        
        # Denormalize
        chunk = chunk * self.act_std.unsqueeze(0) + self.act_mean.unsqueeze(0)
        chunk = chunk.squeeze(0).cpu().numpy()
        chunk = np.clip(chunk, -1, 1)
        
        if verbose:
            print(f"  Action sampled: mean={chunk.mean(axis=0)}, std={chunk.std(axis=0)}")
        
        return chunk[0]  # Return first action


def diagnose():
    """Run full diagnosis"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='runs/latest/ckpt.pt')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 POLICY EVALUATION WITH DIAGNOSTICS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load policy
    policy = SimplePolicyEval(args.ckpt, device)
    
    # Create environment
    env = TwoBlockPickEnv(render=False)
    
    print("\n" + "="*70)
    print("TEST 1: Action Scaling Verification")
    print("="*70)
    
    # Get action from policy
    obs = env.reset(seed=100)
    action = policy.get_action(obs, verbose=True)
    
    # Load demos to compare
    demo_data = np.load('data/demos/demos.npz', allow_pickle=True)
    demo_actions = demo_data['actions']
    demo_action_std = demo_actions.std()
    
    policy_action_std = action[:4].std()  # Position dims only
    ratio = policy_action_std / demo_action_std
    
    print(f"\nDemo action std (pos):   {demo_action_std:.4f}")
    print(f"Policy action std (pos): {policy_action_std:.4f}")
    print(f"Ratio:                   {ratio:.2f}x")
    
    if ratio < 0.3:
        print("❌ CRITICAL: Action suppression! Policy outputs are tiny.")
    elif ratio < 0.7:
        print("⚠️  WARNING: Action suppression. Policy may move too slowly.")
    elif ratio > 2.0:
        print("⚠️  WARNING: Action amplification. Policy may overshoot.")
    else:
        print("✅ GOOD: Action scaling looks reasonable.")
    
    # Test 2: Simple rollout
    print("\n" + "="*70)
    print("TEST 2: Simple Deterministic Rollout")
    print("="*70)
    
    obs = env.reset(seed=100)
    total_reward = 0.0
    ee_start = obs[:3].copy()
    
    for step in range(50):
        action = policy.get_action(obs, verbose=(step == 0))
        result = env.step(action)
        obs = result.obs
        total_reward += result.reward
        
        if step % 10 == 0:
            ee_pos = obs[:3]
            dist_from_start = np.linalg.norm(ee_pos - ee_start)
            print(f"Step {step:3d}: EE moved {dist_from_start:.3f} m, total reward {total_reward:.2f}")
        
        if result.done:
            print(f"Episode finished at step {step}")
            break
    
    print(f"Final reward: {total_reward:.2f}")
    
    ee_final = obs[:3]
    total_dist = np.linalg.norm(ee_final - ee_start)
    print(f"Total EE displacement: {total_dist:.3f} m ({total_dist*100:.1f} cm)")
    
    if total_dist < 0.01:
        print("❌ CRITICAL: Robot didn't move! Action scaling is broken.")
    elif total_dist < 0.1:
        print("⚠️  WARNING: Robot barely moved. Actions might be suppressed.")
    else:
        print("✅ GOOD: Robot is moving.")
    
    # Test 3: Success metric
    print("\n" + "="*70)
    print("TEST 3: Success Metric Check")
    print("="*70)
    
    left_cube_pos = obs[8:11]
    right_cube_pos = obs[15:18]
    
    print(f"Left cube  z: {left_cube_pos[2]:.4f} (success if > 0.52)")
    print(f"Right cube z: {right_cube_pos[2]:.4f} (success if > 0.52)")
    
    if result.info.get('picked_left'):
        print("✅ LEFT CUBE PICKED!")
    elif result.info.get('picked_right'):
        print("✅ RIGHT CUBE PICKED!")
    else:
        print("❌ No cube picked")
    
    env.close()
    
    # Final recommendation
    print("\n" + "="*70)
    print("💡 DIAGNOSIS COMPLETE")
    print("="*70)
    print("""
If you see:
- ❌ "Action suppression" → Model output is too small
  → Possible: Bad normalization, model collapse, or undertrained
  → Fix: Check obs_mean/obs_std in checkpoint match demos
  
- ✅ "Robot is moving" but "No cube picked" → Execution works, policy doesn't
  → Possible: Model needs more training or different data
  → Fix: Train for more epochs (200-500)
  
- ✅ "Cube picked" → Everything works! Multimodality needs tuning
  → Next: Run full evaluation with multiple seeds
    """)


if __name__ == '__main__':
    diagnose()
