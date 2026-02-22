#!/usr/bin/env python
"""
🔍 DIAGNOSIS SCRIPT: Find what's broken in your policy
This script runs simple tests to identify if the problem is:
1. Action scaling / normalization
2. Model collapse
3. Evaluation methodology
4. Data issues
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

def diagnose_demos():
    """Check demo data integrity and statistics"""
    print("\n" + "="*70)
    print("1️⃣  DEMO DATA DIAGNOSIS")
    print("="*70)
    
    demo_path = "data/demos/demos.npz"
    if not Path(demo_path).exists():
        print("❌ Demo file not found!")
        return None
    
    data = np.load(demo_path, allow_pickle=True)
    
    obs = data['obs']           # (N_eps, T, obs_dim)
    acts = data['actions']      # (N_eps, T, act_dim)
    lengths = data['episode_lengths']
    
    print(f"\n✓ Observations shape: {obs.shape}")
    print(f"  - Episodes: {len(obs)}")
    print(f"  - Max timesteps: {obs.shape[1]}")
    print(f"  - Obs dim: {obs.shape[2]}")
    
    print(f"\n✓ Actions shape: {acts.shape}")
    print(f"  - Episodes: {len(acts)}")
    print(f"  - Max timesteps: {acts.shape[1]}")
    print(f"  - Action dim: {acts.shape[2]}")
    
    # Compute statistics
    all_obs = []
    all_acts = []
    
    for ep_idx in range(len(obs)):
        ep_len = int(lengths[ep_idx])
        all_obs.append(obs[ep_idx, :ep_len])
        all_acts.append(acts[ep_idx, :ep_len])
    
    all_obs = np.concatenate(all_obs, axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    
    print(f"\n📊 Observation Statistics:")
    print(f"  Mean: {all_obs.mean(axis=0)[:3]}")  # First 3 dims (EE pos)
    print(f"  Std:  {all_obs.std(axis=0)[:3]}")
    print(f"  Min:  {all_obs.min(axis=0)[:3]}")
    print(f"  Max:  {all_obs.max(axis=0)[:3]}")
    
    print(f"\n📊 Action Statistics:")
    print(f"  Mean: {all_acts.mean(axis=0)}")
    print(f"  Std:  {all_acts.std(axis=0)}")
    print(f"  Min:  {all_acts.min(axis=0)}")
    print(f"  Max:  {all_acts.max(axis=0)}")
    
    # Check for success outcomes
    labels = data.get('labels', None)
    if labels is not None:
        n_left = np.sum(labels == 0)
        n_right = np.sum(labels == 1)
        print(f"\n🎯 Demo Outcomes:")
        print(f"  Left picks:  {n_left} ({n_left/len(labels)*100:.1f}%)")
        print(f"  Right picks: {n_right} ({n_right/len(labels)*100:.1f}%)")
    
    return all_obs, all_acts


def diagnose_environment():
    """Check environment action scaling"""
    print("\n" + "="*70)
    print("2️⃣  ENVIRONMENT ACTION SCALING DIAGNOSIS")
    print("="*70)
    
    env = TwoBlockPickEnv(render=False)
    
    print(f"\n✓ Environment configured:")
    print(f"  - Demo action scale (pos): {env.action_scale_pos} m/step")
    print(f"  - Demo action scale (yaw): {np.degrees(env._action_scale_yaw):.1f} deg/step")
    print(f"  - Action bounds: [-1, 1]")
    
    print(f"\n✓ Physical interpretation:")
    print(f"  - Max XY movement per step: {env.action_scale_pos:.3f} m = {env.action_scale_pos*100:.1f} cm")
    print(f"  - Max Z movement per step: {env.action_scale_pos:.3f} m = {env.action_scale_pos*100:.1f} cm")
    print(f"  - Max rotation per step: {np.degrees(env._action_scale_yaw):.1f} degrees")
    print(f"  - Horizon: 32 timesteps")
    print(f"  - Max reach in 32 steps: {env.action_scale_pos * 32 * 100:.0f} cm")
    
    env.close()


def diagnose_checkpoint(ckpt_path):
    """Check if checkpoint exists and is loadable"""
    print("\n" + "="*70)
    print("3️⃣  CHECKPOINT DIAGNOSIS")
    print("="*70)
    
    if not Path(ckpt_path).exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Loading checkpoint from: {ckpt_path}")
    print(f"  Device: {device}")
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f"✓ Checkpoint loaded successfully")
        
        if 'epoch' in ckpt:
            print(f"  - Epoch: {ckpt['epoch']}")
        
        if 'loss' in ckpt:
            print(f"  - Loss: {ckpt['loss']:.6f}")
        
        if 'config' in ckpt:
            cfg = ckpt['config']
            print(f"  - Horizon: {cfg['horizon']}")
            print(f"  - Obs dim: {cfg['obs_dim']}")
            print(f"  - Act dim: {cfg['act_dim']}")
            print(f"  - N diffusion steps: {cfg.get('n_diffusion_steps', 100)}")
        
        if 'obs_mean' in ckpt and 'act_std' in ckpt:
            obs_mean = ckpt['obs_mean']
            obs_std = ckpt['obs_std']
            act_mean = ckpt['act_mean']
            act_std = ckpt['act_std']
            
            print(f"\n  Normalization stats:")
            print(f"    Obs mean (first 3): {obs_mean[:3]}")
            print(f"    Obs std (first 3): {obs_std[:3]}")
            print(f"    Act mean: {act_mean}")
            print(f"    Act std: {act_std}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("🔍 POLICY DIAGNOSIS TOOL")
    print("="*70)
    
    # 1. Check demos
    demo_info = diagnose_demos()
    
    # 2. Check environment
    diagnose_environment()
    
    # 3. Find latest checkpoint
    print("\n" + "="*70)
    print("4️⃣  FINDING LATEST CHECKPOINT")
    print("="*70)
    
    runs_dir = Path("runs")
    if runs_dir.exists():
        checkpoints = sorted(runs_dir.glob("*/ckpt.pt"), key=lambda x: x.parent.stat().st_mtime, reverse=True)
        if checkpoints:
            latest = checkpoints[0]
            print(f"\n✓ Latest checkpoint: {latest.parent.name}")
            diagnose_checkpoint(str(latest))
        else:
            print("\n❌ No checkpoints found in runs/")
    else:
        print("\n❌ No runs directory found")
    
    # 5. Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    print("""
1. ✅ IF demos look good (high action std):
   → Problem is likely in policy training or evaluation
   → Try: Smaller model (dim=96, n_blocks=2)
   → Train for 100 epochs on GPU (~10 min)

2. ❌ IF demo actions are tiny (std < 0.1):
   → Problem is IN THE DEMOS (bad data collection)
   → Try: Re-collect demos with teleoperation validation

3. ❌ IF normalization stats look wrong:
   → Problem is in data loading
   → Check: Demo file structure, key names

4. 🎯 IF checkpoint loads but evals give 0% success:
   → Problem is EVALUATION (not training!)
   → Try: Simple rollout with deterministic policy
   → Check: Action scaling matches environment

Next step: Run this diagnosis, then contact with results!
    """)


if __name__ == "__main__":
    main()
