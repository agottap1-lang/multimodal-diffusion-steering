import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'runs/latest/ckpt.pt'
print(f"Loading checkpoint: {ckpt_path}")

try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    print("\n" + "="*70)
    print("CHECKPOINT DIAGNOSTIC")
    print("="*70)
    
    print(f"\nEpoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Has EMA weights: {'ema' in ckpt}")
    print(f"Has model weights: {'model' in ckpt}")
    
    cfg = ckpt.get('config', {})
    print(f"\nConfig:")
    print(f"  Horizon: {cfg.get('horizon', 'N/A')}")
    print(f"  n_action_steps: {cfg.get('n_action_steps', 'N/A')}")
    print(f"  Demo path: {cfg.get('demo_path', 'N/A')}")
    print(f"  n_diffusion_steps: {cfg.get('n_diffusion_steps', 'N/A')}")
    
    # Check normalization stats
    if 'obs_mean' in ckpt:
        import numpy as np
        obs_mean = np.array(ckpt['obs_mean'])
        obs_std = np.array(ckpt['obs_std'])
        act_mean = np.array(ckpt['act_mean'])
        act_std = np.array(ckpt['act_std'])
        
        print(f"\nNormalization stats:")
        print(f"  obs_mean: min={obs_mean.min():.3f}, max={obs_mean.max():.3f}")
        print(f"  obs_std: min={obs_std.min():.3f}, max={obs_std.max():.3f}")
        print(f"  act_mean: min={act_mean.min():.3f}, max={act_mean.max():.3f}")
        print(f"  act_std: min={act_std.min():.3f}, max={act_std.max():.3f}")
    
    print("\n" + "="*70)
    print("CHECKPOINT LOOKS VALID")
    print("="*70)
    
except Exception as e:
    print(f"\nERROR loading checkpoint: {e}")
    sys.exit(1)
