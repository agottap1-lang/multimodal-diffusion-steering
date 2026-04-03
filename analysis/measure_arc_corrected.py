"""
Correct arc measurement: Find the sweeping phase and measure arc properly.
If the trajectory makes a large arc and returns, start-to-final-end misses it.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler


def find_sweep_phase_and_measure_arc(ee_positions: np.ndarray):
    """
    Find the sweeping phase and measure arc correctly.
    The arc might be during approach (outward sweep) before returning to cube.
    """
    # Find when lateral position reaches maximum absolute value
    y_positions = ee_positions[:, 1]
    abs_y = np.abs(y_positions)
    max_y_idx = np.argmax(abs_y)
    max_y_val = abs_y[max_y_idx]
    
    # Find phases
    # Phase 1: Start to max lateral deviation (the sweep)
    # Phase 2: Max lateral to end (return/descent)
    
    sweep_phase = ee_positions[:max_y_idx+1]
    
    # Measure arc during sweep phase
    if len(sweep_phase) >= 3:
        start = sweep_phase[0, :2]
        end = sweep_phase[-1, :2]
        
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len > 1e-6:
            line_unit = line_vec / line_len
            perp_distances = []
            for pos in sweep_phase:
                point = pos[:2]
                point_vec = point - start
                proj_len = np.dot(point_vec, line_unit)
                proj_point = start + proj_len * line_unit
                perp_dist = np.linalg.norm(point - proj_point)
                perp_distances.append(perp_dist)
            sweep_arc = max(perp_distances)
        else:
            sweep_arc = 0.0
    else:
        sweep_arc = 0.0
    
    # Also try: measure arc as maximum lateral deviation from home position
    # This is simpler: how far does Y position deviate from starting Y?
    start_y = ee_positions[0, 1]
    max_lateral_deviation = max_y_val
    
    # Also measure: at the max Y point, how much did the trajectory curve?
    # Use cumsum as proxy for "swept distance"
    
    return {
        'max_y_idx': max_y_idx,
        'max_y_val': max_y_val,
        'max_lateral_deviation': max_lateral_deviation,
        'sweep_arc_perpendicular': sweep_arc,
        'sweep_phase_length': len(sweep_phase),
    }


def execute_and_measure_correctly(seed: int, temperature: float, checkpoint_path: str):
    """Execute trajectory and measure arc correctly"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    sampler = DDIMSampler(
        n_steps=cfg.get("n_diffusion_steps", 100),
        beta_start=cfg.get("beta_start", 0.0001),
        beta_end=cfg.get("beta_end", 0.02),
        device=device,
    )
    
    obs_mean = torch.tensor(ckpt["obs_mean"], device=device)
    obs_std = torch.tensor(ckpt["obs_std"], device=device)
    act_mean = ckpt["act_mean"]
    act_std = ckpt["act_std"]
    
    # Execute trajectory
    env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
    obs = env.reset(seed=seed)
    
    ee_positions = []
    actions_executed = []
    action_queue = []
    
    for step in range(250):
        ee_pos = obs[:3].copy()
        ee_positions.append(ee_pos)
        
        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=temperature)[0].cpu().numpy()
            
            actions = seq * act_std + act_mean
            actions[:, -1] = np.clip(actions[:, -1], -1, 1)
            action_queue = actions.tolist()
        
        action = action_queue.pop(0)
        actions_executed.append(action)
        result = env.step(action)
        obs = result.obs
        
        if result.done:
            break
    
    env.close()
    
    ee_positions = np.array(ee_positions)
    actions_executed = np.array(actions_executed)
    
    # Measure arc correctly
    metrics = find_sweep_phase_and_measure_arc(ee_positions)
    
    # Calculate cumsum for comparison
    dy_cumsum = np.cumsum(actions_executed[:, 1])
    cumsum_arc = float(np.max(np.abs(dy_cumsum)))
    
    print(f"\n{'='*80}")
    print(f"CORRECTED ARC MEASUREMENT (seed={seed}, temp={temperature:.2f})")
    print(f"{'='*80}")
    
    print(f"\n📊 TRAJECTORY METRICS:")
    print(f"   Total steps: {len(ee_positions)}")
    print(f"   Start EE: ({ee_positions[0, 0]:.3f}, {ee_positions[0, 1]:.3f}, {ee_positions[0, 2]:.3f})")
    print(f"   End EE:   ({ee_positions[-1, 0]:.3f}, {ee_positions[-1, 1]:.3f}, {ee_positions[-1, 2]:.3f})")
    
    max_y_idx = metrics['max_y_idx']
    print(f"\n🎯 SWEEP ANALYSIS:")
    print(f"   Maximum lateral Y position: {metrics['max_y_val']:.4f}m at step {max_y_idx}")
    print(f"   Position at max Y: ({ee_positions[max_y_idx, 0]:.3f}, {ee_positions[max_y_idx, 1]:.3f}, {ee_positions[max_y_idx, 2]:.3f})")
    print(f"   Sweep phase length: {metrics['sweep_phase_length']} steps")
    print(f"   ")
    print(f"   Arc during sweep (perpendicular): {metrics['sweep_arc_perpendicular']:.4f}m")
    print(f"   Max lateral deviation from home:  {metrics['max_lateral_deviation']:.4f}m")  
    print(f"   Cumsum arc (for comparison):      {cumsum_arc:.4f}m")
    
    # Classification based on maximum lateral deviation
    max_dev = metrics['max_lateral_deviation']
    if max_dev < 0.08:
        arc_class = "00-05 (straight/gentle)"
    elif max_dev < 0.12:
        arc_class = "05-10 (moderate)"
    elif max_dev < 0.18:
        arc_class = "10-15 (pronounced)"
    elif max_dev < 0.24:
        arc_class = "15-19 (approaching large sweep)"
    else:
        arc_class = "15-19 (LARGE SWEEP ✓✓✓)"
    
    print(f"\n🏷️  ARC CLASSIFICATION:")
    print(f"   Based on max lateral deviation: {arc_class}")
    print(f"   ")
    print(f"   Demo arc thresholds (Bézier cp_y_mag):")
    print(f"   Arc 0:  0.05m | Arc 5:  0.11m | Arc 10: 0.17m")
    print(f"   Arc 15: 0.23m | Arc 19: 0.28m")
    print(f"   ")
    if max_dev >= 0.23:
        print(f"   ✓✓✓ This trajectory IS arc 15-19 style!")
    elif max_dev >= 0.18:
        print(f"   ⚠️  Close to arc 15-19, but slightly below threshold")
    else:
        print(f"   ✗ NOT arc 15-19 (needs {0.23 - max_dev:.3f}m more lateral deviation)")
    
    # Create detailed plot
    fig = plt.figure(figsize=(20, 8))
    
    # 1. Top view with sweep phase highlighted
    ax1 = plt.subplot(1, 4, 1)
    ax1.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, alpha=0.4, label='Full trajectory')
    ax1.plot(ee_positions[:max_y_idx+1, 0], ee_positions[:max_y_idx+1, 1], 
             'r-', linewidth=3, alpha=0.8, label='Sweep phase')
    ax1.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=15, label='Start', zorder=5)
    ax1.plot(ee_positions[max_y_idx, 0], ee_positions[max_y_idx, 1], 
             'y*', markersize=25, label=f'Max Y ({metrics["max_y_val"]:.3f}m)', zorder=5)
    ax1.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ko', markersize=10, label='End', zorder=5)
    ax1.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Top View\nMax lateral = {max_dev:.4f}m', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Y position over time
    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(ee_positions[:, 1], 'b-', linewidth=2)
    ax2.axvline(max_y_idx, color='y', linestyle='--', linewidth=2, label='Max Y')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax2.axhline(metrics['max_y_val'], color='r', linestyle='--', alpha=0.5, label=f'Max = {max_dev:.3f}m')
    ax2.axhline(0.23, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Arc 15 threshold (0.23m)')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Y position (m)', fontsize=12)
    ax2.set_title('Lateral Position Over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Absolute Y position
    ax3 = plt.subplot(1, 4, 3)
    ax3.plot(np.abs(ee_positions[:, 1]), 'm-', linewidth=2, label='|Y|')
    ax3.axvline(max_y_idx, color='y', linestyle='--', linewidth=2)
    ax3.axhline(max_dev, color='r', linestyle='--', alpha=0.5, label=f'Max = {max_dev:.3f}m')
    ax3.axhline(0.23, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Arc 15 (0.23m)')
    ax3.axhline(0.28, color='g', linestyle=':', linewidth=2, alpha=0.7, label='Arc 19 (0.28m)')
    ax3.fill_between(range(len(ee_positions)), 0.23, 0.28, alpha=0.2, color='green', label='Arc 15-19 range')
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('|Y| position (m)', fontsize=12)
    ax3.set_title('Absolute Lateral Deviation', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 3D trajectory
    ax4 = plt.subplot(1, 4, 4, projection='3d')
    ax4.plot(ee_positions[:max_y_idx+1, 0], ee_positions[:max_y_idx+1, 1], ee_positions[:max_y_idx+1, 2], 
             'r-', linewidth=3, alpha=0.8, label='Sweep')
    ax4.plot(ee_positions[max_y_idx:, 0], ee_positions[max_y_idx:, 1], ee_positions[max_y_idx:, 2],
             'b-', linewidth=2, alpha=0.4, label='Return/descend')
    ax4.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], 
                c='green', s=100, label='Start', zorder=5)
    ax4.scatter(ee_positions[max_y_idx, 0], ee_positions[max_y_idx, 1], ee_positions[max_y_idx, 2],
                c='yellow', s=200, marker='*', label='Max Y', zorder=5)
    ax4.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
                c='black', s=100, label='End', zorder=5)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Trajectory', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    
    output_dir = Path("runs/arc_measurement_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"corrected_arc_seed{seed}_temp{temperature:.2f}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Plot saved: {fig_path}")
    plt.close()
    
    return max_dev, arc_class


if __name__ == "__main__":
    checkpoint = "runs/diffusion_20260222_195530/ckpt_ep100.pt"
    
    print("="*80)
    print("CORRECTED ARC MEASUREMENT")
    print("="*80)
    print("\nMeasuring arc as MAXIMUM LATERAL DEVIATION during sweep phase")
    print("This matches how you observe arcs visually in videos")
    
    dev1, class1 = execute_and_measure_correctly(1001, 0.81, checkpoint)
    dev2, class2 = execute_and_measure_correctly(1002, 1.12, checkpoint)
    
    print("\n" + "="*80)
    print("FINAL CORRECTED RESULTS")
    print("="*80)
    print(f"\nSeed 1001 (temp=0.81): Max lateral = {dev1:.4f}m → {class1}")
    print(f"Seed 1002 (temp=1.12): Max lateral = {dev2:.4f}m → {class2}")
    print(f"\nDemo arc 15-19: 0.23-0.28m lateral deviation")
    print(f"\nIf max lateral ≥ 0.23m → Trajectory IS arc 15-19 style ✓")
