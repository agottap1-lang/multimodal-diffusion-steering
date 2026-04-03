"""
Find the correct metric that distinguishes:
- Seed 1002 (cumsum=0.79m) = arc 15-19 ✓
- Seed 1001 (cumsum=0.98m) = arc 1-5 ✗

Cumsum is clearly wrong. Need to find what metric matches visual observation.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler


def analyze_trajectory_deeply(seed: int, temperature: float, checkpoint_path: str, expected_class: str):
    """Execute and analyze trajectory with multiple metrics"""
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
    
    for step in range(200):
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
    
    # Calculate various metrics
    dy_cumsum = np.cumsum(actions_executed[:, 1])
    cumsum_metric = float(np.max(np.abs(dy_cumsum)))
    
    # Max lateral deviation from start Y
    max_y_deviation = float(np.max(np.abs(ee_positions[:, 1] - ee_positions[0, 1])))
    
    # Perpendicular distance from start-end line
    start = ee_positions[0, :2]
    end = ee_positions[-1, :2]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len > 1e-6:
        line_unit = line_vec / line_len
        perp_distances = []
        for pos in ee_positions:
            point = pos[:2]
            point_vec = point - start
            proj_len = np.dot(point_vec, line_unit)
            proj_point = start + proj_len * line_unit
            perp_dist = np.linalg.norm(point - proj_point)
            perp_distances.append(perp_dist)
        max_perp = max(perp_distances)
        perp_at_25pct = perp_distances[len(perp_distances)//4] if len(perp_distances) > 4 else 0
        avg_perp_first_half = np.mean(perp_distances[:len(perp_distances)//2])
    else:
        max_perp = 0.0
        perp_at_25pct = 0.0
        avg_perp_first_half = 0.0
    
    # Y variance and smoothness
    y_positions = ee_positions[:, 1]
    y_variance = float(np.var(y_positions))
    y_changes = np.diff(y_positions)
    y_direction_changes = np.sum(np.diff(np.sign(y_changes)) != 0)
    
    # Arc "smoothness" - large arc should have smooth curve, oscillation has many direction changes
    smoothness = 1.0 / (1.0 + y_direction_changes / len(y_positions))
    
    # Peak Y in first 50 steps
    early_max_y = float(np.max(np.abs(y_positions[:min(50, len(y_positions))])))
    
    print(f"\n{'='*80}")
    print(f"TRAJECTORY ANALYSIS: seed={seed}, temp={temperature:.2f}")
    print(f"EXPECTED CLASS: {expected_class}")
    print(f"{'='*80}")
    
    print(f"\n📊 METRICS:")
    print(f"   Cumsum (current):           {cumsum_metric:.4f}m")
    print(f"   Max Y deviation:            {max_y_deviation:.4f}m")
    print(f"   Max perpendicular:          {max_perp:.4f}m")
    print(f"   Perpendicular @ 25%:        {perp_at_25pct:.4f}m")
    print(f"   Avg perp (first half):      {avg_perp_first_half:.4f}m")
    print(f"   Y variance:                 {y_variance:.6f}")
    print(f"   Y direction changes:        {y_direction_changes}")
    print(f"   Smoothness (1/(1+changes)): {smoothness:.4f}")
    print(f"   Early max Y (first 50):     {early_max_y:.4f}m")
    
    # Visualize
    fig = plt.figure(figsize=(18, 10))
    
    # Top view
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax1.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=15, label='Start')
    ax1.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ro', markersize=15, label='End')
    ax1.plot([start[0], end[0]], [start[1], end[1]], 'k--', alpha=0.3, linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Top View\nExpected: {expected_class}', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Y over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(y_positions, 'b-', linewidth=2)
    ax2.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Y position (m)')
    ax2.set_title(f'Y Position\nMax dev={max_y_deviation:.4f}m')
    ax2.grid(True, alpha=0.3)
    
    # Cumsum
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(dy_cumsum, 'g-', linewidth=2)
    ax3.plot(np.abs(dy_cumsum), 'm--', linewidth=2, alpha=0.7, label='|cumsum|')
    ax3.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Cumsum dy (m)')
    ax3.set_title(f'Cumsum\nMax={cumsum_metric:.4f}m')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Perpendicular distance
    ax4 = plt.subplot(2, 3, 4)
    if line_len > 1e-6:
        ax4.plot(perp_distances, 'm-', linewidth=2)
        ax4.axhline(max_perp, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Perpendicular distance (m)')
    ax4.set_title(f'Perpendicular\nMax={max_perp:.4f}m')
    ax4.grid(True, alpha=0.3)
    
    # Y velocity (direction changes indicate oscillation)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(y_changes, 'c-', linewidth=1, alpha=0.7)
    ax5.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Y change (m/step)')
    ax5.set_title(f'Y Velocity\nDirection changes={y_direction_changes}\nSmoothness={smoothness:.3f}')
    ax5.grid(True, alpha=0.3)
    
    # 3D view
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    ax6.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax6.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], c='green', s=100)
    ax6.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2], c='red', s=100)
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_zlabel('Z (m)')
    ax6.set_title('3D Trajectory')
    
    plt.suptitle(f'Seed {seed} (temp={temperature:.2f}) - Expected: {expected_class}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("runs/arc_measurement_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"deep_analysis_seed{seed}_expected_{expected_class}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Plot saved: {fig_path}")
    plt.close()
    
    return {
        'cumsum': cumsum_metric,
        'max_y_dev': max_y_deviation,
        'max_perp': max_perp,
        'perp_at_25pct': perp_at_25pct,
        'avg_perp_first_half': avg_perp_first_half,
        'y_variance': y_variance,
        'direction_changes': y_direction_changes,
        'smoothness': smoothness,
        'early_max_y': early_max_y,
    }


if __name__ == "__main__":
    checkpoint = "runs/diffusion_20260222_195530/ckpt_ep100.pt"
    
    print("="*80)
    print("FINDING CORRECT ARC METRIC")
    print("="*80)
    print("\nUser says:")
    print("  Seed 1002 (cumsum=0.79m) → arc 15-19 ✓ LARGE SWEEP")
    print("  Seed 1001 (cumsum=0.98m) → arc 1-5  ✗ GENTLE/STRAIGHT")
    print("\nCumsum is BROKEN. Finding what metric correctly identifies arcs.")
    
    metrics_1002 = analyze_trajectory_deeply(1002, 1.12, checkpoint, "arc 15-19")
    metrics_1001 = analyze_trajectory_deeply(1001, 0.81, checkpoint, "arc 1-5")
    
    print("\n" + "="*80)
    print("METRIC COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} | {'Seed 1002 (arc 15-19)':<20} | {'Seed 1001 (arc 1-5)':<20}")
    print("-" * 80)
    
    for key in metrics_1002.keys():
        val1002 = metrics_1002[key]
        val1001 = metrics_1001[key]
        if isinstance(val1002, float):
            print(f"{key:<30} | {val1002:>18.4f}   | {val1001:>18.4f}")
        else:
            print(f"{key:<30} | {val1002:>18}   | {val1001:>18}")
    
    print("\n" + "="*80)
    print("FINDINGS")
    print("="*80)
    print("\nLook for metric where:")
    print("  - Seed 1002 value is HIGH (matches arc 15-19)")
    print("  - Seed 1001 value is LOW (matches arc 1-5)")
    print("\nThis will be the CORRECT metric to replace cumsum.")
