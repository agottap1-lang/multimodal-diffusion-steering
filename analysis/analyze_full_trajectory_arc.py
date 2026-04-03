"""
Measure arc from FULL executed trajectory (with replanning), not just first action sequence.
This matches what's actually shown in the videos.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler


def measure_arc_from_ee_trajectory(ee_positions: np.ndarray) -> float:
    """
    Measure arc as max perpendicular distance from start-end line.
    This matches Bézier control point curvature concept.
    
    Args:
        ee_positions: (N, 3) array of EE positions in world space
    """
    if len(ee_positions) < 3:
        return 0.0
    
    # Use XY plane only (lateral motion)
    start = ee_positions[0, :2]
    end = ee_positions[-1, :2]
    
    # Vector from start to end
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-6:
        return 0.0
    
    line_unit = line_vec / line_len
    
    # Calculate perpendicular distance for each point
    max_perp = 0.0
    for pos in ee_positions:
        point = pos[:2]
        # Vector from start to point
        point_vec = point - start
        # Project onto line direction
        proj_len = np.dot(point_vec, line_unit)
        proj_point = start + proj_len * line_unit
        # Perpendicular distance
        perp_dist = np.linalg.norm(point - proj_point)
        max_perp = max(max_perp, perp_dist)
    
    return float(max_perp)


def execute_full_trajectory(seed: int, temperature: float, checkpoint_path: str, n_steps: int = 150):
    """
    Execute full trajectory with replanning and track EE positions.
    """
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
    
    # Execute full trajectory with replanning
    env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
    obs = env.reset(seed=seed)
    
    ee_positions = []
    action_queue = []
    
    for step in range(n_steps):
        # Extract EE position from observation (first 3 elements)
        ee_pos = obs[:3].copy()
        ee_positions.append(ee_pos)
        
        # Replan when queue empty
        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=temperature)[0].cpu().numpy()
            
            actions = seq * act_std + act_mean
            actions[:, -1] = np.clip(actions[:, -1], -1, 1)  # Stabilize gripper
            action_queue = actions.tolist()
        
        action = action_queue.pop(0)
        result = env.step(action)
        obs = result.obs
        
        if result.done:
            break
    
    env.close()
    
    ee_positions = np.array(ee_positions)
    return ee_positions


def analyze_full_trajectory(seed: int, temperature: float, checkpoint_path: str):
    """Analyze full executed trajectory"""
    
    print(f"\nAnalyzing FULL trajectory (seed={seed}, temp={temperature:.2f})")
    print(f"{'='*70}")
    
    # Execute and get EE positions
    ee_positions = execute_full_trajectory(seed, temperature, checkpoint_path, n_steps=150)
    
    # Measure arc
    arc = measure_arc_from_ee_trajectory(ee_positions)
    
    print(f"Total steps: {len(ee_positions)}")
    print(f"Start EE: ({ee_positions[0, 0]:.3f}, {ee_positions[0, 1]:.3f}, {ee_positions[0, 2]:.3f})")
    print(f"End EE:   ({ee_positions[-1, 0]:.3f}, {ee_positions[-1, 1]:.3f}, {ee_positions[-1, 2]:.3f})")
    print(f"\n🎯 Arc (max perpendicular distance): {arc:.4f}m")
    
    # Determine arc class based on perpendicular distance
    if arc < 0.05:
        arc_class = "00-05 (straight/gentle)"
    elif arc < 0.15:
        arc_class = "10-14 (moderate)"
    else:
        arc_class = "15-19 (large sweep)"
    
    print(f"Arc classification: {arc_class}")
    
    # Plot trajectory
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # XY plane (top view)
    axes[0].plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, alpha=0.7)
    axes[0].plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=12, label='Start', zorder=5)
    axes[0].plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ro', markersize=12, label='End', zorder=5)
    
    # Draw start-end line
    axes[0].plot([ee_positions[0, 0], ee_positions[-1, 0]], 
                 [ee_positions[0, 1], ee_positions[-1, 1]], 
                 'k--', alpha=0.3, linewidth=1, label='Direct line')
    
    axes[0].set_xlabel('X position (m)', fontsize=12)
    axes[0].set_ylabel('Y position (m)', fontsize=12)
    axes[0].set_title(f'Top View (XY plane)\nArc = {arc:.4f}m ({arc_class})', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].axis('equal')
    
    # Y over time
    axes[1].plot(ee_positions[:, 1], 'b-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Timestep', fontsize=12)
    axes[1].set_ylabel('Y position (m)', fontsize=12)
    axes[1].set_title('Lateral Position Over Time', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    # 3D view
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = fig.add_subplot(133, projection='3d')
    ax3d.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax3d.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], 
                 c='green', s=100, label='Start', zorder=5)
    ax3d.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2], 
                 c='red', s=100, label='End', zorder=5)
    ax3d.set_xlabel('X (m)', fontsize=10)
    ax3d.set_ylabel('Y (m)', fontsize=10)
    ax3d.set_zlabel('Z (m)', fontsize=10)
    ax3d.set_title('3D Trajectory', fontsize=13)
    ax3d.legend(fontsize=9)
    
    plt.tight_layout()
    
    output_dir = Path("runs/arc_measurement_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"full_trajectory_seed{seed}_temp{temperature:.2f}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {fig_path}")
    plt.close()
    
    return arc, arc_class


if __name__ == "__main__":
    checkpoint = "runs/diffusion_20260222_195530/ckpt_ep100.pt"
    
    print("="*70)
    print("ANALYZING FULL TRAJECTORIES (with replanning)")
    print("="*70)
    print("\nMeasuring arc as max perpendicular distance from start-end line")
    print("(This matches the Bézier control point curvature concept)")
    
    # Analyze the trajectories that were in the videos
    arc1, class1 = analyze_full_trajectory(1001, 0.81, checkpoint)
    arc2, class2 = analyze_full_trajectory(1002, 1.12, checkpoint)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Seed 1001 (temp=0.81): arc = {arc1:.4f}m ({class1})")
    print(f"Seed 1002 (temp=1.12): arc = {arc2:.4f}m ({class2})")
    print("\nDemo arc 15-19: Bézier cp_y_mag = 0.23-0.28m")
    print("If these arcs < 0.23m, policy is NOT producing arc 15-19 style!")
