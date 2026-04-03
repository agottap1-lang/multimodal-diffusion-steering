"""
Analyze why arc measurement doesn't match visual appearance.
Compare different arc measurement methods.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler


def measure_arc_cumsum(actions: np.ndarray) -> float:
    """Current method: cumulative Y displacement"""
    dy_cumsum = np.cumsum(actions[:, 1])
    return float(np.max(np.abs(dy_cumsum)))


def measure_arc_max_perpendicular(actions: np.ndarray) -> float:
    """Alternative: max perpendicular distance from start-end line"""
    # Reconstruct trajectory positions
    positions = np.cumsum(actions[:, :3], axis=0)
    positions = np.vstack([np.zeros(3), positions])  # Add starting position
    
    # Start and end points
    start = positions[0, :2]  # XY only
    end = positions[-1, :2]
    
    # Vector from start to end
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-6:
        return 0.0
    
    line_unit = line_vec / line_len
    
    # Calculate perpendicular distance for each point
    max_perp = 0.0
    for pos in positions:
        point = pos[:2]
        # Project point onto line
        proj_len = np.dot(point - start, line_unit)
        proj_point = start + proj_len * line_unit
        # Perpendicular distance
        perp_dist = np.linalg.norm(point - proj_point)
        max_perp = max(max_perp, perp_dist)
    
    return float(max_perp)


def measure_arc_max_lateral(actions: np.ndarray) -> float:
    """Alternative: max lateral (Y) position reached"""
    # Reconstruct Y positions
    y_positions = np.cumsum(actions[:, 1])
    return float(np.max(np.abs(y_positions)))


def analyze_trajectory(seed: int, temperature: float, checkpoint_path: str):
    """Generate trajectory and compare different arc measurements"""
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
    
    # Generate trajectory
    env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
    obs = env.reset(seed=seed)
    
    obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=temperature)[0].cpu().numpy()
    
    actions = seq * act_std + act_mean
    
    # Calculate different arc measurements
    arc_cumsum = measure_arc_cumsum(actions)
    arc_perp = measure_arc_max_perpendicular(actions)
    arc_lateral = measure_arc_max_lateral(actions)
    
    # Reconstruct trajectory for visualization
    positions = np.cumsum(actions[:, :3], axis=0)
    positions = np.vstack([np.zeros(3), positions])
    
    print(f"\nTrajectory Analysis (seed={seed}, temp={temperature:.2f})")
    print(f"{'='*60}")
    print(f"Arc (cumsum method):          {arc_cumsum:.4f}m")
    print(f"Arc (max perpendicular):      {arc_perp:.4f}m")
    print(f"Arc (max lateral position):   {arc_lateral:.4f}m")
    print(f"Start position: ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f})")
    print(f"End position:   ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})")
    print(f"Net displacement: ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})")
    
    # Plot trajectory
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # XY plane
    axes[0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
    axes[0].plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    axes[0].plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    axes[0].set_xlabel('X position (m)')
    axes[0].set_ylabel('Y position (m)')
    axes[0].set_title(f'Top View\nArc(cumsum)={arc_cumsum:.3f}m, Arc(perp)={arc_perp:.3f}m')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis('equal')
    
    # Y over time
    axes[1].plot(positions[:, 1], 'b-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Y position (m)')
    axes[1].set_title(f'Lateral Position Over Time\nMax |Y|={arc_lateral:.3f}m')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("runs/arc_measurement_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"trajectory_seed{seed}_temp{temperature:.2f}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {fig_path}")
    plt.close()
    
    env.close()
    
    return {
        'arc_cumsum': arc_cumsum,
        'arc_perp': arc_perp,
        'arc_lateral': arc_lateral,
        'positions': positions
    }


if __name__ == "__main__":
    checkpoint = "runs/diffusion_20260222_195530/ckpt_ep100.pt"
    
    print("="*60)
    print("ANALYZING ARC MEASUREMENT METHODS")
    print("="*60)
    
    # Analyze the two generated trajectories
    print("\n### Analyzing policy_arc0.9835m (seed=1001, temp=0.81)")
    result1 = analyze_trajectory(1001, 0.81, checkpoint)
    
    print("\n### Analyzing policy_arc0.7945m (seed=1002, temp=1.12)")
    result2 = analyze_trajectory(1002, 1.12, checkpoint)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("\nIf arc_cumsum is very different from arc_perp or arc_lateral,")
    print("then cumsum method is NOT measuring true arc curvature.")
    print("\nThe correct measurement should match visual arc appearance")
    print("in the demo videos (arc 15-19 = large lateral sweeps).")
