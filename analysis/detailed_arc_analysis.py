"""
Detailed analysis: Compare cumsum vs perpendicular measurements
and check what cumsum actually captures.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler


def execute_with_full_tracking(seed: int, temperature: float, checkpoint_path: str):
    """Execute trajectory with full action and position tracking"""
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
    gripper_values = []
    action_queue = []
    
    for step in range(200):
        ee_pos = obs[:3].copy()
        gripper = obs[7]
        ee_positions.append(ee_pos)
        gripper_values.append(gripper)
        
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
    gripper_values = np.array(gripper_values)
    
    # Calculate cumsum of Y actions
    dy_cumsum = np.cumsum(actions_executed[:, 1])
    cumsum_arc = float(np.max(np.abs(dy_cumsum)))
    
    # Calculate perpendicular distance from full EE trajectory
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
        perp_distances = np.array(perp_distances)
        perp_arc = np.max(perp_distances)
        max_perp_idx = np.argmax(perp_distances)
    else:
        perp_distances = np.zeros(len(ee_positions))
        perp_arc = 0.0
        max_perp_idx = 0
    
    # Find when gripper closes
    gripper_close_idx = len(gripper_values)
    for i in range(len(gripper_values)):
        if gripper_values[i] < 0.5:
            gripper_close_idx = i
            break
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS (seed={seed}, temp={temperature:.2f})")
    print(f"{'='*80}")
    
    print(f"\n📊 MEASUREMENTS:")
    print(f"   Cumsum arc (max |∑dy|):           {cumsum_arc:.4f}m")
    print(f"   Cumsum at step {np.argmax(np.abs(dy_cumsum))}")
    print(f"   ")
    print(f"   Perpendicular arc (max ⊥ dist):   {perp_arc:.4f}m")
    print(f"   Perpendicular peak at step {max_perp_idx}")
    print(f"   ")
    print(f"   Gripper closes at step:           {gripper_close_idx}")
    
    print(f"\n📍 KEY POSITIONS:")
    print(f"   Start EE: ({ee_positions[0, 0]:.3f}, {ee_positions[0, 1]:.3f}, {ee_positions[0, 2]:.3f})")
    print(f"   End EE:   ({ee_positions[-1, 0]:.3f}, {ee_positions[-1, 1]:.3f}, {ee_positions[-1, 2]:.3f})")
    print(f"   Final Y displacement: {ee_positions[-1, 1] - ee_positions[0, 1]:.4f}m")
    print(f"   Peak perpendicular at: ({ee_positions[max_perp_idx, 0]:.3f}, {ee_positions[max_perp_idx, 1]:.3f}, {ee_positions[max_perp_idx, 2]:.3f})")
    
    # Check cumsum before gripper close
    dy_cumsum_before_grip = dy_cumsum[:gripper_close_idx]
    if len(dy_cumsum_before_grip) > 0:
        cumsum_arc_approach = float(np.max(np.abs(dy_cumsum_before_grip)))
        print(f"\n🎯 CUMSUM BEFORE GRIPPER CLOSE:")
        print(f"   Max |∑dy| (steps 0-{gripper_close_idx}): {cumsum_arc_approach:.4f}m")
    
    # Plot detailed analysis
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Top view with perpendicular distances color-coded
    ax1 = plt.subplot(3, 3, 1)
    scatter = ax1.scatter(ee_positions[:, 0], ee_positions[:, 1], 
                         c=perp_distances, cmap='hot', s=30, alpha=0.7)
    ax1.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', alpha=0.3, linewidth=1)
    ax1.plot([start[0], end[0]], [start[1], end[1]], 'k--', linewidth=2, label='Start-End line')
    ax1.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=15, label='Start', zorder=5)
    ax1.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ro', markersize=15, label='End', zorder=5)
    ax1.plot(ee_positions[max_perp_idx, 0], ee_positions[max_perp_idx, 1], 
             'y*', markersize=20, label=f'Max ⊥ ({perp_arc:.3f}m)', zorder=5)
    plt.colorbar(scatter, ax=ax1, label='Perpendicular distance (m)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Top View (colored by ⊥ distance)\nMax ⊥ = {perp_arc:.4f}m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Y position over time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(ee_positions[:, 1], 'b-', linewidth=2, label='Actual Y')
    ax2.axvline(gripper_close_idx, color='r', linestyle='--', linewidth=2, label='Gripper close')
    ax2.axvline(max_perp_idx, color='y', linestyle='--', linewidth=2, label='Max ⊥')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Y position (m)')
    ax2.set_title(f'Y Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumsum of actions
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(dy_cumsum, 'g-', linewidth=2, label='∑dy')
    ax3.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax3.axvline(gripper_close_idx, color='r', linestyle='--', linewidth=2, label='Gripper close')
    max_cumsum_idx = np.argmax(np.abs(dy_cumsum))
    ax3.plot(max_cumsum_idx, dy_cumsum[max_cumsum_idx], 'r*', markersize=20, 
             label=f'Max |∑dy| = {cumsum_arc:.3f}m')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Cumsum dy (m)')
    ax3.set_title(f'Cumulative Y Action\nMax |∑dy| = {cumsum_arc:.4f}m')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Perpendicular distance over time
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(perp_distances, 'm-', linewidth=2, label='⊥ distance')
    ax4.axvline(gripper_close_idx, color='r', linestyle='--', linewidth=2, label='Gripper close')
    ax4.plot(max_perp_idx, perp_arc, 'y*', markersize=20, label=f'Max = {perp_arc:.3f}m')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Perpendicular distance (m)')
    ax4.set_title(f'Perpendicular Distance Over Time\nMax = {perp_arc:.4f}m')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Z position
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(ee_positions[:, 2], 'c-', linewidth=2)
    ax5.axvline(gripper_close_idx, color='r', linestyle='--', linewidth=2, label='Gripper close')
    ax5.axvline(max_perp_idx, color='y', linestyle='--', linewidth=2, label='Max ⊥')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Z position (m)')
    ax5.set_title('Height Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Gripper
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(gripper_values, 'orange', linewidth=2)
    ax6.axhline(0.5, color='k', linestyle='--', alpha=0.3)
    ax6.axvline(gripper_close_idx, color='r', linestyle='--', linewidth=2, label='Close threshold')
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Gripper openness')
    ax6.set_title('Gripper State')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 3D trajectory
    ax7 = plt.subplot(3, 3, 7, projection='3d')
    ax7.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=2, alpha=0.6)
    ax7.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], 
                c='green', s=100, label='Start', zorder=5)
    ax7.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
                c='red', s=100, label='End', zorder=5)
    ax7.scatter(ee_positions[max_perp_idx, 0], ee_positions[max_perp_idx, 1], ee_positions[max_perp_idx, 2],
                c='yellow', s=150, marker='*', label=f'Max ⊥', zorder=5)
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_zlabel('Z (m)')
    ax7.set_title('3D Trajectory')
    ax7.legend()
    
    # 8. Action deltas dx, dy, dz
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(actions_executed[:, 0], label='dx', alpha=0.7)
    ax8.plot(actions_executed[:, 1], label='dy', alpha=0.7, linewidth=2)
    ax8.plot(actions_executed[:, 2], label='dz', alpha=0.7)
    ax8.axvline(gripper_close_idx, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax8.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax8.set_xlabel('Timestep')
    ax8.set_ylabel('Action delta')
    ax8.set_title('Action Deltas (dx, dy, dz)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Comparison: Cumsum vs Perpendicular
    ax9 = plt.subplot(3, 3, 9)
    ax9_twin = ax9.twinx()
    ax9.plot(np.abs(dy_cumsum), 'g-', linewidth=2, label='|∑dy| (cumsum)', alpha=0.7)
    ax9_twin.plot(perp_distances, 'm-', linewidth=2, label='⊥ distance', alpha=0.7)
    ax9.axvline(gripper_close_idx, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax9.set_xlabel('Timestep')
    ax9.set_ylabel('|∑dy| (m)', color='g')
    ax9_twin.set_ylabel('⊥ distance (m)', color='m')
    ax9.tick_params(axis='y', labelcolor='g')
    ax9_twin.tick_params(axis='y', labelcolor='m')
    ax9.set_title(f'Cumsum vs Perpendicular\nCumsum={cumsum_arc:.4f}m, ⊥={perp_arc:.4f}m')
    ax9.grid(True, alpha=0.3)
    lines1, labels1 = ax9.get_legend_handles_labels()
    lines2, labels2 = ax9_twin.get_legend_handles_labels()
    ax9.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    output_dir = Path("runs/arc_measurement_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"detailed_analysis_seed{seed}_temp{temperature:.2f}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Detailed plot saved: {fig_path}")
    plt.close()
    
    return cumsum_arc, perp_arc


if __name__ == "__main__":
    checkpoint = "runs/diffusion_20260222_195530/ckpt_ep100.pt"
    
    print("="*80)
    print("DETAILED ARC MEASUREMENT COMPARISON")
    print("="*80)
    print("\nComparing cumsum(dy) vs perpendicular distance measurements")
    print("Understanding why cumsum shows 0.79m but video looks different")
    
    c1, p1 = execute_with_full_tracking(1001, 0.81, checkpoint)
    c2, p2 = execute_with_full_tracking(1002, 1.12, checkpoint)
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\nSeed 1001 (temp=0.81):")
    print(f"  Cumsum arc:        {c1:.4f}m")
    print(f"  Perpendicular arc: {p1:.4f}m")
    print(f"\nSeed 1002 (temp=1.12):")
    print(f"  Cumsum arc:        {c2:.4f}m")
    print(f"  Perpendicular arc: {p2:.4f}m")
    print(f"\nDemo arc 15-19: 0.23-0.28m (Bézier control point)")
    print("\nKey insight: Cumsum measures cumulative displacement,")
    print("             Perpendicular measures actual curve height from straight line")
