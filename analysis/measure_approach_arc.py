"""
Measure arc during APPROACH phase only (home → above cube).
This matches how demos define arc (Bézier sweep during approach).
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler


def detect_approach_phase_end(ee_positions: np.ndarray, gripper_values: np.ndarray) -> int:
    """
    Detect when approach phase ends (gripper starts closing or descending to cube).
    
    Heuristics:
    1. Gripper starts closing (value drops below 0.5)
    2. Z-height starts descending significantly after rising
    3. Lateral motion stabilizes (reaching above cube)
    """
    # Method 1: Gripper closing
    gripper_close_idx = None
    for i in range(len(gripper_values)):
        if gripper_values[i] < 0.5:  # Gripper closing
            gripper_close_idx = i
            break
    
    # Method 2: Z descent after peak
    z_positions = ee_positions[:, 2]
    if len(z_positions) > 10:
        z_peak_idx = np.argmax(z_positions[:len(z_positions)//2])  # Peak in first half
        # Find when Z drops significantly after peak
        z_descent_idx = None
        z_peak = z_positions[z_peak_idx]
        for i in range(z_peak_idx + 5, len(z_positions)):
            if z_positions[i] < z_peak - 0.05:  # Dropped 5cm
                z_descent_idx = i
                break
    else:
        z_descent_idx = None
    
    # Use earliest indicator
    candidates = [idx for idx in [gripper_close_idx, z_descent_idx] if idx is not None]
    if candidates:
        return min(candidates)
    else:
        return len(ee_positions) // 2  # Default: first half


def measure_arc_from_ee_trajectory(ee_positions: np.ndarray) -> float:
    """Measure arc as max perpendicular distance from start-end line"""
    if len(ee_positions) < 3:
        return 0.0
    
    start = ee_positions[0, :2]
    end = ee_positions[-1, :2]
    
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-6:
        return 0.0
    
    line_unit = line_vec / line_len
    
    max_perp = 0.0
    max_perp_idx = 0
    for i, pos in enumerate(ee_positions):
        point = pos[:2]
        point_vec = point - start
        proj_len = np.dot(point_vec, line_unit)
        proj_point = start + proj_len * line_unit
        perp_dist = np.linalg.norm(point - proj_point)
        if perp_dist > max_perp:
            max_perp = perp_dist
            max_perp_idx = i
    
    return float(max_perp), max_perp_idx


def execute_and_analyze_approach(seed: int, temperature: float, checkpoint_path: str):
    """Execute trajectory and analyze approach arc"""
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
    gripper_values = []
    action_queue = []
    
    for step in range(200):
        ee_pos = obs[:3].copy()
        gripper = obs[7]  # Gripper at index 7
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
        result = env.step(action)
        obs = result.obs
        
        if result.done:
            break
    
    env.close()
    
    ee_positions = np.array(ee_positions)
    gripper_values = np.array(gripper_values)
    
    # Detect approach phase end
    approach_end_idx = detect_approach_phase_end(ee_positions, gripper_values)
    
    # Extract approach phase
    approach_ee = ee_positions[:approach_end_idx]
    
    # Measure full trajectory arc and approach-only arc
    full_arc, full_max_idx = measure_arc_from_ee_trajectory(ee_positions)
    approach_arc, approach_max_idx = measure_arc_from_ee_trajectory(approach_ee)
    
    print(f"\n{'='*75}")
    print(f"TRAJECTORY ANALYSIS (seed={seed}, temp={temperature:.2f})")
    print(f"{'='*75}")
    print(f"\n📊 Full trajectory (0 to {len(ee_positions)} steps):")
    print(f"   Arc: {full_arc:.4f}m")
    print(f"   Start: ({ee_positions[0, 0]:.3f}, {ee_positions[0, 1]:.3f}, {ee_positions[0, 2]:.3f})")
    print(f"   End:   ({ee_positions[-1, 0]:.3f}, {ee_positions[-1, 1]:.3f}, {ee_positions[-1, 2]:.3f})")
    
    print(f"\n🎯 APPROACH PHASE ONLY (0 to {approach_end_idx} steps):")
    print(f"   Arc: {approach_arc:.4f}m")
    print(f"   Start: ({approach_ee[0, 0]:.3f}, {approach_ee[0, 1]:.3f}, {approach_ee[0, 2]:.3f})")
    print(f"   End:   ({approach_ee[-1, 0]:.3f}, {approach_ee[-1, 1]:.3f}, {approach_ee[-1, 2]:.3f})")
    print(f"   Peak arc at step {approach_max_idx}")
    
    # Classification
    if approach_arc < 0.05:
        arc_class = "00-05 (straight/gentle)"
    elif approach_arc < 0.15:
        arc_class = "10-14 (moderate)"
    elif approach_arc < 0.23:
        arc_class = "15-19 (approaching target)"
    else:
        arc_class = "15-19 (LARGE SWEEP ✓)"
    
    print(f"\n   Classification: {arc_class}")
    print(f"   Demo arc 15-19: 0.23-0.28m")
    
    # Plot
    fig = plt.figure(figsize=(20, 6))
    
    # 1. Full trajectory top view
    ax1 = plt.subplot(1, 4, 1)
    ax1.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, alpha=0.5, label='Full')
    ax1.plot(approach_ee[:, 0], approach_ee[:, 1], 'r-', linewidth=3, alpha=0.8, label='Approach')
    ax1.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(approach_ee[-1, 0], approach_ee[-1, 1], 'mo', markersize=10, label='Approach End', zorder=5)
    ax1.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ko', markersize=8, label='Final', zorder=5)
    
    # Draw start-end line for approach
    ax1.plot([approach_ee[0, 0], approach_ee[-1, 0]], 
             [approach_ee[0, 1], approach_ee[-1, 1]], 
             'k--', alpha=0.3, linewidth=1, label='Direct')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Top View\nApproach arc = {approach_arc:.4f}m', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axis('equal')
    
    # 2. Y position over time
    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(ee_positions[:, 1], 'b-', linewidth=2, alpha=0.5, label='Full')
    ax2.axvline(x=approach_end_idx, color='r', linestyle='--', linewidth=2, label='Approach end')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Y position (m)', fontsize=12)
    ax2.set_title(f'Lateral Position\nApproach: 0-{approach_end_idx}', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 3. Z position over time
    ax3 = plt.subplot(1, 4, 3)
    ax3.plot(ee_positions[:, 2], 'b-', linewidth=2, alpha=0.5)
    ax3.axvline(x=approach_end_idx, color='r', linestyle='--', linewidth=2, label='Approach end')
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Z position (m)', fontsize=12)
    ax3.set_title('Height Over Time', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 4. Gripper over time
    ax4 = plt.subplot(1, 4, 4)
    ax4.plot(gripper_values, 'g-', linewidth=2, alpha=0.7)
    ax4.axvline(x=approach_end_idx, color='r', linestyle='--', linewidth=2, label='Approach end')
    ax4.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Timestep', fontsize=12)
    ax4.set_ylabel('Gripper openness', fontsize=12)
    ax4.set_title('Gripper State', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    output_dir = Path("runs/arc_measurement_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"approach_arc_seed{seed}_temp{temperature:.2f}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Plot saved: {fig_path}")
    plt.close()
    
    return approach_arc, arc_class


if __name__ == "__main__":
    checkpoint = "runs/diffusion_20260222_195530/ckpt_ep100.pt"
    
    print("="*75)
    print("MEASURING ARC DURING APPROACH PHASE ONLY")
    print("="*75)
    print("\nApproach phase: Home → Above cube (the sweeping arc motion)")
    print("This matches how demos define arc (Bézier control point)")
    
    arc1, class1 = execute_and_analyze_approach(1001, 0.81, checkpoint)
    arc2, class2 = execute_and_analyze_approach(1002, 1.12, checkpoint)
    
    print("\n" + "="*75)
    print("FINAL COMPARISON")
    print("="*75)
    print(f"Seed 1001 (temp=0.81): Approach arc = {arc1:.4f}m → {class1}")
    print(f"Seed 1002 (temp=1.12): Approach arc = {arc2:.4f}m → {class2}")
    print("\nDemo arc 15-19: Bézier cp_y_mag = 0.23-0.28m")
    print("\nIf approach arc ≥ 0.23m → Policy CAN produce arc 15-19 style ✓")
    print("If approach arc < 0.23m → Policy needs steering to reach arc 15-19 ✗")
