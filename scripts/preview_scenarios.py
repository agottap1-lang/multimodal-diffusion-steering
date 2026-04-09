#!/usr/bin/env python3
"""Preview renders for ALL safety and grounding scenarios.

Generates PNG snapshots + short MP4 videos for every scenario the user wants.
No policy needed — just scripted experts showing what the data will look like.

Safety scenarios (6):
  1. Obstacle LEFT  → legible curved path to RIGHT block (obstacle not in way)
  2. Obstacle LEFT  → predictable straight path to LEFT block (obstacle BLOCKS curve)
  3. Obstacle RIGHT → legible curved path to LEFT block (obstacle not in way)
  4. Obstacle RIGHT → predictable straight path to RIGHT block (obstacle BLOCKS curve)
  5. Obstacle CENTER → predictable straight path to LEFT block
  6. Obstacle CENTER → predictable straight path to RIGHT block

Grounding scenario (2):
  7. 5 colored blocks → pass over 3 waypoints → pick target
  8. 5 colored blocks → different waypoint order → pick different target
"""

from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv, ACT_DIM

OUT_DIR = Path("outputs/previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "videos").mkdir(exist_ok=True)

_EE_HOME = np.array([0.40, 0.0, 0.55], dtype=np.float32)
_TABLE_TOP_Z = 0.4

# ═══════════════════════════════════════════════════════════════
# OBSTACLE PLACEMENT
# ═══════════════════════════════════════════════════════════════

def add_obstacle(env, position, radius=0.035, height=0.12,
                 color=[0.9, 0.1, 0.1, 1.0]):
    """Add a visible, physically-present cylinder obstacle."""
    cid = env._cid
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius,
                                 height=height, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius,
                              length=height, rgbaColor=color,
                              physicsClientId=cid)
    uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis,
                            basePosition=position, physicsClientId=cid)
    return uid


def add_colored_blocks(env, positions_colors):
    """Add multiple colored blocks. Returns list of UIDs."""
    cid = env._cid
    uids = []
    for pos, color in positions_colors:
        col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[0.018, 0.018, 0.018],
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.018, 0.018, 0.018],
                                  rgbaColor=color,
                                  physicsClientId=cid)
        uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=pos, physicsClientId=cid)
        uids.append(uid)
    return uids


# ═══════════════════════════════════════════════════════════════
# SCRIPTED TRAJECTORIES
# ═══════════════════════════════════════════════════════════════

def bezier_curved(start, target_above, lateral_mag, n_pts=200):
    """Legible curved Bézier arc — sweeps laterally toward target side."""
    sign = 1.0 if target_above[1] > 0 else -1.0
    cp = np.array([0.35, sign * lateral_mag, 0.60], dtype=np.float32)
    ts = np.linspace(0, 1, n_pts)
    return np.array([(1-t)**2 * start + 2*(1-t)*t * cp + t**2 * target_above
                     for t in ts], dtype=np.float32)


def straight_line(start, target_above, n_pts=200):
    """Predictable straight-line path."""
    ts = np.linspace(0, 1, n_pts)
    return np.array([start + t * (target_above - start)
                     for t in ts], dtype=np.float32)


def waypoint_path(start, waypoints, target_above, n_pts=200):
    """Path passing through a sequence of waypoints before target."""
    all_pts = [start] + waypoints + [target_above]
    # Allocate points proportionally to segment length
    dists = [np.linalg.norm(all_pts[i+1] - all_pts[i])
             for i in range(len(all_pts)-1)]
    total_dist = sum(dists) + 1e-8
    pts_per_seg = [max(10, int(n_pts * d / total_dist))
                   for d in dists]
    path = []
    for i in range(len(all_pts) - 1):
        ts = np.linspace(0, 1, pts_per_seg[i], endpoint=(i == len(all_pts)-2))
        for t in ts:
            path.append(all_pts[i] + t * (all_pts[i+1] - all_pts[i]))
    return np.array(path, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# EXECUTE PATH AND RENDER
# ═══════════════════════════════════════════════════════════════

def execute_path(env, path_wps, video_path=None, max_steps=400):
    """Follow a waypoint path using position control."""
    if video_path:
        env.record_video(str(video_path), width=640, height=480, fps=30)

    obs = env._get_obs()
    wp_idx = 0

    for step in range(max_steps):
        ee_pos = obs[:3]
        tgt = path_wps[min(wp_idx, len(path_wps)-1)]
        delta = (tgt - ee_pos) / env.action_scale_pos
        action = np.zeros(ACT_DIM, dtype=np.float32)
        action[:3] = np.clip(delta * 0.45, -1, 1)

        # Close gripper + descend when near final waypoint
        if wp_idx >= len(path_wps) - 10:
            action[4] = -1.0  # close gripper
        else:
            action[4] = 1.0   # open

        result = env.step(action)
        obs = result.obs

        if np.linalg.norm(ee_pos - tgt) < 0.012 and wp_idx < len(path_wps) - 1:
            wp_idx += 1

        if result.done:
            break

    if video_path:
        env.stop_video()

    return result.info.get("success_left", 0) > 0.5 or \
           result.info.get("success_right", 0) > 0.5


def snapshot(env, path, label=""):
    """Save a PNG snapshot of current env state."""
    rgb = env.render(width=800, height=600)
    img = Image.fromarray(rgb)
    img.save(str(path))
    print(f"  Snapshot: {path.name}  {label}")


# ═══════════════════════════════════════════════════════════════
# SAFETY SCENARIO PREVIEWS
# ═══════════════════════════════════════════════════════════════

def preview_safety_scenarios():
    print("\n" + "=" * 60)
    print("  SAFETY SCENARIOS — Preview Renders")
    print("=" * 60)

    # Block positions
    left_cube  = np.array([0.50,  0.07, _TABLE_TOP_Z + 0.021])
    right_cube = np.array([0.50, -0.07, _TABLE_TOP_Z + 0.021])

    # Obstacle positions (on table surface, tall enough to block lateral sweeps)
    obstacle_positions = {
        "left":   [0.47,  0.04, _TABLE_TOP_Z + 0.06],  # between center and left block
        "right":  [0.47, -0.04, _TABLE_TOP_Z + 0.06],  # between center and right block
        "center": [0.45,  0.00, _TABLE_TOP_Z + 0.06],  # dead center
    }

    obstacle_colors = {
        "left":   [0.9, 0.1, 0.1, 1.0],  # red
        "right":  [0.9, 0.5, 0.0, 1.0],  # orange
        "center": [0.7, 0.0, 0.7, 1.0],  # purple
    }

    scenarios = [
        # (obs_loc, target, path_type, description)
        ("left",   "right", "legible",     "Obs LEFT → legible curve to RIGHT (safe)"),
        ("left",   "left",  "predictable", "Obs LEFT → straight to LEFT (avoid obstacle)"),
        ("right",  "left",  "legible",     "Obs RIGHT → legible curve to LEFT (safe)"),
        ("right",  "right", "predictable", "Obs RIGHT → straight to RIGHT (avoid obstacle)"),
        ("center", "left",  "predictable", "Obs CENTER → straight to LEFT (avoid obstacle)"),
        ("center", "right", "predictable", "Obs CENTER → straight to RIGHT (avoid obstacle)"),
    ]

    for i, (obs_loc, target, path_type, desc) in enumerate(scenarios):
        print(f"\n  Scenario {i+1}: {desc}")

        env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
        env.reset(seed=42)

        # Add obstacle
        obs_pos = obstacle_positions[obs_loc]
        obs_color = obstacle_colors[obs_loc]
        add_obstacle(env, obs_pos, radius=0.035, height=0.12, color=obs_color)

        # Settle physics
        for _ in range(60):
            p.stepSimulation(physicsClientId=env._cid)

        # Take initial snapshot (shows scene layout)
        snap_path = OUT_DIR / f"safety_{i+1:02d}_{obs_loc}_obs_{target}_{path_type}.png"
        snapshot(env, snap_path, desc)

        # Build path
        target_pos = left_cube if target == "left" else right_cube
        above = target_pos.copy()
        above[2] += 0.10

        if path_type == "legible":
            # Descent path: arc → above → descend → grip → lift
            arc = bezier_curved(_EE_HOME, above, lateral_mag=0.15)
        else:
            arc = straight_line(_EE_HOME, above)

        # Add descent to grasp position
        grasp_pos = target_pos.copy()
        grasp_pos[2] += 0.005
        descent = np.array([above + t * (grasp_pos - above)
                            for t in np.linspace(0, 1, 30)], dtype=np.float32)
        full_path = np.concatenate([arc, descent])

        # Execute and record video
        vid_path = OUT_DIR / "videos" / f"safety_{i+1:02d}_{obs_loc}_{target}_{path_type}.mp4"
        success = execute_path(env, full_path, video_path=str(vid_path))
        print(f"    Video: {vid_path.name}  success={success}")

        env.close()


# ═══════════════════════════════════════════════════════════════
# GROUNDING SCENARIO PREVIEWS
# ═══════════════════════════════════════════════════════════════

def preview_grounding_scenarios():
    print("\n" + "=" * 60)
    print("  GROUNDING SCENARIOS — Preview Renders (5 colored blocks)")
    print("=" * 60)

    # 5 colored blocks spread across table (in addition to the 2 pick cubes)
    # These are WAYPOINT markers the robot must pass over
    waypoint_blocks = [
        ([0.42,  0.10, _TABLE_TOP_Z + 0.02], [0.0, 0.0, 1.0, 1.0], "BLUE"),
        ([0.42, -0.10, _TABLE_TOP_Z + 0.02], [0.0, 0.8, 0.0, 1.0], "GREEN"),
        ([0.48,  0.00, _TABLE_TOP_Z + 0.02], [1.0, 1.0, 0.0, 1.0], "YELLOW"),
        ([0.55,  0.06, _TABLE_TOP_Z + 0.02], [1.0, 0.5, 0.0, 1.0], "ORANGE"),
        ([0.55, -0.06, _TABLE_TOP_Z + 0.02], [0.5, 0.0, 0.5, 1.0], "PURPLE"),
    ]

    # Make pick cubes distinct colors
    pick_colors = {
        "left":  [1.0, 0.2, 0.2, 1.0],   # bright red
        "right": [0.2, 1.0, 1.0, 1.0],    # cyan
    }

    left_cube  = np.array([0.50,  0.07, _TABLE_TOP_Z + 0.021])
    right_cube = np.array([0.50, -0.07, _TABLE_TOP_Z + 0.021])

    # Grounding scenarios: pass over specific waypoints before picking
    scenarios = [
        # (waypoint_indices, target, description)
        ([0, 2, 3], "left",  "Pass BLUE→YELLOW→ORANGE → pick LEFT (red)"),
        ([1, 2, 4], "right", "Pass GREEN→YELLOW→PURPLE → pick RIGHT (cyan)"),
    ]

    for i, (wp_idxs, target, desc) in enumerate(scenarios):
        print(f"\n  Scenario {i+1}: {desc}")

        env = TwoBlockPickEnv(render=False, episode_length=500, cube_jitter=0.0)
        env.reset(seed=42)

        # Color pick cubes
        p.changeVisualShape(env._cube_l_uid, -1,
                            rgbaColor=pick_colors["left"],
                            physicsClientId=env._cid)
        p.changeVisualShape(env._cube_r_uid, -1,
                            rgbaColor=pick_colors["right"],
                            physicsClientId=env._cid)

        # Add 5 waypoint blocks
        wp_positions_colors = [(np.array(wb[0]), wb[1])
                               for wb in waypoint_blocks]
        add_colored_blocks(env, wp_positions_colors)

        # Settle
        for _ in range(60):
            p.stepSimulation(physicsClientId=env._cid)

        # Snapshot of scene
        snap_path = OUT_DIR / f"grounding_{i+1:02d}_scene.png"
        snapshot(env, snap_path, desc)

        # Build waypoint path
        target_pos = left_cube if target == "left" else right_cube
        above_target = target_pos.copy()
        above_target[2] += 0.10

        # Hover height above waypoint blocks
        wp_hover = []
        for wi in wp_idxs:
            wp_pos = np.array(waypoint_blocks[wi][0])
            hover = wp_pos.copy()
            hover[2] = 0.50  # hover above
            wp_hover.append(hover)

        path = waypoint_path(_EE_HOME, wp_hover, above_target)

        # Add descent
        grasp_pos = target_pos.copy()
        grasp_pos[2] += 0.005
        descent = np.array([above_target + t * (grasp_pos - above_target)
                            for t in np.linspace(0, 1, 30)], dtype=np.float32)
        full_path = np.concatenate([path, descent])

        vid_path = OUT_DIR / "videos" / f"grounding_{i+1:02d}_{target}.mp4"
        success = execute_path(env, full_path, video_path=str(vid_path),
                               max_steps=500)
        print(f"    Video: {vid_path.name}  success={success}")

        env.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    preview_safety_scenarios()
    preview_grounding_scenarios()

    print(f"\n{'='*60}")
    print(f"  All previews saved to: {OUT_DIR}")
    print(f"  Snapshots: {OUT_DIR}/*.png")
    print(f"  Videos:    {OUT_DIR}/videos/*.mp4")
    print(f"{'='*60}")
