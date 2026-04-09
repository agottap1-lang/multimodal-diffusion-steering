#!/usr/bin/env python3
"""Collect unified demos for Conditional Diffusion Policy with CFG.

Extended observation (27-d):
  base obs (22) + obstacle_pos (3) + obstacle_flag (1) + behavior_mode (1)
  behavior_mode: +1 = legible (curved), -1 = predictable (straight), 0 = neutral

Demo categories:
  A. Legible (200): curved Bézier arcs, no obstacle, mode=+1
  B. Predictable (100): straight-line paths, no obstacle, mode=-1
  C. Safety-predictable (100): obstacle present, straight paths that avoid it, mode=-1
  D. Grounding-waypoint (100): 3 waypoint blocks, robot passes near them, w/ obstacle_pos
     encoding waypoint position, mode=0

Total: 500 demos, obs_dim=27, act_dim=5

The extended obs lets the policy KNOW where obstacles are and what behavior is requested.
CFG training will randomly zero-out behavior_mode with prob 0.15.

Usage:
  py scripts/collect_demos_cfg.py --preview        # 4 demo videos
  py scripts/collect_demos_cfg.py --seed 0          # full 500 demos
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM
import pybullet as p

# ── constants ────────────────────────────────────────────────────────
_EE_HOME = np.array([0.40, 0.0, 0.55], dtype=np.float32)
_TABLE_TOP_Z = 0.4

EXTENDED_OBS_DIM = 27  # 22 base + 3 obstacle_pos + 1 obstacle_flag + 1 behavior_mode

_N_ARC_PTS     = 200
_N_DESCENT_PTS = 30
_N_GRIP_STEPS  = 40
_N_LIFT_PTS    = 30
_EPISODE_LEN   = 400

_CUBE_HALF     = 0.015
_CUBE_MASS     = 0.08
_CUBE_FRICTION = 2.5


# ═══════════════════════════════════════════════════════════════
# BLOCK CONFIGS (10 configs, same as before)
# ═══════════════════════════════════════════════════════════════

def _build_block_configs() -> list[dict]:
    configs = []
    for dx in [-0.005, 0.0, 0.005]:
        configs.append(dict(ldx=dx, ldy=0.0, rdx=dx, rdy=0.0))
    configs.append(dict(ldx=0.0, ldy=0.004, rdx=0.0, rdy=-0.004))
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(ldx=dx, ldy=dy, rdx=0.0, rdy=0.0))
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(ldx=0.0, ldy=0.0, rdx=dx, rdy=dy))
    assert len(configs) == 10
    return configs


# ═══════════════════════════════════════════════════════════════
# OBSTACLE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

OBSTACLE_CONFIGS = [
    # (position, radius, height, color, name)
    ([0.47,  0.04, _TABLE_TOP_Z + 0.06], 0.035, 0.12, [0.9, 0.1, 0.1, 1.0], "left"),
    ([0.47, -0.04, _TABLE_TOP_Z + 0.06], 0.035, 0.12, [0.9, 0.5, 0.0, 1.0], "right"),
    ([0.45,  0.00, _TABLE_TOP_Z + 0.06], 0.035, 0.12, [0.7, 0.0, 0.7, 1.0], "center"),
    ([0.46,  0.02, _TABLE_TOP_Z + 0.06], 0.030, 0.10, [0.1, 0.5, 0.9, 1.0], "center_left"),
    ([0.46, -0.02, _TABLE_TOP_Z + 0.06], 0.030, 0.10, [0.0, 0.7, 0.0, 1.0], "center_right"),
]


# ═══════════════════════════════════════════════════════════════
# WAYPOINT BLOCK CONFIGS (for grounding)
# ═══════════════════════════════════════════════════════════════

WAYPOINT_BLOCKS = [
    ([0.42,  0.10, _TABLE_TOP_Z + 0.02], [0.0, 0.0, 1.0, 1.0], "BLUE"),
    ([0.42, -0.10, _TABLE_TOP_Z + 0.02], [0.0, 0.8, 0.0, 1.0], "GREEN"),
    ([0.48,  0.00, _TABLE_TOP_Z + 0.02], [1.0, 1.0, 0.0, 1.0], "YELLOW"),
    ([0.55,  0.06, _TABLE_TOP_Z + 0.02], [1.0, 0.5, 0.0, 1.0], "ORANGE"),
    ([0.55, -0.06, _TABLE_TOP_Z + 0.02], [0.5, 0.0, 0.5, 1.0], "PURPLE"),
]


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT HELPERS
# ═══════════════════════════════════════════════════════════════

def add_obstacle(env, pos, radius, height, color):
    cid = env._cid
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius,
                                 height=height, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius,
                              length=height, rgbaColor=color,
                              physicsClientId=cid)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos, physicsClientId=cid)


def add_waypoint_block(env, pos, color):
    cid = env._cid
    col = p.createCollisionShape(p.GEOM_BOX,
                                 halfExtents=[0.018, 0.018, 0.018],
                                 physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_BOX,
                              halfExtents=[0.018, 0.018, 0.018],
                              rgbaColor=color, physicsClientId=cid)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos, physicsClientId=cid)


def get_extended_obs(env, obstacle_pos=None, behavior_mode=0.0):
    """Build 27-d observation: base(22) + obstacle(3) + flag(1) + mode(1)"""
    base_obs = env._get_obs()  # 22-d
    if obstacle_pos is not None:
        obs_ext = np.array(obstacle_pos, dtype=np.float32)[:3]
        obs_flag = np.array([1.0], dtype=np.float32)
    else:
        obs_ext = np.zeros(3, dtype=np.float32)
        obs_flag = np.array([0.0], dtype=np.float32)
    mode = np.array([float(behavior_mode)], dtype=np.float32)
    return np.concatenate([base_obs, obs_ext, obs_flag, mode])


# ═══════════════════════════════════════════════════════════════
# SCRIPTED EXPERTS
# ═══════════════════════════════════════════════════════════════

class _ExpertBase:
    def __init__(self, target, action_scale=0.05):
        assert target in ("left", "right")
        self.target = target
        self.scale = action_scale
        self.phase = 0
        self.wait = 0
        self._arc_wps = None
        self._descent_wps = None
        self._lift_wps = None
        self._wp_idx = 0

    def reset(self):
        self.phase = 0
        self.wait = 0
        self._arc_wps = self._descent_wps = self._lift_wps = None
        self._wp_idx = 0

    def _cube_pos(self, obs):
        return obs[8:11].copy() if self.target == "left" else obs[15:18].copy()

    def _build_descent_and_lift(self, above, cube_pos):
        grasp = cube_pos.copy()
        grasp[2] += 0.005
        self._descent_wps = np.array([
            above + s * (grasp - above)
            for s in np.linspace(0, 1, _N_DESCENT_PTS)
        ], dtype=np.float32)
        lift_top = grasp.copy()
        lift_top[2] = 0.60
        self._lift_wps = np.array([
            grasp + s * (lift_top - grasp)
            for s in np.linspace(0, 1, _N_LIFT_PTS)
        ], dtype=np.float32)

    def _build_waypoints(self, cube_pos):
        raise NotImplementedError

    def act(self, obs):
        ee_pos = obs[0:3]
        cube_pos = self._cube_pos(obs)
        if self._arc_wps is None:
            self._build_waypoints(cube_pos)
        action = np.zeros(ACT_DIM, dtype=np.float32)

        if self.phase == 0:
            tgt = self._arc_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=1.0, speed=0.45)
            if np.linalg.norm(ee_pos - tgt) < 0.012:
                self._wp_idx += 1
                if self._wp_idx >= len(self._arc_wps):
                    self.phase = 1
                    self._wp_idx = 1
        elif self.phase == 1:
            tgt = self._descent_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=1.0, speed=0.35)
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_DESCENT_PTS:
                    self.phase = 2
                    self.wait = 0
        elif self.phase == 2:
            t_frac = self.wait / max(_N_GRIP_STEPS - 1, 1)
            action[4] = 1.0 - 2.0 * t_frac
            self.wait += 1
            if self.wait >= _N_GRIP_STEPS:
                self.phase = 3
                self._wp_idx = 1
        elif self.phase == 3:
            tgt = self._lift_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=-1.0, speed=0.35)
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_LIFT_PTS:
                    self.phase = 4
        else:
            action[4] = -1.0
        return action

    def _goto(self, cur, tgt, grip, speed=1.0):
        delta = (tgt - cur) / self.scale
        a = np.zeros(ACT_DIM, dtype=np.float32)
        a[:3] = np.clip(delta * speed, -1, 1)
        a[4] = grip
        return a


class LegibleExpert(_ExpertBase):
    """Curved Bézier arc that swings toward the target side (legible)."""
    def __init__(self, target, action_scale, cp_y_mag, cp_z=0.60, cp_x=0.35):
        super().__init__(target, action_scale)
        self.cp_y_mag = cp_y_mag
        self.cp_z = cp_z
        self.cp_x = cp_x

    def _build_waypoints(self, cube_pos):
        sign = 1.0 if self.target == "left" else -1.0
        P0 = _EE_HOME.copy()
        P1 = np.array([self.cp_x, sign * self.cp_y_mag, self.cp_z],
                       dtype=np.float32)
        above = cube_pos.copy()
        above[2] += 0.08
        P2 = above
        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array([
            (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2
            for t in ts], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


class PredictableExpert(_ExpertBase):
    """Straight-line path to target (predictable / efficient)."""
    def __init__(self, target, action_scale, approach_h=0.08):
        super().__init__(target, action_scale)
        self.approach_h = approach_h

    def _build_waypoints(self, cube_pos):
        above = cube_pos.copy()
        above[2] += self.approach_h
        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array([_EE_HOME + t * (above - _EE_HOME)
                                  for t in ts], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


class SafetyExpert(_ExpertBase):
    """Straight-line that goes HIGH to clear obstacle, then descends."""
    def __init__(self, target, action_scale, obstacle_pos, approach_h=0.10):
        super().__init__(target, action_scale)
        self.obstacle_pos = np.array(obstacle_pos, dtype=np.float32)
        self.approach_h = approach_h

    def _build_waypoints(self, cube_pos):
        # Go UP first (clear obstacle), then OVER, then descend
        high_pt = _EE_HOME.copy()
        high_pt[2] = max(0.65, self.obstacle_pos[2] + 0.20)
        # Then move to above the target at high altitude
        above = cube_pos.copy()
        above[2] += self.approach_h
        above_high = above.copy()
        above_high[2] = high_pt[2]
        # Straight from home → up → across at altitude → down to above cube
        n1 = _N_ARC_PTS // 3
        n2 = _N_ARC_PTS // 3
        n3 = _N_ARC_PTS - n1 - n2
        seg1 = np.array([_EE_HOME + t * (high_pt - _EE_HOME)
                          for t in np.linspace(0, 1, n1)], dtype=np.float32)
        seg2 = np.array([high_pt + t * (above_high - high_pt)
                          for t in np.linspace(0, 1, n2)], dtype=np.float32)
        seg3 = np.array([above_high + t * (above - above_high)
                          for t in np.linspace(0, 1, n3)], dtype=np.float32)
        self._arc_wps = np.concatenate([seg1, seg2, seg3])
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


class GroundingExpert(_ExpertBase):
    """Pass through waypoint positions before picking target."""
    def __init__(self, target, action_scale, waypoint_positions,
                 approach_h=0.08):
        super().__init__(target, action_scale)
        self.waypoints = [np.array(wp, dtype=np.float32)
                          for wp in waypoint_positions]
        self.approach_h = approach_h

    def _build_waypoints(self, cube_pos):
        # Build path: home → hover over each waypoint → above target
        above = cube_pos.copy()
        above[2] += self.approach_h
        all_points = [_EE_HOME.copy()]
        for wp in self.waypoints:
            hover = wp.copy()
            hover[2] = 0.50  # hover altitude above waypoint
            all_points.append(hover)
        all_points.append(above)

        # Interpolate between consecutive points
        pts_list = []
        n_segs = len(all_points) - 1
        pts_per_seg = max(10, _N_ARC_PTS // n_segs)
        for i in range(n_segs):
            n = pts_per_seg if i < n_segs - 1 else (_N_ARC_PTS - len(pts_list))
            n = max(5, n)
            for t in np.linspace(0, 1, n, endpoint=(i == n_segs - 1)):
                pts_list.append(
                    all_points[i] + t * (all_points[i+1] - all_points[i]))
        self._arc_wps = np.array(pts_list, dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


# ═══════════════════════════════════════════════════════════════
# ARC VARIATIONS
# ═══════════════════════════════════════════════════════════════

def _legible_variations(n=10):
    """10 variations of curved arcs (cp_y_mag 0.05→0.25)."""
    return [dict(cp_y_mag=float(np.interp(i/(n-1), [0,1], [0.05, 0.25])),
                 cp_z=float(np.interp(i/(n-1), [0,1], [0.56, 0.66])),
                 cp_x=float(np.interp(i/(n-1), [0,1], [0.38, 0.30])))
            for i in range(n)]


def _predictable_variations(n=5):
    """5 variations of straight paths (approach height 0.06→0.14)."""
    return [dict(approach_h=float(np.interp(i/(n-1), [0,1], [0.06, 0.14])))
            for i in range(n)]


def _grounding_waypoint_seqs():
    """Waypoint sequences for grounding demos."""
    return [
        [0, 2, 3],  # BLUE → YELLOW → ORANGE → pick left/right
        [1, 2, 4],  # GREEN → YELLOW → PURPLE → pick left/right
        [0, 1, 2],  # BLUE → GREEN → YELLOW → pick
        [3, 2, 1],  # ORANGE → YELLOW → GREEN → pick
        [4, 2, 0],  # PURPLE → YELLOW → BLUE → pick
    ]


# ═══════════════════════════════════════════════════════════════
# RUN ONE EPISODE
# ═══════════════════════════════════════════════════════════════

def _run_episode(env, expert, obstacle_pos, behavior_mode,
                 video_path=None):
    """Run one episode, recording extended 27-d observations."""
    max_T = env.episode_length
    obs_base = env._get_obs()

    if video_path:
        env.record_video(str(video_path))

    obs_buf = np.zeros((max_T, EXTENDED_OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((max_T, ACT_DIM), dtype=np.float32)
    ep_len = 0
    result = None

    for t in range(max_T):
        ext_obs = get_extended_obs(env, obstacle_pos, behavior_mode)
        action = expert.act(obs_base)
        obs_buf[t] = ext_obs
        act_buf[t] = action

        result = env.step(action)
        obs_base = result.obs
        ep_len = t + 1
        if result.done:
            break

    if video_path:
        env.stop_video()

    success = (result.info["success_left"] > 0.5 or
               result.info["success_right"] > 0.5) if result else False
    return obs_buf, act_buf, ep_len, success


# ═══════════════════════════════════════════════════════════════
# MAIN COLLECTION
# ═══════════════════════════════════════════════════════════════

def collect(seed=0, out_path="data/demos/demos_cfg.npz", preview=False):
    block_configs = _build_block_configs()
    leg_vars = _legible_variations(10)
    pred_vars = _predictable_variations(5)
    gnd_seqs = _grounding_waypoint_seqs()

    vid_dir = Path(out_path).parent / "demo_videos_cfg"
    vid_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print("=" * 62)
    print("  Unified CFG Demo Collection (obs_dim=27)")
    print("=" * 62)

    if preview:
        _run_preview(block_configs, leg_vars, pred_vars, gnd_seqs, vid_dir)
        return

    # ── Build full schedule ────────────────────────────────────
    schedule = []

    # A. Legible demos (200): 10 configs × 10 arc vars × 2 sides = 200
    for cfg_id, cfg in enumerate(block_configs):
        for vi, var in enumerate(leg_vars):
            for target in ["left", "right"]:
                schedule.append(dict(
                    category="legible", cfg_id=cfg_id, cfg=cfg,
                    target=target, var=var, var_idx=vi,
                    obs_cfg=None, behavior_mode=1.0))

    # B. Predictable demos (100): 10 configs × 5 vars × 2 sides = 100
    for cfg_id, cfg in enumerate(block_configs):
        for vi, var in enumerate(pred_vars):
            for target in ["left", "right"]:
                schedule.append(dict(
                    category="predictable", cfg_id=cfg_id, cfg=cfg,
                    target=target, var=var, var_idx=vi,
                    obs_cfg=None, behavior_mode=-1.0))

    # C. Safety-predictable (100): 5 obstacle configs × 10 configs × 2 sides / pruned
    # We do: 5 obstacle configs × 10 block configs × 2 sides = 100
    # (pick 1 pred_var per episode randomly)
    safety_items = []
    for oi, obs_cfg in enumerate(OBSTACLE_CONFIGS):
        for cfg_id, cfg in enumerate(block_configs):
            for target in ["left", "right"]:
                safety_items.append(dict(
                    category="safety", cfg_id=cfg_id, cfg=cfg,
                    target=target, var=pred_vars[oi % len(pred_vars)],
                    var_idx=oi, obs_cfg=obs_cfg, behavior_mode=-1.0))
    rng.shuffle(safety_items)
    schedule.extend(safety_items[:100])

    # D. Grounding (100): 5 wp_seqs × 10 configs × 2 sides = 100
    gnd_items = []
    for si, seq in enumerate(gnd_seqs):
        for cfg_id, cfg in enumerate(block_configs):
            for target in ["left", "right"]:
                gnd_items.append(dict(
                    category="grounding", cfg_id=cfg_id, cfg=cfg,
                    target=target, var=dict(seq=seq), var_idx=si,
                    obs_cfg=None, behavior_mode=0.0,
                    wp_seq=seq))
    rng.shuffle(gnd_items)
    schedule.extend(gnd_items[:100])

    total = len(schedule)
    print(f"  Total demos: {total}")
    cats = {}
    for item in schedule:
        cats[item["category"]] = cats.get(item["category"], 0) + 1
    for cat, cnt in sorted(cats.items()):
        print(f"    {cat}: {cnt}")
    print()

    # ── Collect ────────────────────────────────────────────────
    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          cube_half=_CUBE_HALF, cube_mass=_CUBE_MASS,
                          cube_lateral_friction=_CUBE_FRICTION,
                          episode_length=_EPISODE_LEN)

    all_obs = []
    all_act = []
    all_lens = []
    all_labels = []
    all_categories = []
    all_modes = []
    successes = 0
    retries = 0

    pbar = tqdm(total=total, desc="collecting demos")

    for item in schedule:
        cat = item["category"]
        cfg = item["cfg"]
        target = item["target"]
        bmode = item["behavior_mode"]
        obs_cfg = item.get("obs_cfg")

        for attempt in range(5):
            # Reset env
            env.reset(seed=0)
            env.set_cube_offsets(left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                                right_dx=cfg["rdx"], right_dy=cfg["rdy"])

            obstacle_pos = None
            wp_uids = []

            # Add obstacle for safety demos
            if cat == "safety" and obs_cfg is not None:
                obstacle_pos = obs_cfg[0]
                add_obstacle(env, obs_cfg[0], obs_cfg[1], obs_cfg[2], obs_cfg[3])

            # Add waypoint blocks for grounding demos
            if cat == "grounding":
                seq = item.get("wp_seq", [0, 2, 3])
                # Add ALL 5 waypoint blocks visually
                for wb in WAYPOINT_BLOCKS:
                    add_waypoint_block(env, wb[0], wb[1])
                # Use first waypoint in sequence as "obstacle_pos" in obs
                # (repurposed: tells policy where the first waypoint is)
                obstacle_pos = WAYPOINT_BLOCKS[seq[0]][0]

            # Color pick cubes for grounding
            if cat == "grounding":
                p.changeVisualShape(env._cube_l_uid, -1,
                                    rgbaColor=[1.0, 0.2, 0.2, 1.0],
                                    physicsClientId=env._cid)
                p.changeVisualShape(env._cube_r_uid, -1,
                                    rgbaColor=[0.2, 1.0, 1.0, 1.0],
                                    physicsClientId=env._cid)

            # Settle physics
            for _ in range(60):
                p.stepSimulation(physicsClientId=env._cid)

            # Create expert
            var = item["var"]
            if cat == "legible":
                expert = LegibleExpert(target, env.action_scale_pos,
                                       var["cp_y_mag"], var["cp_z"], var["cp_x"])
            elif cat == "predictable":
                expert = PredictableExpert(target, env.action_scale_pos,
                                           var["approach_h"])
            elif cat == "safety":
                expert = SafetyExpert(target, env.action_scale_pos,
                                      obstacle_pos)
            elif cat == "grounding":
                seq = item.get("wp_seq", [0, 2, 3])
                wp_positions = [WAYPOINT_BLOCKS[i][0] for i in seq]
                expert = GroundingExpert(target, env.action_scale_pos,
                                         wp_positions)
            expert.reset()

            # Video for first few of each category
            vpath = None
            if len([x for x in all_categories if x == cat]) < 2:
                vname = f"{cat}_{target}_{item['var_idx']:02d}.mp4"
                vpath = vid_dir / vname

            obs_buf, act_buf, ep_len, success = _run_episode(
                env, expert, obstacle_pos, bmode, video_path=vpath)

            if success:
                break
            retries += 1
            if vpath and vpath.exists():
                vpath.unlink()

        all_obs.append(obs_buf)
        all_act.append(act_buf)
        all_lens.append(ep_len)
        all_labels.append(target)
        all_categories.append(cat)
        all_modes.append(bmode)
        if success:
            successes += 1
        pbar.update(1)

    pbar.close()
    env.close()

    # ── Save ──────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "description": "CFG demos for conditional diffusion policy",
        "obs_dim": EXTENDED_OBS_DIM,
        "act_dim": ACT_DIM,
        "n_demos": total,
        "categories": {cat: sum(1 for c in all_categories if c == cat)
                       for cat in set(all_categories)},
        "success_rate": successes / total,
        "retries": retries,
    }

    np.savez_compressed(
        str(out),
        obs=np.stack(all_obs),
        actions=np.stack(all_act),
        episode_lengths=np.array(all_lens),
        labels=np.array(all_labels),
        categories=np.array(all_categories),
        behavior_modes=np.array(all_modes),
        metadata_json=json.dumps(metadata),
    )

    print(f"\n{'='*62}")
    print(f"  Saved {total} demos to: {out}")
    print(f"  obs_dim={EXTENDED_OBS_DIM}, act_dim={ACT_DIM}")
    for cat, cnt in metadata["categories"].items():
        print(f"    {cat}: {cnt}")
    print(f"  success: {successes}/{total} ({100*successes/total:.0f}%)")
    print(f"  retries: {retries}")
    print(f"  videos: {vid_dir}/")
    print(f"{'='*62}")


def _run_preview(block_configs, leg_vars, pred_vars, gnd_seqs, vid_dir):
    """Run 4 preview demos (1 per category)."""
    print("  PREVIEW MODE: 4 videos\n")
    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          cube_half=_CUBE_HALF, cube_mass=_CUBE_MASS,
                          cube_lateral_friction=_CUBE_FRICTION,
                          episode_length=_EPISODE_LEN)
    cfg = block_configs[1]

    preview_cases = [
        ("legible", "left", dict(category="legible", var=leg_vars[5],
                                  obs_cfg=None, bmode=1.0, seq=None)),
        ("predictable", "right", dict(category="predictable",
                                       var=pred_vars[2], obs_cfg=None,
                                       bmode=-1.0, seq=None)),
        ("safety", "left", dict(category="safety", var=pred_vars[2],
                                 obs_cfg=OBSTACLE_CONFIGS[0], bmode=-1.0,
                                 seq=None)),
        ("grounding", "left", dict(category="grounding", var=None,
                                    obs_cfg=None, bmode=0.0,
                                    seq=[0, 2, 3])),
    ]

    for cat, target, info in preview_cases:
        env.reset(seed=0)
        env.set_cube_offsets(left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                            right_dx=cfg["rdx"], right_dy=cfg["rdy"])

        obstacle_pos = None
        if info["obs_cfg"] is not None:
            obs_c = info["obs_cfg"]
            obstacle_pos = obs_c[0]
            add_obstacle(env, obs_c[0], obs_c[1], obs_c[2], obs_c[3])

        if cat == "grounding":
            for wb in WAYPOINT_BLOCKS:
                add_waypoint_block(env, wb[0], wb[1])
            p.changeVisualShape(env._cube_l_uid, -1,
                                rgbaColor=[1.0, 0.2, 0.2, 1.0],
                                physicsClientId=env._cid)
            p.changeVisualShape(env._cube_r_uid, -1,
                                rgbaColor=[0.2, 1.0, 1.0, 1.0],
                                physicsClientId=env._cid)
            obstacle_pos = WAYPOINT_BLOCKS[info["seq"][0]][0]

        for _ in range(60):
            p.stepSimulation(physicsClientId=env._cid)

        var = info["var"]
        if cat == "legible":
            expert = LegibleExpert(target, env.action_scale_pos,
                                   var["cp_y_mag"], var["cp_z"], var["cp_x"])
        elif cat == "predictable":
            expert = PredictableExpert(target, env.action_scale_pos,
                                       var["approach_h"])
        elif cat == "safety":
            expert = SafetyExpert(target, env.action_scale_pos, obstacle_pos)
        elif cat == "grounding":
            wp_positions = [WAYPOINT_BLOCKS[i][0] for i in info["seq"]]
            expert = GroundingExpert(target, env.action_scale_pos, wp_positions)
        expert.reset()

        vpath = vid_dir / f"preview_{cat}_{target}.mp4"
        _, _, ep_len, success = _run_episode(
            env, expert, obstacle_pos, info["bmode"], video_path=vpath)
        print(f"  {cat:15s} {target:5s}  success={success}  "
              f"steps={ep_len}  → {vpath.name}")

    env.close()
    print(f"\n  Preview done: {vid_dir}/")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("CFG Demo Collection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/demos/demos_cfg.npz")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    collect(seed=args.seed, out_path=args.out, preview=args.preview)


if __name__ == "__main__":
    main()
