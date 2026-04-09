#!/usr/bin/env python3
"""
Collect V2 Demos: Extended 26-d observation, 4 behaviors, CFG-ready.
====================================================================

Observation format (26-d):
  [ee_pos(3), ee_quat(4), gripper(1),            # 8  - robot state
   left_cube_pos(3), left_cube_quat(4),           # 7  - left block
   right_cube_pos(3), right_cube_quat(4),         # 7  - right block
   context_x, context_y, context_z,               # 3  - obstacle/waypoint pos
   behavior_mode]                                  # 1  - +1=legible, -1=predict, 0=grounding

~500 demos total:
  200 legible      (mode=+1, context=[0,0,0]) - curved Bezier arcs
  100 predictable  (mode=-1, context=[0,0,0]) - straight shortest paths
  ~100 safety      (mode=+/-1, context=obstacle_pos) - avoid obstacle
      safety-legible:      obstacle on straight path → take curve (mode=+1)
      safety-predictable:  obstacle on curved path  → take straight (mode=-1)
  100 grounding    (mode=0,  context=waypoint_pos) - hover over waypoint then pick

Usage:
  .venv\\Scripts\\python.exe scripts/collect_demos_v2.py --preview
  .venv\\Scripts\\python.exe scripts/collect_demos_v2.py --collect --seed 0
"""

from __future__ import annotations
import argparse, json, math, sys, pickle
from pathlib import Path
import numpy as np
import pybullet as p
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM

# ── Constants ──────────────────────────────────────────────────────────
EE_HOME       = np.array([0.40, 0.0, 0.55], dtype=np.float32)
TABLE_TOP_Z   = 0.4
OBS_DIM_V2    = 26     # 22 base + 3 context + 1 behavior_mode
N_ARC_PTS     = 200
N_DESCENT_PTS = 30
N_GRIP_STEPS  = 40
N_LIFT_PTS    = 30
EPISODE_LEN   = 400
CUBE_HALF     = 0.015
CUBE_MASS     = 0.08
CUBE_FRICTION = 2.5

# Behavior mode values
MODE_LEGIBLE      = +1.0
MODE_PREDICTABLE  = -1.0
MODE_GROUNDING    =  0.0

# ── Obstacle configs for safety demos ─────────────────────────────────
# Cylinder: cyan, height=0.18 (sits on table, top at z=0.58)
OBSTACLE_RADIUS  = 0.035
OBSTACLE_HEIGHT  = 0.18
OBSTACLE_COLOR   = [0.0, 0.85, 0.85, 1]   # bright cyan

def get_safety_obstacle(safety_type, target):
    """Get obstacle config based on safety scenario and target side.

    safety_type="legible":      obstacle on STRAIGHT path → force curved arc
    safety_type="predictable":  obstacle on CURVED path   → force straight line
    """
    cz = TABLE_TOP_Z + OBSTACLE_HEIGHT / 2   # center so it sits on table
    if safety_type == "legible":
        # Place on the direct line from EE_HOME to target block
        # x=0.46 gives good visual separation from the Bezier arc
        y = 0.02 if target == "left" else -0.02
        return {"pos": [0.46, y, cz], "rgba": OBSTACLE_COLOR,
                "radius": OBSTACLE_RADIUS, "height": OBSTACLE_HEIGHT}
    else:  # predictable
        # Place on the Bezier arc side (blocks the curved path)
        y = 0.11 if target == "left" else -0.11
        return {"pos": [0.38, y, cz], "rgba": OBSTACLE_COLOR,
                "radius": OBSTACLE_RADIUS, "height": OBSTACLE_HEIGHT}


# ── 5 waypoint blocks in a pentagon for grounding ─────────────────────
def _build_waypoint_blocks():
    center_x, center_y = 0.43, 0.0
    radius = 0.06
    colors = [
        ("blue",   [0.1, 0.2, 0.95, 1]),
        ("green",  [0.1, 0.85, 0.1, 1]),
        ("yellow", [0.95, 0.9, 0.1, 1]),
        ("orange", [1.0, 0.55, 0.05, 1]),
        ("purple", [0.65, 0.1, 0.85, 1]),
    ]
    blocks = []
    for i, (name, rgba) in enumerate(colors):
        angle = 2 * math.pi * i / 5 + math.pi / 2   # start from top
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        blocks.append({"name": name,
                       "pos": [round(x, 4), round(y, 4), TABLE_TOP_Z + 0.016],
                       "rgba": rgba})
    return blocks

WAYPOINT_BLOCKS = _build_waypoint_blocks()

# Block placement configs (10 configs, small jitter)
def build_block_configs():
    cfgs = []
    for dx in [-0.005, 0.0, 0.005]:
        cfgs.append(dict(ldx=dx, ldy=0.0, rdx=dx, rdy=0.0))
    cfgs.append(dict(ldx=0.0, ldy=0.004, rdx=0.0, rdy=-0.004))
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        cfgs.append(dict(ldx=dx, ldy=dy, rdx=0.0, rdy=0.0))
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        cfgs.append(dict(ldx=0.0, ldy=0.0, rdx=dx, rdy=dy))
    return cfgs  # 10 configs


# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT HELPERS
# ══════════════════════════════════════════════════════════════════════

def add_obstacle(env, obs_cfg):
    """Add a cylinder obstacle (visual-only, no collision).

    The obstacle is visual-only so the expert's pre-computed Bezier path
    doesn't get blocked by physics.  The policy learns avoidance from the
    trajectory data + obstacle position in the observation.
    """
    cid = env._cid
    r = obs_cfg.get("radius", OBSTACLE_RADIUS)
    h = obs_cfg.get("height", OBSTACLE_HEIGHT)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r,
                              length=h,
                              rgbaColor=obs_cfg["rgba"], physicsClientId=cid)
    uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                            baseVisualShapeIndex=vis,
                            basePosition=obs_cfg["pos"], physicsClientId=cid)
    return uid

def add_waypoint_blocks(env, wp_list):
    """Add colored waypoint blocks. Returns list of uids."""
    cid = env._cid
    uids = []
    for wp in wp_list:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                  rgbaColor=wp["rgba"], physicsClientId=cid)
        uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=wp["pos"], physicsClientId=cid)
        uids.append(uid)
    return uids

def remove_bodies(env, uids):
    """Remove pybullet bodies by uid list."""
    for uid in uids:
        p.removeBody(uid, physicsClientId=env._cid)

def get_obs_v2(env, context_pos=None, behavior_mode=0.0):
    """Get the 26-d extended observation."""
    base_obs = env._get_obs()  # 22-d
    ctx = np.zeros(3, dtype=np.float32) if context_pos is None else np.array(context_pos, dtype=np.float32)
    mode = np.array([behavior_mode], dtype=np.float32)
    return np.concatenate([base_obs, ctx, mode])


# ══════════════════════════════════════════════════════════════════════
# EXPERTS
# ══════════════════════════════════════════════════════════════════════

class ExpertBase:
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

    def cube_pos(self, obs_22):
        return obs_22[8:11].copy() if self.target == "left" else obs_22[15:18].copy()

    def _build_descent_and_lift(self, above, cube_pos):
        grasp = cube_pos.copy(); grasp[2] += 0.005
        self._descent_wps = np.array([
            above + s * (grasp - above) for s in np.linspace(0, 1, N_DESCENT_PTS)
        ], dtype=np.float32)
        lift_top = grasp.copy(); lift_top[2] = 0.60
        self._lift_wps = np.array([
            grasp + s * (lift_top - grasp) for s in np.linspace(0, 1, N_LIFT_PTS)
        ], dtype=np.float32)

    def _build_waypoints(self, cube_pos):
        raise NotImplementedError

    def act(self, obs_22):
        ee = obs_22[0:3]
        cube = self.cube_pos(obs_22)
        if self._arc_wps is None:
            self._build_waypoints(cube)
        action = np.zeros(ACT_DIM, dtype=np.float32)
        if self.phase == 0:
            tgt = self._arc_wps[self._wp_idx]
            action = self._goto(ee, tgt, grip=1.0, speed=0.45)
            if np.linalg.norm(ee - tgt) < 0.012:
                self._wp_idx += 1
                if self._wp_idx >= len(self._arc_wps):
                    self.phase = 1; self._wp_idx = 1
        elif self.phase == 1:
            tgt = self._descent_wps[self._wp_idx]
            action = self._goto(ee, tgt, grip=1.0, speed=0.35)
            if np.linalg.norm(ee - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= N_DESCENT_PTS:
                    self.phase = 2; self.wait = 0
        elif self.phase == 2:
            t_frac = self.wait / max(N_GRIP_STEPS - 1, 1)
            action[4] = 1.0 - 2.0 * t_frac
            self.wait += 1
            if self.wait >= N_GRIP_STEPS:
                self.phase = 3; self._wp_idx = 1
        elif self.phase == 3:
            tgt = self._lift_wps[self._wp_idx]
            action = self._goto(ee, tgt, grip=-1.0, speed=0.35)
            if np.linalg.norm(ee - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= N_LIFT_PTS:
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


class LegibleExpert(ExpertBase):
    """Curved Bezier arc sweeping TOWARD the target side (legible intent)."""
    def __init__(self, target, action_scale, cp_y_mag=0.15, cp_z=0.62,
                 cp_x=0.34, approach_h=0.08):
        super().__init__(target, action_scale)
        self.cp_y_mag = cp_y_mag
        self.cp_z = cp_z
        self.cp_x = cp_x
        self.approach_h = approach_h

    def _build_waypoints(self, cube_pos):
        sign = 1.0 if self.target == "left" else -1.0
        P0 = EE_HOME.copy()
        P1 = np.array([self.cp_x, sign * self.cp_y_mag, self.cp_z], dtype=np.float32)
        above = cube_pos.copy(); above[2] += self.approach_h
        ts = np.linspace(0, 1, N_ARC_PTS)
        self._arc_wps = np.array([
            (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * above for t in ts
        ], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


class PredictableExpert(ExpertBase):
    """STRAIGHT shortest path from EE home to above target (predictable)."""
    def __init__(self, target, action_scale, approach_h=0.08):
        super().__init__(target, action_scale)
        self.approach_h = approach_h

    def _build_waypoints(self, cube_pos):
        P0 = EE_HOME.copy()
        above = cube_pos.copy(); above[2] += self.approach_h
        # Straight line with tiny upward bulge so we don't scrape the table
        mid = (P0 + above) / 2
        mid[2] = max(P0[2], above[2]) + 0.02
        ts = np.linspace(0, 1, N_ARC_PTS)
        self._arc_wps = np.array([
            (1-t)**2 * P0 + 2*(1-t)*t * mid + t**2 * above for t in ts
        ], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


# NOTE: No separate SafetyExpert class!
# Safety-legible  reuses LegibleExpert    (obstacle blocks straight path → curve)
# Safety-predict  reuses PredictableExpert (obstacle blocks curved path  → straight)


class GroundingExpert(ExpertBase):
    """Hover over a WAYPOINT block, then proceed to pick the target.

    Explicit waypoint sequence (not a single Bezier):
      home → above_waypoint → low_hover → above_waypoint → above_target → pick
    """
    def __init__(self, target, action_scale, waypoint_pos, approach_h=0.08):
        super().__init__(target, action_scale)
        self.waypoint_pos = np.array(waypoint_pos, dtype=np.float32)
        self.approach_h = approach_h

    def _build_waypoints(self, cube_pos):
        P0 = EE_HOME.copy()
        # Waypoint hover points
        wp_high = self.waypoint_pos.copy()
        wp_high[2] = TABLE_TOP_Z + 0.12        # high above waypoint
        wp_low = self.waypoint_pos.copy()
        wp_low[2] = TABLE_TOP_Z + 0.05         # close hover (shows intent)
        # Target above
        above = cube_pos.copy()
        above[2] += self.approach_h
        # Transit point between waypoint and target (stay high)
        transit = (wp_high + above) / 2
        transit[2] = max(wp_high[2], above[2]) + 0.02

        # Build smooth multi-segment path
        seg1 = self._lerp_seg(P0, wp_high, 50)       # home → above waypoint
        seg2 = self._lerp_seg(wp_high, wp_low, 20)    # descend to hover
        seg3 = self._lerp_seg(wp_low, wp_high, 15)    # rise back up
        seg4 = self._lerp_seg(wp_high, transit, 25)    # transit toward target
        seg5 = self._lerp_seg(transit, above, 40)      # approach target

        self._arc_wps = np.concatenate([seg1, seg2, seg3, seg4, seg5])
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1

    @staticmethod
    def _lerp_seg(a, b, n):
        return np.array([a + t*(b - a) for t in np.linspace(0, 1, n)],
                        dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_episode(env, expert, context_pos, behavior_mode, vpath=None):
    """Run one demo episode. Returns (obs_v2_buf, act_buf, ep_len, success)."""
    obs_22 = env._get_obs()
    if vpath:
        env.record_video(str(vpath))

    obs_buf = np.zeros((EPISODE_LEN, OBS_DIM_V2), dtype=np.float32)
    act_buf = np.zeros((EPISODE_LEN, ACT_DIM), dtype=np.float32)
    ep_len = 0
    result = None

    for t in range(EPISODE_LEN):
        obs_v2 = get_obs_v2(env, context_pos, behavior_mode)
        action = expert.act(obs_22)  # expert uses base 22-d obs internally
        obs_buf[t] = obs_v2
        act_buf[t] = action
        result = env.step(action)
        obs_22 = result.obs
        ep_len = t + 1
        if result.done:
            break

    if vpath:
        env.stop_video()

    success = (result is not None and
               (result.info.get("success_left", 0) > 0.5 or
                result.info.get("success_right", 0) > 0.5))
    return obs_buf, act_buf, ep_len, success


# ══════════════════════════════════════════════════════════════════════
# LEGIBLE / PREDICTABLE / SAFETY VARIATIONS
# ══════════════════════════════════════════════════════════════════════

def build_legible_variations(n=10):
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        cp_y_mag = float(np.interp(t, [0, 1], [0.08, 0.28]))
        frac = (cp_y_mag - 0.08) / (0.28 - 0.08)
        cp_z = float(0.56 + 0.12 * frac)
        cp_x = float(0.38 - 0.10 * frac)
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.12]))
        variations.append(dict(cp_y_mag=cp_y_mag, cp_z=cp_z,
                               cp_x=cp_x, approach_h=approach_h))
    return variations

def build_predictable_variations(n=5):
    """Straight-line with slight approach height variation."""
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.12]))
        variations.append(dict(approach_h=approach_h))
    return variations

def build_safety_legible_variations(n=5):
    """Arc variations for safety-legible (wider arcs to ensure obstacle clearance)."""
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Use wider arcs (cp_y_mag >= 0.12) to clear center obstacle
        cp_y_mag = float(np.interp(t, [0, 1], [0.12, 0.28]))
        frac = (cp_y_mag - 0.12) / (0.28 - 0.12)
        cp_z = float(0.58 + 0.10 * frac)
        cp_x = float(0.36 - 0.08 * frac)
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.12]))
        variations.append(dict(cp_y_mag=cp_y_mag, cp_z=cp_z,
                               cp_x=cp_x, approach_h=approach_h))
    return variations


# ══════════════════════════════════════════════════════════════════════
# PREVIEW MODE
# ══════════════════════════════════════════════════════════════════════

def run_preview(out_dir):
    out_dir = Path(out_dir)
    vdir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)

    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          cube_half=CUBE_HALF, cube_mass=CUBE_MASS,
                          cube_lateral_friction=CUBE_FRICTION,
                          episode_length=EPISODE_LEN)

    results = []

    def run_and_report(label, expert, ctx, mode, vpath, extra_uids=None):
        expert.reset()
        _, _, ep_len, success = run_episode(env, expert, ctx, mode, vpath)
        tag = "OK" if success else "FAIL"
        print(f"  {tag}  ep_len={ep_len}")
        results.append((label, success, ep_len))
        if extra_uids:
            remove_bodies(env, extra_uids)

    print("=" * 60)
    print("  PREVIEW: V2 Demos (26-d obs, 4 behaviors)")
    print("=" * 60)

    # ── 1. Legible ────────────────────────────────────────────────────
    for tgt in ["left", "right"]:
        label = f"legible_{tgt}"
        print(f"\n-- Legible (Bezier arc → {tgt}) --")
        env.reset(seed=42)
        expert = LegibleExpert(tgt, env.action_scale_pos,
                               cp_y_mag=0.18, cp_z=0.62, cp_x=0.34,
                               approach_h=0.08)
        run_and_report(label, expert, None, MODE_LEGIBLE,
                       vdir / f"legible_{tgt}.mp4")

    # ── 2. Predictable ────────────────────────────────────────────────
    for tgt in ["left", "right"]:
        label = f"predictable_{tgt}"
        print(f"\n-- Predictable (straight → {tgt}) --")
        env.reset(seed=42)
        expert = PredictableExpert(tgt, env.action_scale_pos, approach_h=0.08)
        run_and_report(label, expert, None, MODE_PREDICTABLE,
                       vdir / f"predictable_{tgt}.mp4")

    # ── 3. Safety-Legible (obstacle on straight path → take curve) ────
    for tgt in ["left", "right"]:
        label = f"safety_legible_{tgt}"
        print(f"\n-- Safety-Legible: obstacle on straight, pick {tgt} --")
        env.reset(seed=42)
        obs_cfg = get_safety_obstacle("legible", tgt)
        uid = add_obstacle(env, obs_cfg)
        for _ in range(60):
            p.stepSimulation(physicsClientId=env._cid)
        expert = LegibleExpert(tgt, env.action_scale_pos,
                               cp_y_mag=0.18, cp_z=0.62, cp_x=0.34,
                               approach_h=0.08)
        run_and_report(label, expert, obs_cfg["pos"], MODE_LEGIBLE,
                       vdir / f"safety_legible_{tgt}.mp4",
                       extra_uids=[uid])

    # ── 4. Safety-Predictable (obstacle on arc path → take straight) ──
    for tgt in ["left", "right"]:
        label = f"safety_predictable_{tgt}"
        print(f"\n-- Safety-Predictable: obstacle on arc, pick {tgt} --")
        env.reset(seed=42)
        obs_cfg = get_safety_obstacle("predictable", tgt)
        uid = add_obstacle(env, obs_cfg)
        for _ in range(60):
            p.stepSimulation(physicsClientId=env._cid)
        expert = PredictableExpert(tgt, env.action_scale_pos, approach_h=0.08)
        run_and_report(label, expert, obs_cfg["pos"], MODE_PREDICTABLE,
                       vdir / f"safety_predictable_{tgt}.mp4",
                       extra_uids=[uid])

    # ── 5. Grounding (hover over waypoint → pick target) ──────────────
    for wp in WAYPOINT_BLOCKS[:3]:  # preview 3 of 5
        for tgt in ["left", "right"]:
            label = f"grounding_{wp['name']}_{tgt}"
            print(f"\n-- Grounding: waypoint={wp['name']}, pick {tgt} --")
            env.reset(seed=42)
            p.changeVisualShape(env._cube_l_uid, -1,
                                rgbaColor=[0.1, 0.8, 0.1, 1.0],
                                physicsClientId=env._cid)
            p.changeVisualShape(env._cube_r_uid, -1,
                                rgbaColor=[0.8, 0.1, 0.1, 1.0],
                                physicsClientId=env._cid)
            wp_uids = add_waypoint_blocks(env, WAYPOINT_BLOCKS)
            for _ in range(60):
                p.stepSimulation(physicsClientId=env._cid)
            expert = GroundingExpert(tgt, env.action_scale_pos,
                                    wp["pos"], approach_h=0.08)
            run_and_report(label, expert, wp["pos"], MODE_GROUNDING,
                           vdir / f"grounding_{wp['name']}_{tgt}.mp4",
                           extra_uids=wp_uids)

    env.close()

    # Summary
    n_ok = sum(1 for _, s, _ in results if s)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"  Preview: {n_ok}/{n_total} succeeded")
    for label, success, ep_len in results:
        tag = "OK  " if success else "FAIL"
        print(f"    {tag}  {label:35s}  ep_len={ep_len}")
    print(f"  Videos: {vdir}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════
# FULL COLLECTION
# ══════════════════════════════════════════════════════════════════════

def collect(seed=0, out_path="data/demos/demos_v2.npz"):
    block_configs = build_block_configs()
    leg_vars = build_legible_variations(10)
    pred_vars = build_predictable_variations(5)
    safety_leg_vars = build_safety_legible_variations(5)
    rng = np.random.default_rng(seed)

    vid_dir = Path(out_path).parent / "demo_videos_v2"
    vid_dir.mkdir(parents=True, exist_ok=True)

    schedule = []

    # ── Legible demos (200): 10 configs × 10 vars × 2 sides ──────────
    for cfg_id in range(len(block_configs)):
        for vi, var in enumerate(leg_vars):
            for tgt in ["left", "right"]:
                schedule.append({
                    "type": "legible", "cfg_id": cfg_id, "var_idx": vi,
                    "target": tgt, "var": var,
                    "obs_cfg": None, "wp": None,
                    "mode": MODE_LEGIBLE, "context": [0, 0, 0],
                    "safety_type": None,
                })
    assert len(schedule) == 200

    # ── Predictable demos (100): 10 configs × 5 vars × 2 sides ───────
    for cfg_id in range(len(block_configs)):
        for vi, var in enumerate(pred_vars):
            for tgt in ["left", "right"]:
                schedule.append({
                    "type": "predictable", "cfg_id": cfg_id, "var_idx": vi,
                    "target": tgt, "var": var,
                    "obs_cfg": None, "wp": None,
                    "mode": MODE_PREDICTABLE, "context": [0, 0, 0],
                    "safety_type": None,
                })
    assert len(schedule) == 300

    # ── Safety-Legible demos (~50): obstacle on straight → take curve ─
    # 10 configs × 5 vars × 1 side (alternate left/right)
    for cfg_id in range(len(block_configs)):
        for vi, var in enumerate(safety_leg_vars):
            tgt = "left" if (cfg_id + vi) % 2 == 0 else "right"
            obs_cfg = get_safety_obstacle("legible", tgt)
            schedule.append({
                "type": "safety_legible", "cfg_id": cfg_id, "var_idx": vi,
                "target": tgt, "var": var,
                "obs_cfg": obs_cfg, "wp": None,
                "mode": MODE_LEGIBLE, "context": obs_cfg["pos"],
                "safety_type": "legible",
            })
    n_safety_leg = len(schedule) - 300
    print(f"  Safety-legible demos planned: {n_safety_leg}")

    # ── Safety-Predictable demos (~50): obstacle on arc → take straight
    for cfg_id in range(len(block_configs)):
        for vi, var in enumerate(pred_vars):
            tgt = "left" if (cfg_id + vi) % 2 == 0 else "right"
            obs_cfg = get_safety_obstacle("predictable", tgt)
            schedule.append({
                "type": "safety_predictable", "cfg_id": cfg_id, "var_idx": vi,
                "target": tgt, "var": var,
                "obs_cfg": obs_cfg, "wp": None,
                "mode": MODE_PREDICTABLE, "context": obs_cfg["pos"],
                "safety_type": "predictable",
            })
    n_safety_pred = len(schedule) - 300 - n_safety_leg
    print(f"  Safety-predictable demos planned: {n_safety_pred}")

    # ── Grounding demos (100): 10 configs × 5 waypoints × 2 sides ────
    for cfg_id in range(len(block_configs)):
        for wp in WAYPOINT_BLOCKS:
            for tgt in ["left", "right"]:
                schedule.append({
                    "type": "grounding", "cfg_id": cfg_id, "var_idx": 0,
                    "target": tgt, "var": {"approach_h": 0.08},
                    "obs_cfg": None, "wp": wp,
                    "mode": MODE_GROUNDING, "context": wp["pos"],
                    "safety_type": None,
                })
    total = len(schedule)
    n_grounding = total - 300 - n_safety_leg - n_safety_pred
    print(f"  Grounding demos planned: {n_grounding}")

    print(f"\n{'='*60}")
    print(f"  V2 DEMO COLLECTION")
    print(f"  Legible: 200, Predictable: 100, "
          f"Safety-Leg: {n_safety_leg}, Safety-Pred: {n_safety_pred}, "
          f"Grounding: {n_grounding}")
    print(f"  Total: {total}")
    print(f"  Obs dim: {OBS_DIM_V2} (22 base + 3 context + 1 mode)")
    print(f"{'='*60}\n")

    # Shuffle for robustness
    rng.shuffle(schedule)

    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          cube_half=CUBE_HALF, cube_mass=CUBE_MASS,
                          cube_lateral_friction=CUBE_FRICTION,
                          episode_length=EPISODE_LEN)

    ckpt_path = Path(out_path).with_suffix(".ckpt.pkl")
    all_obs, all_act, all_lens = [], [], []
    all_types, all_targets, all_modes = [], [], []
    done_indices = set()
    retries = 0
    skipped = 0  # episodes excluded due to give-up (data quality guard)

    # Resume from checkpoint
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        all_obs = ckpt["all_obs"]
        all_act = ckpt["all_act"]
        all_lens = ckpt["all_lens"]
        all_types = ckpt["all_types"]
        all_targets = ckpt["all_targets"]
        all_modes = ckpt["all_modes"]
        done_indices = ckpt["done_indices"]
        retries = ckpt["retries"]
        skipped = ckpt.get("skipped", 0)
        print(f"  [RESUME] {len(all_obs)}/{total} from checkpoint (skipped={skipped})\n")

    pbar = tqdm(total=total, initial=len(all_obs), desc="collecting v2 demos")

    for idx, item in enumerate(schedule):
        if idx in done_indices:
            pbar.update(1)
            continue

        cfg = block_configs[item["cfg_id"]]
        added_uids = []

        attempt = 0
        gave_up = False
        obs_buf = act_buf = None
        ep_len = 0
        success = False
        while True:
            env.reset(seed=0)
            env.set_cube_offsets(left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                                right_dx=cfg["rdx"], right_dy=cfg["rdy"])

            # Add scene objects
            if item["type"].startswith("safety"):
                uid = add_obstacle(env, item["obs_cfg"])
                added_uids = [uid]
            elif item["type"] == "grounding":
                # Color pick cubes
                p.changeVisualShape(env._cube_l_uid, -1,
                                    rgbaColor=[0.1, 0.8, 0.1, 1.0],
                                    physicsClientId=env._cid)
                p.changeVisualShape(env._cube_r_uid, -1,
                                    rgbaColor=[0.8, 0.1, 0.1, 1.0],
                                    physicsClientId=env._cid)
                added_uids = add_waypoint_blocks(env, WAYPOINT_BLOCKS)

            # Settle physics
            for _ in range(60):
                p.stepSimulation(physicsClientId=env._cid)

            # Create expert
            var = item["var"]
            if item["type"] == "legible" or item["type"] == "safety_legible":
                expert = LegibleExpert(item["target"], env.action_scale_pos,
                                       **var)
            elif item["type"] == "predictable" or item["type"] == "safety_predictable":
                expert = PredictableExpert(item["target"], env.action_scale_pos,
                                          **var)
            elif item["type"] == "grounding":
                expert = GroundingExpert(item["target"], env.action_scale_pos,
                                        item["wp"]["pos"], **var)

            expert.reset()

            # Record video for first few of each type
            vpath = None
            type_count = sum(1 for t in all_types if t == item["type"])
            if type_count < 5:
                vname = (f"{item['type']}_{item['target']}_"
                         f"cfg{item['cfg_id']:02d}_v{item['var_idx']:02d}.mp4")
                vpath = vid_dir / vname

            obs_buf, act_buf, ep_len, success = run_episode(
                env, expert, item["context"], item["mode"], vpath)

            # Clean up added objects
            if added_uids:
                remove_bodies(env, added_uids)
                added_uids = []

            if success:
                break

            # Failed attempt — discard video and retry
            attempt += 1
            retries += 1
            if vpath and vpath.exists():
                vpath.unlink()
            if attempt > 10:
                # DATA QUALITY GUARD: never write failed episodes to dataset
                print(f"\n  [SKIP] idx={idx} ({item['type']} "
                      f"{item['target']}) gave up after {attempt} attempts — "
                      f"excluded from dataset.")
                gave_up = True
                break

        # Only append verified successful episodes
        if gave_up or not success:
            skipped += 1
            done_indices.add(idx)  # mark done so resume skips it
            pbar.update(1)
            continue

        all_obs.append(obs_buf)
        all_act.append(act_buf)
        all_lens.append(ep_len)
        all_types.append(item["type"])
        all_targets.append(item["target"])
        all_modes.append(item["mode"])
        done_indices.add(idx)
        pbar.update(1)

        # Checkpoint every 50 demos
        if len(all_obs) % 50 == 0:
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "all_obs": all_obs, "all_act": all_act,
                    "all_lens": all_lens, "all_types": all_types,
                    "all_targets": all_targets, "all_modes": all_modes,
                    "done_indices": done_indices, "retries": retries,
                    "skipped": skipped,
                }, f, protocol=4)

    pbar.close()
    env.close()

    # Save
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_by_type = {}
    for t in all_types:
        n_by_type[t] = n_by_type.get(t, 0) + 1

    metadata = {
        "description": "V2 demos: 26-d obs, 4 behaviors, CFG-ready",
        "obs_dim": OBS_DIM_V2,
        "obs_format": ("ee_pos(3), ee_quat(4), grip(1), "
                       "left_pos(3), left_quat(4), "
                       "right_pos(3), right_quat(4), "
                       "context_xyz(3), behavior_mode(1)"),
        "n_demos": len(all_obs),
        "n_by_type": n_by_type,
        "behavior_modes": {"+1": "legible", "-1": "predictable", "0": "grounding"},
        "retries": retries,
        "skipped_give_up": skipped,
    }

    np.savez_compressed(
        str(out),
        obs=np.stack(all_obs),               # (N, 400, 26)
        actions=np.stack(all_act),            # (N, 400, 5)
        episode_lengths=np.array(all_lens),   # (N,)
        types=np.array(all_types),            # (N,) string
        targets=np.array(all_targets),        # (N,) string
        behavior_modes=np.array(all_modes),   # (N,) float
        metadata_json=json.dumps(metadata),
    )

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"\n{'='*60}")
    print(f"  SAVED: {out}")
    print(f"  {len(all_obs)} demos, obs_dim={OBS_DIM_V2}")
    for t, n in sorted(n_by_type.items()):
        print(f"    {t:15s}: {n}")
    print(f"  Retries: {retries}")
    if skipped > 0:
        print(f"  [WARN] Skipped (data quality): {skipped} episodes excluded")
    else:
        print(f"  Skipped (data quality): 0  — all episodes succeeded")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("V2 Demo Collection (26-d obs, 4 behaviors)")
    ap.add_argument("--preview", action="store_true",
                    help="Generate preview videos only")
    ap.add_argument("--collect", action="store_true",
                    help="Run full collection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/demos/demos_v2.npz")
    ap.add_argument("--preview_dir", type=str,
                    default="outputs/previews_v2")
    args = ap.parse_args()

    if args.preview:
        run_preview(args.preview_dir)
    elif args.collect:
        collect(seed=args.seed, out_path=args.out)
    else:
        print("Specify --preview or --collect")
        ap.print_help()


if __name__ == "__main__":
    main()
