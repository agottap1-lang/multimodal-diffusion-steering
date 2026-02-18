#!/usr/bin/env python
"""Collect multimodal demonstrations for TwoBlockPick.

Structure
─────────
  10 block-position configs  ×  40 episodes per config  =  400 demos
    Per config: 20 left-pick  +  20 right-pick
    20 sweeping Bézier arc variations per side

  Multimodality = 50/50 left/right picks (same scene → two modes)
  Legibility    = sweeping Bézier arcs that communicate intent early

  Blocks are placed symmetrically about the y=0 centre line (imaginary
  line from Franka base) and close together (±7 cm).

  Arc trajectories use quadratic Bézier curves with 200 intermediate
  waypoints.  The biggest arcs sweep the robot near its workspace
  limits (y ≈ ±0.28) before curving down to the target cube, producing
  legible curved approach paths of varying size.

Usage:
    python scripts/collect_demos_twoblockpick.py \
        --seed 0 --out data/demos/demos.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM

# ─── constants ────────────────────────────────────────────────────────
_EE_HOME = np.array([0.40, 0.0, 0.55], dtype=np.float32)
_N_ARC_PTS     = 200   # Bézier approach arc waypoints
_N_DESCENT_PTS = 30    # straight descent to cube surface
_N_GRIP_STEPS  = 40    # gradual gripper close (grip ramps +1 → -1)
_N_LIFT_PTS    = 30    # slow lift waypoints
_EPISODE_LEN   = 400   # extended episode for slow graceful motion


# ─── 10 block-position configurations ────────────────────────────────
# Each entry: (left_dx, left_dy, right_dx, right_dy) in metres.
# Small offsets add positional robustness without excessive variation.

def _build_block_configs() -> list[dict]:
    configs: list[dict] = []

    # Type A – both blocks shifted symmetrically  (4 configs)
    for dx in [-0.005, 0.0, 0.005]:
        configs.append(dict(tag="both", ldx=dx, ldy=0.0, rdx=dx, rdy=0.0))
    configs.append(dict(tag="both", ldx=0.0, ldy=0.004, rdx=0.0, rdy=-0.004))

    # Type B – only left block shifted  (3 configs)
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(tag="left", ldx=dx, ldy=dy, rdx=0.0, rdy=0.0))

    # Type C – only right block shifted  (3 configs)
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(tag="right", ldx=0.0, ldy=0.0, rdx=dx, rdy=dy))

    assert len(configs) == 10, f"Expected 10 configs, got {len(configs)}"
    return configs


# ─── 20 Bézier arc variations ────────────────────────────────────────
#
# Each variation stores a MAGNITUDE for the control point lateral
# sweep.  The SIGN is applied at expert construction time:
#   left  picks → cp_y = +magnitude  (sweep left, towards left block)
#   right picks → cp_y = -magnitude  (sweep right, towards right block)
# This ensures arcs always curve towards the correct side.
#
# Bézier:  B(t) = (1-t)²·P₀ + 2(1-t)t·P₁ + t²·P₂
#   P₀ = EE home  [0.40, 0.0, 0.55]
#   P₁ = control point  (determines arc curvature)
#   P₂ = above cube  [cube_x, cube_y, cube_z + approach_h]

def _build_arc_variations(n: int = 20) -> list[dict]:
    """Return *n* arc specs.  cp_y_mag is ALWAYS positive."""
    variations: list[dict] = []
    for i in range(n):
        t = i / max(n - 1, 1)

        # Magnitude of lateral sweep: gentle (0.05) → extreme (0.28)
        cp_y_mag = float(np.interp(t, [0, 1], [0.05, 0.28]))

        # Bigger arcs → higher & more pulled-back control point
        frac = (cp_y_mag - 0.05) / (0.28 - 0.05)    # 0 → 1
        cp_z = float(0.56 + 0.12 * frac)             # 0.56 → 0.68
        cp_x = float(0.38 - 0.10 * frac)             # 0.38 → 0.28

        variations.append(dict(
            cp_x=cp_x,
            cp_y_mag=cp_y_mag,   # sign applied per target side
            cp_z=cp_z,
        ))
    return variations


# ─── scripted expert  ─────────────────────────────────────────────────

class ScriptedExpert:
    """Pick a specific cube via a sweeping Bézier arc approach.

    ALL motion is waypoint-based for uniformly slow, graceful movement:
      Phase 0 – 200-waypoint Bézier arc  (home → sweep → above cube)
      Phase 1 – 30-waypoint descent      (above cube → cube surface)
      Phase 2 – 40-step gradual grip     (gripper ramps +1 → -1)
      Phase 3 – 30-waypoint slow lift    (cube surface → high)
      Phase 4 – Hold

    cp_y sign is set by the caller: + for left picks, - for right picks.
    """

    def __init__(self, target: str, action_scale: float = 0.05,
                 arc_control_point: np.ndarray | None = None,
                 approach_height_range: tuple[float, float] = (0.06, 0.12),
                 rng: np.random.Generator | None = None) -> None:
        assert target in ("left", "right")
        self.target = target
        self.scale = action_scale
        self._arc_cp = (arc_control_point.copy() if arc_control_point is not None
                        else np.array([0.35, 0.0, 0.58], dtype=np.float32))
        self._approach_range = approach_height_range
        self._rng = rng or np.random.default_rng()

        self.phase = 0
        self.wait = 0
        self._cube_pos0: np.ndarray | None = None
        # Waypoint arrays (set once in _build_all_waypoints)
        self._arc_wps: np.ndarray | None = None
        self._descent_wps: np.ndarray | None = None
        self._lift_wps: np.ndarray | None = None
        self._wp_idx = 0
        self._approach_h: float = 0.08

    def reset(self) -> None:
        self.phase = 0
        self.wait = 0
        self._cube_pos0 = None
        self._arc_wps = None
        self._descent_wps = None
        self._lift_wps = None
        self._wp_idx = 0
        lo, hi = self._approach_range
        self._approach_h = float(self._rng.uniform(lo, hi))

    # ── build all waypoints at once ──────────────────────────────────

    def _build_all_waypoints(self, cube_pos: np.ndarray) -> None:
        self._cube_pos0 = cube_pos.copy()

        # 1) Bézier arc: home → control point → above cube
        P0 = _EE_HOME.copy()
        P1 = self._arc_cp.copy()
        P2 = cube_pos.copy()
        P2[2] += self._approach_h

        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array([
            (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
            for t in ts
        ], dtype=np.float32)

        # 2) Descent: above cube → cube surface (straight line down)
        above = P2.copy()
        grasp = cube_pos.copy()
        grasp[2] += 0.005
        self._descent_wps = np.array([
            above + t * (grasp - above)
            for t in np.linspace(0, 1, _N_DESCENT_PTS)
        ], dtype=np.float32)

        # 3) Lift: grasp pos → high
        lift_top = grasp.copy()
        lift_top[2] = 0.60
        self._lift_wps = np.array([
            grasp + t * (lift_top - grasp)
            for t in np.linspace(0, 1, _N_LIFT_PTS)
        ], dtype=np.float32)

        self._wp_idx = 1   # skip first arc wp (already at home)

    # ── main action loop ─────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> np.ndarray:
        ee_pos = obs[0:3]
        cube_pos = obs[8:11].copy() if self.target == "left" else obs[15:18].copy()

        if self._arc_wps is None:
            self._build_all_waypoints(cube_pos)

        action = np.zeros(ACT_DIM, dtype=np.float32)

        if self.phase == 0:
            # Phase 0: Follow 200-waypoint Bézier arc (slow & smooth)
            tgt = self._arc_wps[self._wp_idx]
            speed = 0.45     # uniformly slow
            action = self._goto(ee_pos, tgt, grip=1.0, speed=speed)
            if np.linalg.norm(ee_pos - tgt) < 0.012:
                self._wp_idx += 1
                if self._wp_idx >= _N_ARC_PTS:
                    self.phase = 1
                    self._wp_idx = 1  # skip first descent wp (= last arc wp)

        elif self.phase == 1:
            # Phase 1: 30-waypoint slow descent to cube centre
            tgt = self._descent_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=1.0, speed=0.35)
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_DESCENT_PTS:
                    self.phase = 2
                    self.wait = 0

        elif self.phase == 2:
            # Phase 2: Gradual gripper close over 40 steps (+1 → -1)
            t_frac = self.wait / max(_N_GRIP_STEPS - 1, 1)
            grip_val = 1.0 - 2.0 * t_frac   # +1 → -1
            action[4] = grip_val
            self.wait += 1
            if self.wait >= _N_GRIP_STEPS:
                self.phase = 3
                self._wp_idx = 1  # skip first lift wp (= current pos)

        elif self.phase == 3:
            # Phase 3: 30-waypoint slow lift
            tgt = self._lift_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=-1.0, speed=0.35)
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_LIFT_PTS:
                    self.phase = 4

        else:
            # Phase 4: Hold
            action[4] = -1.0

        return action

    def _goto(self, cur: np.ndarray, tgt: np.ndarray, grip: float,
              speed: float = 1.0) -> np.ndarray:
        delta = (tgt - cur) / self.scale
        a = np.zeros(ACT_DIM, dtype=np.float32)
        a[:3] = np.clip(delta * speed, -1, 1)
        a[4] = grip
        return a


# ─── collection loop ─────────────────────────────────────────────────

def collect(seed: int, out_path: str, cube_jitter: float = 0.0) -> None:
    block_configs = _build_block_configs()            # 10 configs
    arc_variations = _build_arc_variations(20)        # 20 arcs

    n_configs = len(block_configs)                    # 10
    n_arcs    = len(arc_variations)                   # 20
    eps_per_config = 2 * n_arcs                       # 20L + 20R = 40
    total = n_configs * eps_per_config                # 400

    print(f"Plan: {n_configs} block configs × {eps_per_config} eps/config "
          f"= {total} demos  ({total//2}L + {total//2}R)")
    print(f"  block config types: "
          f"{sum(1 for c in block_configs if c['tag']=='both')} both, "
          f"{sum(1 for c in block_configs if c['tag']=='left')} left-only, "
          f"{sum(1 for c in block_configs if c['tag']=='right')} right-only")
    cp_max = max(v["cp_y_mag"] for v in arc_variations)
    print(f"  Bézier arcs: {n_arcs} variations, "
          f"control point Y ±{cp_max*1000:.0f} mm, "
          f"{_N_ARC_PTS} arc + {_N_DESCENT_PTS} descent + "
          f"{_N_GRIP_STEPS} grip + {_N_LIFT_PTS} lift waypoints")
    print(f"  episode length: {_EPISODE_LEN}")
    print(f"  all {total} videos will be saved\n")

    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          episode_length=_EPISODE_LEN)
    max_T = env.episode_length
    rng = np.random.default_rng(seed)

    vid_dir = Path(out_path).parent / "demo_videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    all_obs:        list[np.ndarray] = []
    all_act:        list[np.ndarray] = []
    all_lens:       list[int]        = []
    all_labels:     list[str]        = []
    all_config_ids: list[int]        = []

    successes = 0
    left_ok = right_ok = 0
    retries = 0

    pbar = tqdm(total=total, desc="collecting demos")

    for cfg_id, cfg in enumerate(block_configs):
        # Balanced schedule: pair each arc with L and R
        schedule: list[tuple[str, int]] = []
        for ai in range(n_arcs):
            schedule.append(("left",  ai))
            schedule.append(("right", ai))
        rng.shuffle(schedule)

        for target, arc_idx in schedule:
            ep_rng_seed = int(rng.integers(0, 2**31))

            # Retry until success
            attempt = 0
            while True:
                env.reset(seed=0)
                env.set_cube_offsets(
                    left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                    right_dx=cfg["rdx"], right_dy=cfg["rdy"])
                obs = env._get_obs()

                # Build control point with correct sign for target side
                var = arc_variations[arc_idx]
                sign = 1.0 if target == "left" else -1.0
                cp = np.array([var["cp_x"],
                               sign * var["cp_y_mag"],
                               var["cp_z"]], dtype=np.float32)
                expert = ScriptedExpert(
                    target,
                    action_scale=env.action_scale_pos,
                    arc_control_point=cp,
                    rng=np.random.default_rng(ep_rng_seed + attempt))
                expert.reset()

                vname = f"cfg{cfg_id:02d}_{target}_arc{arc_idx:02d}.mp4"
                vpath = vid_dir / vname
                env.record_video(str(vpath))

                obs_buf = np.zeros((max_T, OBS_DIM), dtype=np.float32)
                act_buf = np.zeros((max_T, ACT_DIM), dtype=np.float32)
                ep_len = 0

                for t in range(max_T):
                    action = expert.act(obs)
                    obs_buf[t] = obs
                    act_buf[t] = action
                    result = env.step(action)
                    obs = result.obs
                    ep_len = t + 1
                    if result.done:
                        break

                env.stop_video()

                success = (result.info["success_left"] > 0.5
                           or result.info["success_right"] > 0.5)
                if success:
                    break
                attempt += 1
                retries += 1
                if vpath.exists():
                    vpath.unlink()

            all_obs.append(obs_buf)
            all_act.append(act_buf)
            all_lens.append(ep_len)
            all_labels.append(target)
            all_config_ids.append(cfg_id)

            successes += 1
            if target == "left":
                left_ok += 1
            else:
                right_ok += 1

            pbar.update(1)

    pbar.close()
    env.close()

    # ── save ──────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metadata as JSON string
    import json
    metadata = {
        "cube_jitter": cube_jitter,
        "episode_length": env.episode_length,
        "action_scale_pos": env.action_scale_pos,
        "action_scale_yaw_deg": np.degrees(env._action_scale_yaw),
        "collection_seed": seed,
        "n_demos": total,
        "n_left": sum(1 for l in all_labels if l == "left"),
        "n_right": sum(1 for l in all_labels if l == "right"),
    }
    
    np.savez_compressed(
        str(out),
        obs=np.stack(all_obs),
        actions=np.stack(all_act),
        episode_lengths=np.array(all_lens),
        labels=np.array(all_labels),
        config_ids=np.array(all_config_ids),
        metadata_json=json.dumps(metadata),  # Store as JSON string
    )

    n_left  = sum(1 for l in all_labels if l == "left")
    n_right = sum(1 for l in all_labels if l == "right")
    n_unique = len(set(all_config_ids))

    print(f"\n{'='*55}")
    print(f"  saved {total} demos to {out}")
    print(f"  unique scenes      : {n_unique}")
    print(f"  total left / right : {n_left} / {n_right}")
    print(f"  left  success      : {left_ok} / {n_left}")
    print(f"  right success      : {right_ok} / {n_right}")
    print(f"  overall success    : {successes}/{total} = {successes/total:.1%}")
    print(f"  retries needed     : {retries}")
    print(f"  videos saved to    : {vid_dir}/")
    print(f"{'='*55}")

    assert left_ok == total // 2, (
        f"Expected {total//2} left successes, got {left_ok}")
    assert right_ok == total // 2, (
        f"Expected {total//2} right successes, got {right_ok}")


# ─── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect 400 multimodal demos (200L+200R) with "
                    "legible Bézier arc trajectories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/demos/demos.npz")
    ap.add_argument("--cube_jitter", type=float, default=0.0,
                    help="Random jitter for cube placement (m). 0=fixed positions, "
                         "0.015=±1.5cm. Must match eval jitter for distribution consistency.")
    args = ap.parse_args()
    collect(args.seed, args.out, cube_jitter=args.cube_jitter)


if __name__ == "__main__":
    main()
