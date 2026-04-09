#!/usr/bin/env python
"""Collect safety + grounding demonstrations for TwoBlockPick.

Adds 200 demos (100 safety + 100 grounding) to complement the existing
400 combined demos (legible/neutral/deceptive).

SAFETY expert (100 demos):
  Quadratic Bézier with control point going EXTRA HIGH (z=0.66→0.74)
  to clear imaginary obstacle zone at (x=0.45, z=0.42).
  5 arc variations × 2 sides × 10 block configs = 100

GROUNDING expert (100 demos):
  Quadratic Bézier where control point passes near a WAYPOINT at
  (x=0.48, y=0.03) before descending to the target block.
  Like legible but biased toward center/waypoint rather than target.
  5 arc variations × 2 sides × 10 block configs = 100

Arc ranges kept within existing demo bounds:
  - z stays within [0.55, 0.74] (existing range is [0.55, 0.68])
  - y stays within [-0.28, 0.28] (existing legible max is 0.28)
  - x stays within [0.28, 0.50] (within existing range)

Usage:
  py scripts/collect_demos_safety_grounding.py --preview
  py scripts/collect_demos_safety_grounding.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM

# ── constants (SAME as collect_demos_combined.py) ──────────────────────
_EE_HOME = np.array([0.40, 0.0, 0.55], dtype=np.float32)

_N_ARC_PTS     = 200
_N_DESCENT_PTS = 30
_N_GRIP_STEPS  = 40
_N_LIFT_PTS    = 30
_EPISODE_LEN   = 400

_CUBE_HALF     = 0.015
_CUBE_MASS     = 0.08
_CUBE_FRICTION = 2.5

STYLE_SAFETY    = 3
STYLE_GROUNDING = 4

CKPT_EVERY = 20


# ── block configs (SAME 10 as combined) ────────────────────────────────

def _build_block_configs() -> list[dict]:
    configs: list[dict] = []
    for dx in [-0.005, 0.0, 0.005]:
        configs.append(dict(tag="both", ldx=dx, ldy=0.0, rdx=dx, rdy=0.0))
    configs.append(dict(tag="both", ldx=0.0, ldy=0.004, rdx=0.0, rdy=-0.004))
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(tag="left", ldx=dx, ldy=dy, rdx=0.0, rdy=0.0))
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(tag="right", ldx=0.0, ldy=0.0, rdx=dx, rdy=dy))
    assert len(configs) == 10
    return configs


# ── arc variations ─────────────────────────────────────────────────────

def _build_safety_variations(n: int = 5) -> list[dict]:
    """Arcs that go EXTRA HIGH over the obstacle zone (x≈0.45, z≈0.42).

    Control point z ranges 0.66→0.74 (vs legible 0.56→0.68).
    Control point x ranges 0.42→0.38 (passes over obstacle region).
    Small lateral bias toward target (cp_y_mag 0.02→0.06) — subtle,
    keeps it close to neutral but with distinct high-arc shape.
    """
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        cp_z     = float(np.interp(t, [0, 1], [0.66, 0.74]))
        cp_x     = float(np.interp(t, [0, 1], [0.42, 0.38]))
        cp_y_mag = float(np.interp(t, [0, 1], [0.02, 0.06]))
        approach_h = float(np.interp(t, [0, 1], [0.08, 0.14]))
        variations.append(dict(cp_x=cp_x, cp_y_mag=cp_y_mag,
                               cp_z=cp_z, approach_h=approach_h))
    return variations


def _build_grounding_variations(n: int = 5) -> list[dict]:
    """Arcs that sweep near a WAYPOINT (x=0.48, y=0.03) before target.

    Uses cubic Bézier: P1 biased toward waypoint, P2 toward target.
    Creates a visible "detour past the center" in the approach.
    wp_y_bias ranges 0.02→0.06 (how close to waypoint center).
    """
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # P1: near waypoint zone (x=0.46→0.48, y biased toward center)
        p1_x     = float(np.interp(t, [0, 1], [0.44, 0.48]))
        p1_y_mag = float(np.interp(t, [0, 1], [0.02, 0.05]))  # slight bias toward center
        p1_z     = float(np.interp(t, [0, 1], [0.58, 0.64]))
        # P2: commit toward target (same side but further along)
        p2_x     = float(np.interp(t, [0, 1], [0.48, 0.50]))
        p2_y_mag = float(np.interp(t, [0, 1], [0.03, 0.06]))  # toward target
        p2_z     = float(np.interp(t, [0, 1], [0.54, 0.50]))
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.10]))
        variations.append(dict(
            p1_x=p1_x, p1_y_mag=p1_y_mag, p1_z=p1_z,
            p2_x=p2_x, p2_y_mag=p2_y_mag, p2_z=p2_z,
            approach_h=approach_h,
        ))
    return variations


# ── helpers ────────────────────────────────────────────────────────────

def _cubic_bezier(t: float, P0, P1, P2, P3):
    return ((1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1
            + 3*(1 - t) * t**2 * P2 + t**3 * P3)


# ── expert base (copied from combined for standalone use) ──────────────

class _ExpertBase:
    def __init__(self, target: str, action_scale: float = 0.05):
        assert target in ("left", "right")
        self.target = target
        self.scale  = action_scale
        self.phase  = 0
        self.wait   = 0
        self._arc_wps     = None
        self._descent_wps = None
        self._lift_wps    = None
        self._wp_idx = 0

    def reset(self):
        self.phase = 0
        self.wait  = 0
        self._arc_wps = self._descent_wps = self._lift_wps = None
        self._wp_idx  = 0

    def _cube_pos(self, obs):
        return obs[8:11].copy() if self.target == "left" else obs[15:18].copy()

    def _build_descent_and_lift(self, above, cube_pos):
        grasp = cube_pos.copy(); grasp[2] += 0.005
        self._descent_wps = np.array([
            above + s * (grasp - above)
            for s in np.linspace(0, 1, _N_DESCENT_PTS)
        ], dtype=np.float32)
        lift_top = grasp.copy(); lift_top[2] = 0.60
        self._lift_wps = np.array([
            grasp + s * (lift_top - grasp)
            for s in np.linspace(0, 1, _N_LIFT_PTS)
        ], dtype=np.float32)

    def _build_waypoints(self, cube_pos):
        raise NotImplementedError

    def act(self, obs):
        ee_pos   = obs[0:3]
        cube_pos = self._cube_pos(obs)
        if self._arc_wps is None:
            self._build_waypoints(cube_pos)

        action = np.zeros(ACT_DIM, dtype=np.float32)

        if self.phase == 0:
            tgt = self._arc_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=1.0, speed=0.45)
            if np.linalg.norm(ee_pos - tgt) < 0.012:
                self._wp_idx += 1
                if self._wp_idx >= _N_ARC_PTS:
                    self.phase = 1; self._wp_idx = 1
        elif self.phase == 1:
            tgt = self._descent_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=1.0, speed=0.35)
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_DESCENT_PTS:
                    self.phase = 2; self.wait = 0
        elif self.phase == 2:
            t_frac = self.wait / max(_N_GRIP_STEPS - 1, 1)
            action[4] = 1.0 - 2.0 * t_frac
            self.wait += 1
            if self.wait >= _N_GRIP_STEPS:
                self.phase = 3; self._wp_idx = 1
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
        a[4]  = grip
        return a


# ── safety expert ──────────────────────────────────────────────────────

class SafetyExpert(_ExpertBase):
    """Quadratic Bézier going EXTRA HIGH to clear obstacle zone.

    Arc peaks at z=0.66→0.74, x≈0.40 (over obstacle at x=0.45, z=0.42).
    Small lateral bias toward target (cp_y_mag 0.02→0.06).
    """

    def __init__(self, target, action_scale, var):
        super().__init__(target, action_scale)
        self._var = var

    def _build_waypoints(self, cube_pos):
        sign = 1.0 if self.target == "left" else -1.0
        P0   = _EE_HOME.copy()
        P1   = np.array([self._var["cp_x"],
                         sign * self._var["cp_y_mag"],
                         self._var["cp_z"]], dtype=np.float32)
        above = cube_pos.copy()
        above[2] += self._var["approach_h"]
        P2 = above.copy()

        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array([
            (1 - t)**2 * P0 + 2*(1 - t)*t * P1 + t**2 * P2
            for t in ts
        ], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


# ── grounding expert ───────────────────────────────────────────────────

class GroundingExpert(_ExpertBase):
    """Cubic Bézier that detours near a WAYPOINT before picking target.

    P1 is biased toward the center/waypoint (y near 0.03).
    P2 then commits toward the actual target block.
    Creates visible "visit waypoint then pick" trajectory.
    """

    def __init__(self, target, action_scale, var):
        super().__init__(target, action_scale)
        self._var = var

    def _build_waypoints(self, cube_pos):
        target_sign = 1.0 if self.target == "left" else -1.0

        P0 = _EE_HOME.copy()
        # P1: biased toward center/waypoint (y near 0.03, OPPOSITE of target)
        # For left target: P1 goes slightly right (toward center)
        # For right target: P1 goes slightly left (toward center)
        center_bias = -target_sign  # opposite of target direction
        P1 = np.array([self._var["p1_x"],
                       center_bias * self._var["p1_y_mag"],
                       self._var["p1_z"]], dtype=np.float32)
        # P2: commits toward actual target
        P2 = np.array([self._var["p2_x"],
                       target_sign * self._var["p2_y_mag"],
                       self._var["p2_z"]], dtype=np.float32)
        above = cube_pos.copy()
        above[2] += self._var["approach_h"]
        P3 = above.copy()

        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array([
            _cubic_bezier(t, P0, P1, P2, P3) for t in ts
        ], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


# ── one episode ────────────────────────────────────────────────────────

def _run_episode(env, expert, vpath):
    max_T = env.episode_length
    obs   = env._get_obs()
    if vpath is not None:
        env.record_video(str(vpath))

    obs_buf = np.zeros((max_T, OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((max_T, ACT_DIM), dtype=np.float32)
    ep_len  = 0
    result  = None

    for t in range(max_T):
        action     = expert.act(obs)
        obs_buf[t] = obs
        act_buf[t] = action
        result     = env.step(action)
        obs        = result.obs
        ep_len     = t + 1
        if result.done:
            break

    if vpath is not None:
        env.stop_video()

    success = (result.info["success_left"] > 0.5 or
               result.info["success_right"] > 0.5)
    return obs_buf, act_buf, ep_len, success


# ── collection ─────────────────────────────────────────────────────────

def collect(seed=0, out_path="data/demos/demos_safety_grounding.npz",
            preview=False):
    block_configs    = _build_block_configs()
    safety_vars      = _build_safety_variations(5)
    grounding_vars   = _build_grounding_variations(5)

    n_configs      = len(block_configs)
    eps_per_config = 20   # 5 safe-L + 5 safe-R + 5 ground-L + 5 ground-R
    total          = n_configs * eps_per_config  # 200

    vid_dir = Path(out_path).parent / "demo_videos_safety_grounding"
    vid_dir.mkdir(parents=True, exist_ok=True)

    style_names = {STYLE_SAFETY: "safety", STYLE_GROUNDING: "grounding"}

    print("=" * 62)
    print("  TwoBlockPick — Safety + Grounding Demo Collection")
    print("=" * 62)
    if preview:
        print("  *** PREVIEW MODE: 2 videos only ***")
    print(f"  Recipe per config: 5 safe-L + 5 safe-R + "
          f"5 ground-L + 5 ground-R = {eps_per_config}")
    print(f"  Total: {n_configs} configs × {eps_per_config} = {total} demos")
    print(f"  Cube: half={_CUBE_HALF}m, mass={_CUBE_MASS}kg, "
          f"friction={_CUBE_FRICTION}")
    print(f"  Videos → {vid_dir}")
    if not preview:
        print(f"  Dataset → {out_path}")
    print()

    env = TwoBlockPickEnv(
        render=False, cube_jitter=0.0,
        cube_half=_CUBE_HALF, cube_mass=_CUBE_MASS,
        cube_lateral_friction=_CUBE_FRICTION,
        episode_length=_EPISODE_LEN,
    )
    rng = np.random.default_rng(seed)

    # ── preview ────────────────────────────────────────────────────
    if preview:
        preview_cases = [
            (STYLE_SAFETY,    "left",  safety_vars[3]),
            (STYLE_GROUNDING, "right", grounding_vars[3]),
        ]
        cfg = block_configs[1]
        print("Running 2 preview episodes ...\n")
        for style, target, var in preview_cases:
            env.reset(seed=0)
            env.set_cube_offsets(left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                                right_dx=cfg["rdx"], right_dy=cfg["rdy"])
            if style == STYLE_SAFETY:
                expert = SafetyExpert(target, env.action_scale_pos, var)
            else:
                expert = GroundingExpert(target, env.action_scale_pos, var)
            expert.reset()

            sname = style_names[style]
            vname = f"preview_{sname}_{target}.mp4"
            vpath = vid_dir / vname
            _, _, ep_len, success = _run_episode(env, expert, vpath)

            if style == STYLE_SAFETY:
                info = f"cp_z={var['cp_z']:.2f} high-arc"
            else:
                info = f"waypoint detour p1_y={var['p1_y_mag']*100:.0f}mm"

            print(f"  {sname:12s} {target:5s}  {info:35s}  "
                  f"success={success}  ep_len={ep_len}  → {vname}")

        env.close()
        print(f"\nPreview done. Videos: {vid_dir}/")
        print("Approve then run:  py scripts/collect_demos_safety_grounding.py --seed 0")
        return

    # ── full collection ────────────────────────────────────────────
    ckpt_path = Path(out_path).with_name(Path(out_path).stem + "_ckpt.pkl")
    done_keys: set[tuple] = set()

    all_obs:         list[np.ndarray] = []
    all_act:         list[np.ndarray] = []
    all_lens:        list[int]        = []
    all_labels:      list[str]        = []
    all_styles:      list[int]        = []
    all_style_names: list[str]        = []
    all_arc_idxs:    list[int]        = []
    all_cfg_ids:     list[int]        = []
    retries = 0

    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        all_obs         = ckpt["all_obs"]
        all_act         = ckpt["all_act"]
        all_lens        = ckpt["all_lens"]
        all_labels      = ckpt["all_labels"]
        all_styles      = ckpt["all_styles"]
        all_style_names = ckpt["all_style_names"]
        all_arc_idxs    = ckpt["all_arc_idxs"]
        all_cfg_ids     = ckpt["all_cfg_ids"]
        retries         = ckpt["retries"]
        done_keys       = ckpt["done_keys"]
        print(f"  [RESUME] {len(all_obs)}/{total} demos from checkpoint\n")

    pbar = tqdm(total=total, initial=len(all_obs),
                desc="collecting safety+grounding demos")

    for cfg_id, cfg in enumerate(block_configs):
        schedule: list[tuple[int, str, int, dict]] = []
        for vi, var in enumerate(safety_vars):
            schedule.append((STYLE_SAFETY, "left",  vi, var))
            schedule.append((STYLE_SAFETY, "right", vi, var))
        for vi, var in enumerate(grounding_vars):
            schedule.append((STYLE_GROUNDING, "left",  vi, var))
            schedule.append((STYLE_GROUNDING, "right", vi, var))

        assert len(schedule) == eps_per_config
        rng.shuffle(schedule)

        for style, target, var_idx, var in schedule:
            key = (cfg_id, style, target, var_idx)
            if key in done_keys:
                pbar.update(1)
                continue

            attempt = 0
            while True:
                env.reset(seed=0)
                env.set_cube_offsets(left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                                    right_dx=cfg["rdx"], right_dy=cfg["rdy"])
                if style == STYLE_SAFETY:
                    expert = SafetyExpert(target, env.action_scale_pos, var)
                else:
                    expert = GroundingExpert(target, env.action_scale_pos, var)
                expert.reset()

                sname = style_names[style]
                vname = (f"cfg{cfg_id:02d}_{sname[:3]}_{target}"
                         f"_v{var_idx:02d}.mp4")
                vpath = vid_dir / vname

                obs_buf, act_buf, ep_len, success = _run_episode(
                    env, expert, vpath)

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
            all_styles.append(style)
            all_style_names.append(sname)
            all_arc_idxs.append(var_idx)
            all_cfg_ids.append(cfg_id)
            done_keys.add(key)

            pbar.update(1)

            if len(all_obs) % CKPT_EVERY == 0:
                with open(ckpt_path, "wb") as f:
                    pickle.dump({
                        "all_obs": all_obs, "all_act": all_act,
                        "all_lens": all_lens, "all_labels": all_labels,
                        "all_styles": all_styles,
                        "all_style_names": all_style_names,
                        "all_arc_idxs": all_arc_idxs,
                        "all_cfg_ids": all_cfg_ids,
                        "retries": retries, "done_keys": done_keys,
                    }, f, protocol=4)

    pbar.close()
    env.close()

    # ── save ──────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_safety    = sum(1 for s in all_styles if s == STYLE_SAFETY)
    n_grounding = sum(1 for s in all_styles if s == STYLE_GROUNDING)
    n_left      = sum(1 for l in all_labels if l == "left")
    n_right     = sum(1 for l in all_labels if l == "right")

    metadata = {
        "description": "Safety + grounding demos for TwoBlockPick",
        "cube_half": _CUBE_HALF,
        "cube_mass": _CUBE_MASS,
        "cube_friction": _CUBE_FRICTION,
        "n_demos": total,
        "n_safety": n_safety,
        "n_grounding": n_grounding,
        "n_left": n_left, "n_right": n_right,
        "style_labels": {"safety": 3, "grounding": 4},
        "safety_z_range_cm": [66, 74],
        "grounding_waypoint": [0.48, 0.03, 0.421],
    }

    if ckpt_path.exists():
        ckpt_path.unlink()

    np.savez_compressed(
        str(out),
        obs             = np.stack(all_obs),
        actions         = np.stack(all_act),
        episode_lengths = np.array(all_lens),
        labels          = np.array(all_labels),
        style_labels    = np.array(all_styles),
        style_names     = np.array(all_style_names),
        arc_idxs        = np.array(all_arc_idxs),
        config_ids      = np.array(all_cfg_ids),
        metadata_json   = json.dumps(metadata),
    )

    print(f"\n{'='*62}")
    print(f"  Saved {total} demos to: {out}")
    print(f"  safety    : {n_safety}")
    print(f"  grounding : {n_grounding}")
    print(f"  left/right: {n_left}/{n_right}")
    print(f"  retries   : {retries}")
    print(f"  videos    : {vid_dir}/")
    print(f"{'='*62}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Collect 200 safety+grounding demos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="data/demos/demos_safety_grounding.npz")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    collect(seed=args.seed, out_path=args.out, preview=args.preview)


if __name__ == "__main__":
    main()
