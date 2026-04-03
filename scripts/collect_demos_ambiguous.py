#!/usr/bin/env python
"""Collect ambiguous-intent demonstrations for TwoBlockPick.

Grounded in:
  - Dragan & Srinivasa (HRI 2013): legibility vs predictability vs ambiguity
  - SLOT-V (RO-MAN 2022): legible manipulation with Franka in simulation
  - Style-Conditioned Diffusion Policy (ICSR 2026): ambiguity-driven style
    switching in diffusion policies — directly aligned with VLM steering goal

Structure
─────────
  400 new demos  (does NOT touch existing demos.npz)
  10 block configs × 40 eps/config = 400
    Per config:
      10 neutral-L    quadratic Bézier, Y-neutral CP → lateral commit at end
      10 neutral-R
      10 deceptive-L  cubic Bézier: first sweeps RIGHT (feint), then LEFT (true pick)
      10 deceptive-R  cubic Bézier: first sweeps LEFT  (feint), then RIGHT (true pick)

Trajectory types
─────────────────
  NEUTRAL (quadratic Bézier):
    B(t) = (1-t)²·P0 + 2(1-t)t·[cp_x, 0, cp_z] + t²·P2(above cube)
    → No lateral bias until very late in the arc. Observer cannot tell
      which block is the target until the descent phase begins.

  DECEPTIVE (cubic Bézier, feint→commit):
    B(t) = (1-t)³·P0 + 3(1-t)²t·P1_feint + 3(1-t)t²·P2_commit + t³·P3
    P1_feint : toward WRONG block (opposite target side)
    P2_commit: toward CORRECT block
    → At t≈0.3 robot is clearly heading toward wrong block.
      At t≈0.7 it has committed to the correct block.
      Maximally ambiguous at t≈0.5.

Grasp stability improvements (smaller cube, higher friction):
  cube_half = 0.015  (3-cm side, was 4-cm — fits cleaner in Panda gripper)
  cube_mass = 0.08   (heavier — less tip under gripper impulse)
  cube_lateral_friction = 2.5  (was 1.5 — less slip)
  spinning/rolling friction   = 0.001 (was 0.01 — stable hold)

Saved dataset:
  data/demos/demos_ambiguous.npz
  data/demos/demo_videos_ambiguous/

Usage:
    # Preview 4 videos (one per type) — APPROVE before full run
    python scripts/collect_demos_ambiguous.py --preview

    # Full collection
    python scripts/collect_demos_ambiguous.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM

# ── constants ──────────────────────────────────────────────────────────
_EE_HOME = np.array([0.40, 0.0, 0.55], dtype=np.float32)

_N_ARC_PTS     = 200   # waypoints along the Bézier arc
_N_DESCENT_PTS = 30    # straight descent to cube surface
_N_GRIP_STEPS  = 40    # gradual gripper close (+1 → -1)
_N_LIFT_PTS    = 30    # slow lift after grasp
_EPISODE_LEN   = 400   # max steps (same as legible demos)

# New cube settings (grasp stability — does NOT affect existing demos)
_CUBE_HALF_NEW     = 0.015   # 3-cm half → 6-cm cube  (was 0.02)
_CUBE_MASS_NEW     = 0.08    # kg  (was 0.05)
_CUBE_FRICTION_NEW = 2.5     # lateral  (was 1.5)


# ── 10 block-position configurations (same as legible demos) ───────────

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

    assert len(configs) == 10
    return configs


# ── 10 neutral-arc variations ──────────────────────────────────────────
# Quadratic Bézier with Y-neutral control point.
# No lateral information until the arc ends and descent begins.

def _build_neutral_variations(n: int = 10) -> list[dict]:
    """Return n neutral arc specs (control point always at y=0)."""
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Control point moves forward and upward as index increases
        cp_x = float(np.interp(t, [0, 1], [0.42, 0.50]))
        cp_z = float(np.interp(t, [0, 1], [0.60, 0.66]))
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.12]))
        variations.append(dict(cp_x=cp_x, cp_z=cp_z, approach_h=approach_h))
    return variations


# ── 10 deceptive-arc variations ────────────────────────────────────────
# Cubic Bézier: P1 on WRONG side (feint), P2 on CORRECT side (commit).
# Sign is applied per target at expert construction time.

def _build_deceptive_variations(n: int = 10) -> list[dict]:
    """Return n deceptive arc specs.  feint_mag and commit_mag are magnitudes;
    signs (wrong/correct side) applied per target by the caller."""
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # feint_mag: how far toward wrong block  (5 cm → 20 cm)
        feint_mag = float(np.interp(t, [0, 1], [0.05, 0.20]))
        # commit_mag: how far toward correct block in P2 (slightly past cube center)
        commit_mag = float(np.interp(t, [0, 1], [0.04, 0.07]))
        # P1 (feint) position
        feint_x = float(np.interp(t, [0, 1], [0.42, 0.44]))
        feint_z = float(np.interp(t, [0, 1], [0.60, 0.65]))
        # P2 (commit) position
        commit_x = float(np.interp(t, [0, 1], [0.47, 0.49]))
        commit_z = float(np.interp(t, [0, 1], [0.55, 0.50]))
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.11]))
        variations.append(dict(
            feint_mag=feint_mag, commit_mag=commit_mag,
            feint_x=feint_x, feint_z=feint_z,
            commit_x=commit_x, commit_z=commit_z,
            approach_h=approach_h,
        ))
    return variations


# ── helper: cubic Bézier ───────────────────────────────────────────────

def _cubic_bezier(t: float,
                  P0: np.ndarray, P1: np.ndarray,
                  P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
    return ((1 - t) ** 3 * P0
            + 3 * (1 - t) ** 2 * t * P1
            + 3 * (1 - t) * t ** 2 * P2
            + t ** 3 * P3)


# ── scripted expert for ambiguous intent ───────────────────────────────

class AmbiguousExpert:
    """Picks a specific cube via a neutral or deceptive approach arc.

    NEUTRAL mode (quadratic Bézier, cp_y = 0):
      Robot moves forward without leaning left or right.
      Target block identity only becomes clear during descent.

    DECEPTIVE mode (cubic Bézier, wrong-side feint → correct-side commit):
      t ∈ [0, 0.4]: robot clearly moves toward WRONG block  (feint)
      t ∈ [0.4, 0.6]: maximally ambiguous (near midpoint)
      t ∈ [0.6, 1.0]: robot commits to CORRECT block
      Descent always goes straight to the correct cube centre.

    Phases (same structure as legible expert):
      0 – 200-waypoint Bézier arc (neutral or deceptive)
      1 – 30-waypoint straight descent to cube surface
      2 – 40-step gradual gripper close
      3 – 30-waypoint slow lift
      4 – Hold
    """

    def __init__(
        self,
        target: str,
        mode: str,                             # "neutral" or "deceptive"
        action_scale: float = 0.05,
        neutral_var: dict | None = None,       # from _build_neutral_variations
        deceptive_var: dict | None = None,     # from _build_deceptive_variations
        rng: np.random.Generator | None = None,
    ) -> None:
        assert target in ("left", "right")
        assert mode in ("neutral", "deceptive")
        self.target = target
        self.mode = mode
        self.scale = action_scale
        self._nvar = neutral_var or {}
        self._dvar = deceptive_var or {}
        self._rng = rng or np.random.default_rng()

        self.phase = 0
        self.wait = 0
        self._arc_wps: np.ndarray | None = None
        self._descent_wps: np.ndarray | None = None
        self._lift_wps: np.ndarray | None = None
        self._wp_idx = 0

    def reset(self) -> None:
        self.phase = 0
        self.wait = 0
        self._arc_wps = None
        self._descent_wps = None
        self._lift_wps = None
        self._wp_idx = 0

    # ── waypoint construction ─────────────────────────────────────────

    def _build_waypoints(self, cube_pos: np.ndarray) -> None:
        P3 = cube_pos.copy()

        if self.mode == "neutral":
            approach_h = self._nvar.get("approach_h", 0.09)
            P3[2] += approach_h
            P0 = _EE_HOME.copy()
            P1 = np.array([self._nvar["cp_x"], 0.0, self._nvar["cp_z"]],
                          dtype=np.float32)
            P2 = P3.copy()
            ts = np.linspace(0, 1, _N_ARC_PTS)
            self._arc_wps = np.array([
                (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
                for t in ts
            ], dtype=np.float32)

        else:  # deceptive
            approach_h = self._dvar.get("approach_h", 0.08)
            P3[2] += approach_h
            P0 = _EE_HOME.copy()

            # Sign: +1 = left side, -1 = right side
            target_sign = 1.0 if self.target == "left" else -1.0
            feint_sign  = -target_sign  # opposite side for feint

            P1 = np.array([
                self._dvar["feint_x"],
                feint_sign * self._dvar["feint_mag"],
                self._dvar["feint_z"],
            ], dtype=np.float32)

            P2 = np.array([
                self._dvar["commit_x"],
                target_sign * self._dvar["commit_mag"],
                self._dvar["commit_z"],
            ], dtype=np.float32)

            ts = np.linspace(0, 1, _N_ARC_PTS)
            self._arc_wps = np.array([
                _cubic_bezier(t, P0, P1, P2, P3)
                for t in ts
            ], dtype=np.float32)

        # Descent: above cube → grasp height (ALWAYS toward correct cube)
        above = P3.copy()
        grasp = cube_pos.copy()
        grasp[2] += 0.005
        self._descent_wps = np.array([
            above + s * (grasp - above)
            for s in np.linspace(0, 1, _N_DESCENT_PTS)
        ], dtype=np.float32)

        # Lift: grasp → high
        lift_top = grasp.copy()
        lift_top[2] = 0.60
        self._lift_wps = np.array([
            grasp + s * (lift_top - grasp)
            for s in np.linspace(0, 1, _N_LIFT_PTS)
        ], dtype=np.float32)

        self._wp_idx = 1  # skip wp 0 (= robot already at home/previous pos)

    # ── action loop ───────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> np.ndarray:
        ee_pos  = obs[0:3]
        cube_pos = obs[8:11].copy() if self.target == "left" else obs[15:18].copy()

        if self._arc_wps is None:
            self._build_waypoints(cube_pos)

        action = np.zeros(ACT_DIM, dtype=np.float32)

        if self.phase == 0:
            tgt = self._arc_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=1.0, speed=0.45)
            if np.linalg.norm(ee_pos - tgt) < 0.012:
                self._wp_idx += 1
                if self._wp_idx >= _N_ARC_PTS:
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
            action[4] = 1.0 - 2.0 * t_frac   # +1 → -1
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

    def _goto(self, cur: np.ndarray, tgt: np.ndarray,
              grip: float, speed: float = 1.0) -> np.ndarray:
        delta = (tgt - cur) / self.scale
        a = np.zeros(ACT_DIM, dtype=np.float32)
        a[:3] = np.clip(delta * speed, -1, 1)
        a[4] = grip
        return a


# ── arc-class label for analysis ───────────────────────────────────────

def _arc_class(arc_type: str, var_idx: int) -> str:
    """Return a human-readable arc class label for post-hoc classification.

    neutral_0-4   : neutral, gentle center approach
    neutral_5-9   : neutral, stronger forward sweep
    deceptive_mild (var 0-4)   : feint 5-12 cm toward wrong block
    deceptive_strong (var 5-9) : feint 12-20 cm toward wrong block
    """
    if arc_type == "neutral":
        return "neutral_early" if var_idx < 5 else "neutral_late"
    else:
        return "deceptive_mild" if var_idx < 5 else "deceptive_strong"


# ── collection loop ────────────────────────────────────────────────────

def collect(seed: int = 0,
            out_path: str = "data/demos/demos_ambiguous.npz",
            preview: bool = False) -> None:
    """Collect 400 ambiguous demos (200L + 200R, 200 neutral + 200 deceptive).

    With preview=True: only run 4 episodes (one per type), save videos only,
    do NOT write .npz.  Approve videos before launching full collection.
    """
    block_configs      = _build_block_configs()          # 10 configs
    neutral_vars       = _build_neutral_variations(10)   # 10 neutral arcs
    deceptive_vars     = _build_deceptive_variations(10) # 10 deceptive arcs

    n_configs     = len(block_configs)     # 10
    eps_per_config = 40                   # 10nL + 10nR + 10dL + 10dR
    total         = n_configs * eps_per_config  # 400

    vid_dir = Path(out_path).parent / "demo_videos_ambiguous"
    vid_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TwoBlockPick – Ambiguous-Intent Demo Collection")
    print("=" * 60)
    if preview:
        print("  *** PREVIEW MODE: 4 videos only, no .npz written ***")
    print(f"  Plan: {n_configs} configs × {eps_per_config} eps = {total} demos")
    print(f"  Trajectory types: neutral (10 variations) + deceptive (10 variations)")
    print(f"  Cube: half={_CUBE_HALF_NEW}m ({_CUBE_HALF_NEW*200:.0f}mm side), "
          f"mass={_CUBE_MASS_NEW}kg, friction={_CUBE_FRICTION_NEW}")
    print(f"  Videos → {vid_dir}")
    if not preview:
        print(f"  Dataset → {out_path}")
    print()

    env = TwoBlockPickEnv(
        render=False,
        cube_jitter=0.0,
        cube_half=_CUBE_HALF_NEW,
        cube_mass=_CUBE_MASS_NEW,
        cube_lateral_friction=_CUBE_FRICTION_NEW,
        episode_length=_EPISODE_LEN,
    )
    max_T = env.episode_length
    rng   = np.random.default_rng(seed)

    all_obs:       list[np.ndarray] = []
    all_act:       list[np.ndarray] = []
    all_lens:      list[int]        = []
    all_labels:    list[str]        = []
    all_arc_types: list[str]        = []
    all_arc_idxs:  list[int]        = []
    all_arc_class: list[str]        = []
    all_cfg_ids:   list[int]        = []

    successes = 0
    left_ok = right_ok = 0
    retries  = 0

    # ── preview: one demo of each type ────────────────────────────────
    if preview:
        preview_cases = [
            ("neutral",   "left",  4),   # neutral-L (middle variation)
            ("neutral",   "right", 4),   # neutral-R
            ("deceptive", "left",  4),   # deceptive-L mild-strong boundary
            ("deceptive", "right", 9),   # deceptive-R strongest feint
        ]
        print(f"Running {len(preview_cases)} preview episodes ...\n")
        cfg = block_configs[0]  # use default block positions (no offset)
        env.reset(seed=0)
        env.set_cube_offsets(
            left_dx=cfg["ldx"], left_dy=cfg["ldy"],
            right_dx=cfg["rdx"], right_dy=cfg["rdy"])
        obs = env._get_obs()

        for arc_type, target, var_idx in preview_cases:
            env.reset(seed=0)
            env.set_cube_offsets(
                left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                right_dx=cfg["rdx"], right_dy=cfg["rdy"])
            obs = env._get_obs()

            if arc_type == "neutral":
                expert = AmbiguousExpert(
                    target, "neutral",
                    action_scale=env.action_scale_pos,
                    neutral_var=neutral_vars[var_idx],
                    rng=np.random.default_rng(42))
            else:
                expert = AmbiguousExpert(
                    target, "deceptive",
                    action_scale=env.action_scale_pos,
                    deceptive_var=deceptive_vars[var_idx],
                    rng=np.random.default_rng(42))
            expert.reset()

            vname = f"preview_{arc_type}_{target}_var{var_idx:02d}.mp4"
            vpath = vid_dir / vname
            env.record_video(str(vpath))

            result = None
            for t in range(max_T):
                action = expert.act(obs)
                result = env.step(action)
                obs = result.obs
                if result.done:
                    break

            env.stop_video()

            picked = ("left"  if result.info["success_left"]  > 0.5 else
                      "right" if result.info["success_right"] > 0.5 else "none")
            feint_info = (f"feint={deceptive_vars[var_idx]['feint_mag']*100:.0f}mm"
                          if arc_type == "deceptive"
                          else f"cp_x={neutral_vars[var_idx]['cp_x']:.2f}")
            print(f"  {arc_type:10s}  {target:5s}  var={var_idx}  "
                  f"{feint_info}  →  picked={picked}  "
                  f"video={vname}")

        env.close()
        print(f"\nPreview done.  Check videos in {vid_dir}/")
        print("If satisfied, run WITHOUT --preview to collect all 400 demos.")
        return

    # ── full collection ────────────────────────────────────────────────
    pbar = tqdm(total=total, desc="collecting ambiguous demos")

    for cfg_id, cfg in enumerate(block_configs):

        # Build balanced schedule per config:
        # 10 neutral-L, 10 neutral-R, 10 deceptive-L, 10 deceptive-R
        schedule: list[tuple[str, str, int]] = []  # (arc_type, target, var_idx)
        for vi in range(len(neutral_vars)):
            schedule.append(("neutral",   "left",  vi))
            schedule.append(("neutral",   "right", vi))
        for vi in range(len(deceptive_vars)):
            schedule.append(("deceptive", "left",  vi))
            schedule.append(("deceptive", "right", vi))
        rng.shuffle(schedule)

        for arc_type, target, var_idx in schedule:
            ep_rng_seed = int(rng.integers(0, 2 ** 31))

            attempt = 0
            while True:
                env.reset(seed=0)
                env.set_cube_offsets(
                    left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                    right_dx=cfg["rdx"], right_dy=cfg["rdy"])
                obs = env._get_obs()

                if arc_type == "neutral":
                    expert = AmbiguousExpert(
                        target, "neutral",
                        action_scale=env.action_scale_pos,
                        neutral_var=neutral_vars[var_idx],
                        rng=np.random.default_rng(ep_rng_seed + attempt))
                else:
                    expert = AmbiguousExpert(
                        target, "deceptive",
                        action_scale=env.action_scale_pos,
                        deceptive_var=deceptive_vars[var_idx],
                        rng=np.random.default_rng(ep_rng_seed + attempt))
                expert.reset()

                vname = (f"cfg{cfg_id:02d}_{arc_type[:3]}_{target}"
                         f"_var{var_idx:02d}.mp4")
                vpath = vid_dir / vname
                env.record_video(str(vpath))

                obs_buf = np.zeros((max_T, OBS_DIM), dtype=np.float32)
                act_buf = np.zeros((max_T, ACT_DIM), dtype=np.float32)
                ep_len  = 0
                result  = None

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

                success = (result.info["success_left"]  > 0.5 or
                           result.info["success_right"] > 0.5)
                if success:
                    break   # keep this episode

                attempt += 1
                retries  += 1
                if vpath.exists():
                    vpath.unlink()   # delete failed attempts

            all_obs.append(obs_buf)
            all_act.append(act_buf)
            all_lens.append(ep_len)
            all_labels.append(target)
            all_arc_types.append(arc_type)
            all_arc_idxs.append(var_idx)
            all_arc_class.append(_arc_class(arc_type, var_idx))
            all_cfg_ids.append(cfg_id)

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

    metadata = {
        "description": "Ambiguous-intent demos: neutral + deceptive arcs",
        "cube_half": _CUBE_HALF_NEW,
        "cube_mass": _CUBE_MASS_NEW,
        "cube_lateral_friction": _CUBE_FRICTION_NEW,
        "n_demos": total,
        "n_left": left_ok,
        "n_right": right_ok,
        "n_neutral": sum(1 for t in all_arc_types if t == "neutral"),
        "n_deceptive": sum(1 for t in all_arc_types if t == "deceptive"),
        "arc_classes": sorted(set(all_arc_class)),
        "references": [
            "Dragan & Srinivasa (HRI 2013) — legibility vs ambiguity",
            "SLOT-V, Wallkotter et al. (RO-MAN 2022) — legible manipulation",
            "Style-Conditioned Diffusion Policy (ICSR 2026)",
        ],
    }

    np.savez_compressed(
        str(out),
        obs              = np.stack(all_obs),               # (N, T, 22)
        actions          = np.stack(all_act),               # (N, T, 5)
        episode_lengths  = np.array(all_lens),              # (N,)
        labels           = np.array(all_labels),            # (N,)  "left"/"right"
        arc_types        = np.array(all_arc_types),         # (N,)  "neutral"/"deceptive"
        arc_idxs         = np.array(all_arc_idxs),          # (N,)  0-9
        arc_classes      = np.array(all_arc_class),         # (N,)  label
        config_ids       = np.array(all_cfg_ids),           # (N,)
        metadata_json    = json.dumps(metadata),
    )

    # ── summary ───────────────────────────────────────────────────────
    n_neutral   = sum(1 for t in all_arc_types if t == "neutral")
    n_deceptive = sum(1 for t in all_arc_types if t == "deceptive")

    print(f"\n{'='*60}")
    print(f"  ✓ saved {total} ambiguous demos to {out}")
    print(f"  total left / right   : {left_ok} / {right_ok}")
    print(f"  neutral / deceptive  : {n_neutral} / {n_deceptive}")
    print(f"  arc classes          :")
    for cls in sorted(set(all_arc_class)):
        cnt = sum(1 for c in all_arc_class if c == cls)
        print(f"    {cls:25s}: {cnt}")
    print(f"  retries needed       : {retries}")
    print(f"  videos saved to      : {vid_dir}/")
    print(f"{'='*60}")
    print()
    print("Next steps:")
    print("  1. Train:  python scripts/train_diffusion_policy.py "
          "--config configs/train.yaml  (point demo_path to demos_ambiguous.npz)")
    print("  2. Eval:   python scripts/eval_multimodality.py "
          "--ckpt runs/latest_ambiguous/ckpt.pt ...")
    print("  3. Classify produced arcs by direction of first 40% of trajectory")


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect 400 ambiguous-intent demos (neutral + deceptive arcs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",  type=str,
                    default="data/demos/demos_ambiguous.npz")
    ap.add_argument("--preview", action="store_true",
                    help="Run 4 preview videos only (one per type). "
                         "No .npz written. Approve before full run.")
    args = ap.parse_args()
    collect(seed=args.seed, out_path=args.out, preview=args.preview)


if __name__ == "__main__":
    main()
