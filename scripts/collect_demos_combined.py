#!/usr/bin/env python
"""Collect combined demonstrations for TwoBlockPick (legible + neutral + deceptive).

Research grounding:
  - Dragan & Srinivasa (HRI 2013): legibility = early lateral commitment
    toward goal; ambiguity = trajectory compatible with multiple goals.
  - SLOT-V (RO-MAN 2022): supervised legible motion on Franka in simulation.
  - Style-Conditioned Diffusion Policy (arXiv:2603.16368, ICSR 2026):
    base policy trained on mixed styles, steered at inference via conditioning.

Dataset recipe  (400 demos, ~6 hours CPU)
─────────────────────────────────────────
  10 block configs × 40 demos/config = 400

  Per config:
    10 legible-L   quadratic Bézier, CP sweeps TOWARD left block
    10 legible-R   quadratic Bézier, CP sweeps TOWARD right block
     5 neutral-L   quadratic Bézier, CP at y=0  (no lateral bias)
     5 neutral-R
     5 deceptive-L cubic Bézier: feints RIGHT → commits LEFT
     5 deceptive-R cubic Bézier: feints LEFT  → commits RIGHT
    ─────────────────────────────────────────────────────────
    40 per config × 10 configs = 400 total

Style distribution:
  legible   : 200 demos  (50%)  ← dominant; policy's default output
  neutral   : 100 demos  (25%)  ← ambiguous but passive
  deceptive : 100 demos  (25%)  ← ambiguous and actively misleading

Style labels stored in .npz (for eval arc-classifier + VLM steering):
  0 = legible
  1 = neutral
  2 = deceptive

Trajectory shapes:
  LEGIBLE  — quadratic Bézier, lateral control point ON the target side.
    B(t) = (1-t)²P0 + 2(1-t)t·[cp_x, ±cp_y, cp_z] + t²·above_target
    cp_y_mag ranges from ±5 cm (gentle) to ±28 cm (dramatic).
    Sign matches target: left pick → cp_y > 0, right pick → cp_y < 0.
    Observer sees lateral commitment by t≈0.3  → legible.

  NEUTRAL  — quadratic Bézier, control point strictly at y=0.
    B(t) = (1-t)²P0 + 2(1-t)t·[cp_x, 0, cp_z] + t²·above_target
    No lateral signal until descent begins  → ambiguous.

  DECEPTIVE — cubic Bézier, P1 on WRONG side, P2 on CORRECT side.
    B(t) = (1-t)³P0 + 3(1-t)²t·P1_feint + 3(1-t)t²·P2_commit + t³·P3
    At t≈0.3: moving toward wrong block.
    At t≈0.5: maximally ambiguous.
    At t≈0.7: committed to correct block.
    Descent always targets correct cube  → pick succeeds.

Cube settings (all demos; does NOT affect existing demos.npz):
  cube_half = 0.015 m  (3 cm → smaller, fits Panda gripper better)
  cube_mass = 0.08 kg  (heavier → less tipping under gripper impulse)
  cube_lateral_friction = 2.5  (vs 1.5 — less slip)

Output:
  data/demos/demos_combined.npz
  data/demos/demo_videos_combined/  (400 videos, one per demo)

Usage:
  # Preview 3 videos (one per style) — check before full run
  py scripts/collect_demos_combined.py --preview

  # Full 400-demo collection
  py scripts/collect_demos_combined.py --seed 0
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

# ── constants ──────────────────────────────────────────────────────────
_EE_HOME = np.array([0.40, 0.0, 0.55], dtype=np.float32)

_N_ARC_PTS     = 200   # Bézier arc waypoints
_N_DESCENT_PTS = 30    # straight descent to cube surface
_N_GRIP_STEPS  = 40    # gradual gripper close (+1 → -1)
_N_LIFT_PTS    = 30    # slow lift after grasp
_EPISODE_LEN   = 400   # max env steps per episode

# Cube settings (stable grasp with smaller cube)
_CUBE_HALF     = 0.015   # 3-cm half-size  (was 0.02 in original demos)
_CUBE_MASS     = 0.08    # kg
_CUBE_FRICTION = 2.5     # lateral friction

# Style label integers (stored in .npz)
STYLE_LEGIBLE   = 0
STYLE_NEUTRAL   = 1
STYLE_DECEPTIVE = 2

CKPT_EVERY = 20   # save checkpoint every N demos to survive interruption


# ── block-position configurations ─────────────────────────────────────

def _build_block_configs() -> list[dict]:
    """10 configs: small positional offsets for placement robustness."""
    configs: list[dict] = []

    # Type A – both blocks shifted symmetrically (4 configs)
    for dx in [-0.005, 0.0, 0.005]:
        configs.append(dict(tag="both", ldx=dx, ldy=0.0, rdx=dx, rdy=0.0))
    configs.append(dict(tag="both", ldx=0.0, ldy=0.004, rdx=0.0, rdy=-0.004))

    # Type B – only left block shifted (3 configs)
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(tag="left", ldx=dx, ldy=dy, rdx=0.0, rdy=0.0))

    # Type C – only right block shifted (3 configs)
    for dx, dy in [(-0.005, 0.0), (0.005, 0.0), (0.0, 0.004)]:
        configs.append(dict(tag="right", ldx=0.0, ldy=0.0, rdx=dx, rdy=dy))

    assert len(configs) == 10
    return configs


# ── arc-variation builders ─────────────────────────────────────────────

def _build_legible_variations(n: int = 10) -> list[dict]:
    """n quadratic Bézier arcs sweeping toward the target side.
    cp_y_mag is ALWAYS positive; sign applied per target at call time.
    Magnitude spans ±5 cm (gentle) → ±28 cm (dramatic)."""
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        cp_y_mag = float(np.interp(t, [0, 1], [0.05, 0.28]))
        frac = (cp_y_mag - 0.05) / (0.28 - 0.05)
        cp_z = float(0.56 + 0.12 * frac)   # 0.56 → 0.68
        cp_x = float(0.38 - 0.10 * frac)   # 0.38 → 0.28
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.12]))
        variations.append(dict(cp_x=cp_x, cp_y_mag=cp_y_mag,
                               cp_z=cp_z, approach_h=approach_h))
    return variations


def _build_neutral_variations(n: int = 5) -> list[dict]:
    """n quadratic Bézier arcs with cp_y = 0 — no lateral information."""
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        cp_x = float(np.interp(t, [0, 1], [0.42, 0.50]))
        cp_z = float(np.interp(t, [0, 1], [0.60, 0.66]))
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.12]))
        variations.append(dict(cp_x=cp_x, cp_z=cp_z, approach_h=approach_h))
    return variations


def _build_deceptive_variations(n: int = 5) -> list[dict]:
    """n cubic Bézier arcs: P1 toward wrong block (feint), P2 toward correct.
    feint_mag signs applied per target at call time."""
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        feint_mag  = float(np.interp(t, [0, 1], [0.05, 0.20]))
        commit_mag = float(np.interp(t, [0, 1], [0.04, 0.07]))
        feint_x    = float(np.interp(t, [0, 1], [0.42, 0.44]))
        feint_z    = float(np.interp(t, [0, 1], [0.60, 0.65]))
        commit_x   = float(np.interp(t, [0, 1], [0.47, 0.49]))
        commit_z   = float(np.interp(t, [0, 1], [0.55, 0.50]))
        approach_h = float(np.interp(t, [0, 1], [0.06, 0.11]))
        variations.append(dict(
            feint_mag=feint_mag, commit_mag=commit_mag,
            feint_x=feint_x, feint_z=feint_z,
            commit_x=commit_x, commit_z=commit_z,
            approach_h=approach_h,
        ))
    return variations


# ── helpers ────────────────────────────────────────────────────────────

def _cubic_bezier(t: float,
                  P0: np.ndarray, P1: np.ndarray,
                  P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
    return ((1 - t) ** 3 * P0
            + 3 * (1 - t) ** 2 * t * P1
            + 3 * (1 - t) * t ** 2 * P2
            + t ** 3 * P3)


# ── scripted experts ───────────────────────────────────────────────────

class _ExpertBase:
    """Shared phase logic for all three expert types."""

    def __init__(self, target: str, action_scale: float = 0.05) -> None:
        assert target in ("left", "right")
        self.target = target
        self.scale  = action_scale
        self.phase  = 0
        self.wait   = 0
        self._arc_wps:     np.ndarray | None = None
        self._descent_wps: np.ndarray | None = None
        self._lift_wps:    np.ndarray | None = None
        self._wp_idx = 0

    def reset(self) -> None:
        self.phase = 0
        self.wait  = 0
        self._arc_wps = self._descent_wps = self._lift_wps = None
        self._wp_idx  = 0

    def _cube_pos(self, obs: np.ndarray) -> np.ndarray:
        return obs[8:11].copy() if self.target == "left" else obs[15:18].copy()

    def _build_descent_and_lift(self, above: np.ndarray,
                                 cube_pos: np.ndarray) -> None:
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

    def _build_waypoints(self, cube_pos: np.ndarray) -> None:
        """Override in subclass to set self._arc_wps and call
        self._build_descent_and_lift(above, cube_pos)."""
        raise NotImplementedError

    def act(self, obs: np.ndarray) -> np.ndarray:
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
                    self.phase = 1
                    self._wp_idx = 1

        elif self.phase == 1:
            tgt = self._descent_wps[self._wp_idx]
            action = self._goto(ee_pos, tgt, grip=1.0, speed=0.35)
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_DESCENT_PTS:
                    self.phase = 2
                    self.wait  = 0

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
        a[4]  = grip
        return a


class LegibleExpert(_ExpertBase):
    """Quadratic Bézier sweeping TOWARD the target side.
    cp_y_mag > 0; sign matches target (left→+, right→-)."""

    def __init__(self, target: str, action_scale: float, var: dict) -> None:
        super().__init__(target, action_scale)
        self._var = var

    def _build_waypoints(self, cube_pos: np.ndarray) -> None:
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
            (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
            for t in ts
        ], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


class NeutralExpert(_ExpertBase):
    """Quadratic Bézier with cp_y = 0 — no lateral signal."""

    def __init__(self, target: str, action_scale: float, var: dict) -> None:
        super().__init__(target, action_scale)
        self._var = var

    def _build_waypoints(self, cube_pos: np.ndarray) -> None:
        P0    = _EE_HOME.copy()
        P1    = np.array([self._var["cp_x"], 0.0, self._var["cp_z"]],
                         dtype=np.float32)
        above = cube_pos.copy()
        above[2] += self._var["approach_h"]
        P2 = above.copy()

        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array([
            (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
            for t in ts
        ], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


class DeceptiveExpert(_ExpertBase):
    """Cubic Bézier: P1 toward WRONG block (feint), P2 toward CORRECT block."""

    def __init__(self, target: str, action_scale: float, var: dict) -> None:
        super().__init__(target, action_scale)
        self._var = var

    def _build_waypoints(self, cube_pos: np.ndarray) -> None:
        target_sign = 1.0 if self.target == "left" else -1.0
        feint_sign  = -target_sign

        P0 = _EE_HOME.copy()
        P1 = np.array([self._var["feint_x"],
                       feint_sign * self._var["feint_mag"],
                       self._var["feint_z"]], dtype=np.float32)
        P2 = np.array([self._var["commit_x"],
                       target_sign * self._var["commit_mag"],
                       self._var["commit_z"]], dtype=np.float32)
        above = cube_pos.copy()
        above[2] += self._var["approach_h"]
        P3 = above.copy()

        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array([
            _cubic_bezier(t, P0, P1, P2, P3) for t in ts
        ], dtype=np.float32)
        self._build_descent_and_lift(above, cube_pos)
        self._wp_idx = 1


# ── expert factory ─────────────────────────────────────────────────────

def _make_expert(style: int, target: str, action_scale: float,
                 var: dict) -> _ExpertBase:
    if style == STYLE_LEGIBLE:
        return LegibleExpert(target, action_scale, var)
    elif style == STYLE_NEUTRAL:
        return NeutralExpert(target, action_scale, var)
    else:
        return DeceptiveExpert(target, action_scale, var)


# ── one episode ────────────────────────────────────────────────────────

def _run_episode(env: TwoBlockPickEnv, expert: _ExpertBase,
                 vpath: Path | None) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Run one episode. Returns (obs_buf, act_buf, ep_len, success)."""
    max_T = env.episode_length
    obs   = env._get_obs()

    if vpath is not None:
        env.record_video(str(vpath))

    obs_buf = np.zeros((max_T, OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((max_T, ACT_DIM), dtype=np.float32)
    ep_len  = 0
    result  = None

    for t in range(max_T):
        action       = expert.act(obs)
        obs_buf[t]   = obs
        act_buf[t]   = action
        result       = env.step(action)
        obs          = result.obs
        ep_len       = t + 1
        if result.done:
            break

    if vpath is not None:
        env.stop_video()

    success = (result.info["success_left"]  > 0.5 or
               result.info["success_right"] > 0.5)
    return obs_buf, act_buf, ep_len, success


# ── collection ─────────────────────────────────────────────────────────

def collect(seed: int = 0,
            out_path: str = "data/demos/demos_combined.npz",
            preview: bool = False) -> None:
    """Collect 400 mixed-style demos.

    preview=True → runs 3 episodes (one per style), saves 3 videos only,
    does NOT write .npz.  Inspect videos first, then run without --preview.
    """
    block_configs   = _build_block_configs()          # 10
    legible_vars    = _build_legible_variations(10)   # 10 arcs (±5→±28 cm)
    neutral_vars    = _build_neutral_variations(5)    #  5 arcs (forward only)
    deceptive_vars  = _build_deceptive_variations(5)  #  5 arcs (feint→commit)

    n_configs      = len(block_configs)   # 10
    eps_per_config = 40                   # 20 leg + 10 neu + 10 dec
    total          = n_configs * eps_per_config  # 400

    vid_dir = Path(out_path).parent / "demo_videos_combined"
    vid_dir.mkdir(parents=True, exist_ok=True)

    style_names = {STYLE_LEGIBLE: "legible",
                   STYLE_NEUTRAL: "neutral",
                   STYLE_DECEPTIVE: "deceptive"}

    print("=" * 62)
    print("  TwoBlockPick — Combined Demo Collection")
    print("=" * 62)
    if preview:
        print("  *** PREVIEW MODE: 3 videos only, no .npz written ***")
    print(f"  Recipe per config: 10 leg-L + 10 leg-R + "
          f"5 neu-L + 5 neu-R + 5 dec-L + 5 dec-R = 40")
    print(f"  Total: {n_configs} configs × {eps_per_config} = {total} demos")
    print(f"  Style split: 200 legible (50%) + 100 neutral (25%) "
          f"+ 100 deceptive (25%)")
    print(f"  Cube: half={_CUBE_HALF}m, mass={_CUBE_MASS}kg, "
          f"friction={_CUBE_FRICTION}")
    print(f"  Videos → {vid_dir}")
    if not preview:
        print(f"  Dataset → {out_path}")
    print()

    env = TwoBlockPickEnv(
        render=False,
        cube_jitter=0.0,
        cube_half=_CUBE_HALF,
        cube_mass=_CUBE_MASS,
        cube_lateral_friction=_CUBE_FRICTION,
        episode_length=_EPISODE_LEN,
    )
    rng = np.random.default_rng(seed)

    # ── preview ────────────────────────────────────────────────────────
    if preview:
        preview_cases = [
            (STYLE_LEGIBLE,   "left",  legible_vars[7]),   # strong sweep
            (STYLE_NEUTRAL,   "right", neutral_vars[2]),    # mid forward
            (STYLE_DECEPTIVE, "left",  deceptive_vars[4]), # strongest feint
        ]
        cfg = block_configs[1]  # default-position config
        print("Running 3 preview episodes ...\n")
        for style, target, var in preview_cases:
            env.reset(seed=0)
            env.set_cube_offsets(left_dx=cfg["ldx"], left_dy=cfg["ldy"],
                                 right_dx=cfg["rdx"], right_dy=cfg["rdy"])
            expert = _make_expert(style, target, env.action_scale_pos, var)
            expert.reset()

            sname  = style_names[style]
            vname  = f"preview_{sname}_{target}.mp4"
            vpath  = vid_dir / vname
            _, _, ep_len, success = _run_episode(env, expert, vpath)

            # Human-readable description of the arc used
            if style == STYLE_LEGIBLE:
                arc_info = f"cp_y={var['cp_y_mag']*100:.0f}mm toward {target}"
            elif style == STYLE_NEUTRAL:
                arc_info = f"cp_y=0, cp_x={var['cp_x']:.2f}"
            else:
                arc_info = f"feint={var['feint_mag']*100:.0f}mm wrong side"

            print(f"  {sname:10s}  {target:5s}  {arc_info:35s}  "
                  f"success={success}  ep_len={ep_len}  → {vname}")

        env.close()
        print(f"\nPreview done. Videos: {vid_dir}/")
        print("Approve then run:  py scripts/collect_demos_combined.py --seed 0")
        return

    # ── full collection ────────────────────────────────────────────────

    ckpt_path = Path(out_path).with_name(Path(out_path).stem + "_ckpt.pkl")
    done_keys: set[tuple] = set()

    # Buffers
    all_obs:    list[np.ndarray] = []
    all_act:    list[np.ndarray] = []
    all_lens:   list[int]        = []
    all_labels: list[str]        = []   # "left" / "right"
    all_styles: list[int]        = []   # 0/1/2
    all_style_names: list[str]   = []
    all_arc_idxs:    list[int]   = []
    all_cfg_ids:     list[int]   = []
    retries = 0

    # Auto-resume if checkpoint exists
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
        print(f"  [RESUME] Checkpoint found: {len(all_obs)}/{total} demos "
              f"already collected — skipping those.\n")

    pbar = tqdm(total=total, initial=len(all_obs),
                desc="collecting combined demos")

    for cfg_id, cfg in enumerate(block_configs):

        # Build the per-config schedule (40 entries):
        #   10 legible-L (arcs 0-9) + 10 legible-R (arcs 0-9)
        #    5 neutral-L  (vars 0-4) +  5 neutral-R  (vars 0-4)
        #    5 deceptive-L(vars 0-4) +  5 deceptive-R(vars 0-4)
        schedule: list[tuple[int, str, int, dict]] = []
        for vi, var in enumerate(legible_vars):
            schedule.append((STYLE_LEGIBLE, "left",  vi, var))
            schedule.append((STYLE_LEGIBLE, "right", vi, var))
        for vi, var in enumerate(neutral_vars):
            schedule.append((STYLE_NEUTRAL, "left",  vi, var))
            schedule.append((STYLE_NEUTRAL, "right", vi, var))
        for vi, var in enumerate(deceptive_vars):
            schedule.append((STYLE_DECEPTIVE, "left",  vi, var))
            schedule.append((STYLE_DECEPTIVE, "right", vi, var))

        assert len(schedule) == eps_per_config, len(schedule)
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

                expert = _make_expert(style, target, env.action_scale_pos, var)
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
                retries  += 1
                if vpath.exists():
                    vpath.unlink()   # discard failed-attempt video

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

            # periodic checkpoint — survive interruption
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

    # ── save ──────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_legible   = sum(1 for s in all_styles if s == STYLE_LEGIBLE)
    n_neutral   = sum(1 for s in all_styles if s == STYLE_NEUTRAL)
    n_deceptive = sum(1 for s in all_styles if s == STYLE_DECEPTIVE)
    n_left      = sum(1 for l in all_labels if l == "left")
    n_right     = sum(1 for l in all_labels if l == "right")

    metadata = {
        "description":  "Combined legible+neutral+deceptive demos for TwoBlockPick",
        "cube_half":    _CUBE_HALF,
        "cube_mass":    _CUBE_MASS,
        "cube_friction": _CUBE_FRICTION,
        "n_demos":      total,
        "n_legible":    n_legible,
        "n_neutral":    n_neutral,
        "n_deceptive":  n_deceptive,
        "n_left":       n_left,
        "n_right":      n_right,
        "style_labels": {"legible": 0, "neutral": 1, "deceptive": 2},
        "legible_arc_range_cm": [5, 28],
        "neutral_cp_x_range":  [0.42, 0.50],
        "deceptive_feint_cm":  [5, 20],
        "references": [
            "Dragan & Srinivasa (HRI 2013) — legibility vs ambiguity",
            "SLOT-V / Wallkotter et al. (RO-MAN 2022)",
            "Style-Conditioned Diffusion Policy (arXiv:2603.16368, ICSR 2026)",
        ],
    }

    # Remove checkpoint after all data is safely in .npz
    if ckpt_path.exists():
        ckpt_path.unlink()

    np.savez_compressed(
        str(out),
        obs             = np.stack(all_obs),           # (N, T, 22)
        actions         = np.stack(all_act),           # (N, T,  5)
        episode_lengths = np.array(all_lens),          # (N,)
        labels          = np.array(all_labels),        # (N,)  "left"/"right"
        style_labels    = np.array(all_styles),        # (N,)  0/1/2
        style_names     = np.array(all_style_names),   # (N,)  str
        arc_idxs        = np.array(all_arc_idxs),      # (N,)  0-based
        config_ids      = np.array(all_cfg_ids),       # (N,)
        metadata_json   = json.dumps(metadata),
    )

    # ── summary ───────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Saved {total} combined demos to: {out}")
    print(f"  left / right          : {n_left} / {n_right}")
    print(f"  legible               : {n_legible}  ({n_legible/total:.0%})")
    print(f"  neutral               : {n_neutral}  ({n_neutral/total:.0%})")
    print(f"  deceptive             : {n_deceptive}  ({n_deceptive/total:.0%})")
    print(f"  retries needed        : {retries}")
    print(f"  videos                : {vid_dir}/  ({total} files)")
    print(f"{'='*62}")
    print()
    print("Next steps:")
    print("  Train:   py scripts/train_diffusion_policy.py --config "
          "configs/train.yaml  (set demo_path=data/demos/demos_combined.npz)")
    print("  Eval:    py scripts/eval_multimodality.py --ckpt "
          "runs/latest/ckpt.pt  (add arc-classifier for style breakdown)")


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect 400 combined demos: 50% legible, "
                    "25% neutral, 25% deceptive.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",  type=str,
                    default="data/demos/demos_combined.npz")
    ap.add_argument("--preview", action="store_true",
                    help="Run 3 preview videos (one per style). "
                         "No .npz written. Inspect before full run.")
    args = ap.parse_args()
    collect(seed=args.seed, out_path=args.out, preview=args.preview)


if __name__ == "__main__":
    main()
