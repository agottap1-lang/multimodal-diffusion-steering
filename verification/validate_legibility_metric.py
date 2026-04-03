#!/usr/bin/env python3
"""
Legibility Metric Validation: Arc00 → Arc19
============================================

Runs the scripted expert for all 20 arc indices, collects EE trajectories,
then computes legibility_metrics.compute_legibility() on each.

Validates:
  1. L_early_intent   increases monotonically with arc index (ICRA metric)
  2. L_geometric      increases monotonically with arc index
  3. L_composite      increases monotonically
  4. path_efficiency  decreases monotonically (wider arc = less efficient)

Plots a figure showing all scores vs arc index.

Research note on t=0..T prefix:
  Dragan et al. (HRI 2013, §4.2) showed subjects a prefix ξ_{0:T_cut}
  and asked which goal the robot was heading for.  T_cut varied from
  the first frame to the full trajectory.  The key dependent variable
  was the *smallest* T_cut at which ≥85% of subjects answered correctly.
  This is why metrics should weight early timesteps heavily (α-decay)
  and why L_early_intent (first 30% of motion) is the most direct
  analogue to the human study protocol.

  The 30% cut-off maps to:
    - 400-step episode × 30% = t=0..120  (steps)
    - At env fps=30, this is the first ~4 seconds of the ~13s video
  which aligns with the t=2..6s range used in the VLM pairwise test.

Usage:
  py -3 scripts/validate_legibility_metric.py
  py -3 scripts/validate_legibility_metric.py --no-sim   # skip env, use saved npz
  py -3 scripts/validate_legibility_metric.py --save-npz # save trajectories to npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.twoblockpick_env import TwoBlockPickEnv, ACT_DIM, _TABLE_TOP_Z
from evaluation.legibility_metrics import compute_legibility

# ── Arc geometry (identical to render_front_view_arcs.py) ─────────────

_EE_HOME       = np.array([0.40, 0.0, 0.55], dtype=np.float32)
_N_ARC_PTS     = 200
_N_DESCENT_PTS = 30
_N_GRIP_STEPS  = 40
_N_LIFT_PTS    = 30
_EPISODE_LEN   = 400
N_ARCS         = 20


def build_arc_cp(arc_idx: int) -> np.ndarray:
    t = arc_idx / max(N_ARCS - 1, 1)
    cp_y_mag = float(np.interp(t, [0, 1], [0.05, 0.28]))
    frac = (cp_y_mag - 0.05) / (0.28 - 0.05)
    cp_z = float(0.56 + 0.12 * frac)
    cp_x = float(0.38 - 0.10 * frac)
    # Left pick: cp_y positive (sweeps left)
    return np.array([cp_x, cp_y_mag, cp_z], dtype=np.float32)


# ── Scripted expert (self-contained) ──────────────────────────────────

class ScriptedExpert:
    def __init__(self, target: str, arc_control_point: np.ndarray):
        self.target   = target
        self.scale    = 0.05
        self._arc_cp  = arc_control_point.copy()
        self.phase    = 0
        self._arc_wps = self._descent_wps = self._lift_wps = None
        self._wp_idx  = 0
        self._approach_h = 0.08

    def reset(self):
        self.phase    = 0
        self._arc_wps = self._descent_wps = self._lift_wps = None
        self._wp_idx  = 0

    def _build_waypoints(self, cube_pos):
        P0 = _EE_HOME.copy()
        P1 = self._arc_cp.copy()
        P2 = cube_pos.copy(); P2[2] += self._approach_h

        ts = np.linspace(0, 1, _N_ARC_PTS)
        self._arc_wps = np.array(
            [(1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2 for t in ts],
            dtype=np.float32)

        above = P2.copy()
        grasp = cube_pos.copy(); grasp[2] += 0.005
        self._descent_wps = np.array(
            [above + t*(grasp - above) for t in np.linspace(0, 1, _N_DESCENT_PTS)],
            dtype=np.float32)

        lift_top = grasp.copy(); lift_top[2] = 0.60
        self._lift_wps = np.array(
            [grasp + t*(lift_top - grasp) for t in np.linspace(0, 1, _N_LIFT_PTS)],
            dtype=np.float32)

        self._wp_idx = 1

    def act(self, obs: np.ndarray) -> np.ndarray:
        ee_pos   = obs[0:3]
        cube_pos = obs[8:11].copy() if self.target == "left" else obs[15:18].copy()

        if self._arc_wps is None:
            self._build_waypoints(cube_pos)

        action = np.zeros(ACT_DIM, dtype=np.float32)

        if self.phase == 0:
            tgt    = self._arc_wps[self._wp_idx]
            delta  = (tgt - ee_pos) / self.scale
            action[:3] = np.clip(delta * 0.45, -1, 1)
            action[4]  = 1.0
            if np.linalg.norm(ee_pos - tgt) < 0.012:
                self._wp_idx += 1
                if self._wp_idx >= _N_ARC_PTS:
                    self.phase = 1; self._wp_idx = 1
        elif self.phase == 1:
            tgt    = self._descent_wps[self._wp_idx]
            delta  = (tgt - ee_pos) / self.scale
            action[:3] = np.clip(delta * 0.35, -1, 1)
            action[4]  = 1.0
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_DESCENT_PTS:
                    self.phase = 2; self._wp_idx = 0
        elif self.phase == 2:
            t_frac = self._wp_idx / max(_N_GRIP_STEPS - 1, 1)
            action[4] = 1.0 - 2.0 * t_frac
            self._wp_idx += 1
            if self._wp_idx >= _N_GRIP_STEPS:
                self.phase = 3; self._wp_idx = 1
        elif self.phase == 3:
            tgt    = self._lift_wps[self._wp_idx]
            delta  = (tgt - ee_pos) / self.scale
            action[:3] = np.clip(delta * 0.35, -1, 1)
            action[4]  = -1.0
            if np.linalg.norm(ee_pos - tgt) < 0.010:
                self._wp_idx += 1
                if self._wp_idx >= _N_LIFT_PTS:
                    self.phase = 4
        else:
            action[4] = -1.0

        return action


# ── Trajectory collection ──────────────────────────────────────────────

def collect_trajectory(arc_idx: int, target: str = "left", seed: int = 0) -> np.ndarray:
    """Run scripted expert and return EE position trajectory (T, 3)."""
    cp = build_arc_cp(arc_idx)
    if target == "right":
        cp[1] = -cp[1]   # mirror y for right pick

    env    = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                             episode_length=_EPISODE_LEN)
    expert = ScriptedExpert(target=target, arc_control_point=cp)
    obs    = env.reset(seed=seed)
    expert.reset()

    positions = [obs[0:3].copy()]       # EE position at each step

    for _ in range(_EPISODE_LEN):
        action = expert.act(obs)
        result = env.step(action)
        obs    = result.obs
        positions.append(obs[0:3].copy())
        if result.done:
            break

    env.close()
    return np.array(positions, dtype=np.float64)   # (T, 3)


# ── Goal positions (from env constants) ─────────────────────────────────
# _CUBE_Z = _TABLE_TOP_Z + _CUBE_HALF + 0.001 = 0.4 + 0.02 + 0.001 = 0.421
# _CUBE_X = 0.50, _CUBE_Y = 0.07
# Left block at y=+0.07, right at y=-0.07

_CUBE_HALF = 0.02
_CUBE_Z    = _TABLE_TOP_Z + _CUBE_HALF + 0.001   # 0.421
_CUBE_X    = 0.50
_CUBE_Y    = 0.07

LEFT_GOAL  = np.array([_CUBE_X,  _CUBE_Y, _CUBE_Z], dtype=np.float64)
RIGHT_GOAL = np.array([_CUBE_X, -_CUBE_Y, _CUBE_Z], dtype=np.float64)

GOALS = np.array([LEFT_GOAL, RIGHT_GOAL])   # goal 0 = left


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="left", choices=["left", "right"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gaussian",
                    choices=["gaussian", "cost_ratio", "both"],
                    help="Observer model for Bayesian posterior")
    ap.add_argument("--save-npz", action="store_true",
                    help="Save trajectories to outputs/legibility_validation/trajectories.npz")
    ap.add_argument("--no-sim", action="store_true",
                    help="Skip simulation; load from saved npz")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip matplotlib plot (just print table)")
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "legibility_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "trajectories.npz"

    # ── Collect / load trajectories ─────────────────────────────────
    if args.no_sim and npz_path.exists():
        print(f"Loading trajectories from {npz_path}")
        data   = np.load(npz_path)
        trajs  = {int(k): data[k] for k in data.files}
    else:
        print(f"Collecting trajectories for arc00–arc{N_ARCS-1:02d} "
              f"(target={args.target}, seed={args.seed}) ...")
        trajs = {}
        for i in range(N_ARCS):
            print(f"  arc{i:02d} ...", end=" ", flush=True)
            traj    = collect_trajectory(i, args.target, args.seed)
            trajs[i] = traj
            print(f"{len(traj)} steps")

        if args.save_npz:
            np.savez(npz_path, **{str(i): trajs[i] for i in range(N_ARCS)})
            print(f"  Saved → {npz_path}")

    # ── Compute legibility metrics ────────────────────────────────────
    true_goal_idx = 0 if args.target == "left" else 1
    results       = []

    print(f"\nComputing legibility (model={args.model}) ...")
    print(f"{'Arc':>5} {'cp_y_mag':>10} {'L_early':>9} {'L_geom[BROKEN]':>14} "
          f"{'L_post':>8} {'L_commit':>9} {'L_comp':>8} {'efficiency':>10}")
    print("─" * 80)

    for i in range(N_ARCS):
        t    = i / max(N_ARCS - 1, 1)
        cp_y = float(np.interp(t, [0, 1], [0.05, 0.28]))

        traj = trajs[i]

        # Only score the arc phase (first 60% of trajectory ≈ the approach)
        # This matches the VLM prefix evaluation (t=0..T_cut)
        arc_end = int(len(traj) * 0.60)
        traj_prefix = traj[:arc_end]

        r = compute_legibility(
            trajectory=traj_prefix,
            goals=GOALS,
            true_goal_idx=true_goal_idx,
            model=args.model,
            return_curves=False,
        )

        row = dict(
            arc_idx=i,
            cp_y_mag=round(cp_y, 4),
            L_early_intent=round(r.L_early_intent, 4),
            L_geometric=round(r.L_geometric, 4),
            L_posterior=round(r.L_posterior, 4),
            L_commitment=round(r.L_commitment, 4),
            L_composite=round(r.L_composite, 4),
            path_efficiency=round(r.path_efficiency, 4),
            n_steps=len(traj_prefix),
        )
        results.append(row)

        print(f"  {i:3d}  {cp_y:10.4f}  {r.L_early_intent:9.4f}  "
              f"{r.L_geometric:14.4f}  {r.L_posterior:8.4f}  "
              f"{r.L_commitment:9.4f}  {r.L_composite:8.4f}  "
              f"{r.path_efficiency:10.4f}")

    # ── Monotonicity analysis ─────────────────────────────────────────
    print("\n── Monotonicity check ────────────────────────────────────────")

    def monotonicity(values, name):
        arr   = np.array(values)
        diffs = np.diff(arr)
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n_ties = int(np.sum(diffs == 0))
        mono  = "MONOTONE ↑" if n_neg == 0 else (
                "MONOTONE ↓" if n_pos == 0 else
                f"non-monotone ({n_pos}↑, {n_neg}↓, {n_ties}=)")
        corr  = float(np.corrcoef(np.arange(N_ARCS), arr)[0, 1])
        print(f"  {name:20s}: {mono}   Pearson r={corr:+.3f}")
        return corr

    metrics_to_check = [
        ([r["L_early_intent"] for r in results],   "L_early_intent     [PRIMARY]"),
        ([r["path_efficiency"] for r in results],  "path_efficiency↓   [SECONDARY]"),
        ([r["L_posterior"]    for r in results],   "L_posterior        [ROBUSTNESS]"),
        ([r["L_composite"]    for r in results],   "L_composite        [AGGREGATE]"),
    ]
    corr_early = monotonicity(*metrics_to_check[0])
    for vals, name in metrics_to_check[1:]:
        monotonicity(vals, name)

    # L_geometric is broken — compute but flag
    print()
    geom_vals = [r["L_geometric"] for r in results]
    geom_corr = float(np.corrcoef(np.arange(N_ARCS), geom_vals)[0, 1])
    geom_mean = float(np.mean(geom_vals))
    print(f"  {'L_geometric [BROKEN]':20s}: saturates to {geom_mean:.4f} for all arcs  "
          f"Pearson r={geom_corr:+.3f}  (not used for ranking)")

    # ── Save results ────────────────────────────────────────────────
    out_json = out_dir / f"metrics_{args.model}.json"
    with open(out_json, "w") as f:
        json.dump({"model": args.model, "target": args.target,
                   "seed": args.seed, "results": results}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(
                f"Legibility Metrics vs Arc Index (model={args.model}, target={args.target})\n"
                f"Research basis: Dragan et al. HRI 2013 — higher arc index = wider Bézier sweep\n"
                f"Metrics computed on first 60% of trajectory (the arc-approach phase)",
                fontsize=11)

            arc_indices = [r["arc_idx"]    for r in results]
            cp_y_mags   = [r["cp_y_mag"]   for r in results]

            # Band shading: 0-4 straight, 5-9 slight, 10-14 moderate, 15-19 high
            band_colors = ["#fee8c8", "#fdd49e", "#d7b5d8", "#c994c7"]
            band_labels = ["Straight\n(0–4)", "Slight\n(5–9)",
                           "Moderate\n(10–14)", "High\n(15–19)"]
            band_ranges = [(0, 4), (5, 9), (10, 14), (15, 19)]

            def add_bands(ax):
                for (lo, hi), col, lbl in zip(band_ranges, band_colors, band_labels):
                    ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.3, color=col, label=lbl)

            metrics_plot = [
                ("L_early_intent", "L_early_intent [PRIMARY]\n(mean P(g*) in first 30% of motion)",
                 "#e41a1c", True),
                ("path_efficiency","path_efficiency [SECONDARY]\n(straight/arc-length, \u2193 = legibility cost)",
                 "#a65628", False),
                ("L_posterior",    "L_posterior [ROBUSTNESS CHECK]\n(time-weighted avg P(g*))",
                 "#4daf4a", True),
                ("L_commitment",   "L_commitment\n(1 \u2212 t_ICP/T, \u03b8=0.80)",
                 "#984ea3", True),
                ("L_composite",    "L_composite [AGGREGATE]\n(weighted combination, excl. L_geometric)",
                 "#ff7f00", True),
                ("L_geometric",    "L_geometric [BROKEN \u2014 DO NOT USE]\n(saturates at 1.0 for all arcs)",
                 "#aaaaaa", True),
            ]

            for ax, (key, title, color, higher_better) in zip(axes.flat, metrics_plot):
                vals = [r[key] for r in results]
                add_bands(ax)
                ax.plot(arc_indices, vals, "o-", color=color, lw=2, ms=5, zorder=3)
                ax.set_xlabel("Arc Index (0=straight, 19=extreme sweep)", fontsize=9)
                ax.set_ylabel(key, fontsize=9)
                ax.set_title(title, fontsize=9)
                ax.set_xlim(-0.5, 19.5)
                ax.set_ylim(-0.02, 1.05)
                ax.grid(True, alpha=0.3, zorder=0)
                corr = float(np.corrcoef(arc_indices, vals)[0, 1])
                direction = "↑ desired" if higher_better else "↓ desired"
                ax.set_xlabel(
                    f"Arc Index   |   Pearson r = {corr:+.3f}   |   {direction}",
                    fontsize=8)

            # Legend on last axis
            axes.flat[-1].legend(
                *axes.flat[0].get_legend_handles_labels(),
                loc="upper right", fontsize=7, title="Arc class")

            plt.tight_layout()
            plot_path = out_dir / f"metric_monotonicity_{args.model}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved → {plot_path}")

        except Exception as e:
            print(f"  Plot skipped: {e}")


if __name__ == "__main__":
    main()
