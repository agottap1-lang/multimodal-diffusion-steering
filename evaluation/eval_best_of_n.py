#!/usr/bin/env python3
"""Best-of-N vs Baseline legibility comparison.

Demonstrates that selecting the most legible trajectory from N candidates
(Best-of-N) yields measurably higher L_early_intent than taking a single
randomly-sampled trajectory (baseline).

The arc trajectories are computed analytically from the Bézier formula
(no PyBullet simulation needed) — identical to the ScriptedExpert arc
phase that produces the training data.  Random jitter is added to the
block position to simulate realistic variation across episodes.

Usage:
    python scripts/eval_best_of_n.py
    python scripts/eval_best_of_n.py --n_candidates 4 8 16 --n_episodes 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.legibility_metrics import compute_legibility

# ── constants (from envs/twoblockpick_env.py and collect_demos) ──────
_EE_HOME = np.array([0.40, 0.0, 0.55], dtype=np.float64)
_CUBE_X  = 0.50
_CUBE_Y  = 0.07    # left block at +0.07, right at -0.07
_CUBE_Z  = 0.421

_GOALS = np.array([
    [_CUBE_X, +_CUBE_Y, _CUBE_Z],   # 0 = left block
    [_CUBE_X, -_CUBE_Y, _CUBE_Z],   # 1 = right block
], dtype=np.float64)

_TRUE_GOAL_IDX = 0   # picking left block (arbitrary, symmetric)
_N_ARC_PTS = 200     # waypoints along Bézier arc (matches ScriptedExpert)


def _arc_variations(n: int = 20) -> list[dict]:
    """Return the same 20 arc specs used in data collection.

    These come directly from ScriptedExpert._build_arc_variations() in
    scripts/collect_demos_twoblockpick.py.  P1 is the Bézier control
    point:
        cp_x  : x-coordinate (decreases 0.38→0.28 as arc widens)
        cp_y_mag : |y| of control point (increases 0.05→0.28)
        cp_z  : z-coordinate (increases 0.56→0.68 as arc rises)
    """
    variations = []
    for i in range(n):
        t = i / max(n - 1, 1)
        cp_y_mag = float(np.interp(t, [0, 1], [0.05, 0.28]))
        frac = (cp_y_mag - 0.05) / (0.28 - 0.05)
        cp_z = float(0.56 + 0.12 * frac)
        cp_x = float(0.38 - 0.10 * frac)
        variations.append(dict(cp_x=cp_x, cp_y_mag=cp_y_mag, cp_z=cp_z,
                               arc_idx=i))
    return variations


def _bezier_arc(arc: dict, cube_pos: np.ndarray,
                approach_h: float = 0.08) -> np.ndarray:
    """Compute the 200-point quadratic Bézier arc trajectory.

    B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2,  t ∈ [0, 1]

    P0 = EE home  [0.40, 0.0, 0.55]  (defined in collect_demos)
    P1 = control point (cp_x, +cp_y_mag, cp_z)  ← picking left block
    P2 = above target cube [cube_x, cube_y, cube_z + approach_h]

    The sign of cp_y is always positive here because we pick the left
    (positive-y) block.  For right picks the sign is flipped.

    Returns (N_ARC_PTS, 3) array of EE positions.
    """
    P0 = _EE_HOME.copy()
    P1 = np.array([arc["cp_x"], +arc["cp_y_mag"], arc["cp_z"]])
    P2 = cube_pos.copy()
    P2[2] += approach_h

    ts = np.linspace(0, 1, _N_ARC_PTS)
    traj = np.array([
        (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
        for t in ts
    ], dtype=np.float64)
    return traj


def _score_arc(arc: dict, cube_pos: np.ndarray,
               approach_h: float, goals: np.ndarray,
               true_goal_idx: int = _TRUE_GOAL_IDX) -> dict:
    """Score one arc using compute_legibility on the first 60% of waypoints.

    Legibility metrics (L_early_intent, L_posterior) are computed on the
    partial trajectory for realistic observer-model semantics.
    Efficiency metrics (RLC, CLC, FLI) use the FULL trajectory so that
    the Dragan cost-ratio comparison C*(x₀,g*) is against the actual goal.
    """
    traj = _bezier_arc(arc, cube_pos, approach_h)
    arc_end = max(2, int(len(traj) * 0.60))
    r_partial = compute_legibility(traj[:arc_end], goals, true_goal_idx,
                                   model="gaussian")
    r_full = compute_legibility(traj, goals, true_goal_idx,
                                model="gaussian")
    return dict(
        arc_idx=arc["arc_idx"],
        cp_y_mag=arc["cp_y_mag"],
        L_early_intent=r_partial.L_early_intent,
        L_posterior=r_partial.L_posterior,
        L_composite=r_partial.L_composite,
        path_efficiency=r_full.path_efficiency,
        relative_legibility_cost=r_full.relative_legibility_cost,
        cumulative_legibility_cost=r_full.cumulative_legibility_cost,
        front_loading_index=r_full.front_loading_index,
    )


def run_comparison(n_candidates_list: list[int], n_episodes: int,
                   seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    arcs = _arc_variations(20)
    n_arcs = len(arcs)
    max_n = max(n_candidates_list)

    # cube_jitter from TwoBlockPickEnv: uniform ±0.01 m per axis
    JITTER = 0.01

    print(f"Evaluating {n_episodes} episodes × up to {max_n} candidates …")
    print(f"Analytical Bézier arcs (no simulation). Cube jitter: ±{JITTER} m\n")

    # ── Pre-compute all arc scores ────────────────────────────────────
    episode_pools: list[list[dict]] = []
    for ep in range(n_episodes):
        # Jitter cube position as in TwoBlockPickEnv
        dx = float(rng.uniform(-JITTER, JITTER))
        dy = float(rng.uniform(-JITTER, JITTER))
        dz = float(rng.uniform(-JITTER, JITTER))
        cube_pos = np.array([_CUBE_X + dx, _CUBE_Y + dy, _CUBE_Z + dz])

        # Approach height: same as ScriptedExpert (uniform 0.06–0.12)
        approach_h = float(rng.uniform(0.06, 0.12))

        # Jitter goals to match cube jitter
        goals = np.array([
            [cube_pos[0], cube_pos[1], cube_pos[2]],     # left (true goal)
            [_CUBE_X + float(rng.uniform(-JITTER, JITTER)),
             -_CUBE_Y + float(rng.uniform(-JITTER, JITTER)),
             _CUBE_Z],
        ])

        # Sample max_n arc indices (uniform over all 20 arcs)
        sampled_idxs = rng.integers(0, n_arcs, size=max_n)
        pool = [
            _score_arc(arcs[int(idx)], cube_pos, approach_h, goals)
            for idx in sampled_idxs
        ]
        episode_pools.append(pool)

    # ── Print comparison table ────────────────────────────────────────
    print("═" * 72)
    print("BEST-OF-N vs BASELINE  (legibility selection criterion: L_early_intent)")
    print("═" * 72)
    print(f"{'Method':<22} {'L_early_intent':>16} {'L_posterior':>12} "
          f"{'path_eff↓':>10}  {'Improvement':>11}")
    print("─" * 72)

    # Baseline: first candidate (random single arc)
    bl_early = [pool[0]["L_early_intent"] for pool in episode_pools]
    bl_post  = [pool[0]["L_posterior"]    for pool in episode_pools]
    bl_eff   = [pool[0]["path_efficiency"] for pool in episode_pools]
    print(f"{'Baseline (N=1)':<22} "
          f"{np.mean(bl_early):>8.4f}±{np.std(bl_early):.4f}  "
          f"{np.mean(bl_post):>7.4f}±{np.std(bl_post):.4f}  "
          f"{np.mean(bl_eff):>5.4f}±{np.std(bl_eff):.4f}  "
          f"{'(reference)':>11}")

    all_results = {"baseline": dict(
        N=1, n_episodes=n_episodes,
        mean_L_early_intent=float(np.mean(bl_early)),
        std_L_early_intent=float(np.std(bl_early)),
        mean_L_posterior=float(np.mean(bl_post)),
        std_L_posterior=float(np.std(bl_post)),
        mean_path_efficiency=float(np.mean(bl_eff)),
    )}

    for N in sorted(n_candidates_list):
        bon_early, bon_post, bon_eff = [], [], []
        for pool in episode_pools:
            best = max(pool[:N], key=lambda m: m["L_early_intent"])
            bon_early.append(best["L_early_intent"])
            bon_post.append(best["L_posterior"])
            bon_eff.append(best["path_efficiency"])

        delta = np.mean(bon_early) - np.mean(bl_early)
        rel   = delta / (np.mean(bl_early) + 1e-9) * 100.0
        print(f"{'Best-of-' + str(N):<22} "
              f"{np.mean(bon_early):>8.4f}±{np.std(bon_early):.4f}  "
              f"{np.mean(bon_post):>7.4f}±{np.std(bon_post):.4f}  "
              f"{np.mean(bon_eff):>5.4f}±{np.std(bon_eff):.4f}  "
              f"{delta:>+7.4f} ({rel:>+5.1f}%)")

        all_results[f"best_of_{N}"] = dict(
            N=N, n_episodes=n_episodes,
            mean_L_early_intent=float(np.mean(bon_early)),
            std_L_early_intent=float(np.std(bon_early)),
            mean_L_posterior=float(np.mean(bon_post)),
            std_L_posterior=float(np.std(bon_post)),
            mean_path_efficiency=float(np.mean(bon_eff)),
            delta_vs_baseline=float(delta),
            relative_gain_pct=float(rel),
        )

    print("─" * 72)
    print("path_eff↓ : lower = curved path (legibility vs efficiency tradeoff)")
    print("Note: λ=0.5 combined score = 0.5·L_early_intent + 0.5·path_efficiency")

    # ── Print tradeoff score ──────────────────────────────────────────
    print("\n── Tradeoff score  (λ=0.5 equal weighting — most defensible): ──────")
    lap_lambda = 0.5
    bl_tradeoff = [
        lap_lambda * pool[0]["L_early_intent"] +
        (1 - lap_lambda) * pool[0]["path_efficiency"]
        for pool in episode_pools
    ]
    print(f"  Baseline (N=1):       {np.mean(bl_tradeoff):.4f} ± {np.std(bl_tradeoff):.4f}")
    for N in sorted(n_candidates_list):
        bon_tradeoff = []
        for pool in episode_pools:
            best = max(pool[:N], key=lambda m:
                       lap_lambda * m["L_early_intent"] +
                       (1 - lap_lambda) * m["path_efficiency"])
            bon_tradeoff.append(
                lap_lambda * best["L_early_intent"] +
                (1 - lap_lambda) * best["path_efficiency"]
            )
        delta_t = np.mean(bon_tradeoff) - np.mean(bl_tradeoff)
        print(f"  Best-of-{N:<2} (tradeoff): {np.mean(bon_tradeoff):.4f} ± "
              f"{np.std(bon_tradeoff):.4f}  Δ={delta_t:+.4f}")

    # ── Legibility spectrum summary ───────────────────────────────────
    print("\n── Arc legibility spectrum (all 20 arcs, analytical) ────────────────")
    goals_ref = _GOALS.copy()
    cube_ref  = np.array([_CUBE_X, _CUBE_Y, _CUBE_Z])
    all_arc_scores = [
        _score_arc(arc, cube_ref, 0.08, goals_ref)
        for arc in arcs
    ]
    print(f"  {'Arc':>4} {'cp_y_mag':>10} {'L_early_intent':>16} "
          f"{'path_eff':>10}  {'RLC':>8}  {'CLC':>8}  {'FLI':>8}  {'LPC':>8}")
    for m in all_arc_scores:
        lpc = m['L_early_intent'] / (m['relative_legibility_cost'] + 0.01)
        print(f"  {m['arc_idx']:>4} {m['cp_y_mag']:>10.4f} "
              f"{m['L_early_intent']:>16.4f} {m['path_efficiency']:>10.4f}  "
              f"{m['relative_legibility_cost']:>8.4f}  "
              f"{m['cumulative_legibility_cost']:>8.4f}  "
              f"{m['front_loading_index']:>8.4f}  {lpc:>8.3f}")

    efficient_arcs  = [m for m in all_arc_scores if m["arc_idx"] <= 5]
    moderate_arcs   = [m for m in all_arc_scores if 6 <= m["arc_idx"] <= 13]
    legible_arcs    = [m for m in all_arc_scores if m["arc_idx"] >= 14]
    print(f"\n  Arc 00-05 (efficient):  L_early={np.mean([m['L_early_intent'] for m in efficient_arcs]):.4f},"
          f"  RLC={np.mean([m['relative_legibility_cost'] for m in efficient_arcs]):.4f}")
    print(f"  Arc 06-13 (moderate):   L_early={np.mean([m['L_early_intent'] for m in moderate_arcs]):.4f},"
          f"  RLC={np.mean([m['relative_legibility_cost'] for m in moderate_arcs]):.4f}")
    print(f"  Arc 14-19 (legible):    L_early={np.mean([m['L_early_intent'] for m in legible_arcs]):.4f},"
          f"  RLC={np.mean([m['relative_legibility_cost'] for m in legible_arcs]):.4f}")
    print("  RLC = Relative Legibility Cost (Dragan 2013): (arc−straight)/straight")
    print("  CLC = Cumulative Legibility Cost: time-integrated suboptimality")
    print("  FLI = Front-Loading Index: fraction of overhead in first 30%")
    print("  LPC = Legibility per Unit Cost: L_early / (RLC+0.01)  [novel efficiency metric]")

    # ── Save ─────────────────────────────────────────────────────────
    import json
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "best_of_n_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n_candidates", type=int, nargs="+", default=[4, 8, 16],
                    help="Number of candidates for Best-of-N (default: 4 8 16)")
    ap.add_argument("--n_episodes", type=int, default=100,
                    help="Episodes to average over (default: 100)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_comparison(
        n_candidates_list=sorted(set(args.n_candidates)),
        n_episodes=args.n_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
