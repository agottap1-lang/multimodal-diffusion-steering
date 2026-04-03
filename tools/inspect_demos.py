#!/usr/bin/env python
"""Inspect demo dataset and guard against too-small / unbalanced data.

Usage:
    python scripts/inspect_demos.py [--path data/demos/demos.npz]

Checks:
  1) prints episode count, total timesteps, obs shape, action shape
  2) per-dim action stats (min / max / mean / std)
  3) demo balance (left vs right count)
  4) HARD-FAIL if episodes < 100 or |left_ratio - 0.5| > 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────


def inspect(path: str) -> bool:
    """Load demos, print stats, return True if all guards pass."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: demo file not found: {p}")
        return False

    data = np.load(str(p), allow_pickle=True)

    obs = data["obs"]                       # (N, T, obs_dim)
    actions = data["actions"]               # (N, T, act_dim)
    ep_lens = data["episode_lengths"]       # (N,)
    labels = data["labels"]                 # (N,)  str

    N, T, obs_dim = obs.shape
    _, _, act_dim = actions.shape

    total_timesteps = int(ep_lens.sum())
    mean_ep_len = float(ep_lens.mean())
    min_ep_len = int(ep_lens.min())
    max_ep_len = int(ep_lens.max())

    print("═══════════════════════════════════════════════")
    print(f"  Demo file : {p}")
    print(f"  Episodes  : {N}")
    print(f"  Max T     : {T}")
    print(f"  Obs dim   : {obs_dim}")
    print(f"  Act dim   : {act_dim}")
    print(f"  Total ts  : {total_timesteps}")
    print(f"  Ep length : mean={mean_ep_len:.1f}  min={min_ep_len}  max={max_ep_len}")
    print("═══════════════════════════════════════════════")

    # ── per-dim action stats (using valid timesteps only) ──────────
    all_acts = []
    for i in range(N):
        L = int(ep_lens[i])
        all_acts.append(actions[i, :L])
    all_acts_cat = np.concatenate(all_acts, axis=0)  # (total_ts, act_dim)

    dim_names = ["dx", "dy", "dz", "dyaw", "grip"]
    print("\n  Per-dim action stats (valid timesteps):")
    print(f"  {'dim':>5s}  {'min':>8s}  {'max':>8s}  {'mean':>8s}  {'std':>8s}")
    for d in range(act_dim):
        col = all_acts_cat[:, d]
        print(f"  {dim_names[d] if d < len(dim_names) else f'd{d}':>5s}"
              f"  {col.min():+8.4f}  {col.max():+8.4f}"
              f"  {col.mean():+8.4f}  {col.std():8.4f}")

    # ── per-dim obs stats ──────────────────────────────────────────
    all_obs = []
    for i in range(N):
        L = int(ep_lens[i])
        all_obs.append(obs[i, :L])
    all_obs_cat = np.concatenate(all_obs, axis=0)

    print(f"\n  Obs stats (total {all_obs_cat.shape[0]} timesteps):")
    print(f"  {'dim':>4s}  {'min':>8s}  {'max':>8s}  {'mean':>8s}  {'std':>8s}")
    obs_names = ["ee_x", "ee_y", "ee_z",
                 "eq_x", "eq_y", "eq_z", "eq_w", "grip",
                 "Lx", "Ly", "Lz", "Lqx", "Lqy", "Lqz", "Lqw",
                 "Rx", "Ry", "Rz", "Rqx", "Rqy", "Rqz", "Rqw"]
    for d in range(obs_dim):
        col = all_obs_cat[:, d]
        nm = obs_names[d] if d < len(obs_names) else f"d{d}"
        print(f"  {nm:>4s}  {col.min():+8.4f}  {col.max():+8.4f}"
              f"  {col.mean():+8.4f}  {col.std():8.4f}")

    # ── demo balance ──────────────────────────────────────────────
    unique, counts = np.unique(labels, return_counts=True)
    label_map = dict(zip(unique, counts))
    n_left = int(label_map.get("left", 0))
    n_right = int(label_map.get("right", 0))
    n_other = N - n_left - n_right

    print(f"\n  Label distribution:")
    print(f"    left  : {n_left}")
    print(f"    right : {n_right}")
    if n_other > 0:
        print(f"    other : {n_other}")

    left_ratio = n_left / N if N > 0 else 0

    # ── episode success check ─────────────────────────────────────
    n_short = int(np.sum(ep_lens < T))  # terminated before max
    print(f"\n  Episodes terminating early (success): {n_short}/{N}"
          f" ({n_short/N:.1%})")

    # ── cube init pos mean/std + symmetry check ────────────────
    # First obs of each episode => initial cube positions
    init_obs = obs[:, 0, :]  # (N, obs_dim)
    lc_y_init = init_obs[:, 9]   # left cube y  (obs index 9)
    rc_y_init = init_obs[:, 16]  # right cube y (obs index 16)
    lc_x_init = init_obs[:, 8]
    rc_x_init = init_obs[:, 15]
    lc_z_init = init_obs[:, 10]
    rc_z_init = init_obs[:, 17]

    print(f"\n  Cube initial positions (first obs of each episode):")
    print(f"    Left  cube: x={lc_x_init.mean():.4f}+/-{lc_x_init.std():.4f}  "
          f"y={lc_y_init.mean():+.4f}+/-{lc_y_init.std():.4f}  "
          f"z={lc_z_init.mean():.4f}+/-{lc_z_init.std():.4f}")
    print(f"    Right cube: x={rc_x_init.mean():.4f}+/-{rc_x_init.std():.4f}  "
          f"y={rc_y_init.mean():+.4f}+/-{rc_y_init.std():.4f}  "
          f"z={rc_z_init.mean():.4f}+/-{rc_z_init.std():.4f}")
    y_sym = abs(lc_y_init.mean() + rc_y_init.mean())
    print(f"    y-symmetry check: |mean(Ly) + mean(Ry)| = {y_sym:.4f}"
          + ("  OK (< 0.02)" if y_sym < 0.02 else "  WARNING: cubes not mirrored!"))

    # ── GUARDS ────────────────────────────────────────────────────
    print("\n── Guards ─────────────────────────────────────")
    passed = True

    if N < 100:
        print(f"  ✗ FAIL: episodes_total={N} < 100 — COLLECT MORE DEMOS FIRST")
        passed = False
    else:
        print(f"  ✓ episodes_total={N} >= 100")

    imbalance = abs(left_ratio - 0.5)
    if imbalance > 0.05:
        print(f"  ✗ FAIL: |left_ratio - 0.5| = {imbalance:.3f} > 0.05 — NOT BALANCED")
        passed = False
    else:
        print(f"  ✓ |left_ratio - 0.5| = {imbalance:.3f} <= 0.05")

    success_rate = n_short / N if N > 0 else 0
    if success_rate < 0.80:
        print(f"  ⚠ WARNING: only {success_rate:.1%} of demos succeeded"
              f" (terminated early)")
    else:
        print(f"  ✓ demo success rate = {success_rate:.1%}")

    print("═══════════════════════════════════════════════")
    if passed:
        print("ALL GUARDS PASSED — safe to train.")
    else:
        print("GUARDS FAILED — fix data before training.")
    print("═══════════════════════════════════════════════")
    return passed


# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="data/demos/demos.npz")
    args = ap.parse_args()
    ok = inspect(args.path)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
