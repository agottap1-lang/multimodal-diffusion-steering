#!/usr/bin/env python3
"""
Step 2: Corrected VLM Steering Experiment
==========================================

Runs three selection modes head-to-head on the same candidate pools:

  1. legible_max_arc  — VLM binary gate + geometry argmax (original, flawed)
  2. max_legibility   — true VLM argmax (direction-aware legibility)
  3. vlm_weighted     — hybrid: VLM legibility * arc bonus

Fixes applied vs. original experiment:
  - Direction-aware legibility: P(correct goal | frames) instead of max(pA, pB)
  - VLM clarity score: prompt now asks for fine-grained clarity (0-1 scale)
  - VLM error tracking: failed calls are flagged, not silently averaged in
  - Fallback fixed: no longer picks MIN arc when VLM fails
  - All modes evaluated on SAME candidate pool (no extra VLM calls)

Output: outputs/step2_corrected/results.json
"""

import argparse, json, sys, time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler

from evaluation.vlm_steering_experiment import (
    load_policy,
    sample_chunk,
    stabilize_gripper,
    infer_block,
    enforce_block_direction,
    run_5sec_rollout,
    score_candidate_vlm,
    measure_arc,
    arc_class,
    is_arc15_19,
    ARC_T1, ARC_T2, ARC_T3,
    GOAL_A, GOAL_B,
)
from scripts.vlm_client import LegibilityScorer


SELECTION_MODES = ["legible_max_arc", "max_legibility", "vlm_weighted"]


def select_candidate(pool, mode, leg_threshold=0.55, max_arc_cap=0.18):
    """Apply a selection mode to a scored candidate pool.

    All modes now use direction-aware legibility_score = P(correct goal | frames).
    """
    # Filter out VLM-failed candidates
    vlm_ok = [c for c in pool if not c["vlm"].get("vlm_error", False)]
    p = vlm_ok if vlm_ok else pool

    if mode == "legible_max_arc":
        legible = [c for c in p if c["vlm"]["legibility_score"] >= leg_threshold]
        if legible:
            safe = [c for c in legible if c["arc"] <= max_arc_cap]
            if safe:
                return max(safe, key=lambda c: c["arc"])
            return min(legible, key=lambda c: c["arc"])
        safe_all = [c for c in p if c["arc"] <= max_arc_cap]
        if safe_all:
            return max(safe_all, key=lambda c: c["arc"])
        return max(p, key=lambda c: c["arc"])  # FIXED: max, not min

    elif mode == "max_legibility":
        return max(p, key=lambda c: c["vlm"]["legibility_score"])

    elif mode == "vlm_weighted":
        def hybrid(c):
            leg = c["vlm"]["legibility_score"]
            arc_bonus = min(c["arc"] / max_arc_cap, 1.0)
            return leg * (1.0 + 0.5 * arc_bonus)
        return max(p, key=hybrid)

    raise ValueError(f"Unknown mode: {mode}")


def replay_to_completion(env, model, sampler, obs_mean, obs_std,
                         act_mean, act_std, device, episode_seed,
                         selected_candidate, max_steps=400):
    """Replay selected candidate's actions, then replan to completion."""
    sel = selected_candidate
    replay_seed = sel["seed"] + 9999
    np.random.seed(replay_seed)
    torch.manual_seed(replay_seed)

    obs = env.reset(seed=episode_seed)
    q = deque(sel["actions"])
    done = False
    steps = 0
    result = None

    while not done and steps < max_steps:
        if len(q) == 0:
            seq = sample_chunk(model, sampler, obs,
                               obs_mean, obs_std, act_mean, act_std, device)
            q.extend(seq)
        action = q.popleft()
        result = env.step(action)
        obs = result.obs
        done = bool(result.done)
        steps += 1

    info = result.info if result else {}
    success = (info.get("success_left", 0) > 0.5 or
               info.get("success_right", 0) > 0.5)
    return success, steps


def run_episode(env, model, sampler, scorer,
                obs_mean, obs_std, act_mean, act_std,
                device, episode_seed, target_block,
                n_candidates=10, max_steps=400):
    """Generate K candidates, VLM-score all, apply all 3 modes, replay each."""

    candidates = []

    # Phase 1: Generate K candidates
    for j in range(n_candidates):
        cseed = episode_seed * 100 + 1000 + j
        np.random.seed(cseed)
        torch.manual_seed(cseed)

        obs = env.reset(seed=episode_seed)
        ro = run_5sec_rollout(env, model, sampler, obs,
                              obs_mean, obs_std, act_mean, act_std, device,
                              target_block, capture_frames=True)

        candidates.append(dict(
            idx=j, seed=cseed,
            arc=ro["arc"], arc_class=arc_class(ro["arc"]),
            is_arc15_19=is_arc15_19(ro["arc"]),
            direction_match=ro["direction_match"],
            frames=ro["frames"],
            actions=ro["actions"],
        ))

    # Phase 2: VLM-score all (direction-aware, with clarity)
    vlm_errors = 0
    for ci, c in enumerate(candidates):
        if not c["frames"]:
            c["vlm"] = dict(legibility_score=0.5, undirected_legibility=0.5,
                            clarity=0.5, pA=0.5, pB=0.5,
                            target_goal="A" if target_block == "LEFT" else "B",
                            choice="C", cue="no_frames",
                            vlm_error=True, error="no frames")
            vlm_errors += 1
            continue
        vid = f"ep{episode_seed}_c{c['idx']}"
        t_vlm = time.time()
        c["vlm"] = score_candidate_vlm(scorer, c["frames"], vid,
                                        target_block=target_block)
        dt_vlm = time.time() - t_vlm
        pA = c["vlm"].get("pA", "?")
        pB = c["vlm"].get("pB", "?")
        clr = c["vlm"].get("clarity", "?")
        err = c["vlm"].get("vlm_error", False)
        print(f"      VLM c{c['idx']}: pA={pA} pB={pB} clr={clr} "
              f"err={err} ({dt_vlm:.1f}s)")
        if err:
            vlm_errors += 1
        # Rate limit: small delay between VLM calls to avoid throttling
        if ci < len(candidates) - 1:
            time.sleep(1.0)

    # Phase 3: Apply all selection modes to same pool
    pool = [c for c in candidates if c["direction_match"]]
    if not pool:
        pool = candidates

    mode_results = {}
    for mode in SELECTION_MODES:
        sel = select_candidate(pool, mode)

        # Phase 4: Replay to completion
        success, steps = replay_to_completion(
            env, model, sampler, obs_mean, obs_std,
            act_mean, act_std, device, episode_seed, sel, max_steps)

        mode_results[mode] = dict(
            selected_idx=sel["idx"],
            arc=float(sel["arc"]),
            arc_class=sel["arc_class"],
            vlm_legibility=float(sel["vlm"]["legibility_score"]),
            vlm_clarity=float(sel["vlm"].get("clarity", 0.5)),
            undirected_leg=float(sel["vlm"].get("undirected_legibility", 0.5)),
            vlm_choice=sel["vlm"]["choice"],
            vlm_cue=sel["vlm"].get("cue", ""),
            success=success,
            steps=steps,
        )

    # Candidate summary
    cand_summary = [
        dict(idx=c["idx"],
             arc=float(c["arc"]),
             arc_class=arc_class(c["arc"]),
             vlm_legibility=float(c["vlm"]["legibility_score"]),
             vlm_clarity=float(c["vlm"].get("clarity", 0.5)),
             undirected_leg=float(c["vlm"].get("undirected_legibility", 0.5)),
             vlm_choice=c["vlm"]["choice"],
             vlm_error=c["vlm"].get("vlm_error", False),
             direction_match=c["direction_match"])
        for c in candidates
    ]

    # Check agreement between modes
    picks = {m: mode_results[m]["selected_idx"] for m in SELECTION_MODES}
    all_same = len(set(picks.values())) == 1

    return dict(
        episode_seed=episode_seed,
        target_block=target_block,
        picks=picks,
        all_same=all_same,
        vlm_errors=vlm_errors,
        n_candidates=n_candidates,
        candidates=cand_summary,
        **mode_results,
    )


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint",
                    default="runs/diffusion_20260222_195530/ckpt_ep100.pt")
    pa.add_argument("--n-rollouts", type=int, default=20)
    pa.add_argument("--n-candidates", type=int, default=10)
    pa.add_argument("--base-seed", type=int, default=100)
    pa.add_argument("--output-dir", default="outputs/step2_corrected")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'█' * 70}")
    print(f"  STEP 2: Corrected VLM Steering (3 modes)")
    print(f"{'█' * 70}")
    print(f"  Modes       : {', '.join(SELECTION_MODES)}")
    print(f"  N episodes  : {args.n_rollouts}")
    print(f"  K candidates: {args.n_candidates}")
    print(f"  VLM calls   : {args.n_rollouts * args.n_candidates}")
    print(f"  Device      : {device}")
    print(f"  FIXES APPLIED:")
    print(f"    ✓ Direction-aware legibility = P(correct goal | frames)")
    print(f"    ✓ VLM clarity score (fine-grained 0-1)")
    print(f"    ✓ VLM errors tracked & excluded from selection")
    print(f"    ✓ Fallback picks MAX arc (not min)")
    print(f"{'█' * 70}\n")

    # Load policy
    print("Loading policy ...")
    model, sampler, obs_m, obs_s, act_m, act_s, cfg = \
        load_policy(args.checkpoint, device)
    print(f"  horizon={cfg['horizon']}  act_dim={cfg['act_dim']}")

    # Init VLM
    print("Initializing VLM ...")
    scorer = LegibilityScorer(model="gemini-2.5-flash")

    # Target blocks
    baseline_path = Path("outputs/vlm_steering_experiment/no_steering_results.json")
    target_blocks = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        for r in bl.get("rollouts", []):
            target_blocks[r["episode_seed"]] = r["target_block"]
        print(f"  Loaded {len(target_blocks)} target blocks from baseline\n")

    # Run
    env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
    all_results = []
    t0 = time.time()
    total_vlm_errors = 0

    for i in range(args.n_rollouts):
        ep = args.base_seed + i

        if ep in target_blocks:
            tgt = target_blocks[ep]
        else:
            obs = env.reset(seed=ep)
            np.random.seed(ep * 100 + 7)
            torch.manual_seed(ep * 100 + 7)
            seq = sample_chunk(model, sampler, obs,
                               obs_m, obs_s, act_m, act_s, device)
            tgt = infer_block(seq)

        print(f"\n  [{i+1:3d}/{args.n_rollouts}]  ep={ep}  tgt={tgt}  "
              f"(K={args.n_candidates})")

        r = run_episode(env, model, sampler, scorer,
                        obs_m, obs_s, act_m, act_s, device,
                        episode_seed=ep, target_block=tgt,
                        n_candidates=args.n_candidates)

        all_results.append(r)
        total_vlm_errors += r["vlm_errors"]

        for mode in SELECTION_MODES:
            mr = r[mode]
            ok = "OK" if mr["success"] else "FAIL"
            print(f"    {mode:20s} → c{mr['selected_idx']}  "
                  f"arc={mr['arc']:.4f}m  leg={mr['vlm_legibility']:.3f}  "
                  f"clr={mr['vlm_clarity']:.2f}  {ok}")

        tag = "ALL_SAME" if r["all_same"] else "DIFFER"
        errs = f"  vlm_err={r['vlm_errors']}" if r["vlm_errors"] else ""
        print(f"    [{tag}]{errs}")

        # Save progress
        with open(out / "progress.json", "w") as f:
            json.dump(dict(completed=i + 1, total=args.n_rollouts,
                           latest_ep=ep), f, indent=2)

    env.close()
    dt = time.time() - t0

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    n = len(all_results)

    summary = dict(
        n_episodes=n,
        n_candidates=args.n_candidates,
        total_vlm_calls=n * args.n_candidates,
        total_vlm_errors=total_vlm_errors,
        vlm_error_rate=round(total_vlm_errors / (n * args.n_candidates), 4),
        runtime_sec=round(dt, 1),
    )

    for mode in SELECTION_MODES:
        successes = sum(1 for r in all_results if r[mode]["success"])
        arcs = [r[mode]["arc"] for r in all_results]
        legs = [r[mode]["vlm_legibility"] for r in all_results]
        clrs = [r[mode]["vlm_clarity"] for r in all_results]
        uds = [r[mode]["undirected_leg"] for r in all_results]

        arc_dist = {"00-04": 0, "05-09": 0, "10-14": 0, "15-19": 0}
        for r in all_results:
            arc_dist[r[mode]["arc_class"]] += 1

        summary[mode] = dict(
            success_rate=round(successes / n, 4),
            success_count=successes,
            mean_arc=round(float(np.mean(arcs)), 5),
            std_arc=round(float(np.std(arcs)), 5),
            mean_directed_legibility=round(float(np.mean(legs)), 4),
            mean_undirected_legibility=round(float(np.mean(uds)), 4),
            mean_clarity=round(float(np.mean(clrs)), 4),
            arc_distribution=arc_dist,
        )

    # Agreement matrix
    for m1 in SELECTION_MODES:
        for m2 in SELECTION_MODES:
            if m1 >= m2:
                continue
            same = sum(1 for r in all_results
                       if r[m1]["selected_idx"] == r[m2]["selected_idx"])
            key = f"agree_{m1}_vs_{m2}"
            summary[key] = dict(same=same, rate=round(same / n, 4))

    # Print results
    print(f"\n\n{'█' * 70}")
    print(f"  STEP 2 RESULTS (CORRECTED)")
    print(f"{'█' * 70}")
    print(f"\n  Episodes: {n}  |  VLM calls: {n * args.n_candidates}  |  "
          f"VLM errors: {total_vlm_errors}  |  Runtime: {dt:.0f}s")

    hdr = f"\n  {'Metric':<28}"
    for mode in SELECTION_MODES:
        hdr += f"{mode:>20}"
    sep = f"  {'─' * 28}" + "─" * 20 * len(SELECTION_MODES)
    print(f"{hdr}\n{sep}")

    for label, key in [("Success rate", "success_rate"),
                       ("Mean arc (m)", "mean_arc"),
                       ("Dir. legibility", "mean_directed_legibility"),
                       ("Undir. legibility", "mean_undirected_legibility"),
                       ("Clarity", "mean_clarity")]:
        row = f"  {label:<28}"
        for mode in SELECTION_MODES:
            v = summary[mode][key]
            if "rate" in key or "legibility" in key or key == "mean_clarity":
                row += f"{v:>19.3f} "
            else:
                row += f"{v:>19.4f} "
        print(row)

    for ac in ["00-04", "05-09", "10-14", "15-19"]:
        row = f"  {'Arc ' + ac:<28}"
        for mode in SELECTION_MODES:
            row += f"{summary[mode]['arc_distribution'][ac]:>19} "
        print(row)

    print(f"\n  Agreement between modes:")
    for m1 in SELECTION_MODES:
        for m2 in SELECTION_MODES:
            if m1 >= m2:
                continue
            k = f"agree_{m1}_vs_{m2}"
            s = summary[k]
            print(f"    {m1} vs {m2}: {s['same']}/{n} ({s['rate']:.0%})")

    print(f"{'█' * 70}\n")

    # Save
    final = dict(
        experiment="step2_corrected_vlm_steering",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        fixes_applied=[
            "direction_aware_legibility",
            "vlm_clarity_score",
            "vlm_error_tracking",
            "fallback_max_arc_not_min",
        ],
        args=vars(args),
        summary=summary,
        episodes=all_results,
    )
    jp = out / "results.json"
    with open(jp, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"  Results → {jp}\n")


if __name__ == "__main__":
    main()
