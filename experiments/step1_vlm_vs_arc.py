#!/usr/bin/env python3
"""
Step 1 Validation:  max_legibility vs legible_max_arc
=====================================================

For each episode:
  1. Generate K candidates (same seeds)
  2. Simulate 5-sec rollout for each
  3. VLM-score ALL candidates (one set of API calls)
  4. Apply BOTH selection modes to the same scored pool
  5. Record which candidate each mode picks, arc, VLM score

This lets us answer: does VLM argmax pick a different candidate
than arc argmax?  If so, is the VLM-selected one actually more
legible by independent VLM evaluation?

Output: outputs/step1_comparison/results.json
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

# ── Reuse functions from vlm_steering_experiment ──────────────────
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


def select_legible_max_arc(pool, leg_threshold=0.55, max_arc_cap=0.18):
    """legible_max_arc mode: VLM as binary filter, argmax arc."""
    legible = [c for c in pool
               if c["vlm"]["legibility_score"] >= leg_threshold]
    if legible:
        safe = [c for c in legible if c["arc"] <= max_arc_cap]
        if safe:
            return max(safe, key=lambda c: c["arc"])
        else:
            return min(legible, key=lambda c: c["arc"])
    else:
        safe_all = [c for c in pool if c["arc"] <= max_arc_cap]
        if safe_all:
            return max(safe_all, key=lambda c: c["arc"])
        else:
            return min(pool, key=lambda c: c["arc"])


def select_max_legibility(pool):
    """max_legibility mode: pure VLM argmax."""
    return max(pool, key=lambda c: c["vlm"]["legibility_score"])


def run_episode_comparison(env, model, sampler, scorer,
                           obs_mean, obs_std, act_mean, act_std,
                           device, episode_seed, target_block,
                           n_candidates=10, max_steps=400):
    """
    Generate K candidates, VLM-score all, apply both selection modes,
    replay BOTH selected candidates to completion.
    """
    candidates = []

    # ── Phase 1: Generate K candidates with 5-sec rollout ────────
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

    # ── Phase 2: VLM-score all candidates (one set of API calls) ─
    for c in candidates:
        if not c["frames"]:
            c["vlm"] = dict(legibility_score=0.5, pA=0.5, pB=0.5,
                            choice="C", cue="no_frames", error="no frames")
            continue
        vid = f"ep{episode_seed}_c{c['idx']}"
        c["vlm"] = score_candidate_vlm(scorer, c["frames"], vid)

    # ── Phase 3: Apply both selection modes ──────────────────────
    pool = [c for c in candidates if c["direction_match"]]
    if not pool:
        pool = candidates

    sel_arc = select_legible_max_arc(pool)
    sel_vlm = select_max_legibility(pool)

    same_pick = (sel_arc["idx"] == sel_vlm["idx"])

    # ── Phase 4: Replay both selected candidates to completion ───
    results = {}
    for mode_name, sel in [("legible_max_arc", sel_arc),
                           ("max_legibility", sel_vlm)]:
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
                                   obs_mean, obs_std, act_mean, act_std,
                                   device)
                q.extend(seq)
            action = q.popleft()
            result = env.step(action)
            obs = result.obs
            done = bool(result.done)
            steps += 1

        info = result.info if result else {}
        success = (info.get("success_left", 0) > 0.5 or
                   info.get("success_right", 0) > 0.5)

        results[mode_name] = dict(
            selected_idx=sel["idx"],
            arc=float(sel["arc"]),
            arc_class=sel["arc_class"],
            vlm_legibility=float(sel["vlm"]["legibility_score"]),
            vlm_choice=sel["vlm"]["choice"],
            success=success,
            steps=steps,
        )

    # ── Candidate summary ────────────────────────────────────────
    cand_summary = [
        dict(idx=c["idx"],
             arc=float(c["arc"]),
             arc_class=arc_class(c["arc"]),
             vlm_legibility=float(c["vlm"]["legibility_score"]),
             vlm_choice=c["vlm"]["choice"],
             direction_match=c["direction_match"])
        for c in candidates
    ]

    return dict(
        episode_seed=episode_seed,
        target_block=target_block,
        same_pick=same_pick,
        legible_max_arc=results["legible_max_arc"],
        max_legibility=results["max_legibility"],
        candidates=cand_summary,
        n_candidates=n_candidates,
    )


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint",
                    default="runs/diffusion_20260222_195530/ckpt_ep100.pt")
    pa.add_argument("--n-rollouts", type=int, default=20)
    pa.add_argument("--n-candidates", type=int, default=10)
    pa.add_argument("--base-seed", type=int, default=100)
    pa.add_argument("--output-dir", default="outputs/step1_comparison")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'█'*70}")
    print(f"  STEP 1:  max_legibility vs legible_max_arc")
    print(f"{'█'*70}")
    print(f"  N episodes  : {args.n_rollouts}")
    print(f"  K candidates: {args.n_candidates}")
    print(f"  VLM calls   : {args.n_rollouts * args.n_candidates}")
    print(f"  Device      : {device}")
    print(f"{'█'*70}\n")

    # ── Load policy ──────────────────────────────────────────────
    print("Loading policy ...")
    model, sampler, obs_m, obs_s, act_m, act_s, cfg = \
        load_policy(args.checkpoint, device)
    print(f"  horizon={cfg['horizon']}  act_dim={cfg['act_dim']}")

    # ── Init VLM + smoke test ────────────────────────────────────
    print("Initializing VLM ...")
    scorer = LegibilityScorer(model="gemini-2.5-flash")

    # ── Determine target blocks (from baseline if available) ─────
    baseline_path = Path("outputs/vlm_steering_experiment/no_steering_results.json")
    target_blocks = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        for r in bl.get("rollouts", []):
            target_blocks[r["episode_seed"]] = r["target_block"]
        print(f"  Loaded {len(target_blocks)} target blocks from baseline\n")

    # ── Run episodes ─────────────────────────────────────────────
    env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
    all_results = []
    t0 = time.time()

    for i in range(args.n_rollouts):
        ep = args.base_seed + i

        # Determine target
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

        r = run_episode_comparison(
            env, model, sampler, scorer,
            obs_m, obs_s, act_m, act_s, device,
            episode_seed=ep, target_block=tgt,
            n_candidates=args.n_candidates)

        all_results.append(r)

        arc_sel = r["legible_max_arc"]
        vlm_sel = r["max_legibility"]
        same = "SAME" if r["same_pick"] else "DIFF"
        print(f"    arc_mode → c{arc_sel['selected_idx']}  "
              f"arc={arc_sel['arc']:.4f}m  leg={arc_sel['vlm_legibility']:.3f}  "
              f"{'OK' if arc_sel['success'] else 'FAIL'}")
        print(f"    vlm_mode → c{vlm_sel['selected_idx']}  "
              f"arc={vlm_sel['arc']:.4f}m  leg={vlm_sel['vlm_legibility']:.3f}  "
              f"{'OK' if vlm_sel['success'] else 'FAIL'}")
        print(f"    [{same}]")

        # Save intermediate
        with open(out / "progress.json", "w") as f:
            json.dump(dict(completed=i+1, total=args.n_rollouts,
                           latest=r), f, indent=2, default=str)

    env.close()
    dt = time.time() - t0

    # ═════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═════════════════════════════════════════════════════════════
    n = len(all_results)
    same_count = sum(1 for r in all_results if r["same_pick"])

    # Per-mode stats
    arc_success = sum(1 for r in all_results if r["legible_max_arc"]["success"])
    vlm_success = sum(1 for r in all_results if r["max_legibility"]["success"])

    arc_arcs = [r["legible_max_arc"]["arc"] for r in all_results]
    vlm_arcs = [r["max_legibility"]["arc"] for r in all_results]

    arc_legs = [r["legible_max_arc"]["vlm_legibility"] for r in all_results]
    vlm_legs = [r["max_legibility"]["vlm_legibility"] for r in all_results]

    # Arc class distributions
    def arc_dist(results, mode):
        d = {"00-04": 0, "05-09": 0, "10-14": 0, "15-19": 0}
        for r in results:
            d[r[mode]["arc_class"]] += 1
        return d

    arc_dist_arc = arc_dist(all_results, "legible_max_arc")
    arc_dist_vlm = arc_dist(all_results, "max_legibility")

    # When they differ, which has higher VLM score?
    diff_episodes = [r for r in all_results if not r["same_pick"]]
    vlm_wins_leg = sum(1 for r in diff_episodes
                       if r["max_legibility"]["vlm_legibility"] >
                          r["legible_max_arc"]["vlm_legibility"])
    arc_wins_arc = sum(1 for r in diff_episodes
                       if r["legible_max_arc"]["arc"] >
                          r["max_legibility"]["arc"])

    summary = dict(
        n_episodes=n,
        n_candidates=args.n_candidates,
        total_vlm_calls=n * args.n_candidates,
        runtime_sec=round(dt, 1),

        same_pick_count=same_count,
        same_pick_rate=round(same_count / n, 4),
        different_pick_count=n - same_count,

        legible_max_arc=dict(
            success_rate=round(arc_success / n, 4),
            success_count=arc_success,
            mean_arc=round(float(np.mean(arc_arcs)), 5),
            std_arc=round(float(np.std(arc_arcs)), 5),
            mean_vlm_legibility=round(float(np.mean(arc_legs)), 4),
            arc_distribution=arc_dist_arc,
        ),
        max_legibility=dict(
            success_rate=round(vlm_success / n, 4),
            success_count=vlm_success,
            mean_arc=round(float(np.mean(vlm_arcs)), 5),
            std_arc=round(float(np.std(vlm_arcs)), 5),
            mean_vlm_legibility=round(float(np.mean(vlm_legs)), 4),
            arc_distribution=arc_dist_vlm,
        ),
        when_different=dict(
            count=len(diff_episodes),
            vlm_mode_has_higher_vlm_score=vlm_wins_leg,
            arc_mode_has_higher_arc=arc_wins_arc,
        ),
    )

    # ── Print results ────────────────────────────────────────────
    print(f"\n\n{'█'*70}")
    print(f"  STEP 1 RESULTS")
    print(f"{'█'*70}")
    print(f"\n  Episodes: {n}  |  VLM calls: {n * args.n_candidates}  |  "
          f"Runtime: {dt:.0f}s")

    print(f"\n  Same pick:      {same_count}/{n}  ({same_count/n:.0%})")
    print(f"  Different pick: {n-same_count}/{n}  ({(n-same_count)/n:.0%})")

    hdr = f"\n  {'Metric':<28}{'legible_max_arc':>18}{'max_legibility':>18}"
    sep = f"  {'─'*28}{'─'*18}{'─'*18}"
    print(f"{hdr}\n{sep}")
    print(f"  {'Success rate':<28}{arc_success/n:>17.0%} {vlm_success/n:>17.0%}")
    print(f"  {'Mean arc (m)':<28}{np.mean(arc_arcs):>17.4f} {np.mean(vlm_arcs):>17.4f}")
    print(f"  {'Mean VLM legibility':<28}{np.mean(arc_legs):>17.3f} {np.mean(vlm_legs):>17.3f}")

    for ac in ["00-04", "05-09", "10-14", "15-19"]:
        print(f"  {'Arc '+ac:<28}{arc_dist_arc[ac]:>17} {arc_dist_vlm[ac]:>17}")

    if diff_episodes:
        print(f"\n  When picks differ ({len(diff_episodes)} episodes):")
        print(f"    VLM mode has higher VLM score: {vlm_wins_leg}/{len(diff_episodes)}")
        print(f"    Arc mode has higher arc:        {arc_wins_arc}/{len(diff_episodes)}")
    print(f"{'█'*70}\n")

    # ── Save ─────────────────────────────────────────────────────
    final = dict(
        experiment="step1_max_legibility_vs_legible_max_arc",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
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
