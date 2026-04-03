#!/usr/bin/env python3
"""Legibility Diffuser + VLM Reranking evaluation.

Three conditions:
  1. Baseline   — unconditioned DiffusionPolicy (no goal, no VLM)
  2. LegDiff    — CFG goal-conditioned (w=3.0, no VLM)
  3. LegDiff+VLM— CFG goal-conditioned + VLM best-of-K reranking

Usage:
  python scripts/eval_legdiff_vlm_rerank.py \\
      --checkpoint runs/legdiff_20260331_021740/ckpt_ep100.pt \\
      --n_episodes 10 --cfg_scale 3.0 --K 3
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pybullet as pb
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# Import all model/sampler/metric code from the WORKING eval script
from evaluation.eval_legibility_diffuser import (
    GOAL_LEFT, GOAL_RIGHT, NULL_GOAL,
    GoalCondDiffusionPolicy, DiffusionPolicy,
    LegDiffDDIMSampler, BaselineDDIMSampler,
    run_baseline_episode, run_legdiff_episode,
    _load_baseline, _load_legdiff, _measure_legibility,
)

STEPS_PER_SEC = 30
DEFAULT_BASELINE = 'runs/diffusion_20260222_195530/ckpt_ep100.pt'
GOAL_A = "pick the left block"
GOAL_B = "pick the right block"


# ══════════════════════════════════════════════════════════════════════
# VLM SCORING
# ══════════════════════════════════════════════════════════════════════

def capture_frame(env, width=480, height=480):
    return env.render(mode='rgb_array', width=width, height=height)


def frame_to_jpeg_bytes(frame_np):
    img = Image.fromarray(frame_np)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def annotate_frame(frame_np, t_sec, idx, total):
    img = Image.fromarray(frame_np.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((5, 5), f"t={t_sec}s ({idx}/{total})", fill=(255, 255, 0), font=font)
    return np.array(img)


def score_candidate_vlm(scorer, frames_bytes, video_id, target_block="RIGHT"):
    target_goal = "A" if target_block == "LEFT" else "B"
    try:
        r = scorer.score_trajectory(
            image_bytes=frames_bytes,
            goal_A=GOAL_A, goal_B=GOAL_B,
            mode="prefix_frames",
            video_id=video_id, t_sec=5.0,
            target_goal=target_goal,
        )
        return dict(
            legibility_score=float(r.get("legibility_score", 0.5)),
            pA=float(r.get("pA", 0.5)),
            pB=float(r.get("pB", 0.5)),
            choice=r.get("choice", "C"),
            cue=str(r.get("cue", "")),
            confidence=int(r.get("confidence", 50)),
            vlm_error=bool(r.get("vlm_error", False)),
            latency_ms=r.get("latency_ms", 0),
        )
    except Exception as exc:
        print(f"      [VLM error: {exc}]")
        return dict(legibility_score=0.5, pA=0.5, pB=0.5, choice="C",
                    cue=f"ERROR: {exc}", confidence=0, vlm_error=True, latency_ms=0)


# ══════════════════════════════════════════════════════════════════════
# VLM EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════

def _save_env_state(env):
    """Save both PyBullet and Python-side env state."""
    return dict(
        pb_state=pb.saveState(physicsClientId=env._cid),
        target_pos=env._target_pos.copy(),
        target_yaw=float(env._target_yaw),
        grip_cmd=float(env._grip_cmd),
        episode_steps=env._episode_steps,
        picked_left=env._picked_left,
        picked_right=env._picked_right,
    )


def _restore_env_state(env, saved):
    """Restore both PyBullet and Python-side env state."""
    pb.restoreState(saved['pb_state'], physicsClientId=env._cid)
    env._target_pos = saved['target_pos'].copy()
    env._target_yaw = saved['target_yaw']
    env._grip_cmd = saved['grip_cmd']
    env._episode_steps = saved['episode_steps']
    env._picked_left = saved['picked_left']
    env._picked_right = saved['picked_right']


def run_legdiff_vlm_episode(
    model, sampler, obs_mean, obs_std, act_mean, act_std,
    device, scorer, cfg_scale=3.0, K=3,
    n_sampling_steps=10, max_steps=400, sim_steps=32,
):
    """LegDiff + VLM reranking episode.

    At the FIRST replan:
      1. Detect goal (unconditioned)
      2. Generate K candidate CFG action chunks
      3. For each: save state -> simulate one chunk -> capture frames
      4. VLM scores each
      5. Restore state -> execute best candidate
    Subsequent replans: single CFG sample (no VLM, for speed).
    """
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset()
    queue: deque = deque(maxlen=model.horizon)
    ee_traj = []
    success = False
    last_obs = obs
    committed_goal = None
    vlm_used = False
    vlm_scores = []
    selected_idx = -1
    first_replan = True

    for step in range(max_steps):
        ee_traj.append(obs[0:3].copy())
        if len(queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            # Detect goal on first plan
            if committed_goal is None:
                gid = sampler.detect_goal(model, obs_t, obs, n_sampling_steps)
                committed_goal = torch.tensor([gid], dtype=torch.long, device=device)

            if first_replan and scorer is not None:
                # ── VLM reranking on first replan ──
                first_replan = False
                vlm_used = True
                saved = _save_env_state(env)
                target_block = "LEFT" if committed_goal.item() == GOAL_LEFT else "RIGHT"
                candidates = []

                for k in range(K):
                    # Each sample gets different noise -> different trajectory
                    chunk = sampler.sample_cfg(model, obs_t, committed_goal,
                                               cfg_scale=cfg_scale,
                                               n_sampling_steps=n_sampling_steps)
                    chunk_np = chunk[0].cpu().numpy()
                    actions_raw = chunk_np * act_std + act_mean

                    # Restore state and simulate this candidate
                    _restore_env_state(env, saved)
                    frames_bytes = []
                    n_sim = min(sim_steps, len(actions_raw))
                    for s in range(n_sim):
                        env.step(actions_raw[s])
                        # Capture frame every 10 steps + 1st step
                        if s == 0 or (s + 1) % 10 == 0:
                            frame = capture_frame(env)
                            ann = annotate_frame(frame, round((s+1)/30, 1),
                                                 len(frames_bytes)+1, 4)
                            frames_bytes.append(frame_to_jpeg_bytes(ann))

                    candidates.append(dict(
                        idx=k, chunk_np=chunk_np,
                        actions_raw=actions_raw,
                        frames_bytes=frames_bytes,
                    ))

                # Score with VLM
                for c in candidates:
                    if c['frames_bytes']:
                        c['vlm'] = score_candidate_vlm(
                            scorer, c['frames_bytes'],
                            f"cand_{c['idx']}", target_block)
                    else:
                        c['vlm'] = dict(legibility_score=0.5, vlm_error=True)
                    vlm_scores.append(c['vlm'])
                    time.sleep(0.3)  # Rate limit

                # Select best candidate
                best = max(candidates, key=lambda c: c['vlm']['legibility_score'])
                selected_idx = best['idx']

                # Restore state and execute best candidate's actions
                _restore_env_state(env, saved)
                pb.removeState(saved['pb_state'], physicsClientId=env._cid)

                for a in best['actions_raw']:
                    queue.append(a)
            else:
                # Regular CFG sample (no VLM)
                chunk = sampler.sample_cfg(model, obs_t, committed_goal,
                                           cfg_scale=cfg_scale,
                                           n_sampling_steps=n_sampling_steps)
                for a in (chunk[0].cpu().numpy() * act_std + act_mean):
                    queue.append(a)

        action = queue.popleft()
        result = env.step(action)
        obs = result.obs; last_obs = obs
        success = result.info.get('success_left', 0) > 0.5 or result.info.get('success_right', 0) > 0.5
        if result.done:
            break

    env.close()
    le, goal = _measure_legibility(np.array(ee_traj), last_obs)
    return dict(
        success=success, steps=step+1, l_early=le, true_goal=goal,
        committed_goal='left' if committed_goal.item() == GOAL_LEFT else 'right',
        vlm_used=vlm_used,
        vlm_scores=[s.get('legibility_score', 0.5) for s in vlm_scores],
        selected_idx=selected_idx,
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--checkpoint', required=True,
                    help='LegDiff checkpoint (train_legibility_diffuser.py)')
    ap.add_argument('--baseline_checkpoint', default=DEFAULT_BASELINE)
    ap.add_argument('--n_episodes', type=int, default=10)
    ap.add_argument('--cfg_scale', type=float, default=3.0)
    ap.add_argument('--K', type=int, default=3,
                    help='Number of CFG candidates for VLM reranking')
    ap.add_argument('--skip_baseline', action='store_true')
    ap.add_argument('--skip_legdiff', action='store_true')
    ap.add_argument('--skip_vlm', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"  Legibility Diffuser + VLM Reranking Evaluation")
    print(f"{'='*65}")
    print(f"  Device       : {device}")
    print(f"  LegDiff ckpt : {args.checkpoint}")
    print(f"  CFG scale w  : {args.cfg_scale}")
    print(f"  VLM K        : {args.K}")
    print(f"  Episodes     : {args.n_episodes}")
    print(f"{'='*65}\n")

    # Load LegDiff model using the WORKING loader
    ld_model, ld_ckpt = _load_legdiff(args.checkpoint, device)
    ld_cfg = ld_ckpt['config']
    ld_sampler = LegDiffDDIMSampler(
        ld_cfg['n_diffusion_steps'], ld_cfg['beta_start'], ld_cfg['beta_end'],
        device, cfg_scale=args.cfg_scale)
    ld_obs_mean = np.array(ld_ckpt['obs_mean'], dtype=np.float32)
    ld_obs_std  = np.array(ld_ckpt['obs_std'],  dtype=np.float32)
    ld_act_mean = np.array(ld_ckpt['act_mean'], dtype=np.float32)
    ld_act_std  = np.array(ld_ckpt['act_std'],  dtype=np.float32)
    print(f"  LegDiff: {sum(p.numel() for p in ld_model.parameters()):,} params, "
          f"epoch {ld_ckpt['epoch']}, loss {ld_ckpt['loss']:.6f}")

    # Load VLM scorer
    scorer = None
    if not args.skip_vlm:
        try:
            from scripts.vlm_client import LegibilityScorer
            scorer = LegibilityScorer(model='gemini-2.5-flash')
            print(f"  VLM: LegibilityScorer initialized (gemini-2.5-flash)")
        except Exception as e:
            print(f"  VLM: FAILED to initialize ({e})")
            print(f"  Skipping VLM condition.")

    # ── 1. Baseline ──────────────────────────────────────────────
    baseline_results = []
    if not args.skip_baseline:
        bl_model, bl_ckpt = _load_baseline(args.baseline_checkpoint, device)
        bl_cfg = bl_ckpt['config']
        bl_sampler = BaselineDDIMSampler(
            bl_cfg['n_diffusion_steps'], bl_cfg['beta_start'], bl_cfg['beta_end'], device)
        bl_obs_mean = np.array(bl_ckpt['obs_mean'], dtype=np.float32)
        bl_obs_std  = np.array(bl_ckpt['obs_std'],  dtype=np.float32)
        bl_act_mean = np.array(bl_ckpt['act_mean'], dtype=np.float32)
        bl_act_std  = np.array(bl_ckpt['act_std'],  dtype=np.float32)
        print(f"  Baseline: {sum(p.numel() for p in bl_model.parameters()):,} params")

        print(f"\n-- BASELINE (unconditioned) -- {args.n_episodes} episodes --")
        for ep in range(args.n_episodes):
            r = run_baseline_episode(bl_model, bl_sampler,
                                     bl_obs_mean, bl_obs_std, bl_act_mean, bl_act_std,
                                     device, n_sampling_steps=10)
            baseline_results.append(r)
            tick = 'OK' if r['success'] else 'X '
            print(f"  Ep {ep+1:>2}/{args.n_episodes} [{tick}]  "
                  f"L_early={r['l_early']:.4f}  goal={r['true_goal']}  steps={r['steps']}")

    # ── 2. LegDiff only ──────────────────────────────────────────
    legdiff_results = []
    if not args.skip_legdiff:
        print(f"\n-- LEGDIFF (CFG w={args.cfg_scale}) -- {args.n_episodes} episodes --")
        for ep in range(args.n_episodes):
            r = run_legdiff_episode(ld_model, ld_sampler,
                                    ld_obs_mean, ld_obs_std, ld_act_mean, ld_act_std,
                                    device, cfg_scale=args.cfg_scale, n_sampling_steps=10)
            legdiff_results.append(r)
            tick = 'OK' if r['success'] else 'X '
            print(f"  Ep {ep+1:>2}/{args.n_episodes} [{tick}]  "
                  f"L_early={r['l_early']:.4f}  "
                  f"committed={r.get('committed_goal','?')}  "
                  f"actual={r['true_goal']}  steps={r['steps']}")

    # ── 3. LegDiff + VLM reranking ───────────────────────────────
    vlm_results = []
    if scorer is not None:
        print(f"\n-- LEGDIFF+VLM (CFG w={args.cfg_scale}, K={args.K}) -- "
              f"{args.n_episodes} episodes --")
        for ep in range(args.n_episodes):
            t0 = time.time()
            r = run_legdiff_vlm_episode(
                ld_model, ld_sampler,
                ld_obs_mean, ld_obs_std, ld_act_mean, ld_act_std,
                device, scorer,
                cfg_scale=args.cfg_scale, K=args.K,
                n_sampling_steps=10)
            dt = time.time() - t0
            vlm_results.append(r)
            tick = 'OK' if r['success'] else 'X '
            vlm_scores_str = ', '.join(f"{s:.2f}" for s in r.get('vlm_scores', []))
            print(f"  Ep {ep+1:>2}/{args.n_episodes} [{tick}]  "
                  f"L_early={r['l_early']:.4f}  "
                  f"committed={r.get('committed_goal','?')}  "
                  f"actual={r['true_goal']}  "
                  f"selected=c{r.get('selected_idx', '?')}  "
                  f"vlm=[{vlm_scores_str}]  "
                  f"{dt:.0f}s")

    # ── Comparison ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  COMPARISON")
    print(f"{'='*65}")
    for label, results in [("Baseline", baseline_results),
                           ("LegDiff", legdiff_results),
                           ("LegDiff+VLM", vlm_results)]:
        if results:
            le = np.mean([r['l_early'] for r in results])
            sr = np.mean([r['success'] for r in results])
            print(f"  {label:<15}  L_early={le:.4f}  Success={sr:.1%}")

    # ── Save ─────────────────────────────────────────────────────
    def _s(v):
        if isinstance(v, (bool, np.bool_)): return bool(v)
        if isinstance(v, (float, np.floating)): return float(v)
        if isinstance(v, (int, np.integer)): return int(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    def clean(results):
        out = []
        for r in results:
            out.append({k: _s(v) for k, v in r.items()
                        if k not in ('chunk_np', 'actions_raw')})
        return out

    out = dict(
        checkpoint=str(args.checkpoint),
        baseline_checkpoint=str(args.baseline_checkpoint),
        cfg_scale=args.cfg_scale, K=args.K,
        n_episodes=args.n_episodes,
        baseline=clean(baseline_results),
        legdiff=clean(legdiff_results),
        legdiff_vlm=clean(vlm_results),
    )
    out_dir = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'legdiff_vlm_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == '__main__':
    main()
