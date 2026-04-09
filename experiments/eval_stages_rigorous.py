#!/usr/bin/env python3
"""
Rigorous Stage 1 / 2 / 3 Evaluation
====================================

Scientific evaluation pipeline with:
- **Paired comparisons**: Same env_seed × sample_seed for all methods
- **Stage isolation**: Each stage tested independently
- **Ablation**: Individual VLM criteria contribution
- **Guidance scale sweep**: Find optimal w per scoring function
- **Statistical tests**: Paired t-test, confidence intervals
- **Gradient diagnostics**: Verify gradient structure

Stage 1: VLM code synthesis — can Gemini produce a valid, discriminative scoring function?
Stage 2: Classifier guidance integration — does the VLM function improve legibility under gradient guidance?
Stage 3: VLM text reranking — does post-hoc selection further improve?

Usage:
  python experiments/eval_stages_rigorous.py --stage 1      # Stage 1 only
  python experiments/eval_stages_rigorous.py --stage 2      # Stage 2 (includes sweep)
  python experiments/eval_stages_rigorous.py --stage 3      # Stage 3
  python experiments/eval_stages_rigorous.py --stage all    # Full pipeline
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv
from evaluation.eval_legibility_guided import (
    DDIMSampler,
    DiffusionPolicy,
    GuidedDDIMSampler,
    l_early_intent_torch,
)

# ── constants ────────────────────────────────────────────────────────────
ACTION_SCALE = 0.05
DEFAULT_CKPT = 'runs/diffusion_20260222_195530/ckpt_ep100.pt'

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ═════════════════════════════════════════════════════════════════════════
# PAIRED EPISODE RUNNER  (deterministic seed control)
# ═════════════════════════════════════════════════════════════════════════

def run_paired_episode(
    model: DiffusionPolicy,
    sampler,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_mean: np.ndarray,
    act_std: np.ndarray,
    device: torch.device,
    env_seed: int,
    sample_seed: int,
    guided: bool = False,
    n_sampling_steps: int = 10,
    max_steps: int = 400,
) -> dict:
    """Run one episode with FIXED seeds for reproducible pairing."""
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset(seed=env_seed)
    torch.manual_seed(sample_seed)

    action_queue: deque = deque(maxlen=model.horizon)
    ee_trajectory: List[np.ndarray] = []
    guided_score_vals: List[float] = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_trajectory.append(obs[0:3].copy())

        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)

            if guided:
                ee_start = torch.tensor(obs[0:3], dtype=torch.float32, device=device)
                left_goal = torch.tensor(obs[8:11], dtype=torch.float32, device=device)
                right_goal = torch.tensor(obs[15:18], dtype=torch.float32, device=device)
                goals_t = torch.stack([left_goal, right_goal])

                act_seq, s_val = sampler.sample(
                    model, obs_t, ee_start, goals_t,
                    n_sampling_steps=n_sampling_steps)
                guided_score_vals.append(s_val)
            else:
                act_seq = sampler.sample(model, obs_t,
                                         n_sampling_steps=n_sampling_steps)

            chunk = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in chunk:
                action_queue.append(a)

        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs
        last_obs = obs
        success = (result.info.get('success_left', 0) > 0.5 or
                   result.info.get('success_right', 0) > 0.5)
        if result.done:
            break

    env.close()

    # Compute actual L_early from executed EE trajectory
    l_early, rlc, true_goal = 0.0, 0.0, 'unknown'
    if len(ee_trajectory) >= 4:
        traj_arr = np.array(ee_trajectory)
        goals_np = np.stack([last_obs[8:11], last_obs[15:18]], axis=0)
        from evaluation.legibility_metrics import compute_legibility
        r0 = compute_legibility(traj_arr, goals_np, true_goal_idx=0, model='gaussian')
        r1 = compute_legibility(traj_arr, goals_np, true_goal_idx=1, model='gaussian')
        if r0.L_early_intent >= r1.L_early_intent:
            l_early = r0.L_early_intent
            rlc = r0.relative_legibility_cost
            true_goal = 'left'
        else:
            l_early = r1.L_early_intent
            rlc = r1.relative_legibility_cost
            true_goal = 'right'

    return dict(
        env_seed=env_seed,
        sample_seed=sample_seed,
        success=bool(success),
        steps=step + 1,
        l_early_actual=float(l_early),
        rlc_actual=float(rlc),
        true_goal=true_goal,
        guided_score_mean=(float(np.mean(guided_score_vals))
                           if guided_score_vals else 0.0),
    )


# ═════════════════════════════════════════════════════════════════════════
# GENERIC GUIDED SAMPLER (accepts any scoring function)
# ═════════════════════════════════════════════════════════════════════════

class GuidedDDIMSampler:
    """DDIM with arbitrary differentiable guidance function."""

    def __init__(self, n_steps, beta_start, beta_end, device,
                 score_fn: Callable, guidance_scale: float = 10.0,
                 grad_clip: float = 1.0):
        self.device = device
        self.score_fn = score_fn
        self.guidance_scale = guidance_scale
        self.grad_clip = grad_clip
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def sample(self, model, obs, ee_pos_start, goals, n_sampling_steps=10):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.linspace(
            0, len(self.alphas_cumprod) - 1, n_sampling_steps,
            device=self.device).long()
        timesteps = torch.flip(timesteps, [0])
        final_score = 0.0

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (self.alphas_cumprod[timesteps[i + 1]]
                          if i < len(timesteps) - 1
                          else torch.tensor(1.0, device=self.device))
            sqrt_ab = torch.sqrt(alpha_t)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_t)

            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                eps_pred = model(x_in, t_batch, obs)
                x0_pred = (x_in - sqrt_1m_ab * eps_pred) / sqrt_ab
                delta_pos = x0_pred[0, :, :3] * ACTION_SCALE
                ee_traj = torch.cumsum(delta_pos, dim=0) + ee_pos_start

                with torch.no_grad():
                    s0 = self.score_fn(ee_traj.detach(), goals, 0).item()
                    s1 = self.score_fn(ee_traj.detach(), goals, 1).item()
                true_goal = 0 if s0 >= s1 else 1

                score = self.score_fn(ee_traj, goals, true_goal)
                grad = torch.autograd.grad(score, x_in)[0]

            final_score = float(score.item())

            with torch.no_grad():
                g = grad.detach()
                gn = g.norm()
                if gn > self.grad_clip:
                    g = g * (self.grad_clip / (gn + 1e-8))
                guided_eps = eps_pred.detach() - self.guidance_scale * sqrt_1m_ab * g
                x0_guided = (x - sqrt_1m_ab * guided_eps) / sqrt_ab
                if i < len(timesteps) - 1:
                    x = (torch.sqrt(alpha_prev) * x0_guided
                         + torch.sqrt(1.0 - alpha_prev) * guided_eps)
                else:
                    x = x0_guided

        return x, final_score


# ═════════════════════════════════════════════════════════════════════════
# STAGE 1: VLM CODE SYNTHESIS — ISOLATED VALIDATION
# ═════════════════════════════════════════════════════════════════════════

def stage1_validate_function(fn: Callable, device: torch.device) -> dict:
    """Rigorous validation of a scoring function, independent of any rollout.

    Tests:
    1. Gradient exists and has non-trivial magnitude
    2. Legible trajectory scores > ambiguous trajectory (discrimination)
    3. Score correlates with known L_early ordering across multiple scenarios
    4. Gradient direction analysis: does gradient point toward goal?
    5. Monotonicity: score increases as trajectory becomes more legible
    """
    results = {"passed": True, "tests": {}}
    goals = torch.tensor([[0.50, -0.07, 0.42],
                           [0.50,  0.07, 0.42]], device=device)

    # ── Test 1: Gradient existence and magnitude ──
    ee = torch.randn(32, 3, device=device, requires_grad=True)
    try:
        s = fn(ee, goals, 0)
        grad = torch.autograd.grad(s, ee)[0]
        grad_norm = float(grad.norm().item())
        results["tests"]["gradient_exists"] = {
            "passed": grad_norm > 1e-10,
            "grad_norm": grad_norm,
            "score": float(s.item()),
        }
    except Exception as e:
        results["tests"]["gradient_exists"] = {"passed": False, "error": str(e)}
        results["passed"] = False
        return results

    # ── Test 2: Discrimination — legible vs ambiguous ──
    def make_traj(y_offset):
        """Trajectory from home toward left goal with lateral offset."""
        t = torch.zeros(32, 3, device=device)
        for i in range(32):
            f = i / 31.0
            t[i] = torch.tensor([0.40 + f * 0.10,
                                  y_offset * f,
                                  0.55 - f * 0.13])
        return t.requires_grad_(True)

    # Going left (toward goal 0)
    traj_left = make_traj(-0.07)      # legible for goal 0
    traj_center = make_traj(0.0)      # ambiguous
    traj_right = make_traj(+0.07)     # legible for goal 1

    s_left = fn(traj_left, goals, 0).item()
    s_center = fn(traj_center, goals, 0).item()
    s_right = fn(traj_right, goals, 0).item()

    disc_pass = (s_left > s_center > s_right)
    results["tests"]["discrimination"] = {
        "passed": disc_pass,
        "score_left_to_goal0": round(s_left, 4),
        "score_center_to_goal0": round(s_center, 4),
        "score_right_to_goal0": round(s_right, 4),
        "ordering": "left > center > right" if disc_pass else "WRONG",
    }
    if not disc_pass:
        results["passed"] = False

    # ── Test 3: Symmetry — same function works for both goals ──
    s_right_g1 = fn(traj_right, goals, 1).item()
    s_center_g1 = fn(traj_center, goals, 1).item()
    s_left_g1 = fn(traj_left, goals, 1).item()

    sym_pass = (s_right_g1 > s_center_g1 > s_left_g1)
    results["tests"]["symmetry"] = {
        "passed": sym_pass,
        "score_right_to_goal1": round(s_right_g1, 4),
        "score_center_to_goal1": round(s_center_g1, 4),
        "score_left_to_goal1": round(s_left_g1, 4),
    }
    if not sym_pass:
        results["passed"] = False

    # ── Test 4: Monotonicity — increasing lateral deviation ──
    offsets = [0.0, -0.02, -0.04, -0.06, -0.08]
    scores_mono = []
    for off in offsets:
        t = make_traj(off)
        s = fn(t, goals, 0).item()
        scores_mono.append(round(s, 4))

    # Should be monotonically increasing for goal 0 (more negative y = closer)
    is_monotonic = all(scores_mono[i] <= scores_mono[i+1]
                       for i in range(len(scores_mono)-1))
    results["tests"]["monotonicity_goal0"] = {
        "passed": is_monotonic,
        "offsets": offsets,
        "scores": scores_mono,
    }

    # ── Test 5: Gradient direction analysis ──
    # At the starting position, gradient should push y-displacement negative
    # (toward left goal at y=-0.07) when true_goal=0
    home_traj = make_traj(0.0)  # center trajectory
    s_home = fn(home_traj, goals, 0)
    grad_home = torch.autograd.grad(s_home, home_traj)[0]  # (32, 3)
    early_grad_y = float(grad_home[:10, 1].mean().item())  # mean y-gradient in first 30%

    # For goal 0 (left, y=-0.07), gradient should push y negative
    grad_dir_pass = early_grad_y < 0
    results["tests"]["gradient_direction"] = {
        "passed": grad_dir_pass,
        "early_mean_grad_y": round(early_grad_y, 6),
        "expected": "negative (push toward y=-0.07 for left goal)",
    }

    # ── Test 6: Score range ──
    range_pass = (0.0 <= s_left <= 1.0 and 0.0 <= s_center <= 1.0
                  and 0.0 <= s_right <= 1.0)
    results["tests"]["score_range"] = {
        "passed": range_pass,
        "min": min(s_left, s_center, s_right),
        "max": max(s_left, s_center, s_right),
    }

    return results


def stage1_compare_to_handcrafted(vlm_fn: Callable, device: torch.device) -> dict:
    """Compare VLM scoring function to hand-crafted l_early_intent_torch
    on a grid of synthetic trajectories. No environment rollouts needed."""
    goals = torch.tensor([[0.50, -0.07, 0.42],
                           [0.50,  0.07, 0.42]], device=device)

    offsets = np.linspace(-0.10, 0.10, 21)
    vlm_scores = []
    hc_scores = []

    for y_off in offsets:
        t = torch.zeros(32, 3, device=device)
        for i in range(32):
            f = i / 31.0
            t[i] = torch.tensor([0.40 + f * 0.10,
                                  float(y_off) * f,
                                  0.55 - f * 0.13])

        vs = vlm_fn(t, goals, 0).item()
        hs = l_early_intent_torch(t, goals, 0).item()
        vlm_scores.append(vs)
        hc_scores.append(hs)

    # Correlation between VLM and hand-crafted
    corr = float(np.corrcoef(vlm_scores, hc_scores)[0, 1])

    return {
        "offsets": offsets.tolist(),
        "vlm_scores": [round(s, 4) for s in vlm_scores],
        "handcrafted_scores": [round(s, 4) for s in hc_scores],
        "correlation": round(corr, 4),
        "vlm_dynamic_range": round(max(vlm_scores) - min(vlm_scores), 4),
        "hc_dynamic_range": round(max(hc_scores) - min(hc_scores), 4),
    }


# ═════════════════════════════════════════════════════════════════════════
# STAGE 2: CLASSIFIER GUIDANCE INTEGRATION — PAIRED EVALUATION + SWEEP
# ═════════════════════════════════════════════════════════════════════════

def stage2_paired_eval(
    model, obs_mean, obs_std, act_mean, act_std, device,
    score_fn: Callable, label: str,
    guidance_scales: List[float],
    seed_pairs: List[Tuple[int, int]],
    n_sampling_steps: int = 10,
    n_diff: int = 100, beta_s: float = 1e-4, beta_e: float = 0.1,
) -> dict:
    """Run paired evaluation across multiple guidance scales."""
    results = {}

    for w in guidance_scales:
        print(f"\n  ── {label} w={w:.1f} ── {len(seed_pairs)} paired episodes ──")
        sampler = GuidedDDIMSampler(
            n_diff, beta_s, beta_e, device,
            score_fn=score_fn, guidance_scale=w, grad_clip=1.0,
        )
        episodes = []
        for ep_idx, (es, ss) in enumerate(seed_pairs):
            r = run_paired_episode(
                model, sampler, obs_mean, obs_std, act_mean, act_std, device,
                env_seed=es, sample_seed=ss, guided=True,
                n_sampling_steps=n_sampling_steps,
            )
            episodes.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"    Ep {ep_idx+1:>2}/{len(seed_pairs)} {tick}  "
                  f"L_early={r['l_early_actual']:.4f}  "
                  f"goal={r['true_goal']}  "
                  f"seeds=({es},{ss})")

        l_vals = np.array([r['l_early_actual'] for r in episodes])
        s_vals = np.array([r['success'] for r in episodes])
        results[f"w{w}"] = {
            "guidance_scale": w,
            "success_rate": float(s_vals.mean()),
            "l_early_mean": float(l_vals.mean()),
            "l_early_std": float(l_vals.std()),
            "l_early_median": float(np.median(l_vals)),
            "l_early_min": float(l_vals.min()),
            "l_early_max": float(l_vals.max()),
            "episodes": episodes,
        }
        print(f"    Summary: success={s_vals.mean():.0%}  "
              f"L_early={l_vals.mean():.4f} ± {l_vals.std():.4f}")

    return results


def stage2_baseline_eval(
    model, obs_mean, obs_std, act_mean, act_std, device,
    seed_pairs: List[Tuple[int, int]],
    n_sampling_steps: int = 10,
    n_diff: int = 100, beta_s: float = 1e-4, beta_e: float = 0.1,
) -> dict:
    """Run baseline (unguided) on the same seed pairs."""
    print(f"\n  ── BASELINE (w=0) ── {len(seed_pairs)} paired episodes ──")
    sampler = DDIMSampler(n_diff, beta_s, beta_e, device)
    episodes = []
    for ep_idx, (es, ss) in enumerate(seed_pairs):
        r = run_paired_episode(
            model, sampler, obs_mean, obs_std, act_mean, act_std, device,
            env_seed=es, sample_seed=ss, guided=False,
            n_sampling_steps=n_sampling_steps,
        )
        episodes.append(r)
        tick = '✓' if r['success'] else '✗'
        print(f"    Ep {ep_idx+1:>2}/{len(seed_pairs)} {tick}  "
              f"L_early={r['l_early_actual']:.4f}  "
              f"goal={r['true_goal']}  seeds=({es},{ss})")

    l_vals = np.array([r['l_early_actual'] for r in episodes])
    s_vals = np.array([r['success'] for r in episodes])
    print(f"    Summary: success={s_vals.mean():.0%}  "
          f"L_early={l_vals.mean():.4f} ± {l_vals.std():.4f}")

    return {
        "success_rate": float(s_vals.mean()),
        "l_early_mean": float(l_vals.mean()),
        "l_early_std": float(l_vals.std()),
        "episodes": episodes,
    }


# ═════════════════════════════════════════════════════════════════════════
# STAGE 3: VLM TEXT RERANKING
# ═════════════════════════════════════════════════════════════════════════

def stage3_rerank_candidates(
    model, sampler, obs_mean, obs_std, act_mean, act_std, device,
    env_seed: int, sample_seed: int,
    n_candidates: int = 5,
    n_sampling_steps: int = 10,
    max_steps: int = 400,
    vlm_rerank_fn: Optional[Callable] = None,
) -> dict:
    """Generate N candidate trajectories, rerank with VLM text analysis.

    For each candidate:
    1. Run guided classifier-guidance sampling to get action chunk
    2. Forward-simulate to get EE trajectory prefix
    3. Compute trajectory statistics (arc, velocity, L_early)
    4. VLM selects best candidate based on text description
    5. Execute the chosen candidate

    If vlm_rerank_fn is None, uses L_early as the selection criterion (oracle).
    """
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset(seed=env_seed)

    action_queue: deque = deque(maxlen=model.horizon)
    ee_trajectory: List[np.ndarray] = []
    rerank_info: List[dict] = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_trajectory.append(obs[0:3].copy())

        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            ee_start = torch.tensor(obs[0:3], dtype=torch.float32, device=device)
            left_goal = torch.tensor(obs[8:11], dtype=torch.float32, device=device)
            right_goal = torch.tensor(obs[15:18], dtype=torch.float32, device=device)
            goals_t = torch.stack([left_goal, right_goal])

            # Generate N candidates with different random seeds
            candidates = []
            candidate_scores = []
            base_seed = sample_seed + step * 1000

            for c in range(n_candidates):
                torch.manual_seed(base_seed + c)
                act_seq, s_val = sampler.sample(
                    model, obs_t, ee_start, goals_t,
                    n_sampling_steps=n_sampling_steps)
                chunk_np = act_seq[0].cpu().numpy() * act_std + act_mean

                # Compute trajectory stats for this candidate
                delta_pos = act_seq[0, :, :3].detach().cpu().numpy() * ACTION_SCALE
                ee_pred = np.cumsum(delta_pos, axis=0) + obs[0:3]

                # Compute L_early for predicted trajectory
                ee_pred_t = torch.tensor(ee_pred, dtype=torch.float32, device=device)
                with torch.no_grad():
                    le0 = l_early_intent_torch(ee_pred_t, goals_t, 0).item()
                    le1 = l_early_intent_torch(ee_pred_t, goals_t, 1).item()
                pred_l_early = max(le0, le1)

                # Trajectory statistics for VLM reranking
                arc_mag = float(np.linalg.norm(
                    ee_pred[int(len(ee_pred)*0.3)] - ee_pred[0]))
                y_displacement = float(ee_pred[int(len(ee_pred)*0.3), 1] - ee_pred[0, 1])
                mean_speed = float(np.linalg.norm(np.diff(ee_pred, axis=0), axis=1).mean())

                candidates.append(chunk_np)
                candidate_scores.append({
                    "idx": c,
                    "guided_score": float(s_val),
                    "pred_l_early": float(pred_l_early),
                    "arc_magnitude": arc_mag,
                    "y_displacement": y_displacement,
                    "mean_speed": mean_speed,
                })

            # Rerank: select best candidate
            if vlm_rerank_fn is not None:
                best_idx = vlm_rerank_fn(candidate_scores)
            else:
                # Oracle: pick candidate with highest predicted L_early
                best_idx = max(range(n_candidates),
                              key=lambda i: candidate_scores[i]["pred_l_early"])

            rerank_info.append({
                "step": step,
                "n_candidates": n_candidates,
                "best_idx": best_idx,
                "scores": candidate_scores,
            })

            chosen = candidates[best_idx]
            for a in chosen:
                action_queue.append(a)

        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs
        last_obs = obs
        success = (result.info.get('success_left', 0) > 0.5 or
                   result.info.get('success_right', 0) > 0.5)
        if result.done:
            break

    env.close()

    # Compute actual L_early
    l_early, rlc, true_goal = 0.0, 0.0, 'unknown'
    if len(ee_trajectory) >= 4:
        traj_arr = np.array(ee_trajectory)
        goals_np = np.stack([last_obs[8:11], last_obs[15:18]], axis=0)
        from evaluation.legibility_metrics import compute_legibility
        r0 = compute_legibility(traj_arr, goals_np, true_goal_idx=0, model='gaussian')
        r1 = compute_legibility(traj_arr, goals_np, true_goal_idx=1, model='gaussian')
        if r0.L_early_intent >= r1.L_early_intent:
            l_early, rlc, true_goal = r0.L_early_intent, r0.relative_legibility_cost, 'left'
        else:
            l_early, rlc, true_goal = r1.L_early_intent, r1.relative_legibility_cost, 'right'

    return dict(
        env_seed=env_seed, sample_seed=sample_seed,
        success=bool(success), steps=step + 1,
        l_early_actual=float(l_early), rlc_actual=float(rlc),
        true_goal=true_goal,
        n_candidates=n_candidates,
        rerank_info=rerank_info,
    )


# ═════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def paired_ttest(a: List[float], b: List[float]) -> dict:
    """Paired t-test with confidence interval."""
    a, b = np.array(a), np.array(b)
    diff = b - a
    n = len(diff)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    se = std_diff / np.sqrt(n) if n > 1 else 0.0
    t_stat = mean_diff / se if se > 1e-12 else 0.0

    # Two-tailed p-value approximation (normal for n >= 20)
    from math import erfc
    p_val = erfc(abs(t_stat) / np.sqrt(2)) if n >= 2 else 1.0

    ci_95 = 1.96 * se
    return {
        "mean_diff": round(mean_diff, 5),
        "std_diff": round(std_diff, 5),
        "t_stat": round(t_stat, 3),
        "p_value": round(float(p_val), 5),
        "ci_95": [round(mean_diff - ci_95, 5), round(mean_diff + ci_95, 5)],
        "n": n,
        "significant_at_005": float(p_val) < 0.05,
    }


# ═════════════════════════════════════════════════════════════════════════
# VLM TEXT RERANKING (Gemini call)
# ═════════════════════════════════════════════════════════════════════════

def build_rerank_prompt(candidate_scores: List[dict]) -> str:
    """Build a text prompt for Gemini to select the most legible candidate."""
    lines = [
        "You are a robotics trajectory legibility expert.",
        "A robot arm is picking one of two blocks (left or right) on a table.",
        "Below are trajectory candidates from a diffusion policy. Each shows",
        "statistics from the FIRST 30% of predicted motion.",
        "",
        "Select the candidate whose early motion MOST CLEARLY reveals which",
        "block the robot intends to pick. A legible trajectory:",
        "- Moves decisively toward one block early",
        "- Has large lateral displacement toward the target",
        "- Has high arc magnitude (purposeful curve)",
        "",
        "Candidates:",
    ]
    for c in candidate_scores:
        lines.append(
            f"  #{c['idx']}: arc_magnitude={c['arc_magnitude']:.4f}  "
            f"y_displacement={c['y_displacement']:.4f}  "
            f"mean_speed={c['mean_speed']:.5f}  "
            f"guided_score={c['guided_score']:.4f}"
        )
    lines.append("")
    lines.append("Output ONLY the candidate number (0-indexed integer). No explanation.")
    return "\n".join(lines)


def make_gemini_reranker(api_key: str, model_name: str = "gemini-2.5-flash"):
    """Create a VLM reranking function that calls Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def rerank_fn(candidate_scores: List[dict]) -> int:
        prompt = build_rerank_prompt(candidate_scores)
        config = types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        )
        try:
            resp = client.models.generate_content(
                model=model_name, contents=prompt, config=config)
            text = resp.text.strip()
            # Extract integer
            import re
            match = re.search(r'(\d+)', text)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < len(candidate_scores):
                    return idx
        except Exception as e:
            print(f"      [VLM rerank error: {e}]")

        # Fallback: pick highest guided_score
        return max(range(len(candidate_scores)),
                  key=lambda i: candidate_scores[i]["guided_score"])

    return rerank_fn


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def load_vlm_function(path: Path, device: torch.device) -> Callable:
    """Load VLM-generated scoring function from file."""
    code = path.read_text(encoding='utf-8')
    sandbox = {"torch": torch, "math": math, "nn": nn}
    # Also handle F import that Gemini sometimes adds
    import torch.nn.functional as F
    sandbox["F"] = F
    exec(code, sandbox)
    fn = sandbox.get("vlm_legibility_score")
    if fn is None:
        raise RuntimeError(f"vlm_legibility_score not found in {path}")
    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--stage', default='all', choices=['1', '2', '3', 'all'])
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--n_episodes', type=int, default=20,
                    help='Episodes per condition (paired)')
    ap.add_argument('--n_sampling_steps', type=int, default=10)
    ap.add_argument('--vlm_fn_path', type=str,
                    default='outputs/stage1/vlm_score_fn.py')
    ap.add_argument('--api_key', type=str, default=None)
    ap.add_argument('--n_candidates', type=int, default=5,
                    help='Candidates for Stage 3 reranking')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(__file__).parent.parent / 'outputs' / 'rigorous_eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    run_all = args.stage == 'all'

    print(f"\n{'='*72}")
    print("  RIGOROUS STAGE EVALUATION — Paired Comparisons")
    print(f"{'='*72}")
    print(f"  Stage(s)     : {args.stage}")
    print(f"  Device       : {device}")
    print(f"  Episodes     : {args.n_episodes} (paired seeds)")
    print(f"{'='*72}\n")

    # Generate deterministic seed pairs
    rng = np.random.RandomState(42)
    seed_pairs = [(int(rng.randint(0, 10000)), int(rng.randint(0, 100000)))
                  for _ in range(args.n_episodes)]
    print(f"  Seed pairs: {seed_pairs[:5]}... ({len(seed_pairs)} total)")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'], act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    obs_mean = np.array(ckpt['obs_mean'], dtype=np.float32)
    obs_std  = np.array(ckpt['obs_std'],  dtype=np.float32)
    act_mean = np.array(ckpt['act_mean'], dtype=np.float32)
    act_std  = np.array(ckpt['act_std'],  dtype=np.float32)
    n_diff = cfg.get('n_diffusion_steps', 100)
    beta_s = cfg.get('beta_start', 1e-4)
    beta_e = cfg.get('beta_end', 0.1)

    all_outputs = {"seed_pairs": seed_pairs, "checkpoint": str(args.checkpoint)}

    # ═════════════════════════════════════════════════════════════════
    # STAGE 1: VLM Code Synthesis — Isolated Validation
    # ═════════════════════════════════════════════════════════════════
    if args.stage == '1' or run_all:
        print(f"\n{'='*72}")
        print("  STAGE 1: VLM Code Synthesis — Can Gemini produce a valid,")
        print("           discriminative guidance function?")
        print(f"{'='*72}")
        print("  H_S1: The VLM-generated function will:")
        print("    (a) Pass all gradient and range checks")
        print("    (b) Correctly discriminate legible vs ambiguous trajectories")
        print("    (c) Correlate with hand-crafted L_early (r > 0.8)")
        print("    (d) Have gradient pointing toward the correct goal")

        vlm_fn_path = Path(args.vlm_fn_path)
        if not vlm_fn_path.exists():
            print(f"\n  ERROR: VLM function not found at {vlm_fn_path}")
            print("  Run stage1_vlm_guidance.py first to generate the function.")
            sys.exit(1)

        vlm_fn = load_vlm_function(vlm_fn_path, device)
        print(f"\n  Loaded VLM function from {vlm_fn_path}")

        # Validation tests
        print("\n  ── Stage 1 Validation Tests ──")
        s1_val = stage1_validate_function(vlm_fn, device)
        for name, test in s1_val["tests"].items():
            status = "PASS ✓" if test["passed"] else "FAIL ✗"
            detail = {k: v for k, v in test.items() if k != "passed"}
            print(f"    {status}  {name}: {detail}")

        # Comparison with hand-crafted
        print("\n  ── Stage 1 Correlation with Hand-Crafted ──")
        s1_corr = stage1_compare_to_handcrafted(vlm_fn, device)
        print(f"    Pearson r = {s1_corr['correlation']:.4f}")
        print(f"    VLM dynamic range  = {s1_corr['vlm_dynamic_range']:.4f}")
        print(f"    HC dynamic range   = {s1_corr['hc_dynamic_range']:.4f}")

        # Also validate hand-crafted for comparison
        print("\n  ── Hand-Crafted Validation (reference) ──")
        hc_val = stage1_validate_function(l_early_intent_torch, device)
        for name, test in hc_val["tests"].items():
            status = "PASS ✓" if test["passed"] else "FAIL ✗"
            detail = {k: v for k, v in test.items() if k != "passed"}
            print(f"    {status}  {name}: {detail}")

        # Stage 1 verdict
        print(f"\n  ── STAGE 1 VERDICT ──")
        s1_pass = s1_val["passed"]
        corr_pass = s1_corr["correlation"] > 0.8
        print(f"    H_S1(a) All checks pass       : {'CONFIRMED ✓' if s1_pass else 'REJECTED ✗'}")
        print(f"    H_S1(b) Discrimination         : {'CONFIRMED ✓' if s1_val['tests']['discrimination']['passed'] else 'REJECTED ✗'}")
        print(f"    H_S1(c) Correlation r > 0.8    : {'CONFIRMED ✓' if corr_pass else 'REJECTED ✗'} (r={s1_corr['correlation']:.4f})")
        print(f"    H_S1(d) Gradient direction     : {'CONFIRMED ✓' if s1_val['tests']['gradient_direction']['passed'] else 'REJECTED ✗'}")

        all_outputs["stage1"] = {
            "validation": s1_val,
            "correlation": s1_corr,
            "handcrafted_validation": hc_val,
        }

        s1_path = out_dir / 'stage1_results.json'
        with open(s1_path, 'w') as f:
            json.dump(all_outputs.get("stage1", {}), f, indent=2)
        print(f"\n  Stage 1 results → {s1_path}")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 2: Classifier Guidance Integration — Paired Evaluation + Scale Sweep
    # ═════════════════════════════════════════════════════════════════
    if args.stage == '2' or run_all:
        print(f"\n{'='*72}")
        print("  STAGE 2: Classifier Guidance Integration — Does gradient guidance with")
        print("           VLM function improve legibility?")
        print(f"{'='*72}")
        print("  H_S2: At the optimal guidance scale w*:")
        print("    (a) VLM-guided classifier guidance success ≥ 95% (no degradation)")
        print("    (b) VLM-guided L_early > baseline L_early (p < 0.05)")
        print("    (c) VLM-guided L_early ≈ hand-crafted classifier guidance L_early")
        print("    (d) Optimal w* for VLM function may differ from hand-crafted")

        vlm_fn_path = Path(args.vlm_fn_path)
        vlm_fn = load_vlm_function(vlm_fn_path, device)

        # 2a. Baseline (same seeds)
        baseline = stage2_baseline_eval(
            model, obs_mean, obs_std, act_mean, act_std, device,
            seed_pairs, n_sampling_steps=args.n_sampling_steps,
            n_diff=n_diff, beta_s=beta_s, beta_e=beta_e)

        # 2b. Hand-crafted classifier guidance sweep
        hc_scales = [5.0, 10.0, 15.0]
        hc_results = stage2_paired_eval(
            model, obs_mean, obs_std, act_mean, act_std, device,
            score_fn=l_early_intent_torch, label="Hand-crafted",
            guidance_scales=hc_scales, seed_pairs=seed_pairs,
            n_sampling_steps=args.n_sampling_steps,
            n_diff=n_diff, beta_s=beta_s, beta_e=beta_e)

        # 2c. VLM function sweep
        vlm_scales = [5.0, 10.0, 15.0]
        vlm_results = stage2_paired_eval(
            model, obs_mean, obs_std, act_mean, act_std, device,
            score_fn=vlm_fn, label="VLM-guided",
            guidance_scales=vlm_scales, seed_pairs=seed_pairs,
            n_sampling_steps=args.n_sampling_steps,
            n_diff=n_diff, beta_s=beta_s, beta_e=beta_e)

        # 2d. Statistical analysis
        print(f"\n{'='*72}")
        print("  STAGE 2: STATISTICAL ANALYSIS")
        print(f"{'='*72}")

        bl_l = [r['l_early_actual'] for r in baseline['episodes']]

        # Find best scale: require ≥95% success first, then max L_early
        def _pick_best(rd, min_sr=0.95):
            viable = {k: v for k, v in rd.items() if v['success_rate'] >= min_sr}
            if not viable:  # fallback: pick highest success
                return max(rd, key=lambda k: rd[k]['success_rate'])
            return max(viable, key=lambda k: viable[k]['l_early_mean'])
        best_hc_key = _pick_best(hc_results)
        best_vlm_key = _pick_best(vlm_results)

        best_hc = hc_results[best_hc_key]
        best_vlm = vlm_results[best_vlm_key]

        hc_l = [r['l_early_actual'] for r in best_hc['episodes']]
        vlm_l = [r['l_early_actual'] for r in best_vlm['episodes']]

        # Paired t-tests (same seeds!)
        test_vlm_vs_bl = paired_ttest(bl_l, vlm_l)
        test_vlm_vs_hc = paired_ttest(hc_l, vlm_l)
        test_hc_vs_bl = paired_ttest(bl_l, hc_l)

        print(f"\n  Comparison Table:")
        print(f"  {'Method':<30} {'Success':>8} {'L_early':>10} {'std':>8} {'Best w':>8}")
        print(f"  {'-'*64}")
        print(f"  {'Baseline (w=0)':<30} {baseline['success_rate']:>7.0%} "
              f"{baseline['l_early_mean']:>10.4f} {baseline['l_early_std']:>8.4f} {'—':>8}")
        print(f"  {'Hand-crafted (best)':<30} {best_hc['success_rate']:>7.0%} "
              f"{best_hc['l_early_mean']:>10.4f} {best_hc['l_early_std']:>8.4f} "
              f"{best_hc['guidance_scale']:>8.1f}")
        print(f"  {'VLM-guided (best)':<30} {best_vlm['success_rate']:>7.0%} "
              f"{best_vlm['l_early_mean']:>10.4f} {best_vlm['l_early_std']:>8.4f} "
              f"{best_vlm['guidance_scale']:>8.1f}")

        print(f"\n  Paired t-tests (same {len(seed_pairs)} seed pairs):")
        print(f"    VLM vs Baseline:      Δ={test_vlm_vs_bl['mean_diff']:+.4f}  "
              f"p={test_vlm_vs_bl['p_value']:.4f}  "
              f"{'significant' if test_vlm_vs_bl['significant_at_005'] else 'NOT significant'}")
        print(f"    HC vs Baseline:       Δ={test_hc_vs_bl['mean_diff']:+.4f}  "
              f"p={test_hc_vs_bl['p_value']:.4f}  "
              f"{'significant' if test_hc_vs_bl['significant_at_005'] else 'NOT significant'}")
        print(f"    VLM vs Hand-crafted:  Δ={test_vlm_vs_hc['mean_diff']:+.4f}  "
              f"p={test_vlm_vs_hc['p_value']:.4f}  "
              f"{'significant' if test_vlm_vs_hc['significant_at_005'] else 'NOT significant'}")

        # Verdict
        print(f"\n  ── STAGE 2 VERDICT ──")
        print(f"    H_S2(a) VLM success ≥ 95%     : {'CONFIRMED ✓' if best_vlm['success_rate'] >= 0.95 else 'REJECTED ✗'} ({best_vlm['success_rate']:.0%})")
        print(f"    H_S2(b) VLM > baseline (p<.05): {'CONFIRMED ✓' if test_vlm_vs_bl['significant_at_005'] else 'REJECTED ✗'} (p={test_vlm_vs_bl['p_value']:.4f})")
        vlm_close_to_hc = abs(test_vlm_vs_hc['mean_diff']) < 0.02
        print(f"    H_S2(c) VLM ≈ HC (|Δ|<0.02)  : {'CONFIRMED ✓' if vlm_close_to_hc else 'REJECTED ✗'} (Δ={test_vlm_vs_hc['mean_diff']:+.4f})")
        w_differs = (best_vlm['guidance_scale'] != best_hc['guidance_scale'])
        print(f"    H_S2(d) Optimal w* differs     : {'CONFIRMED ✓' if w_differs else 'REJECTED ✗'} (VLM:{best_vlm['guidance_scale']}, HC:{best_hc['guidance_scale']})")

        # Guidance scale sweep table
        print(f"\n  Guidance Scale Sweep:")
        print(f"  {'w':>6}  {'HC success':>10}  {'HC L_early':>10}  "
              f"{'VLM success':>11}  {'VLM L_early':>11}")
        print(f"  {'-'*60}")
        for w in sorted(set(hc_scales + vlm_scales)):
            hk = f"w{w}"
            hc_row = hc_results.get(hk, {})
            vlm_row = vlm_results.get(hk, {})
            print(f"  {w:>6.1f}  "
                  f"{hc_row.get('success_rate', 0):>9.0%}  "
                  f"{hc_row.get('l_early_mean', 0):>10.4f}  "
                  f"{vlm_row.get('success_rate', 0):>10.0%}  "
                  f"{vlm_row.get('l_early_mean', 0):>11.4f}")

        all_outputs["stage2"] = {
            "baseline": baseline,
            "handcrafted": hc_results,
            "vlm_guided": vlm_results,
            "ttest_vlm_vs_baseline": test_vlm_vs_bl,
            "ttest_vlm_vs_handcrafted": test_vlm_vs_hc,
            "ttest_hc_vs_baseline": test_hc_vs_bl,
            "best_hc_scale": best_hc['guidance_scale'],
            "best_vlm_scale": best_vlm['guidance_scale'],
        }

        s2_path = out_dir / 'stage2_results.json'
        with open(s2_path, 'w') as f:
            json.dump(all_outputs.get("stage2", {}), f, indent=2, default=str)
        print(f"\n  Stage 2 results → {s2_path}")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 3: VLM Text Reranking
    # ═════════════════════════════════════════════════════════════════
    if args.stage == '3' or run_all:
        print(f"\n{'='*72}")
        print("  STAGE 3: VLM Text Reranking — Does post-hoc candidate")
        print("           selection further improve legibility?")
        print(f"{'='*72}")
        print("  H_S3: Best-of-N with VLM reranking will:")
        print("    (a) Maintain ≥ 95% success rate")
        print("    (b) Improve L_early by +1-3% over single-sample guidance")
        print("    (c) VLM reranking ≈ oracle L_early selection (r > 0.8)")

        vlm_fn_path = Path(args.vlm_fn_path)
        vlm_fn = load_vlm_function(vlm_fn_path, device)

        api_key = (args.api_key
                   or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY"))

        # Use oracle reranking (select by L_early) if no API key
        if api_key:
            print(f"  Using Gemini text reranking (N={args.n_candidates})")
            reranker = make_gemini_reranker(api_key)
        else:
            print(f"  No API key — using oracle reranking (N={args.n_candidates})")
            reranker = None

        # Best VLM guidance scale from Stage 2 (default to 10)
        best_w = 10.0
        s2_path = out_dir / 'stage2_results.json'
        if s2_path.exists():
            s2_data = json.loads(s2_path.read_text())
            best_w = s2_data.get('best_vlm_scale', 10.0)

        sampler = GuidedDDIMSampler(
            n_diff, beta_s, beta_e, device,
            score_fn=vlm_fn, guidance_scale=best_w, grad_clip=1.0)

        # 3a. Single-sample guided (N=1, same seeds)
        print(f"\n  ── Guided single-sample (N=1, w={best_w}) ──")
        single_results = []
        for ep_idx, (es, ss) in enumerate(seed_pairs):
            r = run_paired_episode(
                model, sampler, obs_mean, obs_std, act_mean, act_std, device,
                env_seed=es, sample_seed=ss, guided=True,
                n_sampling_steps=args.n_sampling_steps)
            single_results.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"    Ep {ep_idx+1:>2}/{len(seed_pairs)} {tick}  "
                  f"L_early={r['l_early_actual']:.4f}")

        # 3b. Oracle reranking (N=K, select best L_early)
        print(f"\n  ── Oracle reranking (N={args.n_candidates}) ──")
        oracle_results = []
        for ep_idx, (es, ss) in enumerate(seed_pairs):
            r = stage3_rerank_candidates(
                model, sampler, obs_mean, obs_std, act_mean, act_std, device,
                env_seed=es, sample_seed=ss,
                n_candidates=args.n_candidates,
                n_sampling_steps=args.n_sampling_steps,
                vlm_rerank_fn=None)  # Oracle
            oracle_results.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"    Ep {ep_idx+1:>2}/{len(seed_pairs)} {tick}  "
                  f"L_early={r['l_early_actual']:.4f}")

        # 3c. VLM text reranking (if API key available)
        vlm_rerank_results = []
        if api_key:
            print(f"\n  ── VLM text reranking (N={args.n_candidates}) ──")
            for ep_idx, (es, ss) in enumerate(seed_pairs):
                r = stage3_rerank_candidates(
                    model, sampler, obs_mean, obs_std, act_mean, act_std, device,
                    env_seed=es, sample_seed=ss,
                    n_candidates=args.n_candidates,
                    n_sampling_steps=args.n_sampling_steps,
                    vlm_rerank_fn=reranker)
                vlm_rerank_results.append(r)
                tick = '✓' if r['success'] else '✗'
                print(f"    Ep {ep_idx+1:>2}/{len(seed_pairs)} {tick}  "
                      f"L_early={r['l_early_actual']:.4f}")

        # Analysis
        print(f"\n{'='*72}")
        print("  STAGE 3: RESULTS")
        print(f"{'='*72}")

        single_l = [r['l_early_actual'] for r in single_results]
        single_s = [r['success'] for r in single_results]
        oracle_l = [r['l_early_actual'] for r in oracle_results]
        oracle_s = [r['success'] for r in oracle_results]

        print(f"\n  {'Method':<35} {'Success':>8} {'L_early':>10} {'std':>8}")
        print(f"  {'-'*65}")
        print(f"  {'Guided single-sample (N=1)':<35} {np.mean(single_s):>7.0%} "
              f"{np.mean(single_l):>10.4f} {np.std(single_l):>8.4f}")
        print(f"  {'Oracle reranking (N=' + str(args.n_candidates) + ')':<35} "
              f"{np.mean(oracle_s):>7.0%} "
              f"{np.mean(oracle_l):>10.4f} {np.std(oracle_l):>8.4f}")

        if vlm_rerank_results:
            vlm_rr_l = [r['l_early_actual'] for r in vlm_rerank_results]
            vlm_rr_s = [r['success'] for r in vlm_rerank_results]
            print(f"  {'VLM text reranking (N=' + str(args.n_candidates) + ')':<35} "
                  f"{np.mean(vlm_rr_s):>7.0%} "
                  f"{np.mean(vlm_rr_l):>10.4f} {np.std(vlm_rr_l):>8.4f}")

        # Paired tests
        test_oracle_vs_single = paired_ttest(single_l, oracle_l)
        print(f"\n  Oracle vs Single: Δ={test_oracle_vs_single['mean_diff']:+.4f}  "
              f"p={test_oracle_vs_single['p_value']:.4f}")

        if vlm_rerank_results:
            test_vlm_rr_vs_single = paired_ttest(single_l, vlm_rr_l)
            test_vlm_rr_vs_oracle = paired_ttest(oracle_l, vlm_rr_l)
            print(f"  VLM-RR vs Single: Δ={test_vlm_rr_vs_single['mean_diff']:+.4f}  "
                  f"p={test_vlm_rr_vs_single['p_value']:.4f}")
            print(f"  VLM-RR vs Oracle: Δ={test_vlm_rr_vs_oracle['mean_diff']:+.4f}  "
                  f"p={test_vlm_rr_vs_oracle['p_value']:.4f}")

        # Verdict
        print(f"\n  ── STAGE 3 VERDICT ──")
        print(f"    H_S3(a) Success ≥ 95%          : {'CONFIRMED ✓' if np.mean(oracle_s) >= 0.95 else 'REJECTED ✗'}")
        oracle_improves = test_oracle_vs_single['mean_diff'] > 0.01
        print(f"    H_S3(b) Oracle +1-3% L_early   : {'CONFIRMED ✓' if oracle_improves else 'REJECTED ✗'} (Δ={test_oracle_vs_single['mean_diff']:+.4f})")

        all_outputs["stage3"] = {
            "single_sample": {
                "success_rate": float(np.mean(single_s)),
                "l_early_mean": float(np.mean(single_l)),
                "l_early_std": float(np.std(single_l)),
                "episodes": single_results,
            },
            "oracle_rerank": {
                "n_candidates": args.n_candidates,
                "success_rate": float(np.mean(oracle_s)),
                "l_early_mean": float(np.mean(oracle_l)),
                "l_early_std": float(np.std(oracle_l)),
                "episodes": oracle_results,
            },
            "ttest_oracle_vs_single": test_oracle_vs_single,
        }
        if vlm_rerank_results:
            all_outputs["stage3"]["vlm_rerank"] = {
                "success_rate": float(np.mean(vlm_rr_s)),
                "l_early_mean": float(np.mean(vlm_rr_l)),
                "l_early_std": float(np.std(vlm_rr_l)),
                "episodes": vlm_rerank_results,
            }
            all_outputs["stage3"]["ttest_vlm_vs_single"] = test_vlm_rr_vs_single
            all_outputs["stage3"]["ttest_vlm_vs_oracle"] = test_vlm_rr_vs_oracle

        s3_path = out_dir / 'stage3_results.json'
        with open(s3_path, 'w') as f:
            json.dump(all_outputs.get("stage3", {}), f, indent=2, default=str)
        print(f"\n  Stage 3 results → {s3_path}")

    # ═════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════
    full_path = out_dir / 'full_results.json'
    with open(full_path, 'w') as f:
        json.dump(all_outputs, f, indent=2, default=str)
    print(f"\n  Full results → {full_path}")
    print(f"\n{'='*72}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
