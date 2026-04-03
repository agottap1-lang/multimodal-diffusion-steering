#!/usr/bin/env python3
"""
Verification: Stage123 Hypothesis Flaws
========================================

Tests three specific flaws identified in STAGE123_HYPOTHESIS.md:

FLAW 2 — VLM improvement is confounded by multi-criteria, not VLM per se.
    Test: hand-crafted 4-criteria function vs VLM-generated 4-criteria function
    at the same guidance scale. If hand-crafted 4-criteria ≈ VLM 4-criteria,
    the VLM contributed nothing beyond the criteria count.

FLAW 3 — Stage 1 validation test is too weak (trivially satisfied by any
    proximity-based function). Test: construct a hard case where Gaussian
    proximity can't distinguish two trajectories but directional criteria can.

FLAW 5 — Baseline success inconsistency. Re-run baseline with fixed seed
    to test reproducibility and identify the source of 85% vs 95% gap.

Usage
-----
  python experiments/verify_stage123_flaws.py
  python experiments/verify_stage123_flaws.py --n_episodes 10 --guidance_scale 10.0
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# ── re-use model architecture from eval_legibility_guided ───────────────────
from evaluation.eval_legibility_guided import (
    DiffusionPolicy,
    DDIMSampler,
    LPSDDIMSampler,
    l_early_intent_torch,
    ACTION_SCALE,
    OBS_EE_POS,
    OBS_LEFT_POS,
    OBS_RIGHT_POS,
    DEFAULT_CKPT,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# HAND-CRAFTED 4-CRITERIA FUNCTION  (Flaw 2 ablation)
# Same criteria as the VLM-generated function, written by hand.
# Reference: STAGE123_HYPOTHESIS.md "VLM-Generated Function Details"
# ══════════════════════════════════════════════════════════════════════════════

def handcrafted_4criteria(
    ee_traj: torch.Tensor,   # (H, 3)
    goals: torch.Tensor,     # (K, 3)
    true_goal_idx: int = 0,
    early_frac: float = 0.30,
) -> torch.Tensor:
    """4-criteria legibility function constructed by hand from the same criteria
    the VLM chose.  Used to test whether the VLM's function adds anything beyond
    what a human could write with the same criteria list (Flaw 2 ablation).

    Criteria (same weights as VLM-generated function):
      1. Gaussian Proximity P_prox        (w=0.35)  — Dragan 2013 baseline
      2. Directional Alignment P_dir      (w=0.30)  — velocity · goal direction
      3. Lateral Separation P_lat         (w=0.25)  — movement away from non-goal
      4. Speed Commitment P_speed         (w=0.10)  — velocity magnitude
    """
    H = ee_traj.shape[0]
    K = goals.shape[0]
    early_end = max(1, int(H * early_frac))
    early = ee_traj[:early_end]            # (E, 3)
    non_goal = 1 - true_goal_idx           # index of the distractor goal

    # Auto-calibrate sigma from inter-goal distance
    d = torch.cdist(goals, goals)
    mask = d > 1e-6
    d_min = d[mask].min() if mask.any() else torch.tensor(0.14, device=goals.device)
    sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))

    # ── 1. Gaussian Proximity ──────────────────────────────────────────────
    diff    = early.unsqueeze(1) - goals.unsqueeze(0)    # (E, K, 3)
    sq_dist = (diff ** 2).sum(-1)                         # (E, K)
    log_p   = -sq_dist / (2.0 * sigma ** 2)
    post    = torch.softmax(log_p, dim=-1)                # (E, K)
    p_prox  = post[:, true_goal_idx].mean()

    # ── 2. Directional Alignment ──────────────────────────────────────────
    # velocity = finite difference along trajectory
    if early.shape[0] >= 2:
        vel = early[1:] - early[:-1]                      # (E-1, 3)
        vel_mag   = vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vel_unit  = vel / vel_mag                          # (E-1, 3)

        goal_dir  = goals[true_goal_idx] - early[:-1]     # (E-1, 3)
        gdir_mag  = goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        gdir_unit = goal_dir / gdir_mag

        cos_align = (vel_unit * gdir_unit).sum(-1)         # (E-1,)  in [-1, 1]
        p_dir = (cos_align + 1.0).mul(0.5).mean()          # shift to [0, 1]
    else:
        p_dir = torch.tensor(0.5, device=goals.device)

    # ── 3. Lateral Separation ─────────────────────────────────────────────
    # Signed lateral position: projection onto axis perpendicular to table-forward
    # and in the plane of the two goals.
    goal_axis  = goals[true_goal_idx] - goals[non_goal]   # (3,)
    axis_mag   = goal_axis.norm().clamp(min=1e-8)
    axis_unit  = goal_axis / axis_mag

    # Lateral displacement of EE toward true_goal (signed projection)
    ee_from_non = early - goals[non_goal].unsqueeze(0)    # (E, 3)
    lat_proj    = (ee_from_non * axis_unit.unsqueeze(0)).sum(-1)  # (E,)

    # Normalise by inter-goal distance so it's scale-invariant
    p_lat = torch.sigmoid(lat_proj / (axis_mag + 1e-8) * 4.0).mean()

    # ── 4. Speed Commitment ───────────────────────────────────────────────
    if early.shape[0] >= 2:
        speeds = (early[1:] - early[:-1]).norm(dim=-1)     # (E-1,)
        v_char = sigma / 3.0                               # characteristic speed
        p_speed = torch.sigmoid((speeds - v_char) / (v_char + 1e-8)).mean()
    else:
        p_speed = torch.tensor(0.5, device=goals.device)

    # ── Weighted combination ──────────────────────────────────────────────
    score = 0.35 * p_prox + 0.30 * p_dir + 0.25 * p_lat + 0.10 * p_speed
    return score.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# HARD DISCRIMINATION TEST  (Flaw 3 ablation)
# Trajectories that Gaussian-only can't distinguish but directional can.
# ══════════════════════════════════════════════════════════════════════════════

def run_flaw3_discrimination_test(device):
    """Test whether the Stage 1 validation test is trivially easy.

    Construct trajectories at equal proximity to both goals but with different
    directional intentions.  The Gaussian-only (1-criterion) function should
    FAIL to distinguish them; the 4-criteria function should succeed.
    """
    print("\n" + "="*65)
    print("  FLAW 3: Stage 1 validation test hardness analysis")
    print("="*65)

    goals = torch.tensor([[0.50, -0.07, 0.42],
                           [0.50,  0.07, 0.42]], device=device)

    # ── Case A: EASY (used in Stage 1 validation) ─────────────────────────
    # "Legible": straight toward left goal  ← trivially different proximity
    legible_easy = torch.zeros(32, 3, device=device)
    for i in range(32):
        frac = i / 31.0
        legible_easy[i] = torch.tensor([0.40 + frac*0.10, 0.0 - frac*0.07, 0.55 - frac*0.13])

    # "Ambiguous": straight down center  ← different proximity from step 1
    ambig_easy = torch.zeros(32, 3, device=device)
    for i in range(32):
        frac = i / 31.0
        ambig_easy[i] = torch.tensor([0.40 + frac*0.10, 0.0, 0.55 - frac*0.13])

    # ── Case B: HARD (new test) ────────────────────────────────────────────
    # Both trajectories are equidistant from both goals in the early window,
    # but one points toward the left goal (directionally committed) while
    # the other points toward the right goal (wrong direction).
    # This is the midpoint of the two goals at the start of the early window.
    mid_y  = 0.0   # midpoint of goals at y=0
    H      = 32
    E_end  = int(H * 0.30)  # early window = first ~10 steps

    # Trajectory commits LEFT: starts at midpoint, curves left
    traj_left = torch.zeros(H, 3, device=device)
    for i in range(H):
        frac = i / (H - 1)
        # Linear interpolation EE start → left goal
        traj_left[i] = torch.tensor([0.50, mid_y - frac * 0.07, 0.50 - frac * 0.08])

    # Trajectory commits RIGHT: starts at midpoint, curves right
    traj_right = torch.zeros(H, 3, device=device)
    for i in range(H):
        frac = i / (H - 1)
        # Linear interpolation EE start → right goal
        traj_right[i] = torch.tensor([0.50, mid_y + frac * 0.07, 0.50 - frac * 0.08])

    print("\n  Case A: Stage 1 validation (original easy test)")
    print(f"  {'Function':<35}  {'Legible':>8}  {'Ambiguous':>10}  {'Gap':>8}  {'Pass?':>6}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}")

    for name, fn, true_idx in [
        ("1-criteria (Gaussian only)",    l_early_intent_torch,  0),
        ("4-criteria (hand-crafted)", handcrafted_4criteria, 0),
    ]:
        s_leg = fn(legible_easy, goals, true_idx).item()
        s_amb = fn(ambig_easy,   goals, true_idx).item()
        gap   = s_leg - s_amb
        passed = "✓" if gap > 0.01 else "✗"
        print(f"  {name:<35}  {s_leg:>8.4f}  {s_amb:>10.4f}  {gap:>+8.4f}  {passed:>6}")

    print("\n  Case B: HARD test (equidistant start, directional only)")
    print(f"  {'Function':<35}  {'Traj→LEFT':>10}  {'Traj→RIGHT':>11}  {'Gap':>8}  {'Pass?':>6}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*6}")

    for name, fn, true_idx in [
        ("1-criteria (Gaussian only)",    l_early_intent_torch,  0),
        ("4-criteria (hand-crafted)", handcrafted_4criteria, 0),
    ]:
        # true_goal_idx=0 = left goal  →  traj_left should score higher
        s_left  = fn(traj_left,  goals, true_goal_idx=0).item()
        s_right = fn(traj_right, goals, true_goal_idx=0).item()
        gap     = s_left - s_right
        passed  = "✓" if gap > 0.01 else "✗"
        print(f"  {name:<35}  {s_left:>10.4f}  {s_right:>11.4f}  {gap:>+8.4f}  {passed:>6}")

    print()
    print("  INTERPRETATION:")
    print("  - Case A: if 1-criteria passes, the Stage 1 validation is trivially easy.")
    print("  - Case B: if 1-criteria fails but 4-criteria passes, multi-criteria adds real signal.")
    print("  - If 1-criteria passes BOTH, validation tests nothing.")


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER (shared)
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(model, sampler, obs_mean, obs_std, act_mean, act_std, device,
                n_sampling_steps=10, max_steps=400, seed=None):
    if seed is not None:
        np.random.seed(seed)
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset()
    action_queue = deque(maxlen=model.horizon)
    ee_trajectory: List[np.ndarray] = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_trajectory.append(obs[0:3].copy())
        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t    = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            if hasattr(sampler, 'sample') and callable(sampler.sample):
                try:
                    # guided samplers need ee_start + goals
                    ee_start  = torch.tensor(obs[OBS_EE_POS], dtype=torch.float32, device=device)
                    goals_t   = torch.stack([
                        torch.tensor(obs[OBS_LEFT_POS],  dtype=torch.float32, device=device),
                        torch.tensor(obs[OBS_RIGHT_POS], dtype=torch.float32, device=device),
                    ])
                    result_t  = sampler.sample(model, obs_t, ee_start, goals_t,
                                               n_sampling_steps=n_sampling_steps)
                    act_seq   = result_t[0] if isinstance(result_t, tuple) else result_t
                except TypeError:
                    act_seq = sampler.sample(model, obs_t,
                                             n_sampling_steps=n_sampling_steps)
            chunk = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in chunk:
                action_queue.append(a)
        action  = action_queue.popleft()
        result  = env.step(action)
        obs     = result.obs
        last_obs = obs
        success  = (result.info.get('success_left', 0) > 0.5
                    or result.info.get('success_right', 0) > 0.5)
        if result.done:
            break

    env.close()

    if len(ee_trajectory) >= 4:
        traj_arr = np.array(ee_trajectory)
        goals_np = np.stack([last_obs[8:11], last_obs[15:18]], axis=0)
        from evaluation.legibility_metrics import compute_legibility
        r0 = compute_legibility(traj_arr, goals_np, true_goal_idx=0, model='gaussian')
        r1 = compute_legibility(traj_arr, goals_np, true_goal_idx=1, model='gaussian')
        best = r0 if r0.L_early_intent >= r1.L_early_intent else r1
        l_early = best.L_early_intent
    else:
        l_early = 0.0

    return dict(success=success, steps=step + 1, l_early=float(l_early))


class HandcraftedGuidedSampler:
    """Classifier-guidance sampler using hand-crafted 4-criteria function."""

    label = '4-criteria HC (Flaw 2 ablation)'

    def __init__(self, n_steps, beta_start, beta_end, device,
                 guidance_scale=10.0, grad_clip=1.0):
        self.gs = guidance_scale
        self.gc = grad_clip
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.acp = torch.cumprod(alphas, dim=0)

    def sample(self, model, obs, ee_start, goals, n_sampling_steps=10):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        ts = torch.flip(
            torch.linspace(0, len(self.acp) - 1, n_sampling_steps,
                           device=self.device).long(), [0])

        for i, t in enumerate(ts):
            t_batch = t.repeat(B)
            alpha_t    = self.acp[t]
            alpha_prev = self.acp[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=self.device)
            sqrt_ab    = torch.sqrt(alpha_t)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_t)

            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                eps_pred = model(x_in, t_batch, obs)
                x0_pred  = (x_in - sqrt_1m_ab * eps_pred) / sqrt_ab
                delta_pos = x0_pred[0, :, :3] * ACTION_SCALE
                ee_traj   = torch.cumsum(delta_pos, dim=0) + ee_start
                # Infer goal (detached)
                with torch.no_grad():
                    s0 = handcrafted_4criteria(ee_traj.detach(), goals, 0).item()
                    s1 = handcrafted_4criteria(ee_traj.detach(), goals, 1).item()
                tg = 0 if s0 >= s1 else 1
                score = handcrafted_4criteria(ee_traj, goals, tg)
                grad  = torch.autograd.grad(score, x_in)[0]

            with torch.no_grad():
                g  = grad.detach()
                gn = g.norm()
                if gn > self.gc:
                    g = g * (self.gc / (gn + 1e-8))
                guided_eps = eps_pred.detach() - self.gs * sqrt_1m_ab * g
                x0_g = (x - sqrt_1m_ab * guided_eps) / sqrt_ab
                if i < len(ts) - 1:
                    x = torch.sqrt(alpha_prev) * x0_g + torch.sqrt(1.0 - alpha_prev) * guided_eps
                else:
                    x = x0_g

        return x, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--n_episodes', type=int, default=10)
    ap.add_argument('--guidance_scale', type=float, default=10.0)
    ap.add_argument('--n_sampling_steps', type=int, default=10)
    ap.add_argument('--skip_rollouts', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*65)
    print("  Stage123 Hypothesis Flaws Verification")
    print("="*65)
    print(f"  Device  : {device}  |  Episodes : {args.n_episodes}  |  w={args.guidance_scale}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt['config']
    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'], act_dim=cfg['act_dim'], horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256), n_blocks=cfg.get('n_blocks', 3),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    obs_mean = np.array(ckpt['obs_mean'], dtype=np.float32)
    obs_std  = np.array(ckpt['obs_std'],  dtype=np.float32)
    act_mean = np.array(ckpt['act_mean'], dtype=np.float32)
    act_std  = np.array(ckpt['act_std'],  dtype=np.float32)
    n_diff = cfg.get('n_diffusion_steps', 100)
    b_s    = cfg.get('beta_start', 1e-4)
    b_e    = cfg.get('beta_end', 0.1)

    # ── Flaw 3: discrimination test ──────────────────────────────────────────
    run_flaw3_discrimination_test(device)

    if args.skip_rollouts:
        print("Skipping rollout comparison (--skip_rollouts).")
        return

    # ── Flaw 5: baseline reproducibility ────────────────────────────────────
    print("\n" + "="*65)
    print("  FLAW 5: Baseline reproducibility (85% vs 95% discrepancy)")
    print("="*65)
    baseline_sampler = DDIMSampler(n_diff, b_s, b_e, device)
    baseline_results = []
    for ep in range(args.n_episodes):
        r = run_episode(model, baseline_sampler, obs_mean, obs_std, act_mean, act_std,
                        device, n_sampling_steps=args.n_sampling_steps)
        baseline_results.append(r)
        tick = '✓' if r['success'] else '✗'
        print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  L_early={r['l_early']:.4f}")
    bl_s = np.mean([r['success'] for r in baseline_results])
    bl_l = np.mean([r['l_early'] for r in baseline_results])
    print(f"\n  Baseline: success={bl_s:.1%}  L_early={bl_l:.4f}")
    if bl_s >= 0.90:
        print("  → Consistent with DPS_HYPOTHESIS.md (95%) — Stage1 85% was an outlier.")
    elif bl_s <= 0.87:
        print("  → Consistent with Stage1 result (85%) — DPS_HYPOTHESIS.md 95% was an outlier.")
    else:
        print("  → Intermediate — both runs have noise; baseline is ~88-92% range.")

    # ── Flaw 2: multi-criteria ablation ─────────────────────────────────────
    print("\n" + "="*65)
    print("  FLAW 2: 1-criteria vs 4-criteria (hand-crafted ablation)")
    print("="*65)
    print("  If 4-criteria HC ≈ VLM 4-criteria, VLM added nothing.")

    sampler_1c = LPSDDIMSampler(n_diff, b_s, b_e, device,
                                 guidance_scale=args.guidance_scale, grad_clip=1.0)
    sampler_4c = HandcraftedGuidedSampler(n_diff, b_s, b_e, device,
                                           guidance_scale=args.guidance_scale, grad_clip=1.0)

    all_results = {}
    for label, sampler in [
        ("1-criteria HC (Gaussian only, w=%.0f)" % args.guidance_scale, sampler_1c),
        ("4-criteria HC (ablation, w=%.0f)" % args.guidance_scale,      sampler_4c),
    ]:
        print(f"\n── {label} ── {args.n_episodes} episodes ──")
        results = []
        for ep in range(args.n_episodes):
            r = run_episode(model, sampler, obs_mean, obs_std, act_mean, act_std,
                            device, n_sampling_steps=args.n_sampling_steps)
            results.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  L_early={r['l_early']:.4f}")
        l_vals = [r['l_early'] for r in results]
        s_vals = [r['success'] for r in results]
        print(f"  → success={np.mean(s_vals):.1%}  L_early={np.mean(l_vals):.4f}±{np.std(l_vals):.4f}")
        all_results[label] = {
            'l_early_mean': float(np.mean(l_vals)),
            'l_early_std':  float(np.std(l_vals)),
            'success_rate': float(np.mean(s_vals)),
        }

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  Baseline (DDIM):              success={bl_s:.1%}  L_early={bl_l:.4f}")
    for k, v in all_results.items():
        delta = v['l_early_mean'] - bl_l
        print(f"  {k[:50]:<50}  success={v['success_rate']:.1%}  "
              f"L_early={v['l_early_mean']:.4f}  Δ={delta:+.4f}")

    labels = list(all_results.keys())
    if len(labels) == 2:
        l1 = all_results[labels[0]]['l_early_mean']
        l4 = all_results[labels[1]]['l_early_mean']
        delta_4_over_1 = l4 - l1
        vlm_stage1_delta = 0.9486 - 0.9431  # from STAGE123_HYPOTHESIS.md
        print(f"\n  4-criteria HC vs 1-criteria HC: ΔL_early = {delta_4_over_1:+.4f}")
        print(f"  VLM 4-criteria vs HC 1-criteria (Stage1 paper): ΔL_early = {vlm_stage1_delta:+.4f}")
        if abs(delta_4_over_1 - vlm_stage1_delta) < 0.005:
            print("\n  FLAW 2 CONFIRMED: multi-criteria alone explains the VLM improvement.")
            print("  The VLM contributed no additional signal beyond the criteria count.")
        elif delta_4_over_1 > vlm_stage1_delta + 0.005:
            print("\n  FLAW 2 PARTIALLY CONFIRMED: 4-criteria HC beats 1-criteria by MORE than VLM did.")
            print("  VLM-generated implementation may actually be worse than hand-crafted equivalent.")
        else:
            print("\n  FLAW 2 UNCLEAR: VLM showed larger gain than criteria count alone. Needs more episodes.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'verify_stage123_flaws.json'
    payload = {
        'guidance_scale': args.guidance_scale,
        'n_episodes':     args.n_episodes,
        'baseline':       {'success_rate': float(bl_s), 'l_early_mean': float(bl_l)},
        'ablation':       all_results,
        'flaw2_delta_4criteria_hc': float(list(all_results.values())[1]['l_early_mean']
                                          - list(all_results.values())[0]['l_early_mean'])
                          if len(all_results) == 2 else None,
        'flaw5_baseline_success': float(bl_s),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == '__main__':
    main()
