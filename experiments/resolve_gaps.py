#!/usr/bin/env python3
"""
Resolve Critical Gaps: Arc Classification + Reverse-Steering
=============================================================

Addresses 3 high-severity issues from HONEST_ASSESSMENT.md:

1. **Arc classification** on guided trajectories (never computed before)
   - Saves full EE trajectories and classifies into arc categories
   - arc00-05: max lateral < 0.05m (straight)
   - arc10-14: 0.05m <= max lateral < 0.15m (moderate)
   - arc15-19: max lateral >= 0.15m (large sweep)

2. **Reverse-steering** test (w < 0)
   - Negative guidance should produce PREDICTABLE (straight) trajectories
   - Proves causal control: w>0 = legible arcs, w<0 = straight paths

3. **Per-episode trajectory analysis**
   - Saves full trajectory data for downstream VLM visual evaluation

Usage:
  python experiments/resolve_gaps.py                 # Run all conditions
  python experiments/resolve_gaps.py --arc_only      # Arc classification only
  python experiments/resolve_gaps.py --reverse_only  # Reverse-steering only
  python experiments/resolve_gaps.py --n_episodes 10 # Fewer episodes
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv
from evaluation.eval_legibility_guided import (
    DDIMSampler,
    DiffusionPolicy,
    LPSDDIMSampler,
    l_early_intent_torch,
)

ACTION_SCALE = 0.05
DEFAULT_CKPT = 'runs/diffusion_20260222_195530/ckpt_ep100.pt'

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── Arc Classification ─────────────────────────────────────────────────

def classify_arc(ee_trajectory: np.ndarray) -> dict:
    """Classify a trajectory's arc by max cumulative lateral Y-displacement.

    Categories (from analyze_arc_structure.py):
      arc00-05: max_arc < 0.05m   (straight approach)
      arc10-14: 0.05 <= max_arc < 0.15m (moderate curve)
      arc15-19: max_arc >= 0.15m  (large lateral sweep)
    """
    if len(ee_trajectory) < 2:
        return {"max_arc": 0.0, "category": "arc00-05"}

    # Compute Y-displacement from start position
    y_start = ee_trajectory[0, 1]
    y_displacements = ee_trajectory[:, 1] - y_start
    max_arc = float(np.max(np.abs(y_displacements)))

    if max_arc < 0.05:
        category = "arc00-05"
    elif max_arc < 0.15:
        category = "arc10-14"
    else:
        category = "arc15-19"

    return {
        "max_arc": max_arc,
        "category": category,
        "y_displacement_range": float(np.ptp(y_displacements)),
        "y_final": float(y_displacements[-1]),
    }


def arc_distribution(episodes: List[dict]) -> dict:
    """Compute arc category distribution from a list of episode results."""
    cats = [ep["arc"]["category"] for ep in episodes]
    total = len(cats)
    dist = {
        "arc00-05": cats.count("arc00-05"),
        "arc10-14": cats.count("arc10-14"),
        "arc15-19": cats.count("arc15-19"),
    }
    pct = {k: round(v / total * 100, 1) if total > 0 else 0
           for k, v in dist.items()}
    return {"counts": dist, "percentages": pct, "total": total}


# ─── Episode Runner (saves full trajectory) ─────────────────────────────

def run_episode_with_trajectory(
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
    """Run one episode, saving full EE trajectory for arc analysis."""
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset(seed=env_seed)
    torch.manual_seed(sample_seed)

    action_queue: deque = deque(maxlen=model.horizon)
    ee_trajectory: List[np.ndarray] = []
    guided_scores: List[float] = []
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
                guided_scores.append(s_val)
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

    # Compute L_early from executed trajectory
    traj_arr = np.array(ee_trajectory)
    l_early, rlc, true_goal = 0.0, 0.0, 'unknown'
    if len(ee_trajectory) >= 4:
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

    # Arc classification
    arc_info = classify_arc(traj_arr)

    return dict(
        env_seed=env_seed,
        sample_seed=sample_seed,
        success=bool(success),
        steps=step + 1,
        l_early_actual=float(l_early),
        rlc_actual=float(rlc),
        true_goal=true_goal,
        guided_score_mean=(float(np.mean(guided_scores)) if guided_scores else 0.0),
        arc=arc_info,
        ee_trajectory=traj_arr.tolist(),  # full trajectory for downstream analysis
    )


# ─── Generic Guided Sampler (supports negative w for reverse-steering) ──

class FlexibleGuidedSampler:
    """DDIM with classifier guidance. Supports negative guidance_scale
    for reverse-steering (push AWAY from legibility).
    """

    def __init__(self, n_steps, beta_start, beta_end, device,
                 score_fn: Callable,
                 guidance_scale: float = 10.0,
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
                # guidance_scale can be negative for reverse-steering
                guided_eps = eps_pred.detach() - self.guidance_scale * sqrt_1m_ab * g
                x0_guided = (x - sqrt_1m_ab * guided_eps) / sqrt_ab
                if i < len(timesteps) - 1:
                    x = (torch.sqrt(alpha_prev) * x0_guided
                         + torch.sqrt(1.0 - alpha_prev) * guided_eps)
                else:
                    x = x0_guided

        return x, final_score


# ─── Load VLM Score Function ────────────────────────────────────────────

def load_vlm_fn(path: str, device: torch.device):
    """Dynamically load VLM-generated scoring function."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("vlm_fn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.vlm_legibility_score


# ─── Print Summary ──────────────────────────────────────────────────────

def print_condition_summary(label: str, episodes: List[dict]):
    """Print summary table for one condition."""
    n = len(episodes)
    successes = sum(1 for ep in episodes if ep["success"])
    l_vals = [ep["l_early_actual"] for ep in episodes]
    arcs = arc_distribution(episodes)

    print(f"\n  {label}")
    print(f"  {'─'*58}")
    print(f"  Success: {successes}/{n} ({successes/n*100:.0f}%)")
    print(f"  L_early: {np.mean(l_vals):.4f} ± {np.std(l_vals):.4f}")
    print(f"  Arc distribution:")
    for cat in ["arc00-05", "arc10-14", "arc15-19"]:
        cnt = arcs["counts"][cat]
        pct = arcs["percentages"][cat]
        bar = "█" * int(pct / 5)
        print(f"    {cat}: {cnt:3d} ({pct:5.1f}%) {bar}")
    max_arcs = [ep["arc"]["max_arc"] for ep in episodes]
    print(f"  Max arc: {np.mean(max_arcs):.4f}m ± {np.std(max_arcs):.4f}m")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--n_episodes', type=int, default=20,
                    help='Episodes per condition (paired seeds)')
    ap.add_argument('--n_sampling_steps', type=int, default=10)
    ap.add_argument('--vlm_fn_path', default='outputs/stage1/vlm_score_fn.py')
    ap.add_argument('--arc_only', action='store_true',
                    help='Only run arc classification (skip reverse-steering)')
    ap.add_argument('--reverse_only', action='store_true',
                    help='Only run reverse-steering (skip arc classification)')
    args = ap.parse_args()

    run_arc = not args.reverse_only
    run_reverse = not args.arc_only

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(__file__).parent.parent / 'outputs' / 'gap_resolution'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print("  RESOLVE CRITICAL GAPS — Arc Classification + Reverse-Steering")
    print(f"{'='*72}")
    print(f"  Device    : {device}")
    print(f"  Episodes  : {args.n_episodes} (paired seeds)")
    print(f"  Run arc   : {run_arc}")
    print(f"  Run reverse: {run_reverse}")
    print(f"{'='*72}\n")

    # ── Deterministic seed pairs (same as rigorous eval) ──
    rng = np.random.RandomState(42)
    seed_pairs = [(int(rng.randint(0, 10000)), int(rng.randint(0, 100000)))
                  for _ in range(args.n_episodes)]

    # ── Load model ──
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

    # ── Load VLM score function ──
    vlm_fn = load_vlm_fn(args.vlm_fn_path, device)
    print(f"  VLM function loaded from: {args.vlm_fn_path}")

    # ── Build samplers ──
    baseline_sampler = DDIMSampler(n_diff, beta_s, beta_e, device)

    # Hand-crafted at w=10 (best operating point with >=95% success)
    hc_sampler = FlexibleGuidedSampler(
        n_diff, beta_s, beta_e, device,
        score_fn=l_early_intent_torch, guidance_scale=10.0)

    # VLM at w=10
    vlm_sampler = FlexibleGuidedSampler(
        n_diff, beta_s, beta_e, device,
        score_fn=vlm_fn, guidance_scale=10.0)

    all_results = {"seed_pairs": seed_pairs}

    # ══════════════════════════════════════════════════════════════════
    # PART 1: ARC CLASSIFICATION ON ALL CONDITIONS
    # ══════════════════════════════════════════════════════════════════
    if run_arc:
        print(f"\n{'='*72}")
        print("  PART 1: ARC CLASSIFICATION ACROSS CONDITIONS")
        print(f"{'='*72}")

        conditions = [
            ("baseline",       baseline_sampler, False),
            ("hc_w10",         hc_sampler,       True),
            ("vlm_w10",        vlm_sampler,      True),
        ]

        for cond_name, sampler, is_guided in conditions:
            print(f"\n  ── {cond_name.upper()} ── {args.n_episodes} episodes ──")
            episodes = []
            for idx, (env_s, samp_s) in enumerate(seed_pairs):
                r = run_episode_with_trajectory(
                    model, sampler, obs_mean, obs_std, act_mean, act_std,
                    device, env_s, samp_s,
                    guided=is_guided,
                    n_sampling_steps=args.n_sampling_steps)
                episodes.append(r)
                arc_cat = r["arc"]["category"]
                mark = "✓" if r["success"] else "✗"
                print(f"    [{idx+1:2d}/{args.n_episodes}] seed=({env_s},{samp_s}) "
                      f"{mark} L={r['l_early_actual']:.3f} "
                      f"arc={r['arc']['max_arc']:.4f}m ({arc_cat})")

            print_condition_summary(cond_name.upper(), episodes)
            all_results[cond_name] = {
                "success_rate": sum(1 for e in episodes if e["success"]) / len(episodes),
                "l_early_mean": float(np.mean([e["l_early_actual"] for e in episodes])),
                "l_early_std": float(np.std([e["l_early_actual"] for e in episodes])),
                "arc_distribution": arc_distribution(episodes),
                "max_arc_mean": float(np.mean([e["arc"]["max_arc"] for e in episodes])),
                "max_arc_std": float(np.std([e["arc"]["max_arc"] for e in episodes])),
                "episodes": episodes,
            }

        # ── Comparison Table ──
        print(f"\n\n{'='*72}")
        print("  ARC CLASSIFICATION COMPARISON")
        print(f"{'='*72}")
        print(f"  {'Condition':<20} {'Success':>8} {'L_early':>8} "
              f"{'arc00-05':>9} {'arc10-14':>9} {'arc15-19':>9} {'MeanArc':>8}")
        print(f"  {'─'*68}")
        for cond_name in ["baseline", "hc_w10", "vlm_w10"]:
            r = all_results[cond_name]
            d = r["arc_distribution"]["percentages"]
            print(f"  {cond_name:<20} {r['success_rate']:>7.0%} {r['l_early_mean']:>8.4f} "
                  f"{d['arc00-05']:>8.1f}% {d['arc10-14']:>8.1f}% {d['arc15-19']:>8.1f}% "
                  f"{r['max_arc_mean']:>7.4f}m")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: REVERSE-STEERING TEST (w < 0)
    # ══════════════════════════════════════════════════════════════════
    if run_reverse:
        print(f"\n\n{'='*72}")
        print("  PART 2: REVERSE-STEERING TEST (w < 0)")
        print(f"  If w > 0 produces legible arcs, w < 0 should produce STRAIGHT paths.")
        print(f"{'='*72}")

        reverse_scales = [-5.0, -10.0]
        forward_scale = 10.0

        # Forward reference (w=+10, HC)
        fwd_name = "forward_hc_w10"
        if fwd_name not in all_results:
            print(f"\n  ── FORWARD HC w=+{forward_scale} ── {args.n_episodes} episodes ──")
            fwd_sampler = FlexibleGuidedSampler(
                n_diff, beta_s, beta_e, device,
                score_fn=l_early_intent_torch, guidance_scale=forward_scale)
            episodes_fwd = []
            for idx, (env_s, samp_s) in enumerate(seed_pairs):
                r = run_episode_with_trajectory(
                    model, fwd_sampler, obs_mean, obs_std, act_mean, act_std,
                    device, env_s, samp_s, guided=True,
                    n_sampling_steps=args.n_sampling_steps)
                episodes_fwd.append(r)
                mark = "✓" if r["success"] else "✗"
                print(f"    [{idx+1:2d}/{args.n_episodes}] {mark} "
                      f"L={r['l_early_actual']:.3f} arc={r['arc']['max_arc']:.4f}m")
            print_condition_summary(f"FORWARD HC w=+{forward_scale}", episodes_fwd)
            all_results[fwd_name] = {
                "guidance_scale": forward_scale,
                "success_rate": sum(1 for e in episodes_fwd if e["success"]) / len(episodes_fwd),
                "l_early_mean": float(np.mean([e["l_early_actual"] for e in episodes_fwd])),
                "arc_distribution": arc_distribution(episodes_fwd),
                "max_arc_mean": float(np.mean([e["arc"]["max_arc"] for e in episodes_fwd])),
                "episodes": episodes_fwd,
            }

        # Reverse conditions
        for w in reverse_scales:
            rev_name = f"reverse_hc_w{w}"
            print(f"\n  ── REVERSE HC w={w} ── {args.n_episodes} episodes ──")
            rev_sampler = FlexibleGuidedSampler(
                n_diff, beta_s, beta_e, device,
                score_fn=l_early_intent_torch, guidance_scale=w)
            episodes_rev = []
            for idx, (env_s, samp_s) in enumerate(seed_pairs):
                r = run_episode_with_trajectory(
                    model, rev_sampler, obs_mean, obs_std, act_mean, act_std,
                    device, env_s, samp_s, guided=True,
                    n_sampling_steps=args.n_sampling_steps)
                episodes_rev.append(r)
                mark = "✓" if r["success"] else "✗"
                print(f"    [{idx+1:2d}/{args.n_episodes}] {mark} "
                      f"L={r['l_early_actual']:.3f} arc={r['arc']['max_arc']:.4f}m")
            print_condition_summary(f"REVERSE HC w={w}", episodes_rev)
            all_results[rev_name] = {
                "guidance_scale": w,
                "success_rate": sum(1 for e in episodes_rev if e["success"]) / len(episodes_rev),
                "l_early_mean": float(np.mean([e["l_early_actual"] for e in episodes_rev])),
                "l_early_std": float(np.std([e["l_early_actual"] for e in episodes_rev])),
                "arc_distribution": arc_distribution(episodes_rev),
                "max_arc_mean": float(np.mean([e["arc"]["max_arc"] for e in episodes_rev])),
                "max_arc_std": float(np.std([e["arc"]["max_arc"] for e in episodes_rev])),
                "episodes": episodes_rev,
            }

        # ── Reverse-Steering Comparison ──
        print(f"\n\n{'='*72}")
        print("  REVERSE-STEERING COMPARISON")
        print(f"{'='*72}")
        rev_conds = ["baseline"]
        if "hc_w10" in all_results:
            rev_conds.append("hc_w10")
        elif fwd_name in all_results:
            rev_conds.append(fwd_name)
        for w in reverse_scales:
            rev_conds.append(f"reverse_hc_w{w}")

        print(f"  {'Condition':<25} {'Success':>8} {'L_early':>8} "
              f"{'arc00-05':>9} {'arc15-19':>9} {'MeanArc':>8}")
        print(f"  {'─'*68}")
        for cond_name in rev_conds:
            if cond_name not in all_results:
                continue
            r = all_results[cond_name]
            d = r["arc_distribution"]["percentages"]
            print(f"  {cond_name:<25} {r['success_rate']:>7.0%} {r['l_early_mean']:>8.4f} "
                  f"{d['arc00-05']:>8.1f}% {d['arc15-19']:>8.1f}% "
                  f"{r['max_arc_mean']:>7.4f}m")

        # Check if reverse actually shifts distribution
        if fwd_name in all_results and f"reverse_hc_w{reverse_scales[0]}" in all_results:
            fwd_arcs = [e["arc"]["max_arc"] for e in all_results[fwd_name]["episodes"]]
            rev_arcs = [e["arc"]["max_arc"] for e in
                        all_results[f"reverse_hc_w{reverse_scales[0]}"]["episodes"]]
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(fwd_arcs, rev_arcs)
            print(f"\n  Forward vs Reverse arc magnitude:")
            print(f"    Forward mean arc: {np.mean(fwd_arcs):.4f}m")
            print(f"    Reverse mean arc: {np.mean(rev_arcs):.4f}m")
            print(f"    t-stat={t_stat:.3f}, p={p_val:.6f}")
            if p_val < 0.05:
                print(f"    SIGNIFICANT — reverse steering changes arc shape (p<0.05)")
            else:
                print(f"    NOT significant — reverse steering has no arc effect")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════

    # Strip full trajectories for the summary file (they're large)
    summary = {}
    for key, val in all_results.items():
        if isinstance(val, dict) and "episodes" in val:
            summary[key] = {k: v for k, v in val.items() if k != "episodes"}
            # Keep per-episode arc + metrics but not full trajectories
            summary[key]["episodes"] = [
                {k: v for k, v in ep.items() if k != "ee_trajectory"}
                for ep in val["episodes"]
            ]
        else:
            summary[key] = val

    summary_path = out_dir / 'gap_resolution_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    # Save full trajectories separately (for visual VLM eval later)
    traj_path = out_dir / 'trajectories.npz'
    traj_data = {}
    for key, val in all_results.items():
        if isinstance(val, dict) and "episodes" in val:
            for i, ep in enumerate(val["episodes"]):
                if "ee_trajectory" in ep:
                    traj_data[f"{key}_ep{i}"] = np.array(ep["ee_trajectory"])
    if traj_data:
        np.savez_compressed(traj_path, **traj_data)
        print(f"  Trajectories saved to: {traj_path}")

    print(f"\n{'='*72}")
    print("  DONE")
    print(f"{'='*72}\n")


if __name__ == '__main__':
    main()
