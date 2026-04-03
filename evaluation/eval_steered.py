#!/usr/bin/env python
"""Inference-time legibility steering for the existing unconditional diffusion policy.

Uses the EXACT same model, sampler, and rollout pipeline as eval_with_videos.py
(which achieves 92% success), adding best-of-N candidate selection with
legibility scoring on top.

Approach:
  - At each replan step, sample N candidate action chunks from the
    unconditional policy (one batched DDIM pass).
  - Score each candidate for lateral alignment with the desired target.
  - Select the highest-scoring candidate and execute the full horizon.
  - NO retraining — uses the existing checkpoint as-is.

Usage:
    py scripts/eval_steered.py --target left  --n_candidates 16
    py scripts/eval_steered.py --target right --n_candidates 16
    py scripts/eval_steered.py --target both  --n_candidates 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv


# ═══════════════════════════════════════════════════════════════════════
# MODEL — identical to eval_with_videos.py
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class UNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.shortcut = (nn.Linear(in_dim, out_dim)
                         if in_dim != out_dim else nn.Identity())
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.act = nn.Mish()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h.transpose(1, 2)
        h = self.norm1(h)
        h = h.transpose(1, 2)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        return self.act(h + self.shortcut(x))


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128),
            nn.Linear(128, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(act_dim, hidden_dim)

        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i + 1], hidden_dim)
            for i in range(len(dims) - 1)
        ])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i + 1] + dims[i + 1], dims[i], hidden_dim)
            for i in range(len(dims) - 2, -1, -1)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, noisy_act, timestep, obs):
        B = noisy_act.shape[0]
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_embed(obs)
        x = self.input_proj(noisy_act)
        x = x + obs_emb.unsqueeze(1)
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            skip_connections.append(x)
        x = self.bottleneck(x, t_emb)
        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=-1)
            x = block(x, t_emb)
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════
# DDIM SAMPLER — identical to eval_with_videos.py
# ═══════════════════════════════════════════════════════════════════════

class DDIMSampler:
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs, n_sampling_steps=10, temperature=1.0,
               initial_noise=None):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim

        if initial_noise is not None:
            x = initial_noise
        else:
            x = torch.randn(B, H, A, device=self.device) * temperature

        all_steps = torch.arange(len(self.alphas_cumprod), device=self.device)
        timesteps = torch.linspace(
            0, len(all_steps) - 1, n_sampling_steps, device=self.device
        ).long()
        timesteps = torch.flip(timesteps, [0])

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            pred_noise = model(x, t_batch, obs)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else torch.tensor(1.0, device=self.device)
            )

            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(
                alpha_t
            )

            if i < len(timesteps) - 1:
                x = (
                    torch.sqrt(alpha_prev) * x0_pred
                    + torch.sqrt(1 - alpha_prev) * pred_noise
                )
            else:
                x = x0_pred

        return x


# ═══════════════════════════════════════════════════════════════════════
# LEGIBILITY SCORING — uses v2-calibrated proxy (not raw dy heuristic)
# ═══════════════════════════════════════════════════════════════════════

from scripts.trajectory_legibility_proxy import (
    proxy_score_chunk,
    DEFAULT_WEIGHTS,
    load_weights,
)

# Module-level weights, set in evaluate() if --proxy_weights is provided
_proxy_weights: dict | None = None


def legibility_score(chunk: np.ndarray, target_sign: float,
                     obs: np.ndarray) -> float:
    """Score an action chunk using geometry-based proxy.

    NOTE: This uses spatial heuristics (arc magnitude proxy), NOT VLM scoring.
    For true VLM-based legibility, use score_candidate_vlm() from
    evaluation/vlm_steering_experiment.py.
    """
    return proxy_score_chunk(chunk, target_sign, obs,
                             action_scale=0.05,
                             weights=_proxy_weights)


# ═══════════════════════════════════════════════════════════════════════
# STEERED ROLLOUT — same rollout logic as eval_with_videos.py
#   + best-of-N selection at each replan step
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def steered_rollout(
    env, model, sampler,
    obs_mean, obs_std, act_mean, act_std,
    target_sign: float,
    n_candidates: int = 16,
    n_sampling_steps: int = 10,
    max_steps: int = 400,
    video_path: str | None = None,
    device=None,
) -> Dict:
    """Run one steered episode using best-of-N selection.

    Rollout mechanics (action queue, denorm, stepping) are identical to
    eval_with_videos.py — only the candidate selection is added.
    """
    obs = env.reset()
    first_obs = obs.copy()
    ee_trajectory = [obs[:3].copy()]

    if video_path:
        env.record_video(video_path, width=640, height=480, fps=30)

    action_queue = deque(maxlen=model.horizon)
    steps = 0
    success = False
    plan_count = 0

    result = None
    while not success and steps < max_steps:
        if len(action_queue) == 0:
            # Normalize observation
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(
                obs_norm, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # ── Best-of-N ────────────────────────────────────────
            obs_batch = obs_t.expand(n_candidates, -1)

            # One batched DDIM pass for all N candidates
            candidates = sampler.sample(
                model, obs_batch, n_sampling_steps=n_sampling_steps
            )  # (N, H, A) normalized

            # Denormalize (numpy, no clipping — matches eval_with_videos)
            cands_np = candidates.cpu().numpy()           # (N, H, A)
            cands_np = cands_np * act_std + act_mean

            # Score each candidate
            scores = np.array([
                legibility_score(cands_np[i], target_sign, obs)
                for i in range(n_candidates)
            ])

            best_idx = int(np.argmax(scores))
            chosen = cands_np[best_idx]
            plan_count += 1

            if plan_count <= 3:
                top3 = np.argsort(scores)[::-1][:3]
                tag = (f"[plan {plan_count}] best=#{best_idx} "
                       f"score={scores[best_idx]:.4f} "
                       f"dy_mean={chosen[:,1].mean():+.5f}")
                print(f"    {tag}")

            # Fill the FULL action queue (like eval_with_videos)
            for a in chosen:
                action_queue.append(a)

        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs
        success = (
            result.info.get("success_left", 0) > 0.5
            or result.info.get("success_right", 0) > 0.5
        )
        steps += 1
        ee_trajectory.append(obs[:3].copy())
        if result.done:
            break

    # Flush video BEFORE closing the env
    env.stop_video()
    env.close()

    # Determine outcome
    if result is None:
        return {"outcome": "failure", "steps": 0, "success": False,
                "plans": 0, "ee_trajectory": np.zeros((1, 3))}

    info = result.info
    picked_left = info.get("picked_left", False) or info.get("success_left", 0) > 0.5
    picked_right = info.get("picked_right", False) or info.get("success_right", 0) > 0.5
    if picked_left:
        outcome = "left_success"
    elif picked_right:
        outcome = "right_success"
    else:
        outcome = "failure"

    return {
        "outcome": outcome,
        "steps": steps,
        "success": picked_left or picked_right,
        "plans": plan_count,
        "ee_trajectory": np.array(ee_trajectory),
        "obs_first": first_obs,
    }


# ═══════════════════════════════════════════════════════════════════════
# UNSTEERED ROLLOUT — identical to eval_with_videos.py (for baseline)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def unsteered_rollout(
    env, model, sampler, obs_mean, obs_std, act_mean, act_std,
    n_sampling_steps: int = 10, max_steps: int = 400, device=None,
    video_path: str | None = None,
) -> Dict:
    """Run one unsteered episode (identical to eval_with_videos.py)."""
    obs = env.reset()
    first_obs = obs.copy()
    ee_trajectory = [obs[:3].copy()]

    if video_path:
        env.record_video(video_path, width=640, height=480, fps=30)

    action_queue = deque(maxlen=model.horizon)
    steps = 0
    success = False
    result = None

    while not success and steps < max_steps:
        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(
                obs_norm, dtype=torch.float32, device=device
            ).unsqueeze(0)
            act_seq = sampler.sample(model, obs_t, n_sampling_steps=n_sampling_steps)
            act_seq = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in act_seq:
                action_queue.append(a)

        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs
        success = (
            result.info.get("success_left", 0) > 0.5
            or result.info.get("success_right", 0) > 0.5
        )
        steps += 1
        ee_trajectory.append(obs[:3].copy())
        if result.done:
            break

    env.stop_video()
    env.close()
    if result is None:
        return {"outcome": "failure", "steps": 0, "success": False,
                "ee_trajectory": np.zeros((1, 3))}

    info = result.info
    picked_left = info.get("picked_left", False) or info.get("success_left", 0) > 0.5
    picked_right = info.get("picked_right", False) or info.get("success_right", 0) > 0.5
    if picked_left:
        outcome = "left_success"
    elif picked_right:
        outcome = "right_success"
    else:
        outcome = "failure"
    return {"outcome": outcome, "steps": steps, "success": picked_left or picked_right,
            "ee_trajectory": np.array(ee_trajectory), "obs_first": first_obs}


# ═══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"Checkpoint: {args.ckpt}")
    print(f"  Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.6f}")

    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg.get("hidden_dim", 256),
        n_blocks=3,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"horizon={model.horizon}")

    sampler = DDIMSampler(
        cfg["n_diffusion_steps"], cfg["beta_start"], cfg["beta_end"], device
    )

    # Normalization stats (numpy — matches eval_with_videos.py)
    obs_mean = ckpt["obs_mean"]
    obs_std = ckpt["obs_std"]
    act_mean = ckpt["act_mean"]
    act_std = ckpt["act_std"]

    # Load proxy weights if provided
    global _proxy_weights
    if hasattr(args, "proxy_weights") and args.proxy_weights:
        _proxy_weights = load_weights(args.proxy_weights)
        print(f"  Proxy weights: {args.proxy_weights}")
    else:
        _proxy_weights = DEFAULT_WEIGHTS
        print(f"  Proxy weights: DEFAULT (hand-tuned, v2-aligned)")

    targets = ["left", "right"] if args.target == "both" else [args.target]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    # ── Run unsteered baseline first ──────────────────────────────
    if args.run_baseline:
        print(f"\n{'='*65}")
        print(f"  UNSTEERED BASELINE — {args.n_baseline} episodes")
        print(f"{'='*65}")
        base_left = base_right = base_fail = 0
        for ep in range(args.n_baseline):
            env = TwoBlockPickEnv(render=False, episode_length=400,
                                  cube_jitter=0.0)
            res = unsteered_rollout(
                env, model, sampler, obs_mean, obs_std, act_mean, act_std,
                n_sampling_steps=args.n_sampling_steps, device=device
            )
            if res["outcome"] == "left_success":
                base_left += 1
            elif res["outcome"] == "right_success":
                base_right += 1
            else:
                base_fail += 1
            print(f"  baseline ep={ep:3d}  {res['outcome']:14s}  "
                  f"({res['steps']} steps)")

        base_total = args.n_baseline
        base_success = base_left + base_right
        print(f"\n  Baseline: {base_success}/{base_total} success "
              f"({base_success/base_total:.0%}), "
              f"L={base_left} R={base_right} F={base_fail}")
        if base_success > 0:
            print(f"  Left/Right split: {base_left}/{base_success} left "
                  f"({base_left/base_success:.0%})")

    # ── Run steered evaluation ────────────────────────────────────
    for target_side in targets:
        target_sign = 1.0 if target_side == "left" else -1.0

        print(f"\n{'='*65}")
        print(f"  STEERED — target = {target_side.upper()}, "
              f"N = {args.n_candidates} candidates")
        print(f"{'='*65}")

        vid_dir = out_dir / f"videos_{target_side}"
        vid_dir.mkdir(parents=True, exist_ok=True)

        for ep in range(args.n_episodes):
            env = TwoBlockPickEnv(render=False, episode_length=400,
                                  cube_jitter=0.0)

            video_tmp = str(vid_dir / f"_tmp_ep{ep:03d}.mp4")

            res = steered_rollout(
                env, model, sampler,
                obs_mean, obs_std, act_mean, act_std,
                target_sign=target_sign,
                n_candidates=args.n_candidates,
                n_sampling_steps=args.n_sampling_steps,
                max_steps=400,
                video_path=video_tmp if ep < args.n_videos else None,
                device=device,
            )

            res["target"] = target_side
            res["episode"] = ep
            all_results.append(res)

            correct = (
                (target_side == "left" and res["outcome"] == "left_success")
                or (target_side == "right" and res["outcome"] == "right_success")
            )
            tag = (
                "CORRECT"
                if correct
                else ("WRONG" if "success" in res["outcome"] else "FAIL")
            )

            print(
                f"  ep={ep:3d} target={target_side} "
                f"outcome={res['outcome']:14s} [{tag}] "
                f"({res['steps']} steps, {res['plans']} plans)"
            )

            # Rename video
            if Path(video_tmp).exists():
                final = (
                    vid_dir
                    / f"{tag.lower()}_ep{ep:03d}_{res['outcome']}"
                      f"_{res['steps']}steps.mp4"
                )
                try:
                    if final.exists():
                        final.unlink()
                    Path(video_tmp).rename(final)
                except OSError:
                    shutil.move(video_tmp, str(final))

    # ── Summary ───────────────────────────────────────────────────
    _print_summary(all_results, targets)
    _save_results(all_results, out_dir)


# ═══════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════

def _print_summary(results: List[Dict], targets: List[str]) -> None:
    print(f"\n{'='*65}")
    print(f"  STEERING RESULTS SUMMARY")
    print(f"{'='*65}")

    for target in targets:
        subset = [r for r in results if r["target"] == target]
        total = len(subset)
        if total == 0:
            continue

        correct = sum(
            1
            for r in subset
            if (target == "left" and r["outcome"] == "left_success")
            or (target == "right" and r["outcome"] == "right_success")
        )
        wrong = sum(
            1
            for r in subset
            if (target == "left" and r["outcome"] == "right_success")
            or (target == "right" and r["outcome"] == "left_success")
        )
        fail = sum(1 for r in subset if r["outcome"] == "failure")
        success = correct + wrong

        print(f"\n  Target = {target.upper()} ({total} episodes)")
        print(f"    Correct block  : {correct:3d} / {total}  "
              f"({correct / total:.1%})")
        print(f"    Wrong block    : {wrong:3d} / {total}  "
              f"({wrong / total:.1%})")
        print(f"    Failure        : {fail:3d} / {total}  "
              f"({fail / total:.1%})")
        if success > 0:
            print(f"    Steering acc   : {correct:3d} / {success} successes  "
                  f"({correct / success:.1%})")

    total = len(results)
    if total == 0:
        return

    correct_all = sum(
        1
        for r in results
        if (r["target"] == "left" and r["outcome"] == "left_success")
        or (r["target"] == "right" and r["outcome"] == "right_success")
    )
    success_all = sum(1 for r in results if "success" in r["outcome"])
    fail_all = total - success_all

    print(f"\n  OVERALL ({total} episodes)")
    print(f"    Success rate   : {success_all:3d} / {total}  "
          f"({success_all / total:.1%})")
    if success_all > 0:
        print(f"    Steering acc   : {correct_all:3d} / {success_all} successes  "
              f"({correct_all / success_all:.1%})")
    print(f"    Failure rate   : {fail_all:3d} / {total}  "
          f"({fail_all / total:.1%})")

    if success_all > 0:
        acc = correct_all / success_all
        print(f"\n  Random baseline  : 50% steering accuracy")
        print(f"  Achieved         : {acc:.1%} steering accuracy")
        if acc >= 0.8:
            print(f"  >>> STRONG LEGIBILITY STEERING <<<")
        elif acc >= 0.65:
            print(f"  >>> MODERATE LEGIBILITY STEERING <<<")
        elif acc > 0.55:
            print(f"  >>> WEAK STEERING EFFECT <<<")
        else:
            print(f"  >>> NO STEERING EFFECT <<<")


def _save_results(results: List[Dict], out_dir: Path) -> None:
    # Filter out numpy arrays for CSV serialization
    csv_keys = [k for k in results[0].keys()
                if not isinstance(results[0][k], np.ndarray)]
    csv_path = out_dir / "steered_results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in csv_keys if k in r})
        print(f"\n  Saved {csv_path}")

    metrics: Dict = {"total": len(results), "per_target": {}}
    for target in set(r["target"] for r in results):
        subset = [r for r in results if r["target"] == target]
        t = len(subset)
        correct = sum(
            1
            for r in subset
            if (target == "left" and r["outcome"] == "left_success")
            or (target == "right" and r["outcome"] == "right_success")
        )
        success = sum(1 for r in subset if "success" in r["outcome"])
        metrics["per_target"][target] = {
            "total": t,
            "correct": correct,
            "success": success,
            "failure": t - success,
            "steering_accuracy": round(correct / success, 4) if success else None,
        }

    total_correct = sum(m["correct"] for m in metrics["per_target"].values())
    total_success = sum(m["success"] for m in metrics["per_target"].values())
    metrics["overall_steering_accuracy"] = (
        round(total_correct / total_success, 4) if total_success else None
    )
    metrics["overall_success_rate"] = (
        round(total_success / len(results), 4) if results else None
    )

    mp = out_dir / "steered_metrics.json"
    mp.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved {mp}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Steered diffusion policy — inference-time legibility"
    )
    ap.add_argument(
        "--ckpt",
        default="runs/diffusion_20260222_195530/ckpt_ep100.pt",
    )
    ap.add_argument("--target", default="both",
                    choices=["left", "right", "both"])
    ap.add_argument("--n_candidates", type=int, default=16)
    ap.add_argument("--n_episodes", type=int, default=25,
                    help="Episodes per target direction")
    ap.add_argument("--n_videos", type=int, default=25,
                    help="Max videos to record per target")
    ap.add_argument("--n_sampling_steps", type=int, default=10)
    ap.add_argument("--run_baseline", action="store_true", default=True)
    ap.add_argument("--no_baseline", dest="run_baseline",
                    action="store_false")
    ap.add_argument("--n_baseline", type=int, default=20,
                    help="Unsteered baseline episodes")
    ap.add_argument("--proxy_weights", type=str, default=None,
                    help="Path to fitted proxy weights JSON "
                         "(from generate_benchmark.py --step fit)")
    ap.add_argument("--out_dir", default="outputs/steered")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
