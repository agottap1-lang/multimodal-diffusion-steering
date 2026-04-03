#!/usr/bin/env python
"""Evaluate a BC (MLP) baseline policy on TwoBlockPick.

Same K × M evaluation protocol as eval_multimodality.py but for BC.
Since BC is deterministic, different sample_seeds have NO effect —
the policy always outputs the same action for the same observation.
This demonstrates that BC cannot express multimodality.

Usage:
    python scripts/eval_bc.py --ckpt runs/bc_latest/bc_ckpt.pt --K 10 --M 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM
from scripts.train_bc import MLPPolicy


# ── BC Policy Runner ─────────────────────────────────────────────────

class BCPolicyRunner:
    def __init__(self, ckpt_path: str, device: torch.device) -> None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.device = device

        self.model = MLPPolicy(
            obs_dim=cfg["obs_dim"],
            act_dim=cfg["act_dim"],
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=4,
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.obs_mean = torch.tensor(ckpt["obs_mean"], dtype=torch.float32,
                                     device=device)
        self.obs_std = torch.tensor(ckpt["obs_std"], dtype=torch.float32,
                                    device=device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = (obs_t - self.obs_mean) / self.obs_std
        obs_t = obs_t.unsqueeze(0)
        action = self.model(obs_t).squeeze(0).cpu().numpy()
        return np.clip(action, -1, 1)


# ── rollout ──────────────────────────────────────────────────────────

def rollout(
    env: TwoBlockPickEnv,
    policy: BCPolicyRunner,
    env_seed: int,
    sample_seed: int,  # ignored — BC is deterministic
    video_path: str | None = None,
) -> Dict:
    obs = env.reset(seed=env_seed)

    if video_path:
        env.record_video(video_path)

    total_reward = 0.0
    for t in range(env.episode_length):
        action = policy.act(obs)
        result = env.step(action)
        obs = result.obs
        total_reward += result.reward
        if result.done:
            break

    if video_path:
        env.stop_video()

    info = result.info
    if info["picked_left"]:
        outcome = "left_success"
    elif info["picked_right"]:
        outcome = "right_success"
    else:
        outcome = "failure"

    return {
        "env_seed": env_seed,
        "sample_seed": sample_seed,
        "outcome": outcome,
        "steps": t + 1,
        "reward": total_reward,
    }


# ── evaluate ─────────────────────────────────────────────────────────

def evaluate(
    ckpt_path: str,
    K: int = 10,
    M: int = 5,
    n_videos: int = 5,
    out_dir: str = "outputs/bc",
    video_dir: str = "outputs/bc/videos",
    env_seed_start: int = 100,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"NOTE: BC is deterministic — sample_seed has no effect.")
    print(f"Running {K} env seeds × {M} sample seeds = {K * M} rollouts ...\n")

    policy = BCPolicyRunner(ckpt_path, device)
    env = TwoBlockPickEnv(render=False)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    video_count = 0
    env_seeds = list(range(env_seed_start, env_seed_start + K))

    for es in env_seeds:
        for mi in range(M):
            ss = 5000 + mi * 137
            vp = None
            if video_count < n_videos:
                vp = str(Path(video_dir) / f"bc_env{es}_s{ss}.mp4")
                video_count += 1

            res = rollout(env, policy, es, ss, video_path=vp)
            results.append(res)

            tag = res["outcome"]
            print(f"  env_seed={es}  sample={ss:5d}  => {tag:14s}  ({res['steps']} steps)")

    env.close()

    # ── aggregate ────────────────────────────────────────────────────
    n_left = sum(1 for r in results if r["outcome"] == "left_success")
    n_right = sum(1 for r in results if r["outcome"] == "right_success")
    n_fail = sum(1 for r in results if r["outcome"] == "failure")
    total = len(results)
    n_success = n_left + n_right
    success_rate = n_success / total if total > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  BC Overall ({total} rollouts)")
    print(f"{'='*55}")
    print(f"  left_success :  {n_left:3d}  ({n_left/total:.1%})")
    print(f"  right_success:  {n_right:3d}  ({n_right/total:.1%})")
    print(f"  failure      :  {n_fail:3d}  ({n_fail/total:.1%})")
    print(f"  success_rate :  {success_rate:.1%}")

    # Per-seed: check whether BC always picks the same side
    per_seed: Dict[int, Dict[str, int]] = {}
    for r in results:
        es = r["env_seed"]
        per_seed.setdefault(es, {"left_success": 0, "right_success": 0, "failure": 0})
        per_seed[es][r["outcome"]] += 1

    print(f"\n  Per env_seed:")
    all_same = True
    for es in sorted(per_seed):
        c = per_seed[es]
        ls, rs, fl = c["left_success"], c["right_success"], c["failure"]
        modes = (ls > 0) + (rs > 0)
        if modes > 1:
            all_same = False
        print(f"    {es}: L={ls} R={rs} F={fl}")

    if all_same:
        print(f"\n  BC always picks the SAME side per env_seed (no multimodality)")
        print(f"  This confirms that multimodality requires stochastic inference.")
    else:
        print(f"\n  WARNING: BC shows variation — check for non-determinism.")

    # ── save bc_metrics.json ─────────────────────────────────────────
    metrics = {
        "total_rollouts": total,
        "left_success": n_left,
        "right_success": n_right,
        "failure": n_fail,
        "success_rate": round(success_rate, 4),
        "p_left_given_success": round(n_left / n_success, 4) if n_success > 0 else None,
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "deterministic": all_same,
    }
    metrics_path = Path(out_dir) / "bc_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n  saved {metrics_path}")

    # ── save results.csv ─────────────────────────────────────────────
    csv_path = Path(out_dir) / "bc_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "env_seed", "sample_seed", "outcome", "steps", "reward"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  saved {csv_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bc_latest/bc_ckpt.pt")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--n_videos", type=int, default=5)
    ap.add_argument("--out_dir", default="outputs/bc")
    ap.add_argument("--video_dir", default="outputs/bc/videos")
    ap.add_argument("--env_seed_start", type=int, default=100)
    args = ap.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        K=args.K,
        M=args.M,
        n_videos=args.n_videos,
        out_dir=args.out_dir,
        video_dir=args.video_dir,
        env_seed_start=args.env_seed_start,
    )


if __name__ == "__main__":
    main()
