#!/usr/bin/env python
"""Evaluate steerable diffusion policy: does changing the steering vector
actually change the trajectory direction?

Tests:
1. Same obs + left steering -> picks left block?
2. Same obs + right steering -> picks right block?
3. Same obs + varied curvature -> different arc shapes?
4. Classifier-free guidance scale sweep

Produces: success rates, bimodality metrics, per-steering-value outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.train_steered_diffusion import SteerableNoiseNet, SteeredDDPMSchedule


class SteeredPolicyRunner:
    """Wrapper for steerable diffusion policy inference."""
    
    def __init__(self, ckpt_path: str, device: torch.device,
                 sampling_method: str = 'ddim',
                 ddim_eta: float = 0.3,
                 ddim_steps: int | None = None,
                 cfg_scale: float = 0.0) -> None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        
        assert ckpt.get("model_type") == "steerable", \
            f"Expected steerable checkpoint, got {ckpt.get('model_type')}"
        
        self.device = device
        self.horizon = cfg["horizon"]
        self.n_action_steps = cfg["n_action_steps"]
        self.act_dim = cfg["act_dim"]
        self.sampling_method = sampling_method
        self.ddim_eta = ddim_eta
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        
        steer_dim = ckpt.get("steer_dim", 3)
        steer_embed_dim = ckpt.get("steer_embed_dim", 64)
        
        self.model = SteerableNoiseNet(
            obs_dim=cfg["obs_dim"],
            act_dim=cfg["act_dim"],
            horizon=cfg["horizon"],
            steer_dim=steer_dim,
            steer_embed_dim=steer_embed_dim,
            hidden_dim=cfg["hidden_dim"],
            n_blocks=cfg["n_blocks"],
            time_embed_dim=cfg["time_embed_dim"],
            steer_dropout=0.0,  # No dropout at inference
        ).to(device)
        
        # Load EMA weights
        if "ema" in ckpt:
            ema_sd = ckpt["ema"]
            model_sd = ckpt["model"]
            merged = {}
            for k, v in model_sd.items():
                merged[k] = ema_sd.get(k, v)
            self.model.load_state_dict(merged)
            print("  Loaded EMA weights")
        else:
            self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.schedule = SteeredDDPMSchedule(
            cfg["n_diffusion_steps"],
            cfg["beta_start"], cfg["beta_end"], device)
        
        self.obs_mean = torch.tensor(ckpt["obs_mean"], dtype=torch.float32, device=device)
        self.obs_std = torch.tensor(ckpt["obs_std"], dtype=torch.float32, device=device)
        self.act_mean = torch.tensor(ckpt["act_mean"], dtype=torch.float32, device=device)
        self.act_std = torch.tensor(ckpt["act_std"], dtype=torch.float32, device=device)
        
        self._action_queue: list[np.ndarray] = []
        self._current_steer: torch.Tensor | None = None
        
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded steered policy (epoch {epoch}, cfg_scale={cfg_scale})")
    
    def reset(self, steer_vector: np.ndarray | None = None):
        """Reset policy with a new steering vector."""
        self._action_queue = []
        if steer_vector is not None:
            self._current_steer = torch.tensor(
                steer_vector, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            self._current_steer = None
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        if len(self._action_queue) == 0:
            self._plan(obs)
        return self._action_queue.pop(0)
    
    @torch.no_grad()
    def _plan(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = (obs_t - self.obs_mean) / self.obs_std
        obs_t = obs_t.unsqueeze(0)
        
        steer = self._current_steer
        if steer is None:
            steer = torch.zeros(1, 3, device=self.device)
        
        chunk = self.schedule.sample_steered(
            self.model, obs_t, steer,
            self.horizon, self.act_dim,
            method=self.sampling_method,
            eta=self.ddim_eta,
            ddim_steps=self.ddim_steps,
            cfg_scale=self.cfg_scale)
        
        chunk = chunk * self.act_std.unsqueeze(0) + self.act_mean.unsqueeze(0)
        chunk = chunk.squeeze(0).cpu().numpy()
        chunk = np.clip(chunk, -1, 1)
        
        n_steps = min(self.n_action_steps, self.horizon)
        self._action_queue = [chunk[i] for i in range(n_steps)]


def rollout(env: TwoBlockPickEnv, policy: SteeredPolicyRunner,
            steer_vector: np.ndarray,
            env_seed: int, sample_seed: int,
            max_steps: int = 400,
            execute_steps: int = 8,
            video_path: str | None = None) -> Dict:
    """Run a single rollout with given steering."""
    torch.manual_seed(sample_seed)
    np.random.seed(sample_seed)
    
    policy.reset(steer_vector)
    policy.n_action_steps = execute_steps
    obs = env.reset(seed=env_seed)
    
    writer = None
    if video_path:
        import cv2
        frames = []
    
    outcome = "fail"
    for step in range(max_steps):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        
        if video_path:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if done:
            if info.get("left_success"):
                outcome = "left_success"
            elif info.get("right_success"):
                outcome = "right_success"
            break
    
    if video_path and frames:
        import cv2
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
    
    ee_pos = obs[:3]
    return {
        "outcome": outcome,
        "env_seed": env_seed,
        "sample_seed": sample_seed,
        "steer_vector": steer_vector.tolist(),
        "steps": step + 1,
        "final_ee": ee_pos.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate steered diffusion policy")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--K", type=int, default=5, help="Number of env seeds")
    ap.add_argument("--M", type=int, default=10, help="Samples per seed per steering")
    ap.add_argument("--output", type=str, default="outputs/steering_eval")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--execute_steps", type=int, default=8)
    ap.add_argument("--sampling_method", type=str, default="ddim")
    ap.add_argument("--ddim_eta", type=float, default=0.3)
    ap.add_argument("--cfg_scale", type=float, default=1.5,
                    help="Classifier-free guidance scale (0=no guidance)")
    ap.add_argument("--videos", type=int, default=5, help="Number of videos to save")
    args = ap.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    policy = SteeredPolicyRunner(
        args.ckpt, device,
        sampling_method=args.sampling_method,
        ddim_eta=args.ddim_eta,
        cfg_scale=args.cfg_scale)
    
    env = TwoBlockPickEnv(render=False, episode_length=args.max_steps)
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define steering conditions to test
    steering_conditions = {
        "steer_left": np.array([1.0, 0.5, 0.0], dtype=np.float32),
        "steer_right": np.array([-1.0, 0.5, 1.0], dtype=np.float32),
        "steer_left_strong_curve": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        "steer_right_strong_curve": np.array([-1.0, 1.0, 1.0], dtype=np.float32),
        "steer_left_weak_curve": np.array([1.0, 0.2, 0.0], dtype=np.float32),
        "steer_right_weak_curve": np.array([-1.0, 0.2, 1.0], dtype=np.float32),
        "no_steering": np.array([0.0, 0.5, 0.5], dtype=np.float32),
    }
    
    env_seeds = list(range(100, 100 + args.K))
    all_results = {}
    video_count = 0
    
    for cond_name, steer_vec in steering_conditions.items():
        print(f"\n{'='*60}")
        print(f"Testing: {cond_name}  steer={steer_vec.tolist()}")
        print(f"{'='*60}")
        
        results = []
        for es in env_seeds:
            for mi in range(args.M):
                ss = 5000 + mi * 137
                
                vp = None
                if video_count < args.videos:
                    vp = str(out_dir / "videos" / f"{cond_name}_env{es}_s{ss}.mp4")
                    video_count += 1
                
                res = rollout(env, policy, steer_vec,
                              es, ss, args.max_steps, args.execute_steps, vp)
                results.append(res)
                
                if vp and Path(vp).exists():
                    final = str(Path(vp).with_stem(f"{Path(vp).stem}_{res['outcome']}"))
                    try:
                        Path(vp).rename(final)
                    except:
                        pass
        
        n_left = sum(1 for r in results if r["outcome"] == "left_success")
        n_right = sum(1 for r in results if r["outcome"] == "right_success")
        n_fail = sum(1 for r in results if r["outcome"] == "fail")
        total = len(results)
        
        success_rate = (n_left + n_right) / total
        p_left = n_left / (n_left + n_right) if (n_left + n_right) > 0 else 0.5
        
        print(f"  Success: {n_left + n_right}/{total} = {success_rate:.1%}")
        print(f"  Left: {n_left}  Right: {n_right}  Fail: {n_fail}")
        print(f"  P(left|success): {p_left:.2f}")
        
        all_results[cond_name] = {
            "steer_vector": steer_vec.tolist(),
            "n_left": n_left,
            "n_right": n_right,
            "n_fail": n_fail,
            "success_rate": round(success_rate, 4),
            "p_left": round(p_left, 4),
            "results": results,
        }
    
    env.close()
    
    # Summary
    print(f"\n{'='*70}")
    print("STEERING EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<30} {'Success':>8} {'P(left)':>8} {'Left':>6} {'Right':>6} {'Fail':>6}")
    print("-" * 70)
    for cond, data in all_results.items():
        print(f"{cond:<30} {data['success_rate']:>7.1%} {data['p_left']:>8.2f} "
              f"{data['n_left']:>6} {data['n_right']:>6} {data['n_fail']:>6}")
    
    # Key metrics
    left_sr = all_results.get("steer_left", {}).get("success_rate", 0)
    right_sr = all_results.get("steer_right", {}).get("success_rate", 0)
    left_pleft = all_results.get("steer_left", {}).get("p_left", 0)
    right_pleft = all_results.get("steer_right", {}).get("p_left", 1)
    
    steerability = abs(left_pleft - right_pleft)
    print(f"\nSteerability score: {steerability:.2f}")
    print(f"  (|P(left|steer_left) - P(left|steer_right)| = |{left_pleft:.2f} - {right_pleft:.2f}|)")
    print(f"  1.0 = perfect steering, 0.0 = no effect")
    
    # Save results
    summary = {
        "steerability": round(steerability, 4),
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "results"}
                       for k, v in all_results.items()},
        "config": {
            "K": args.K,
            "M": args.M,
            "cfg_scale": args.cfg_scale,
            "sampling_method": args.sampling_method,
            "ddim_eta": args.ddim_eta,
        },
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Full results
    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
