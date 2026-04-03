#!/usr/bin/env python
"""Evaluate multimodality of a trained diffusion policy for TwoBlockPick.

For each of K environment seeds, run M rollouts with different diffusion
sampling seeds.  Records which cube was picked and produces statistics,
plots, videos, and montages.

═══════════════════════════════════════════════════════════════════════
DIAGNOSIS (2026-02-14):
═══════════════════════════════════════════════════════════════════════
OBSERVED: 0% success rate across all configurations (DDIM eta=0/0.3, 
          jitter=0/0.015, dynamic_mpc=True/False)

HYPOTHESES (ranked by likelihood):

#1 [ACTION SCALING] - Model outputs tiny actions → robot barely moves
   Evidence: diagnostic shows ee moves 1cm in 10 steps (should be ~40cm)
   Test: Compare action.std() to demo action.std() (expect ~0.3-0.5)
   Fix: Add action_scale multiplier or investigate normalization mismatch
   Command: py scripts/diagnose_rollout.py --verbose

#2 [DDIM BUG] - DDIM implementation collapses actions to near-zero
   Evidence: All DDIM tests fail; DDPM might work
   Test: Run with --sampling_method ddpm (bypass DDIM math entirely)
   Fix: Debug p_sample_ddim() or revert to DDPM
   Command: py scripts/eval_multimodality.py --sampling_method ddpm --K 2 --M 2

#3 [DYNAMIC MPC STALE QUEUE] - execute_steps changes but old actions still used
   Evidence: Dynamic MPC implemented but queue not cleared on n_action_steps change
   Test: Compare dynamic_mpc=True vs execute_steps=1 (pure replanning)
   Fix: Clear action queue when execute_steps changes
   Command: py scripts/eval_multimodality.py --execute_steps 1 --K 2 --M 2

IMPLEMENTED FIXES:
- [✓] Checkpoint epoch verification (fail fast on mismatch)
- [✓] Action magnitude diagnostics (print min/max/std per rollout)
- [✓] Dynamic MPC queue clearing (force immediate replan)
- [✓] DDPM fallback test command
═══════════════════════════════════════════════════════════════════════

Usage:
    python scripts/eval_multimodality.py \
        --ckpt runs/latest/ckpt.pt --K 10 --M 20 --execute_steps 8

──────────────────────────────────────────────────────────
Observation dim : 22
    ee_pos(3) + ee_quat(4) + grip(1) +
    left_cube_pos(3) + left_cube_quat(4) +
    right_cube_pos(3) + right_cube_quat(4)
Action dim      : 5   (dx, dy, dz, dyaw, grip)  each in [-1, 1]
Action scaling  : pos *= 0.05 m/step, yaw *= 15 deg/step
Cubes           : x=0.50, y=+/-0.07, jitter +/-0.015 (seeded per env_seed)
Success cond    : cube_z > 0.52
Episode length  : 400 steps max (demos ~303 steps, legible Bézier arcs)
Diffusion       : 100-step DDPM, beta 0.0001->0.1
  Noise net     : 6 ResBlock MLP, hidden=256, FiLM time
  Action chunk  : horizon=16, n_action_steps=8 (default; overridden by execute_steps)
  Act norm      : identity (mean=0,std=1)
  Obs norm      : per-dim mean/std, std floored at 0.01

SEED SEPARATION (critical for multimodality proof):
  env_seed      : controls cube placement jitter (via env._rng)
                  -> identical initial state for same env_seed
  sample_seed   : controls diffusion sampling noise (torch.manual_seed)
                  -> set ONCE per rollout at the start; NOT reseeded per plan
──────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv, OBS_DIM, ACT_DIM
from scripts.train_diffusion_policy import DDPMSchedule, NoiseNet


# ── policy wrapper ───────────────────────────────────────────────────

class DiffusionPolicyRunner:
    """Wraps a trained NoiseNet for receding-horizon (MPC) inference.

    Supports temporal ensembling: when enabled, overlapping action chunks
    are averaged with exponential weighting (newer plans get more weight).
    This smooths transitions between plans without blending modes.
    """

    def __init__(self, ckpt_path: str, device: torch.device,
                 verbose: bool = False,
                 temporal_ensemble: bool = True,
                 ensemble_decay: float = 0.7,
                 ensemble_grip: bool = False,
                 sampling_method: str = 'ddim',
                 ddim_eta: float = 0.0,
                 ddim_steps: int | None = None,
                 multimodal_selection: bool = False,
                 n_candidates: int = 5) -> None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        
        # FIX #1: Checkpoint verification - fail fast if suspicious
        epoch = ckpt.get("epoch", "unknown")
        print(f"  [CKPT] Loaded epoch {epoch}, horizon={cfg['horizon']}, sampling={sampling_method}, eta={ddim_eta}")
        if epoch == "unknown" or (isinstance(epoch, int) and epoch < 100):
            print(f"  [WARNING] Checkpoint epoch={epoch} may be undertrained!")
        
        # Multimodal selection mode
        self.multimodal_selection = multimodal_selection
        self.n_candidates = n_candidates
        if multimodal_selection:
            print(f"  [MULTIMODAL SELECTION] Enabled with {n_candidates} candidates per replan")
            print(f"  [NOTE] Using placeholder value function (random selection for now)")
            print(f"         To plug in VLM value function, modify _value_function() method")
        
        self.device = device
        self.horizon = cfg["horizon"]
        self.n_action_steps = cfg["n_action_steps"]
        self.act_dim = cfg["act_dim"]
        self.verbose = verbose
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_decay = ensemble_decay
        self.ensemble_grip = ensemble_grip  # Whether to ensemble gripper (causes smearing if True)
        self.sampling_method = sampling_method  # 'ddpm' or 'ddim'
        self.ddim_eta = ddim_eta
        self.ddim_steps = ddim_steps  # None = use all n_diffusion_steps

        self.model = NoiseNet(
            obs_dim=cfg["obs_dim"],
            act_dim=cfg["act_dim"],
            horizon=cfg["horizon"],
            hidden_dim=cfg["hidden_dim"],
            n_blocks=cfg["n_blocks"],
            time_embed_dim=cfg["time_embed_dim"],
        ).to(device)

        # Load EMA weights if available, otherwise fall back to raw model
        if "ema" in ckpt:
            ema_sd = ckpt["ema"]
            model_sd = ckpt["model"]
            # EMA keys are param names; map them into state_dict
            merged = {}
            for k, v in model_sd.items():
                merged[k] = ema_sd.get(k, v)
            self.model.load_state_dict(merged)
            print("  loaded EMA weights for inference")
        else:
            self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.schedule = DDPMSchedule(
            cfg["n_diffusion_steps"],
            cfg["beta_start"], cfg["beta_end"], device)

        self.obs_mean = torch.tensor(ckpt["obs_mean"], dtype=torch.float32,
                                     device=device)
        self.obs_std = torch.tensor(ckpt["obs_std"], dtype=torch.float32,
                                    device=device)
        self.act_mean = torch.tensor(ckpt["act_mean"], dtype=torch.float32,
                                     device=device)
        self.act_std = torch.tensor(ckpt["act_std"], dtype=torch.float32,
                                    device=device)

        self._action_queue: List[np.ndarray] = []
        self._plan_count: int = 0
        # Temporal ensembling state
        self._pending_chunks: List[np.ndarray] = []  # future planned chunks
        self._chunk_offsets: List[int] = []  # how many steps consumed from each
        
        # FIX #2: Track action statistics for diagnosis
        self._action_stats: List[float] = []  # std of each planned chunk
        
        # FIX #3: Load demo action statistics for comparison
        self._demo_std: float | None = None
        self._demo_abs_mean: float | None = None
        demo_path = cfg.get("demo_path", "data/demos/demos.npz")
        try:
            if Path(demo_path).exists():
                demo_data = np.load(demo_path, allow_pickle=True)
                demo_actions = demo_data["actions"]  # (N, T, act_dim)
                # Compute statistics over position deltas (dx, dy, dz, dyaw) across all demos
                all_pos_actions = demo_actions[:, :, :4].reshape(-1, 4)
                self._demo_std = float(all_pos_actions.std())
                self._demo_abs_mean = float(np.abs(all_pos_actions).mean())
                print(f"  [DEMO STATS] action std={self._demo_std:.4f}, abs_mean={self._demo_abs_mean:.4f}")
        except Exception as e:
            print(f"  [WARNING] Could not load demo stats: {e}")

    def reset(self) -> None:
        self._action_queue = []
        self._plan_count = 0
        self._pending_chunks = []
        self._chunk_offsets = []
        self._action_stats = []
    
    def set_execute_steps(self, new_steps: int) -> None:
        """FIX #3: Update n_action_steps and force immediate replanning by clearing queue."""
        if new_steps != self.n_action_steps:
            self.n_action_steps = new_steps
            self._action_queue = []  # Clear queue to force replan with new execute_steps

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return the next action.  Plans a new chunk when queue is empty."""
        if len(self._action_queue) == 0:
            if self.multimodal_selection:
                self._plan_with_selection(obs)
            else:
                self._plan(obs)
        return self._action_queue.pop(0)

    @torch.no_grad()
    def _plan(self, obs: np.ndarray) -> None:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = (obs_t - self.obs_mean) / self.obs_std
        obs_t = obs_t.unsqueeze(0)

        chunk = self.schedule.sample(self.model, obs_t,
                                      self.horizon, self.act_dim,
                                      method=self.sampling_method,
                                      eta=self.ddim_eta,
                                      ddim_steps=self.ddim_steps)
        chunk = chunk * self.act_std.unsqueeze(0) + self.act_mean.unsqueeze(0)
        chunk = chunk.squeeze(0).cpu().numpy()
        chunk = np.clip(chunk, -1, 1)
        
        # FIX A: Action magnitude diagnostics - MOTION DIMS ONLY (avoid gripper contamination)
        pos_chunk = chunk[:, :3]  # dx, dy, dz only (no yaw, no gripper)
        pos_std = pos_chunk.std()
        pos_abs_mean = np.abs(pos_chunk).mean()
        pos_min = pos_chunk.min()
        pos_max = pos_chunk.max()
        
        # Yaw separately (if present in non-gripper dims)
        yaw_std = chunk[:, 3].std() if chunk.shape[1] > 3 else 0.0
        
        self._action_stats.append(pos_std)
        
        # Print diagnostics for first 3 plans or if action collapse detected
        should_print = (self._plan_count < 3) or (pos_std < 0.1)
        
        if should_print:
            ratio_str = ""
            if self._demo_std is not None and self._demo_std > 0:
                ratio = pos_std / self._demo_std
                ratio_str = f" (demo_std={self._demo_std:.4f}, ratio={ratio:.2f})"
            
            print(f"  [PLAN #{self._plan_count + 1}] pos_std={pos_std:.4f}, pos_abs_mean={pos_abs_mean:.4f}, yaw_std={yaw_std:.4f}{ratio_str}")
            print(f"            pos_range=[{pos_min:+.3f}, {pos_max:+.3f}]")
            
            # Demo-relative warnings (if demo_std available)
            if self._demo_std is not None and self._demo_std > 0:
                # CRITICAL: policy outputs < 30% of demo std
                if pos_std < 0.3 * self._demo_std:
                    print(f"  [CRITICAL] Severe action suppression! Policy std is {pos_std/self._demo_std:.1%} of demo std.")
                    print(f"            Expected: {self._demo_std:.4f}, Got: {pos_std:.4f}")
                    print(f"            → Model may have collapsed or normalization is broken")
                    print(f"            → Run scripts/bc_sanity_check.py to diagnose")
                # WARNING: policy outputs < 70% of demo std
                elif pos_std < 0.7 * self._demo_std:
                    print(f"  [WARNING] Action suppression. Policy std is {pos_std/self._demo_std:.1%} of demo std.")
                    print(f"            Expected: {self._demo_std:.4f}, Got: {pos_std:.4f}")
                    print(f"            → May indicate temporal ensembling over-smoothing or model undertrained")
                # WARNING: policy outputs > 200% of demo std
                elif pos_std > 2.0 * self._demo_std:
                    print(f"  [WARNING] Action amplification! Policy std is {pos_std/self._demo_std:.1%} of demo std.")
                    print(f"            Expected: {self._demo_std:.4f}, Got: {pos_std:.4f}")
                    print(f"            → Possible thrashing, overshoot, or out-of-distribution observations")
            # Fallback warnings if demo_std unavailable (legacy behavior, but demo-agnostic)
            else:
                if pos_std < 0.005:
                    print(f"  [CRITICAL] Policy outputs near-zero (std < 0.005).")
                    print(f"            → Model may have collapsed")
                elif pos_std < 0.01:
                    print(f"  [WARNING] Policy outputs very small (std < 0.01).")

        self._plan_count += 1
        if self.verbose and self._plan_count <= 3:
            mn = chunk.mean(axis=0)
            sd = chunk.std(axis=0)
            lo = chunk.min(axis=0)
            hi = chunk.max(axis=0)
            print(f"    [plan #{self._plan_count}] chunk stats (H={chunk.shape[0]}):")
            for i, name in enumerate(["dx", "dy", "dz", "dyaw", "grip"]):
                print(f"      {name}: mean={mn[i]:+.4f} std={sd[i]:.4f} "
                      f"range=[{lo[i]:+.4f}, {hi[i]:+.4f}]")

        if self.temporal_ensemble and self._pending_chunks:
            # Average overlapping actions with exponential decay
            # New chunk gets weight 1.0, older pending chunks get decay^age
            blended = chunk.copy()
            for ci, (old_chunk, offset) in enumerate(
                    zip(self._pending_chunks, self._chunk_offsets)):
                remaining = old_chunk[offset:]
                overlap = min(len(remaining), self.horizon)
                if overlap <= 0:
                    continue
                age = len(self._pending_chunks) - ci  # older = higher age
                w_old = self.ensemble_decay ** age
                for j in range(overlap):
                    # Blend continuous dims (dx, dy, dz, dyaw)
                    blended[j, :4] = (w_old * remaining[j, :4] + blended[j, :4]) / (1.0 + w_old)
                    # Gripper: use most recent (no ensemble) or ensemble based on flag
                    if self.ensemble_grip:
                        blended[j, 4] = (w_old * remaining[j, 4] + blended[j, 4]) / (1.0 + w_old)
                    # else: keep blended[j, 4] as is (most recent value from new chunk)
            chunk = blended

        # Store for future ensembling
        self._pending_chunks.append(chunk)
        self._chunk_offsets.append(0)

        # Build action queue from first n_action_steps
        actions = []
        for i in range(self.n_action_steps):
            actions.append(chunk[i])
        self._action_queue = actions

        # Advance offsets and prune expired chunks
        new_chunks, new_offsets = [], []
        for c, o in zip(self._pending_chunks, self._chunk_offsets):
            new_o = o + self.n_action_steps
            if new_o < len(c):
                new_chunks.append(c)
                new_offsets.append(new_o)
        self._pending_chunks = new_chunks
        self._chunk_offsets = new_offsets
    
    def _value_function(self, obs: np.ndarray, chunk: np.ndarray) -> float:
        """Placeholder value function for scoring action chunks.
        
        This is a stub that returns a random score. Replace with:
          - VLM-based scoring (e.g., CLIP similarity to goal description)
          - Learned value function (trained Q-network or reward predictor)
          - Heuristic scoring (e.g., distance to target, collision avoidance)
        
        Args:
            obs: Current observation (22-dim for TwoBlockPick)
            chunk: Action chunk to score (horizon × act_dim)
        
        Returns:
            score: Higher is better (arbitrary scale)
        """
        # PLACEHOLDER: Random scoring for now
        # In practice, you would:
        #   1. Simulate chunk forward (or use learned dynamics model)
        #   2. Score final state with VLM/reward model
        #   3. Return expected value
        
        # Simple heuristic: prefer chunks with moderate action magnitudes
        # (too small = action collapse, too large = instability)
        chunk_std = chunk[:, :4].std()
        target_std = 0.35  # Target action std (from demos)
        score = -abs(chunk_std - target_std)  # Penalize deviation from target
        
        return float(score)
    
    def _plan_with_selection(self, obs: np.ndarray) -> None:
        """Plan with multimodal selection: sample N candidates, pick best.
        
        This enables mode-mixing: the policy can sample diverse strategies
        (e.g., left vs right approach) and select the best according to a
        value function (e.g., VLM scoring).
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = (obs_t - self.obs_mean) / self.obs_std
        obs_t = obs_t.unsqueeze(0)
        
        candidates = []
        scores = []
        
        # Sample N candidate chunks with different diffusion seeds
        base_seed = int(torch.randint(0, 1000000, (1,)).item())
        for i in range(self.n_candidates):
            # Set unique seed for this candidate
            torch.manual_seed(base_seed + i * 137)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(base_seed + i * 137)
            
            # Sample chunk
            chunk = self.schedule.sample(self.model, obs_t,
                                        self.horizon, self.act_dim,
                                        method=self.sampling_method,
                                        eta=self.ddim_eta,
                                        ddim_steps=self.ddim_steps)
            chunk = chunk * self.act_std.unsqueeze(0) + self.act_mean.unsqueeze(0)
            chunk = chunk.squeeze(0).cpu().numpy()
            chunk = np.clip(chunk, -1, 1)
            
            # Score chunk
            score = self._value_function(obs, chunk)
            
            candidates.append(chunk)
            scores.append(score)
        
        # Pick best candidate
        best_idx = int(np.argmax(scores))
        chunk = candidates[best_idx]
        
        if self.verbose or self._plan_count < 3:
            print(f"  [MULTIMODAL SELECT] plan #{self._plan_count + 1}: "
                  f"sampled {self.n_candidates} candidates, "
                  f"picked #{best_idx} (score={scores[best_idx]:.4f})")
            print(f"    scores: {[f'{s:.3f}' for s in scores]}")
        
        # Diagnostics (same as _plan)
        chunk_std = chunk[:, :4].std()
        chunk_abs_mean = np.abs(chunk[:, :4]).mean()
        self._action_stats.append(chunk_std)
        
        if chunk_std < 0.1:
            print(f"  [WARNING] Selected chunk has low std={chunk_std:.4f}")
        
        self._plan_count += 1
        
        # Apply temporal ensembling if enabled
        if self.temporal_ensemble and self._pending_chunks:
            blended = chunk.copy()
            for ci, (old_chunk, offset) in enumerate(
                    zip(self._pending_chunks, self._chunk_offsets)):
                remaining = old_chunk[offset:]
                overlap = min(len(remaining), self.horizon)
                if overlap <= 0:
                    continue
                age = len(self._pending_chunks) - ci
                w_old = self.ensemble_decay ** age
                for j in range(overlap):
                    blended[j, :4] = (w_old * remaining[j, :4] + blended[j, :4]) / (1.0 + w_old)
                    if self.ensemble_grip:
                        blended[j, 4] = (w_old * remaining[j, 4] + blended[j, 4]) / (1.0 + w_old)
            chunk = blended
        
        # Store and build action queue (same as _plan)
        self._pending_chunks.append(chunk)
        self._chunk_offsets.append(0)
        
        actions = []
        for i in range(self.n_action_steps):
            actions.append(chunk[i])
        self._action_queue = actions
        
        new_chunks, new_offsets = [], []
        for c, o in zip(self._pending_chunks, self._chunk_offsets):
            new_o = o + self.n_action_steps
            if new_o < len(c):
                new_chunks.append(c)
                new_offsets.append(new_o)
        self._pending_chunks = new_chunks
        self._chunk_offsets = new_offsets


# ── rollout ──────────────────────────────────────────────────────────

def rollout(
    env: TwoBlockPickEnv,
    policy: DiffusionPolicyRunner,
    env_seed: int,
    sample_seed: int,
    execute_steps: int = 8,
    dynamic_mpc: bool = False,
    mpc_far_threshold: float = 0.15,
    mpc_near_threshold: float = 0.05,
    video_path: str | None = None,
    commit_steps: int = 0,
    commit_thresh: float = 0.2,
    commit_mag: float = 0.8,
    max_steps: int = 400,
    log_chunks: bool = False,
    log_ee_displacement: bool = False,
) -> Dict:
    """Run one episode.

    Seeding contract:
      - env_seed  : passed to env.reset(seed=...) to fix cube placement.
      - sample_seed : torch.manual_seed(sample_seed) called ONCE here.
                      All subsequent DDPM sampling draws from this stream.
    """
    obs = env.reset(seed=env_seed)
    policy.reset()
    
    # Dynamic MPC: Initial execute_steps, will be updated per step if enabled
    policy.n_action_steps = execute_steps

    # ── Set diffusion RNG ONCE for the entire rollout ──
    torch.manual_seed(sample_seed)

    if video_path:
        env.record_video(video_path)

    # Commitment nudge RNG (deterministic per sample_seed)
    commit_rng = np.random.default_rng(sample_seed)
    committed_sign: float | None = None

    # Tracking for diagnostics
    ee_positions = []  # Track EE positions for displacement calculation
    if log_ee_displacement:
        ee_positions.append(obs[:3].copy())
    
    # Track chunk stats for first 5 plans
    chunk_stats_logged = 0
    
    # Track observation z-scores at replanning steps (for debugging OOD observations)
    obs_z_scores = []  # List of mean(|z-score|) at each replanning step

    total_reward = 0.0
    for t in range(max_steps):
        # FIX #3: Dynamic MPC with queue clearing
        if dynamic_mpc:
            ee_pos = obs[:3]  # ee x,y,z (indices 0-2)
            left_cube = obs[8:11]  # left cube x,y,z
            right_cube = obs[15:18]  # right cube x,y,z
            dist_left = np.linalg.norm(ee_pos[:2] - left_cube[:2])  # xy distance
            dist_right = np.linalg.norm(ee_pos[:2] - right_cube[:2])
            dist_to_nearest = min(dist_left, dist_right)
            
            if dist_to_nearest > mpc_far_threshold:      # Far: approach
                current_execute_steps = execute_steps  # Use base value (16)
            elif dist_to_nearest > mpc_near_threshold:   # Near: descent
                current_execute_steps = max(4, execute_steps // 4)
            else:                                        # Very near: precision grasp
                current_execute_steps = 1
            
            # Clear queue if execute_steps changed to force immediate replan
            policy.set_execute_steps(current_execute_steps)
        
        # Track if this is a replanning step (action queue was empty before act())
        is_replanning_step = len(policy._action_queue) == 0
        
        action = policy.act(obs)  # no seed argument — uses global torch RNG
        
        # Compute observation z-score magnitude at replanning steps
        if is_replanning_step:
            obs_normalized = (obs - policy.obs_mean.cpu().numpy()) / policy.obs_std.cpu().numpy()
            obs_z_magnitude = np.mean(np.abs(obs_normalized))
            obs_z_scores.append(obs_z_magnitude)
        
        # Log chunk stats if enabled (first 5 plans)
        if log_chunks and chunk_stats_logged < 5 and len(policy._pending_chunks) > 0:
            chunk = policy._pending_chunks[-1]  # Most recent chunk
            chunk_pos = chunk[:, :4]  # Position deltas
            chunk_std = chunk_pos.std()
            chunk_abs_mean = np.abs(chunk_pos).mean()
            chunk_min = chunk_pos.min()
            chunk_max = chunk_pos.max()
            print(f"    [t={t:3d}, plan #{policy._plan_count}] chunk: "
                  f"std={chunk_std:.4f}, abs_mean={chunk_abs_mean:.4f}, "
                  f"range=[{chunk_min:+.3f}, {chunk_max:+.3f}]")
            chunk_stats_logged += 1

        # ── Commitment: nudge dy away from center early on ──
        if t < commit_steps:
            if abs(action[1]) < commit_thresh:
                if committed_sign is None:
                    committed_sign = 1.0 if commit_rng.random() > 0.5 else -1.0
                action[1] = committed_sign * commit_mag
            elif committed_sign is None:
                committed_sign = float(np.sign(action[1]))

        result = env.step(action)
        obs = result.obs
        total_reward += result.reward
        
        # Track EE positions every step
        if log_ee_displacement:
            ee_positions.append(obs[:3].copy())
        
        if result.done:
            break
    
    # Log EE displacement every 10 steps
    if log_ee_displacement and len(ee_positions) > 10:
        print(f"    EE displacement (meters) per 10-step window:")
        for i in range(0, len(ee_positions) - 10, 10):
            disp = np.linalg.norm(ee_positions[i + 10] - ee_positions[i])
            print(f"      steps {i:3d}-{i+10:3d}: {disp:.4f} m")
    
    # Log observation z-score statistics (debugging OOD observations)
    if len(obs_z_scores) > 0:
        avg_z = np.mean(obs_z_scores)
        max_z = np.max(obs_z_scores)
        print(f"    Obs z-score magnitude: avg={avg_z:.2f}, max={max_z:.2f} "
              f"(computed at {len(obs_z_scores)} replanning steps)")
        if max_z > 10.0:
            print(f"      [WARNING] High z-score detected! Observations may be out-of-distribution.")

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


# ── evaluation ───────────────────────────────────────────────────────

def evaluate(
    ckpt_path: str,
    K: int = 10,
    M: int = 10,
    n_videos: int = 10,
    out_dir: str = "outputs",
    video_dir: str = "outputs/videos",
    execute_steps: int = 8,
    dynamic_mpc: bool = False,
    mpc_far_threshold: float = 0.15,
    mpc_near_threshold: float = 0.05,
    env_seed_start: int = 100,
    verbose: bool = False,
    temporal_ensemble: bool = True,
    ensemble_grip: bool = False,
    sampling_method: str = 'ddim',
    ddim_eta: float = 0.0,
    ddim_steps: int | None = None,
    cube_jitter: float = 0.015,
    max_steps: int = 400,
    log_chunks: bool = False,
    log_ee_displacement: bool = False,
    verify_scaling: bool = False,
    multimodal_selection: bool = False,
    n_candidates: int = 5,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n\n"
            "===============================================================\n"
            "  ERROR: CUDA not available!\n"
            "  Evaluation requires GPU. Please activate venv with CUDA PyTorch:\n"
            "    .venv\\Scripts\\Activate.ps1\n"
            "===============================================================\n"
        )
    device = torch.device("cuda")
    print(f"device: {device}")
    print(f"execute_steps: {execute_steps}" + (" (base, will adapt dynamically)" if dynamic_mpc else ""))
    print(f"dynamic_mpc: {dynamic_mpc}")
    if dynamic_mpc:
        print(f"  far_threshold: {mpc_far_threshold}m -> execute_steps={execute_steps}")
        print(f"  near_threshold: {mpc_near_threshold}m -> execute_steps=4")
        print(f"  very_near: <{mpc_near_threshold}m -> execute_steps=1")
    print(f"max_steps: {max_steps}")
    print(f"temporal_ensemble: {temporal_ensemble}")
    print(f"ensemble_grip: {ensemble_grip}")
    print(f"sampling_method: {sampling_method}")
    if sampling_method == 'ddim':
        print(f"ddim_eta: {ddim_eta}")
        print(f"ddim_steps: {ddim_steps if ddim_steps is not None else 'all (100)'}")
    print(f"cube_jitter: {cube_jitter} m")
    print(f"log_chunks: {log_chunks}" + (" (first 5 plans per rollout)" if log_chunks else ""))
    print(f"log_ee_displacement: {log_ee_displacement}" + (" (per 10-step window)" if log_ee_displacement else ""))
    print(f"multimodal_selection: {multimodal_selection}" + (f" ({n_candidates} candidates)" if multimodal_selection else ""))
    
    # WARNING: Multimodal selection should only be used after achieving >50% success
    if multimodal_selection:
        print(f"\n{'='*70}")
        print(f"⚠️  MULTIMODAL SELECTION MODE ENABLED")
        print(f"{'='*70}")
        print(f"  This mode samples {n_candidates} candidate chunks per replan and selects")
        print(f"  the best according to a value function.")
        print(f"  ")
        print(f"  CURRENT: Using placeholder value function (prefers moderate action std)")
        print(f"  TODO: Replace _value_function() with VLM or learned value model")
        print(f"  ")
        print(f"  NOTE: Only use this after achieving >50% success in deterministic mode!")
        print(f"        Otherwise, action collapse will dominate all candidates.")
        print(f"{'='*70}\n")
    
    # Load checkpoint and check for jitter mismatch
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    demo_path = ckpt["config"].get("demo_path", "data/demos/demos.npz")
    
    # Try to load demo metadata
    demo_jitter = None
    if Path(demo_path).exists():
        try:
            import json
            demo_data = np.load(demo_path, allow_pickle=True)
            if 'metadata_json' in demo_data:
                metadata = json.loads(str(demo_data['metadata_json']))
                demo_jitter = metadata.get('cube_jitter', None)
        except Exception:
            pass  # Metadata not available (old demos)
    
    # Warn about jitter mismatch
    if demo_jitter is not None and abs(demo_jitter - cube_jitter) > 1e-6:
        print("\n" + "="*70)
        print("⚠️  WARNING: TRAIN/EVAL DISTRIBUTION MISMATCH")
        print("="*70)
        print(f"  Demos were collected with cube_jitter={demo_jitter:.4f}m")
        print(f"  Evaluation is using cube_jitter={cube_jitter:.4f}m")
        print(f"  ")
        print(f"  Mismatch = {abs(demo_jitter - cube_jitter)*100:.2f}cm")
        print(f"  ")
        print(f"  This distribution shift may significantly reduce success rate.")
        print(f"  ")
        print(f"  Recommendation: Use --cube_jitter {demo_jitter}")
        print("="*70 + "\n")
    
    # WARNING: Check if multimodality is testable
    if sampling_method == 'ddim' and ddim_eta == 0.0:
        print("\n" + "="*70)
        print("⚠️  WARNING: MULTIMODALITY CANNOT BE TESTED")
        print("="*70)
        print("  sampling_method='ddim' with eta=0.0 is DETERMINISTIC.")
        print("  Different sample_seeds will produce IDENTICAL trajectories.")
        print("  ")
        print("  To test multimodality, use one of:")
        print("    --sampling_method ddpm              (stochastic baseline)")
        print("    --sampling_method ddim --ddim_eta 0.3   (controlled stochasticity)")
        print("    --sampling_method ddim --ddim_eta 1.0   (equivalent to DDPM)")
        print("="*70 + "\n")

    policy = DiffusionPolicyRunner(ckpt_path, device, verbose=verbose,
                                    temporal_ensemble=temporal_ensemble,
                                    ensemble_grip=ensemble_grip,
                                    sampling_method=sampling_method,
                                    ddim_eta=ddim_eta,
                                    ddim_steps=ddim_steps,
                                    multimodal_selection=multimodal_selection,
                                    n_candidates=n_candidates)
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=cube_jitter)

    # Action scaling verification
    if verify_scaling:
        print(f"\n{'='*70}")
        print(f"ACTION SCALING VERIFICATION")
        print(f"{'='*70}")
        
        # Get demo metadata
        demo_data = np.load(demo_path, allow_pickle=True)
        demo_action_scale_pos = None
        demo_action_scale_yaw = None
        
        if 'metadata_json' in demo_data:
            import json
            metadata = json.loads(str(demo_data['metadata_json']))
            demo_action_scale_pos = metadata.get('action_scale_pos', None)
            demo_action_scale_yaw = metadata.get('action_scale_yaw_deg', None)
        
        # Get eval env scaling
        eval_action_scale_pos = env.action_scale_pos
        eval_action_scale_yaw = np.degrees(env._action_scale_yaw)
        
        print(f"  Demo collection scaling:")
        if demo_action_scale_pos is not None:
            print(f"    action_scale_pos: {demo_action_scale_pos:.4f} m/step")
        else:
            print(f"    action_scale_pos: [NOT RECORDED IN METADATA]")
        
        if demo_action_scale_yaw is not None:
            print(f"    action_scale_yaw: {demo_action_scale_yaw:.1f} deg/step")
        else:
            print(f"    action_scale_yaw: [NOT RECORDED IN METADATA]")
        
        print(f"\n  Eval environment scaling:")
        print(f"    action_scale_pos: {eval_action_scale_pos:.4f} m/step")
        print(f"    action_scale_yaw: {eval_action_scale_yaw:.1f} deg/step")
        
        # Check for mismatch
        if demo_action_scale_pos is not None:
            if abs(demo_action_scale_pos - eval_action_scale_pos) > 1e-6:
                print(f"\n  ⚠️  WARNING: ACTION SCALING MISMATCH DETECTED!")
                print(f"     Demo pos scale: {demo_action_scale_pos:.4f} m/step")
                print(f"     Eval pos scale: {eval_action_scale_pos:.4f} m/step")
                print(f"     Ratio: {eval_action_scale_pos / demo_action_scale_pos:.2f}x")
                print(f"     This will cause the robot to move faster/slower than demos!")
            else:
                print(f"\n  ✓ Action scaling matches between demo collection and eval")
        
        print(f"{'='*70}\n")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    video_count = 0
    env_seeds = list(range(env_seed_start, env_seed_start + K))

    print(f"\nRunning {K} env seeds x {M} sample seeds = {K * M} rollouts ...")
    for es in env_seeds:
        for mi in range(M):
            ss = 5000 + mi * 137  # deterministic, varied sample seeds

            # Record video? Use temp name, rename after outcome known.
            vp = None
            vp_temp = None
            if video_count < n_videos:
                vp_temp = str(Path(video_dir) / f"_tmp_e{es}_s{ss}.mp4")
                vp = vp_temp
                video_count += 1

            res = rollout(env, policy, es, ss,
                          execute_steps=execute_steps,
                          dynamic_mpc=dynamic_mpc,
                          mpc_far_threshold=mpc_far_threshold,
                          mpc_near_threshold=mpc_near_threshold,
                          video_path=vp,
                          max_steps=max_steps,
                          log_chunks=log_chunks and (mi < 2),  # Only log first 2 rollouts per env seed
                          log_ee_displacement=log_ee_displacement and (mi < 2))
            results.append(res)

            # Rename video to include outcome
            if vp_temp and Path(vp_temp).exists():
                final_name = (Path(video_dir) /
                              f"rollout_env{es}_sample{ss}_{res['outcome']}.mp4")
                try:
                    if final_name.exists():
                        final_name.unlink()
                    Path(vp_temp).rename(final_name)
                except OSError:
                    shutil.move(str(vp_temp), str(final_name))

            tag = res["outcome"]
            print(f"  env_seed={es}  sample={ss:5d}  => {tag:14s}  ({res['steps']} steps)")

    env.close()

    # ── aggregate stats ─────────────────────────────────────────────
    n_left = sum(1 for r in results if r["outcome"] == "left_success")
    n_right = sum(1 for r in results if r["outcome"] == "right_success")
    n_fail = sum(1 for r in results if r["outcome"] == "failure")
    total = len(results)
    n_success = n_left + n_right
    success_rate = n_success / total if total > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  Overall ({total} rollouts)")
    print(f"{'='*55}")
    print(f"  left_success :  {n_left:3d}  ({n_left/total:.1%})")
    print(f"  right_success:  {n_right:3d}  ({n_right/total:.1%})")
    print(f"  failure      :  {n_fail:3d}  ({n_fail/total:.1%})")
    print(f"  success_rate :  {success_rate:.1%}")

    # ── per-seed analysis ───────────────────────────────────────────
    per_seed: Dict[int, Dict[str, int]] = {}
    for r in results:
        es = r["env_seed"]
        per_seed.setdefault(es, {"left_success": 0, "right_success": 0, "failure": 0})
        per_seed[es][r["outcome"]] += 1

    entropy_rows: List[Dict] = []
    collapse_seeds = 0
    multimodal_seeds = 0

    print(f"\n{'='*55}")
    print(f"  Per env_seed breakdown")
    print(f"{'='*55}")
    print(f"  {'seed':>6s}  {'succ':>4s}  {'L':>3s}  {'R':>3s}  {'F':>3s}  "
          f"{'p(L|S)':>6s}  {'H':>5s}  {'flag':>10s}")
    print(f"  {'-'*50}")

    for es in sorted(per_seed):
        c = per_seed[es]
        ls = c["left_success"]
        rs = c["right_success"]
        fl = c["failure"]
        successes = ls + rs

        # p(left | success)
        p_left = ls / successes if successes > 0 else float("nan")

        # entropy over {left, right} — only meaningful when successes >= 5
        if successes >= 5:
            probs = np.array([ls, rs], dtype=np.float64) / successes
            ent = -sum(p * math.log2(p + 1e-12) for p in probs if p > 0)
        else:
            ent = float("nan")

        # collapse flag
        if successes >= 5:
            dominant = max(ls, rs) / successes
            if dominant > 0.9:
                flag = "COLLAPSE"
                collapse_seeds += 1
            elif min(ls, rs) >= 1:
                flag = "BIMODAL"
                multimodal_seeds += 1
            else:
                flag = "UNIMODAL"
        elif successes == 0:
            flag = "NO_SUCCESS"
        else:
            flag = "LOW_N"

        p_left_str = f"{p_left:.2f}" if not math.isnan(p_left) else "  n/a"
        ent_str = f"{ent:.3f}" if not math.isnan(ent) else "  n/a"

        print(f"  {es:6d}  {successes:4d}  {ls:3d}  {rs:3d}  {fl:3d}  "
              f"{p_left_str:>6s}  {ent_str:>5s}  {flag:>10s}")

        entropy_rows.append({
            "env_seed": es,
            "successes": successes,
            "left_success": ls,
            "right_success": rs,
            "failure": fl,
            "p_left_given_success": round(p_left, 4) if not math.isnan(p_left) else "",
            "entropy_success_only": round(ent, 4) if not math.isnan(ent) else "",
            "collapse_flag": flag,
        })

    # ── summary ─────────────────────────────────────────────────────
    valid_ents = [r["entropy_success_only"]
                  for r in entropy_rows
                  if r["entropy_success_only"] != ""]
    mean_ent = float(np.mean(valid_ents)) if valid_ents else float("nan")

    ent_display = f"{mean_ent:.3f}" if not math.isnan(mean_ent) else "n/a"
    print(f"\n  mean entropy (seeds with >=5 successes): {ent_display}")
    print(f"  multimodal seeds (both L+R): {multimodal_seeds}/{K}")
    print(f"  collapsed seeds  (>90% one): {collapse_seeds}/{K}")

    if multimodal_seeds > 0:
        print(f"\n  >>> MULTIMODALITY CONFIRMED on {multimodal_seeds} seed(s) <<<")
    else:
        print(f"\n  >>> NO multimodal seeds found — check model or increase M. <<<")

    # ── save outputs/metrics.json ───────────────────────────────────
    metrics = {
        "total_rollouts": total,
        "left_success": n_left,
        "right_success": n_right,
        "failure": n_fail,
        "success_rate": round(success_rate, 4),
        "p_left_given_success": round(n_left / n_success, 4) if n_success > 0 else None,
        "mean_entropy": round(mean_ent, 4) if not math.isnan(mean_ent) else None,
        "multimodal_seeds": multimodal_seeds,
        "collapse_seeds": collapse_seeds,
        "K": K,
        "M": M,
        "execute_steps": execute_steps,
        "per_seed": {str(k): v for k, v in per_seed.items()},
    }
    metrics_path = Path(out_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n  saved {metrics_path}")

    # ── save outputs/results.csv ────────────────────────────────────
    csv_path = Path(out_dir) / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "env_seed", "sample_seed", "outcome", "steps", "reward"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  saved {csv_path}")

    # ── save outputs/entropy_by_seed.csv ────────────────────────────
    ent_path = Path(out_dir) / "entropy_by_seed.csv"
    with open(ent_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "env_seed", "successes", "left_success", "right_success",
            "failure", "p_left_given_success", "entropy_success_only",
            "collapse_flag"])
        writer.writeheader()
        writer.writerows(entropy_rows)
    print(f"  saved {ent_path}")

    # ── plot ─────────────────────────────────────────────────────────
    _plot(per_seed, out_dir)

    # ── montage for one bimodal seed ────────────────────────────────
    _montage(results, video_dir, out_dir, per_seed)


def _plot(per_seed: Dict[int, Dict[str, int]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    seeds = sorted(per_seed.keys())
    lefts = [per_seed[s]["left_success"] for s in seeds]
    rights = [per_seed[s]["right_success"] for s in seeds]
    fails = [per_seed[s]["failure"] for s in seeds]

    x = np.arange(len(seeds))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(seeds) * 1.2), 4))
    ax.bar(x - w, lefts, w, label="Left", color="#4e79a7")
    ax.bar(x, rights, w, label="Right", color="#e15759")
    ax.bar(x + w, fails, w, label="Fail", color="#bab0ac")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
    ax.set_xlabel("Environment seed")
    ax.set_ylabel("Count")
    ax.set_title("Per-seed left / right / failure distribution")
    ax.legend()
    fig.tight_layout()

    for name in ["multimodality_bar.png", "left_right_hist.png"]:
        fig.savefig(Path(out_dir) / name, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(out_dir) / 'multimodality_bar.png'}")


def _montage(
    results: List[Dict],
    video_dir: str,
    out_dir: str,
    per_seed: Dict[int, Dict[str, int]],
) -> None:
    """If ffmpeg is available, create a montage for one bimodal env_seed."""
    # Find a seed with both left and right
    bimodal_seed = None
    for es in sorted(per_seed):
        c = per_seed[es]
        if c["left_success"] >= 1 and c["right_success"] >= 1:
            bimodal_seed = es
            break
    if bimodal_seed is None:
        print("  no bimodal seed found — skipping montage")
        return

    # Gather video files for that seed
    vdir = Path(video_dir)
    vids = sorted(vdir.glob(f"rollout_env{bimodal_seed}_sample*_*.mp4"))
    if len(vids) < 2:
        print(f"  only {len(vids)} video(s) for env_seed={bimodal_seed} — skipping montage")
        return

    # Check ffmpeg
    if shutil.which("ffmpeg") is None:
        print("  ffmpeg not found — skipping montage (videos still saved individually)")
        return

    # Build montage: tile up to 4 videos in a 2x2 grid
    vids = vids[:4]
    montage_path = Path(out_dir) / f"montage_env{bimodal_seed}.mp4"

    if len(vids) == 1:
        shutil.copy(str(vids[0]), str(montage_path))
    else:
        # Build ffmpeg filter for tiling
        inputs = []
        for v in vids:
            inputs.extend(["-i", str(v)])

        n = len(vids)
        if n == 2:
            filt = "[0:v][1:v]hstack=inputs=2[v]"
        elif n == 3:
            filt = ("[0:v][1:v]hstack=inputs=2[top];"
                    "[2:v]pad=iw*2:ih[bot];"
                    "[top][bot]vstack=inputs=2[v]")
        else:  # 4
            filt = ("[0:v][1:v]hstack=inputs=2[top];"
                    "[2:v][3:v]hstack=inputs=2[bot];"
                    "[top][bot]vstack=inputs=2[v]")

        cmd = (["ffmpeg", "-y"] + inputs +
               ["-filter_complex", filt, "-map", "[v]",
                "-c:v", "libx264", "-crf", "23",
                str(montage_path)])
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            print(f"  saved montage: {montage_path}")
        except Exception as e:
            print(f"  montage failed: {e}")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate multimodality of diffusion policy")
    ap.add_argument("--ckpt", type=str, default="runs/latest/ckpt.pt")
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--K", type=int, default=10,
                    help="number of env seeds")
    ap.add_argument("--M", type=int, default=10,
                    help="number of sample seeds per env seed")
    ap.add_argument("--n_videos", type=int, default=10,
                    help="max videos to record (0 = none)")
    ap.add_argument("--out_dir", type=str, default="outputs",
                    help="directory for metrics/csv/plots")
    ap.add_argument("--video_dir", type=str, default="outputs/videos",
                    help="directory for rollout videos")
    ap.add_argument("--execute_steps", type=int, default=8,
                    help="actions to execute before replanning (1/2/4/8)")
    ap.add_argument("--dynamic_mpc", action="store_true",
                    help="enable dynamic MPC: adapt execute_steps based on proximity to cubes")
    ap.add_argument("--mpc_far_threshold", type=float, default=0.15,
                    help="distance threshold (m) for far/approach phase (uses base execute_steps)")
    ap.add_argument("--mpc_near_threshold", type=float, default=0.05,
                    help="distance threshold (m) for near/descent phase (uses execute_steps=4)")
    ap.add_argument("--env_seed_start", type=int, default=100,
                    help="first env seed")
    ap.add_argument("--verbose", action="store_true",
                    help="print action chunk stats for first 3 plans")
    ap.add_argument("--temporal_ensemble", action="store_true",
                    help="average overlapping action chunks for smoother actions")
    ap.add_argument("--ensemble_grip", action="store_true",
                    help="enable ensembling for gripper (default: False to avoid smearing 0/1)")
    ap.add_argument("--sampling_method", type=str, default="ddim",
                    choices=["ddpm", "ddim"],
                    help="sampling method: ddpm (stochastic) or ddim (deterministic, smoother)")
    ap.add_argument("--ddim_eta", type=float, default=0.0,
                    help="DDIM stochasticity: eta=0 (deterministic, can't test multimodality), "
                         "eta>0 (stochastic, enables multimodality), eta=1 (~ DDPM)")
    ap.add_argument("--ddim_steps", type=int, default=None,
                    help="Number of DDIM inference steps (default: use all n_diffusion_steps). "
                         "Typical values: 20-50 for fast sampling. Lower=faster but may reduce quality.")
    ap.add_argument("--cube_jitter", type=float, default=0.015,
                    help="Random jitter for cube placement (m). 0=fixed positions, 0.015=±1.5cm. "
                         "MUST match demo collection jitter for distribution consistency.")
    ap.add_argument("--max_steps", type=int, default=400,
                    help="max steps per rollout (400 — demos are ~303 steps)")
    ap.add_argument("--log_chunks", action="store_true",
                    help="log first 5 planned chunks' statistics (std, abs_mean, min/max) for first 2 rollouts per env seed")
    ap.add_argument("--log_ee_displacement", action="store_true",
                    help="log end-effector displacement per 10-step window for first 2 rollouts per env seed")
    ap.add_argument("--verify_scaling", action="store_true",
                    help="verify action scaling matches between demo collection and eval (checks for scaling bugs)")
    ap.add_argument("--multimodal_selection", action="store_true",
                    help="enable multimodal selection: sample N candidates per replan, pick best via value function. "
                         "Only use after >50%% deterministic success! Requires stochastic sampling (ddim_eta>0 or ddpm).")
    ap.add_argument("--n_candidates", type=int, default=5,
                    help="number of candidate chunks to sample when multimodal_selection is enabled (default: 5)")
    args = ap.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        dynamic_mpc=args.dynamic_mpc,
        mpc_far_threshold=args.mpc_far_threshold,
        mpc_near_threshold=args.mpc_near_threshold,
        K=args.K,
        M=args.M,
        n_videos=args.n_videos,
        out_dir=args.out_dir,
        video_dir=args.video_dir,
        execute_steps=args.execute_steps,
        env_seed_start=args.env_seed_start,
        verbose=args.verbose,
        temporal_ensemble=args.temporal_ensemble,
        ensemble_grip=args.ensemble_grip,
        sampling_method=args.sampling_method,
        ddim_eta=args.ddim_eta,
        ddim_steps=args.ddim_steps,
        cube_jitter=args.cube_jitter,
        max_steps=args.max_steps,
        log_chunks=args.log_chunks,
        log_ee_displacement=args.log_ee_displacement,
        verify_scaling=args.verify_scaling,
        multimodal_selection=args.multimodal_selection,
        n_candidates=args.n_candidates,
    )


if __name__ == "__main__":
    main()
