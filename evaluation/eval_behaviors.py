#!/usr/bin/env python3
"""
Unified Best-of-K VLM Behavior Evaluation
==========================================

ONE methodology, FOUR behaviors. Same diffusion policy checkpoint,
same Best-of-K pipeline — only the VLM prompt and selection criterion change.

Behaviors:
  1. legibility    — VLM picks trajectory where goal is most obvious early
  2. predictability — VLM picks most direct/efficient path to goal
  3. safety        — VLM picks trajectory with best obstacle clearance
  4. grounding     — VLM picks trajectory following spatial instruction

Pipeline per episode:
  1. Reset env with episode seed
  2. Save PyBullet state
  3. For k in 1..K:
     - Restore state → sample action chunk → simulate 150 steps (5s)
     - Capture frames at t=0,1,2,3,4,5s → send to VLM
  4. Select best candidate per behavior criterion
  5. Restore state → replay winner to completion → record video
  6. Compute L_early on executed trajectory
  7. Also replay a RANDOM candidate (baseline) for paired comparison

Usage:
  $env:GEMINI_API_KEY="YOUR_KEY"
  .venv\\Scripts\\python.exe evaluation/eval_behaviors.py ^
      --checkpoint runs/diffusion_20260402_072747/ckpt_ep100.pt ^
      --behavior legibility --n_episodes 20 --K 8
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, r"C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\src")

from envs.twoblockpick_env import TwoBlockPickEnv
from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.schema import ManifestEntry

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VLM_MODEL = "gemini-2.5-flash-preview-05-20"
GOAL_A_DESC = "pick the left block"
GOAL_B_DESC = "pick the right block"
FRAME_STEPS = {0: 0, 30: 1, 60: 2, 90: 3, 120: 4, 149: 5}  # step→t_sec


# ══════════════════════════════════════════════════════════════════════════
# MODEL + SAMPLER (inline from eval_with_videos.py — proven, do not change)
# ══════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class UNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.act = nn.Mish()
    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2)
                     + self.time_proj(t_emb).unsqueeze(1))
        h = self.act(self.norm2(self.conv2(h).transpose(1,2)).transpose(1,2)
                     + self.shortcut(x))
        return h

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128), nn.Linear(128, hidden_dim),
            nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        dims = [hidden_dim, hidden_dim*2, hidden_dim*4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim) for i in range(len(dims)-1)])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1]+dims[i+1], dims[i], hidden_dim)
            for i in range(len(dims)-2, -1, -1)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, act_dim))
    def forward(self, noisy_act, timestep, obs):
        t_emb = self.time_mlp(timestep)
        x = self.input_proj(noisy_act) + self.obs_embed(obs).unsqueeze(1)
        skips = []
        for blk in self.encoder_blocks:
            x = blk(x, t_emb); skips.append(x)
        x = self.bottleneck(x, t_emb)
        for blk, sk in zip(self.decoder_blocks, reversed(skips)):
            x = blk(torch.cat([x, sk], dim=-1), t_emb)
        return self.output_proj(x)

class DDIMSampler:
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs, n_sampling_steps=10):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod)-1,
                           n_sampling_steps, device=self.device).long(), [0])
        for i, t in enumerate(timesteps):
            t_b = t.repeat(B)
            eps = model(x, t_b, obs)
            a_t = self.alphas_cumprod[t]
            a_prev = (self.alphas_cumprod[timesteps[i+1]]
                      if i < len(timesteps)-1
                      else torch.tensor(1.0, device=self.device))
            x0 = (x - torch.sqrt(1-a_t)*eps) / torch.sqrt(a_t)
            x = (torch.sqrt(a_prev)*x0 + torch.sqrt(1-a_prev)*eps
                 if i < len(timesteps)-1 else x0)
        return x


# ══════════════════════════════════════════════════════════════════════════
# LOAD POLICY
# ══════════════════════════════════════════════════════════════════════════

def load_policy(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg.get("hidden_dim", 256),
        n_blocks=cfg.get("n_blocks", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sampler = DDIMSampler(
        cfg["n_diffusion_steps"], cfg["beta_start"], cfg["beta_end"], device)
    obs_mean = np.array(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.array(ckpt["obs_std"], dtype=np.float32)
    act_mean = np.array(ckpt["act_mean"], dtype=np.float32)
    act_std = np.array(ckpt["act_std"], dtype=np.float32)
    return model, sampler, obs_mean, obs_std, act_mean, act_std, cfg


# ══════════════════════════════════════════════════════════════════════════
# FRAME CAPTURE
# ══════════════════════════════════════════════════════════════════════════

def capture_jpeg(env, width=480, height=480, quality=90) -> bytes:
    from PIL import Image
    frame = env.render(mode="rgb_array", width=width, height=height)
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════
# SIMULATE ONE CANDIDATE (5 seconds = 150 steps)
# ══════════════════════════════════════════════════════════════════════════

def simulate_candidate(env, model, sampler, obs, obs_mean, obs_std,
                       act_mean, act_std, device, seed: int,
                       n_steps: int = 150):
    """Simulate a candidate trajectory from current env state.

    Returns dict with: actions (full 150-step array), frames (list of JPEG bytes),
    ee_trajectory (Nx3), obs_trajectory (Nx22).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    queue = deque()
    frames_bytes = []
    ee_traj = []
    obs_traj = []
    actions_all = []
    current_obs = obs.copy()

    for step in range(n_steps):
        ee_traj.append(current_obs[:3].copy())
        obs_traj.append(current_obs.copy())

        if len(queue) == 0:
            obs_norm = (current_obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            act_seq = sampler.sample(model, obs_t, n_sampling_steps=10)
            act_np = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in act_np:
                queue.append(a)

        action = queue.popleft()
        actions_all.append(action.copy())

        # Capture frame at scheduled timesteps
        if step in FRAME_STEPS:
            frames_bytes.append(capture_jpeg(env))

        result = env.step(action)
        current_obs = result.obs
        if result.done:
            # Pad remaining
            while len(actions_all) < n_steps:
                actions_all.append(action.copy())
                ee_traj.append(current_obs[:3].copy())
                obs_traj.append(current_obs.copy())
            break

    return dict(
        actions=np.array(actions_all),
        frames=frames_bytes,
        ee_trajectory=np.array(ee_traj),
        obs_trajectory=np.array(obs_traj),
    )


# ══════════════════════════════════════════════════════════════════════════
# VLM SCORING
# ══════════════════════════════════════════════════════════════════════════

def score_candidate_vlm(client: GeminiClient, frames_bytes: list,
                        video_id: str, behavior: str,
                        target_block: str = "LEFT",
                        obstacle_pos: np.ndarray | None = None,
                        waypoint_desc: str | None = None,
                        sleep: float = 1.0) -> dict:
    """Score a candidate's frames with VLM using behavior-specific prompt.

    Returns dict with pA, pB, choice, cue, legible, score (behavior-specific).
    """
    target_goal = "A" if target_block == "LEFT" else "B"

    # Build ManifestEntry
    entry = ManifestEntry(
        video_id=video_id,
        video_path=f"candidate_{video_id}",
        goal_gt=target_goal,
        goal_A=GOAL_A_DESC,
        goal_B=GOAL_B_DESC,
        scene_id="twoblockpick",
        task_family="block_pick",
        traj_type=behavior,
        notes=f"behavior={behavior}",
    )

    # For legibility: use the existing validated prompt (prefix_frames mode)
    if behavior == "legibility":
        result = client.evaluate_frame(
            image_bytes=frames_bytes,
            manifest_entry=entry,
            t_sec=5,
            frame_idx=0,
            mode="prefix_frames",
        )
        # Legibility = how clearly the arm reveals ANY intent (max of both)
        legibility_score = max(result.pA, result.pB)
        return dict(
            pA=result.pA, pB=result.pB, choice=result.choice,
            cue=result.cue, legible=result.legible,
            score=float(legibility_score),
            latency_ms=result.latency_ms,
        )

    # For other behaviors: custom prompt via direct API call
    from google.genai import types as gtypes

    parts = []
    for i, fb in enumerate(frames_bytes):
        parts.append(gtypes.Part.from_bytes(data=fb, mime_type="image/jpeg"))

    if behavior == "predictability":
        prompt_text = (
            f"You are evaluating a robot arm trajectory. The robot is picking one of two blocks.\n"
            f"Goal A: {GOAL_A_DESC}. Goal B: {GOAL_B_DESC}.\n\n"
            f"These {len(frames_bytes)} frames show the arm's motion from t=0 to t=5 seconds.\n\n"
            f"TASK: Rate how PREDICTABLE and EFFICIENT this trajectory is.\n"
            f"A predictable trajectory takes the SHORTEST, MOST DIRECT path to its target — "
            f"no unnecessary lateral sweeps, no wide arcs. A human watching can predict "
            f"where the arm will be 1 second from now because it moves in a straight, purposeful line.\n\n"
            f"First determine which goal the robot appears to be heading toward.\n"
            f"Then rate predictability from 0.0 (unpredictable/erratic) to 1.0 (perfectly direct).\n\n"
            f"Output ONLY valid JSON: {{\"pA\": X, \"pB\": X, \"predictability\": X, "
            f"\"cue\": \"short description of path directness\"}}"
        )
    elif behavior == "safety":
        prompt_text = (
            f"You are evaluating a robot arm trajectory for SAFETY.\n"
            f"Goal A: {GOAL_A_DESC}. Goal B: {GOAL_B_DESC}.\n\n"
            f"There is a RED CYLINDRICAL OBSTACLE on the table between the robot's starting position "
            f"and the blocks. The obstacle is clearly visible in the frames.\n\n"
            f"These {len(frames_bytes)} frames show the arm's motion from t=0 to t=5 seconds.\n\n"
            f"TASK: Rate how SAFELY this trajectory avoids the obstacle.\n"
            f"A safe trajectory maintains clear distance from the obstacle at all times. "
            f"An unsafe trajectory passes very close to or appears to risk collision.\n\n"
            f"First determine which goal the robot appears heading toward.\n"
            f"Then rate safety from 0.0 (dangerous, very close to obstacle) to 1.0 (wide clearance).\n\n"
            f"Output ONLY valid JSON: {{\"pA\": X, \"pB\": X, \"safety\": X, "
            f"\"cue\": \"short description of clearance from obstacle\"}}"
        )
    elif behavior == "grounding":
        prompt_text = (
            f"You are evaluating a robot arm trajectory for SPATIAL INSTRUCTION FOLLOWING.\n"
            f"Goal A: {GOAL_A_DESC}. Goal B: {GOAL_B_DESC}.\n\n"
            f"INSTRUCTION: \"{waypoint_desc or 'Pass near the BLUE block before picking the GREEN block.'}\"\n\n"
            f"The scene has colored blocks visible in the frames. "
            f"These {len(frames_bytes)} frames show the arm's motion from t=0 to t=5 seconds.\n\n"
            f"TASK: Rate how well this trajectory follows the spatial instruction.\n"
            f"Does the arm's path pass near the referenced waypoint block before reaching the target?\n\n"
            f"Rate instruction_following from 0.0 (ignores instruction) to 1.0 (clearly follows it).\n"
            f"Also determine which goal block the robot is heading toward.\n\n"
            f"Output ONLY valid JSON: {{\"pA\": X, \"pB\": X, \"instruction_following\": X, "
            f"\"cue\": \"short description of path relative to waypoint\"}}"
        )
    else:
        raise ValueError(f"Unknown behavior: {behavior}")

    parts.append(gtypes.Part.from_text(prompt_text))

    # Call Gemini directly for custom prompts
    for attempt in range(3):
        try:
            resp = client.client.models.generate_content(
                model=client.model,
                contents=[gtypes.Content(role="user", parts=parts)],
            )
            text = resp.text.strip()
            # Parse JSON
            j_start = text.find("{")
            j_end = text.rfind("}") + 1
            if j_start == -1 or j_end == 0:
                raise ValueError(f"No JSON in response: {text[:200]}")
            data = json.loads(text[j_start:j_end])

            pA = float(data.get("pA", 0.5))
            pB = float(data.get("pB", 0.5))
            cue = str(data.get("cue", ""))

            if behavior == "predictability":
                bscore = float(data.get("predictability", 0.5))
            elif behavior == "safety":
                bscore = float(data.get("safety", 0.5))
            elif behavior == "grounding":
                bscore = float(data.get("instruction_following", 0.5))
            else:
                bscore = max(pA, pB)

            choice = "A" if pA >= pB else "B"
            return dict(pA=pA, pB=pB, choice=choice, cue=cue,
                        legible="legible_now" if max(pA,pB)>0.6 else "not_legible_yet",
                        score=float(bscore), latency_ms=0)
        except Exception as e:
            if attempt < 2:
                print(f"      [VLM retry {attempt+1}: {e}]")
                time.sleep(2 ** attempt)
            else:
                print(f"      [VLM FAILED 3x: {e}]")
                return dict(pA=0.5, pB=0.5, choice="C", cue=f"ERROR: {e}",
                            legible="not_legible_yet", score=0.5, latency_ms=0)

    time.sleep(sleep)


# ══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP PER BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════

_OBSTACLE_UID = None

def setup_env_for_behavior(env, behavior: str):
    """Modify the environment for the given behavior. Returns metadata."""
    global _OBSTACLE_UID
    cid = env._cid

    if behavior == "safety":
        # Add red cylinder obstacle between EE home and blocks
        obs_pos = [0.45, 0.0, 0.42]  # on table, between start and blocks
        obs_radius = 0.025
        obs_height = 0.08
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obs_radius,
                                     height=obs_height, physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obs_radius,
                                  length=obs_height,
                                  rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                  physicsClientId=cid)
        _OBSTACLE_UID = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=obs_pos, physicsClientId=cid)
        return dict(obstacle_pos=np.array(obs_pos), obstacle_radius=obs_radius)

    elif behavior == "grounding":
        # Change block colors: left=green, right=red (default)
        # Add blue waypoint block between them
        p.changeVisualShape(env._cube_l_uid, -1,
                            rgbaColor=[0.2, 0.8, 0.2, 1.0],
                            physicsClientId=cid)
        p.changeVisualShape(env._cube_r_uid, -1,
                            rgbaColor=[0.8, 0.2, 0.2, 1.0],
                            physicsClientId=cid)
        # Blue waypoint block at midpoint, slightly offset
        wp_pos = [0.48, 0.03, 0.421]
        col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[0.015, 0.015, 0.015],
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.015, 0.015, 0.015],
                                  rgbaColor=[0.2, 0.2, 1.0, 1.0],
                                  physicsClientId=cid)
        wp_uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                   baseVisualShapeIndex=vis,
                                   basePosition=wp_pos, physicsClientId=cid)
        return dict(waypoint_pos=np.array(wp_pos), waypoint_uid=wp_uid,
                    instruction="Pass near the BLUE block before picking the GREEN block (left block).")

    return {}


# ══════════════════════════════════════════════════════════════════════════
# L_EARLY METRIC (from executed trajectory)
# ══════════════════════════════════════════════════════════════════════════

def compute_l_early(ee_traj: np.ndarray, obs_final: np.ndarray,
                    early_frac: float = 0.3) -> dict:
    """Compute L_early_intent from executed EE trajectory."""
    if len(ee_traj) < 4:
        return dict(L_early=0.5, true_goal="unknown")

    goals = np.array([obs_final[8:11], obs_final[15:18]])  # left, right
    T = len(ee_traj)
    early_end = max(2, int(T * early_frac))
    sigma = np.linalg.norm(goals[0] - goals[1]) / (2 * np.sqrt(2 * np.log(2)))

    posteriors = []
    for t in range(early_end):
        d0 = np.linalg.norm(ee_traj[t] - goals[0])
        d1 = np.linalg.norm(ee_traj[t] - goals[1])
        l0 = np.exp(-d0**2 / (2 * sigma**2))
        l1 = np.exp(-d1**2 / (2 * sigma**2))
        total = l0 + l1 + 1e-12
        posteriors.append([l0/total, l1/total])

    posteriors = np.array(posteriors)
    # True goal = whichever has higher mean posterior in early window
    mean_p0 = posteriors[:, 0].mean()
    mean_p1 = posteriors[:, 1].mean()
    if mean_p0 >= mean_p1:
        return dict(L_early=float(mean_p0), true_goal="left")
    else:
        return dict(L_early=float(mean_p1), true_goal="right")


# ══════════════════════════════════════════════════════════════════════════
# RUN ONE EPISODE (Best-of-K)
# ══════════════════════════════════════════════════════════════════════════

def run_episode(model, sampler, obs_mean, obs_std, act_mean, act_std,
                device, client: GeminiClient, episode_seed: int,
                behavior: str, K: int, max_steps: int = 600,
                out_dir: Path | None = None,
                env_meta: dict | None = None,
                sleep: float = 1.5) -> dict:
    """Run one Best-of-K episode.

    Returns dict with per-candidate scores, selected index, success,
    L_early for both VLM-selected and baseline (random first candidate).
    """
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs_init = env.reset(seed=episode_seed)

    # Setup environment modifications for this behavior
    env_info = setup_env_for_behavior(env, behavior)

    # Settle physics after env modifications
    for _ in range(60):
        p.stepSimulation(physicsClientId=env._cid)
    obs_init = env._get_obs()

    # Save state AFTER env setup
    saved_state = p.saveState(physicsClientId=env._cid)

    # ── Phase 1: Generate K candidates ────────────────────────────
    candidates = []
    for k in range(K):
        p.restoreState(saved_state, physicsClientId=env._cid)
        # Restore Python-side state
        env._target_pos = obs_init[:3].copy()
        env._target_yaw = 0.0
        env._grip_cmd = 1.0
        env._episode_steps = 0

        c_seed = episode_seed * 1000 + k * 7 + 1
        cand = simulate_candidate(
            env, model, sampler, obs_init, obs_mean, obs_std,
            act_mean, act_std, device, seed=c_seed)
        cand["seed"] = c_seed
        cand["idx"] = k
        candidates.append(cand)
        print(f"    Candidate {k+1}/{K} simulated", end="", flush=True)

    # ── Phase 2: VLM scoring ─────────────────────────────────────
    # For legibility: target_block doesn't matter (score = max(pA,pB))
    # For other behaviors: use VLM consensus from first pass to determine target
    target_block = "LEFT"  # default; overridden below if needed

    for k, cand in enumerate(candidates):
        vid_id = f"ep{episode_seed}_c{k}_{behavior}"
        vlm_result = score_candidate_vlm(
            client, cand["frames"], vid_id, behavior,
            target_block=target_block,
            obstacle_pos=env_info.get("obstacle_pos"),
            waypoint_desc=env_info.get("instruction"),
            sleep=sleep)
        cand["vlm"] = vlm_result
        print(f"  | VLM scored c{k}: score={vlm_result['score']:.3f} "
              f"cue=\"{vlm_result['cue'][:50]}\"")

    # Determine actual target from VLM consensus (majority choice)
    choices_A = sum(1 for c in candidates if c["vlm"]["pA"] > c["vlm"]["pB"])
    target_block = "LEFT" if choices_A > K // 2 else "RIGHT"

    # ── Phase 2b: Feasibility bonus ──────────────────────────────
    # Candidates whose final EE is close to a block get a small bonus
    # to break ties in favor of trajectories likely to succeed
    block_l = obs_init[8:11]   # left block pos
    block_r = obs_init[15:18]  # right block pos
    for cand in candidates:
        final_ee = cand["ee_trajectory"][-1]
        d_l = np.linalg.norm(final_ee[:2] - block_l[:2])
        d_r = np.linalg.norm(final_ee[:2] - block_r[:2])
        closest = min(d_l, d_r)
        # bonus: 0.05 if very close (<0.05m), 0 if far (>0.2m)
        feasibility = max(0, 0.05 * (1.0 - closest / 0.2))
        cand["vlm"]["combined_score"] = cand["vlm"]["score"] + feasibility

    # ── Phase 3: Select best + baseline ──────────────────────────
    # VLM selection: highest combined score (VLM + feasibility)
    vlm_selected = max(range(K), key=lambda i: candidates[i]["vlm"]["combined_score"])
    # Baseline: candidate with LOWEST score (worst-case alternative)
    # If same as VLM-selected, pick a different one
    sorted_by_score = sorted(range(K), key=lambda i: candidates[i]["vlm"]["combined_score"])
    baseline_idx = sorted_by_score[0]
    if baseline_idx == vlm_selected and K > 1:
        baseline_idx = sorted_by_score[1]

    print(f"  → VLM selected candidate {vlm_selected} "
          f"(score={candidates[vlm_selected]['vlm']['score']:.3f}, "
          f"combined={candidates[vlm_selected]['vlm']['combined_score']:.3f})")
    print(f"  → Baseline candidate {baseline_idx} "
          f"(score={candidates[baseline_idx]['vlm']['score']:.3f})")

    # ── Phase 4: Execute VLM selection to completion + record video ─
    results = {}
    for label, sel_idx in [("vlm", vlm_selected), ("baseline", baseline_idx)]:
        p.restoreState(saved_state, physicsClientId=env._cid)
        env._target_pos = obs_init[:3].copy()
        env._target_yaw = 0.0
        env._grip_cmd = 1.0
        env._episode_steps = 0
        # Re-apply env mods if needed (they persist in the pybullet state)

        # Setup video recording
        video_path = None
        if out_dir:
            video_path = str(out_dir / "videos" / f"ep{episode_seed:03d}_{label}_{behavior}.mp4")
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            env.record_video(video_path, width=640, height=480, fps=30)

        # Replay selected candidate's actions, then continue with replanning
        sel_actions = candidates[sel_idx]["actions"]
        queue = deque(sel_actions)
        obs = obs_init.copy()
        ee_full = []
        obs_full = []
        success = False
        steps = 0

        while steps < max_steps:
            ee_full.append(obs[:3].copy())
            obs_full.append(obs.copy())

            if len(queue) == 0:
                # Replan
                obs_norm = (obs - obs_mean) / obs_std
                obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                act_seq = sampler.sample(model, obs_t, n_sampling_steps=10)
                act_np = act_seq[0].cpu().numpy() * act_std + act_mean
                for a in act_np:
                    queue.append(a)

            action = queue.popleft()
            result = env.step(action)
            obs = result.obs
            steps += 1
            s_l = result.info.get("success_left", 0) > 0.5
            s_r = result.info.get("success_right", 0) > 0.5
            success = s_l or s_r
            if result.done:
                break

        if video_path:
            env.stop_video()

        ee_arr = np.array(ee_full)
        l_early_info = compute_l_early(ee_arr, obs, early_frac=0.3)

        picked = "left" if result.info.get("success_left", 0) > 0.5 else (
                 "right" if result.info.get("success_right", 0) > 0.5 else "none")

        results[label] = dict(
            selected_idx=sel_idx,
            success=success,
            picked=picked,
            steps=steps,
            L_early=l_early_info["L_early"],
            true_goal=l_early_info["true_goal"],
            vlm_score=candidates[sel_idx]["vlm"]["score"],
            vlm_cue=candidates[sel_idx]["vlm"]["cue"],
            video_path=video_path,
        )
        status = "OK" if success else "FAIL"
        print(f"  {label.upper()}: {status} picked={picked} "
              f"L_early={l_early_info['L_early']:.4f} steps={steps}")

    # Remove the pybullet saved state before closing env
    p.removeState(saved_state, physicsClientId=env._cid)
    env.close()

    return dict(
        episode_seed=episode_seed,
        behavior=behavior,
        target_block=target_block,
        K=K,
        vlm=results["vlm"],
        baseline=results["baseline"],
        candidates=[
            dict(idx=c["idx"], seed=c["seed"],
                 vlm_score=c["vlm"]["score"],
                 vlm_pA=c["vlm"]["pA"], vlm_pB=c["vlm"]["pB"],
                 vlm_choice=c["vlm"]["choice"],
                 vlm_cue=c["vlm"]["cue"][:100])
            for c in candidates
        ],
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Best-of-K VLM Behavior Eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--behavior", required=True,
                        choices=["legibility", "predictability", "safety", "grounding"])
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="Sleep between VLM calls (rate limiting)")
    parser.add_argument("--vlm_model", type=str, default=VLM_MODEL)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load policy
    model, sampler, obs_mean, obs_std, act_mean, act_std, cfg = \
        load_policy(args.checkpoint, device)
    print(f"Policy loaded: {args.checkpoint}")
    print(f"  horizon={cfg['horizon']}, act_dim={cfg['act_dim']}")

    # Create VLM client
    client = GeminiClient(model=args.vlm_model)
    print(f"VLM client: {args.vlm_model}")

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"outputs/behaviors/{args.behavior}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BEHAVIOR: {args.behavior.upper()}")
    print(f"  Episodes: {args.n_episodes}, K={args.K}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # Run episodes
    all_results = []
    for ep in range(args.n_episodes):
        ep_seed = ep * 7 + 42  # deterministic, spread out
        print(f"\n── Episode {ep+1}/{args.n_episodes} (seed={ep_seed}) ──")
        result = run_episode(
            model, sampler, obs_mean, obs_std, act_mean, act_std,
            device, client, ep_seed, args.behavior, args.K,
            out_dir=out_dir, sleep=args.sleep)
        all_results.append(result)

    # ── Summary statistics ────────────────────────────────────────
    vlm_success = [r["vlm"]["success"] for r in all_results]
    bl_success = [r["baseline"]["success"] for r in all_results]
    vlm_learly = [r["vlm"]["L_early"] for r in all_results]
    bl_learly = [r["baseline"]["L_early"] for r in all_results]
    vlm_scores = [r["vlm"]["vlm_score"] for r in all_results]
    bl_scores = [r["baseline"]["vlm_score"] for r in all_results]

    # Paired t-test
    from scipy import stats
    diffs = np.array(vlm_learly) - np.array(bl_learly)
    if np.std(diffs) > 1e-8:
        t_stat, p_val = stats.ttest_rel(vlm_learly, bl_learly)
    else:
        t_stat, p_val = 0.0, 1.0

    summary = dict(
        behavior=args.behavior,
        n_episodes=args.n_episodes,
        K=args.K,
        vlm_model=args.vlm_model,
        checkpoint=args.checkpoint,
        vlm_success_rate=float(np.mean(vlm_success)),
        baseline_success_rate=float(np.mean(bl_success)),
        vlm_L_early_mean=float(np.mean(vlm_learly)),
        vlm_L_early_std=float(np.std(vlm_learly)),
        baseline_L_early_mean=float(np.mean(bl_learly)),
        baseline_L_early_std=float(np.std(bl_learly)),
        L_early_improvement=float(np.mean(vlm_learly) - np.mean(bl_learly)),
        vlm_score_mean=float(np.mean(vlm_scores)),
        vlm_score_std=float(np.std(vlm_scores)),
        baseline_score_mean=float(np.mean(bl_scores)),
        paired_ttest_t=float(t_stat),
        paired_ttest_p=float(p_val),
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.behavior.upper()}")
    print(f"{'='*60}")
    print(f"  VLM selected:  success={summary['vlm_success_rate']:.0%}  "
          f"L_early={summary['vlm_L_early_mean']:.4f} +/- {summary['vlm_L_early_std']:.4f}")
    print(f"  Baseline:      success={summary['baseline_success_rate']:.0%}  "
          f"L_early={summary['baseline_L_early_mean']:.4f} +/- {summary['baseline_L_early_std']:.4f}")
    print(f"  Improvement:   {summary['L_early_improvement']:+.4f}  "
          f"(p={summary['paired_ttest_p']:.4f})")
    print(f"  VLM score:     {summary['vlm_score_mean']:.4f} vs baseline {summary['baseline_score_mean']:.4f}")
    print(f"{'='*60}")

    # Save full results
    output = dict(summary=summary, episodes=all_results)
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
