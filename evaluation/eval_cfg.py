#!/usr/bin/env python3
"""
CFG Evaluation Pipeline for V2 (26-d obs) Diffusion Policy
===========================================================

Evaluates 4 behaviors using Classifier-Free Guidance at inference:
  1. Legibility   (lambda=+2)  - curved Bezier arcs toward target
  2. Predictability (lambda=-2) - straight shortest path
  3. Safety       (lambda=+/-1) - avoid obstacle via curve or straight
  4. Grounding    (lambda=0)   - hover over waypoint then pick

CFG inference formula:
  eps = eps_uncond + lambda * (eps_cond - eps_uncond)
  where eps_uncond is computed with conditioning dims zeroed out.

Usage:
  .venv\\Scripts\\python.exe evaluation/eval_cfg.py ^
      --checkpoint runs/cfg_YYYYMMDD_HHMMSS/ckpt_ep200.pt ^
      --behavior legibility --n_episodes 20

  .venv\\Scripts\\python.exe evaluation/eval_cfg.py ^
      --checkpoint runs/cfg_YYYYMMDD_HHMMSS/ckpt_ep200.pt ^
      --behavior all --n_episodes 10
"""

from __future__ import annotations

import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pybullet as p
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from envs.twoblockpick_env import TwoBlockPickEnv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ─────────────────────────────────────────────────────────
OBS_DIM_V2       = 26
ACT_DIM          = 5
CFG_COND_START   = 22    # indices [22:26] = context + mode
TABLE_TOP_Z      = 0.4
OBSTACLE_RADIUS  = 0.035
OBSTACLE_HEIGHT  = 0.18
OBSTACLE_COLOR   = [0.0, 0.85, 0.85, 1]

# 5 waypoint blocks in pentagon (same as collect_demos_v2)
def _build_waypoint_blocks():
    center_x, center_y = 0.43, 0.0
    radius = 0.06
    colors = [
        ("blue",   [0.1, 0.2, 0.95, 1]),
        ("green",  [0.1, 0.85, 0.1, 1]),
        ("yellow", [0.95, 0.9, 0.1, 1]),
        ("orange", [1.0, 0.55, 0.05, 1]),
        ("purple", [0.65, 0.1, 0.85, 1]),
    ]
    blocks = []
    for i, (name, rgba) in enumerate(colors):
        angle = 2 * math.pi * i / 5 + math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        blocks.append({"name": name,
                       "pos": [round(x, 4), round(y, 4), TABLE_TOP_Z + 0.016],
                       "rgba": rgba})
    return blocks

WAYPOINT_BLOCKS = _build_waypoint_blocks()


# ══════════════════════════════════════════════════════════════════════
# MODEL (must match train_cfg.py exactly)
# ══════════════════════════════════════════════════════════════════════

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
        h = self.conv1(x)
        h = h.transpose(1, 2); h = self.norm1(h); h = h.transpose(1, 2)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        h = self.conv2(h)
        h = h.transpose(1, 2); h = self.norm2(h); h = h.transpose(1, 2)
        return self.act(h + self.shortcut(x))

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.obs_dim = obs_dim; self.act_dim = act_dim; self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128), nn.Linear(128, hidden_dim),
            nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i+1], hidden_dim)
            for i in range(len(dims) - 1)])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i+1] + dims[i+1], dims[i], hidden_dim)
            for i in range(len(dims) - 2, -1, -1)])
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


# ══════════════════════════════════════════════════════════════════════
# CFG DDIM SAMPLER
# ══════════════════════════════════════════════════════════════════════

class CFGDDIMSampler:
    """DDIM sampler with Classifier-Free Guidance.

    At each denoising step:
      eps = eps_uncond + cfg_lambda * (eps_cond - eps_uncond)

    cfg_lambda > 0: amplifies conditioning (e.g., more legible/curved)
    cfg_lambda < 0: inverts conditioning (e.g., more predictable/straight)
    cfg_lambda = 0: unconditional (baseline)
    cfg_lambda = 1: standard conditional (no amplification)
    """

    def __init__(self, n_steps, beta_start, beta_end, device, eta=0.5):
        self.device = device
        self.eta = eta
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs_cond, obs_uncond, cfg_lambda=1.0,
               n_sampling_steps=20):
        """Sample with CFG.

        obs_cond:   (B, obs_dim) normalized obs WITH conditioning
        obs_uncond: (B, obs_dim) normalized obs with cond dims zeroed
        cfg_lambda: guidance strength
        """
        B = obs_cond.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)

        timesteps = torch.flip(
            torch.linspace(0, len(self.alphas_cumprod) - 1,
                           n_sampling_steps, device=self.device).long(), [0])

        for i, t in enumerate(timesteps):
            t_b = t.repeat(B)

            # Two forward passes: conditional and unconditional
            eps_cond = model(x, t_b, obs_cond)
            eps_uncond = model(x, t_b, obs_uncond)

            # CFG combination
            eps = eps_uncond + cfg_lambda * (eps_cond - eps_uncond)

            a_t = self.alphas_cumprod[t]
            a_prev = (self.alphas_cumprod[timesteps[i + 1]]
                      if i < len(timesteps) - 1
                      else torch.tensor(1.0, device=self.device))

            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)

            if i < len(timesteps) - 1:
                sigma = self.eta * torch.sqrt(
                    (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
                dir_xt = torch.sqrt(
                    torch.clamp(1 - a_prev - sigma ** 2, min=0)) * eps
                noise = torch.randn_like(x) * sigma
                x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
            else:
                x = x0_pred

        return x


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADING
# ══════════════════════════════════════════════════════════════════════

def load_policy(ckpt_path, device, eta=0.5):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    obs_dim = ckpt.get('obs_dim', OBS_DIM_V2)
    act_dim = ckpt.get('act_dim', ACT_DIM)
    horizon = ckpt.get('horizon', 32)
    hidden_dim = ckpt.get('hidden_dim', 256)
    n_blocks = ckpt.get('n_blocks', 6)

    model = DiffusionPolicy(obs_dim, act_dim, horizon, hidden_dim, n_blocks).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    sampler = CFGDDIMSampler(
        n_steps=ckpt.get('n_diffusion_steps', 100),
        beta_start=ckpt.get('beta_start', 1e-4),
        beta_end=ckpt.get('beta_end', 0.1),
        device=device, eta=eta)

    stats = {
        'obs_mean': ckpt['obs_mean'],
        'obs_std':  ckpt['obs_std'],
        'act_mean': ckpt['act_mean'],
        'act_std':  ckpt['act_std'],
    }
    return model, sampler, stats, ckpt


# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT HELPERS
# ══════════════════════════════════════════════════════════════════════

def add_obstacle_visual(env, pos, rgba=None):
    cid = env._cid
    rgba = rgba or OBSTACLE_COLOR
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=OBSTACLE_RADIUS,
                              length=OBSTACLE_HEIGHT, rgbaColor=rgba,
                              physicsClientId=cid)
    uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                            baseVisualShapeIndex=vis,
                            basePosition=pos, physicsClientId=cid)
    return uid

def add_waypoint_blocks(env, wp_list):
    cid = env._cid
    uids = []
    for wp in wp_list:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                  rgbaColor=wp["rgba"], physicsClientId=cid)
        uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=wp["pos"], physicsClientId=cid)
        uids.append(uid)
    return uids

def remove_bodies(env, uids):
    for uid in uids:
        p.removeBody(uid, physicsClientId=env._cid)

def get_obs_v2(env, context_pos=None, behavior_mode=0.0):
    base = env._get_obs()
    ctx = np.zeros(3, dtype=np.float32) if context_pos is None else np.array(context_pos, dtype=np.float32)
    mode = np.array([behavior_mode], dtype=np.float32)
    return np.concatenate([base, ctx, mode])


# ══════════════════════════════════════════════════════════════════════
# L_EARLY METRIC
# ══════════════════════════════════════════════════════════════════════

def compute_l_early(ee_traj, left_pos, right_pos, early_frac=0.3):
    """Bayesian intent prediction from early trajectory."""
    if len(ee_traj) < 4:
        return 0.5
    goals = np.array([left_pos, right_pos])
    T = len(ee_traj)
    early_end = max(2, int(T * early_frac))
    sigma = np.linalg.norm(goals[0] - goals[1]) / (2 * np.sqrt(2 * np.log(2)))
    posteriors = []
    for t in range(early_end):
        ee = ee_traj[t]
        dists = np.array([np.linalg.norm(ee - g) for g in goals])
        likes = np.exp(-0.5 * (dists / max(sigma, 1e-6)) ** 2)
        post = likes / (likes.sum() + 1e-12)
        posteriors.append(post)
    # L_early = average confidence toward the true goal (index 0 = left)
    return float(np.mean([p[0] for p in posteriors]))


# ══════════════════════════════════════════════════════════════════════
# ROLLOUT WITH CFG
# ══════════════════════════════════════════════════════════════════════

def rollout_cfg(env, model, sampler, stats, context_pos, behavior_mode,
                cfg_lambda, target="left", max_steps=400,
                n_action_steps=8, n_sampling_steps=20,
                video_path=None):
    """Run one episode with CFG inference."""
    device = next(model.parameters()).device
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    act_mean, act_std = stats['act_mean'], stats['act_std']
    horizon = model.horizon

    if video_path:
        env.record_video(str(video_path))

    obs_22 = env._get_obs()
    ee_traj = []
    total_steps = 0
    success = False

    for chunk_start in range(0, max_steps, n_action_steps):
        obs_v2 = get_obs_v2(env, context_pos, behavior_mode)
        obs_norm = (obs_v2 - obs_mean) / obs_std
        obs_cond = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

        # Unconditional obs: zero out conditioning dims (after normalization)
        obs_uncond_np = obs_norm.copy()
        obs_uncond_np[CFG_COND_START:] = 0.0  # zeroed = mean after z-score
        obs_uncond = torch.tensor(obs_uncond_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Sample action chunk with CFG
        act_chunk = sampler.sample(model, obs_cond, obs_uncond,
                                   cfg_lambda=cfg_lambda,
                                   n_sampling_steps=n_sampling_steps)
        act_chunk = act_chunk[0].cpu().numpy()  # (H, A)

        # Denormalize
        act_chunk = act_chunk * act_std + act_mean

        # Execute n_action_steps
        for j in range(min(n_action_steps, horizon)):
            if total_steps >= max_steps:
                break
            action = act_chunk[j]
            action[:4] = np.clip(action[:4], -1, 1)
            action[4] = np.clip(action[4], -1, 1)

            result = env.step(action)
            obs_22 = result.obs
            ee_traj.append(obs_22[:3].copy())
            total_steps += 1

            if result.done:
                success = (result.info.get("success_left", 0) > 0.5 or
                           result.info.get("success_right", 0) > 0.5)
                break

        if result.done:
            break

    if video_path:
        env.stop_video()

    return {
        "success": success,
        "total_steps": total_steps,
        "ee_traj": np.array(ee_traj) if ee_traj else np.zeros((0, 3)),
        "final_obs": obs_22,
    }


# ══════════════════════════════════════════════════════════════════════
# BEHAVIOR EVALUATION
# ══════════════════════════════════════════════════════════════════════

def eval_legibility(env, model, sampler, stats, n_episodes, cfg_lambda,
                    out_dir, seeds):
    """Legible behavior: curved Bezier arcs. mode=+1, lambda > 0."""
    results = []
    for i, seed in enumerate(seeds[:n_episodes]):
        env.reset(seed=int(seed))
        target = "left" if i % 2 == 0 else "right"
        vpath = out_dir / f"legible_{i:03d}_{target}.mp4"

        r = rollout_cfg(env, model, sampler, stats,
                        context_pos=None, behavior_mode=1.0,
                        cfg_lambda=cfg_lambda, target=target,
                        video_path=vpath)

        left_pos = r["final_obs"][8:11]
        right_pos = r["final_obs"][15:18]
        l_early = compute_l_early(r["ee_traj"], left_pos, right_pos)
        if target == "right":
            l_early = 1.0 - l_early  # flip for right target

        results.append({
            "episode": i, "target": target,
            "success": r["success"], "steps": r["total_steps"],
            "L_early": round(l_early, 4),
        })
        tag = "OK" if r["success"] else "FAIL"
        print(f"  [{i+1}/{n_episodes}] {tag} target={target} "
              f"L_early={l_early:.3f} steps={r['total_steps']}")

    return results


def eval_predictability(env, model, sampler, stats, n_episodes, cfg_lambda,
                        out_dir, seeds):
    """Predictable: straight shortest path. mode=-1, lambda < 0."""
    results = []
    for i, seed in enumerate(seeds[:n_episodes]):
        env.reset(seed=int(seed))
        target = "left" if i % 2 == 0 else "right"
        vpath = out_dir / f"predictable_{i:03d}_{target}.mp4"

        r = rollout_cfg(env, model, sampler, stats,
                        context_pos=None, behavior_mode=-1.0,
                        cfg_lambda=cfg_lambda, target=target,
                        video_path=vpath)

        # Path efficiency: ratio of straight-line distance to actual path length
        traj = r["ee_traj"]
        if len(traj) > 1:
            straight_dist = np.linalg.norm(traj[-1] - traj[0])
            path_len = sum(np.linalg.norm(traj[j+1] - traj[j])
                           for j in range(len(traj)-1))
            efficiency = straight_dist / max(path_len, 1e-6)
        else:
            efficiency = 0

        results.append({
            "episode": i, "target": target,
            "success": r["success"], "steps": r["total_steps"],
            "path_efficiency": round(efficiency, 4),
        })
        tag = "OK" if r["success"] else "FAIL"
        print(f"  [{i+1}/{n_episodes}] {tag} target={target} "
              f"efficiency={efficiency:.3f} steps={r['total_steps']}")

    return results


def eval_safety(env, model, sampler, stats, n_episodes, out_dir, seeds):
    """Safety: obstacle on path → choose alternative route.

    Two sub-cases tested:
      - Obstacle on straight path → model should curve (legible, lambda>0)
      - Obstacle on curved path  → model should go straight (pred, lambda<0)
    """
    results = []
    for i, seed in enumerate(seeds[:n_episodes]):
        env.reset(seed=int(seed))
        target = "left" if i % 2 == 0 else "right"

        # Alternate between the two safety scenarios
        if i % 4 < 2:
            # Obstacle on straight path → should take curved (legible) route
            safety_type = "legible"
            cfg_lambda = 2.0
            mode = 1.0
            y = 0.02 if target == "left" else -0.02
            obs_pos = [0.46, y, TABLE_TOP_Z + OBSTACLE_HEIGHT / 2]
        else:
            # Obstacle on curved path → should take straight (pred) route
            safety_type = "predictable"
            cfg_lambda = -2.0
            mode = -1.0
            y = 0.11 if target == "left" else -0.11
            obs_pos = [0.38, y, TABLE_TOP_Z + OBSTACLE_HEIGHT / 2]

        uid = add_obstacle_visual(env, obs_pos)
        for _ in range(30):
            p.stepSimulation(physicsClientId=env._cid)

        vpath = out_dir / f"safety_{i:03d}_{safety_type}_{target}.mp4"

        r = rollout_cfg(env, model, sampler, stats,
                        context_pos=obs_pos, behavior_mode=mode,
                        cfg_lambda=cfg_lambda, target=target,
                        video_path=vpath)

        # Check if trajectory stayed clear of obstacle
        traj = r["ee_traj"]
        obs_np = np.array(obs_pos)
        if len(traj) > 0:
            dists = np.linalg.norm(traj[:, :2] - obs_np[:2], axis=1)
            min_clearance = float(dists.min())
            collision = min_clearance < OBSTACLE_RADIUS
        else:
            min_clearance = 0; collision = True

        remove_bodies(env, [uid])

        results.append({
            "episode": i, "target": target, "safety_type": safety_type,
            "success": r["success"], "steps": r["total_steps"],
            "min_clearance": round(min_clearance, 4),
            "collision": collision,
        })
        tag = "OK" if r["success"] and not collision else "FAIL"
        print(f"  [{i+1}/{n_episodes}] {tag} type={safety_type} target={target} "
              f"clearance={min_clearance:.3f} collision={collision} "
              f"steps={r['total_steps']}")

    return results


def eval_grounding(env, model, sampler, stats, n_episodes, cfg_lambda,
                   out_dir, seeds):
    """Grounding: hover over waypoint then pick. mode=0."""
    results = []
    for i, seed in enumerate(seeds[:n_episodes]):
        env.reset(seed=int(seed))
        target = "left" if i % 2 == 0 else "right"
        wp = WAYPOINT_BLOCKS[i % len(WAYPOINT_BLOCKS)]

        p.changeVisualShape(env._cube_l_uid, -1,
                            rgbaColor=[0.1, 0.8, 0.1, 1.0],
                            physicsClientId=env._cid)
        p.changeVisualShape(env._cube_r_uid, -1,
                            rgbaColor=[0.8, 0.1, 0.1, 1.0],
                            physicsClientId=env._cid)
        wp_uids = add_waypoint_blocks(env, WAYPOINT_BLOCKS)
        for _ in range(30):
            p.stepSimulation(physicsClientId=env._cid)

        vpath = out_dir / f"grounding_{i:03d}_{wp['name']}_{target}.mp4"

        r = rollout_cfg(env, model, sampler, stats,
                        context_pos=wp["pos"], behavior_mode=0.0,
                        cfg_lambda=cfg_lambda, target=target,
                        video_path=vpath)

        # Check hovering: did EE pass close to the waypoint?
        traj = r["ee_traj"]
        wp_pos = np.array(wp["pos"])
        if len(traj) > 0:
            dists = np.linalg.norm(traj - wp_pos, axis=1)
            min_wp_dist = float(dists.min())
            hovered = min_wp_dist < 0.06  # within 6cm of waypoint
        else:
            min_wp_dist = 999; hovered = False

        remove_bodies(env, wp_uids)

        results.append({
            "episode": i, "target": target, "waypoint": wp["name"],
            "success": r["success"], "steps": r["total_steps"],
            "min_wp_dist": round(min_wp_dist, 4),
            "hovered": hovered,
        })
        tag = "OK" if r["success"] and hovered else "FAIL"
        print(f"  [{i+1}/{n_episodes}] {tag} wp={wp['name']} target={target} "
              f"wp_dist={min_wp_dist:.3f} hovered={hovered} "
              f"steps={r['total_steps']}")

    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("CFG Behavior Evaluation (26-d V2)")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--behavior', default='all',
                    choices=['legibility', 'predictability', 'safety',
                             'grounding', 'all'])
    ap.add_argument('--n_episodes', type=int, default=20)
    ap.add_argument('--cfg_lambda_leg', type=float, default=2.0,
                    help='CFG lambda for legibility')
    ap.add_argument('--cfg_lambda_pred', type=float, default=-2.0,
                    help='CFG lambda for predictability')
    ap.add_argument('--cfg_lambda_ground', type=float, default=1.0,
                    help='CFG lambda for grounding')
    ap.add_argument('--eta', type=float, default=0.5,
                    help='DDIM stochasticity')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model, sampler, stats, ckpt = load_policy(args.checkpoint, device, args.eta)
    print(f"Model: obs_dim={model.obs_dim}, act_dim={model.act_dim}, "
          f"horizon={model.horizon}")
    print(f"Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?')}")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/eval_cfg_{ts}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 100000, size=1000)

    env = TwoBlockPickEnv(render=False, cube_jitter=0.0,
                          cube_half=0.015, cube_mass=0.08,
                          cube_lateral_friction=2.5,
                          episode_length=400)

    all_results = {}
    behaviors = (['legibility', 'predictability', 'safety', 'grounding']
                 if args.behavior == 'all'
                 else [args.behavior])

    for beh in behaviors:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {beh.upper()}")
        print(f"{'='*60}")
        beh_dir = out_dir / beh
        beh_dir.mkdir(exist_ok=True)

        if beh == 'legibility':
            res = eval_legibility(env, model, sampler, stats, args.n_episodes,
                                  args.cfg_lambda_leg, beh_dir, seeds)
        elif beh == 'predictability':
            res = eval_predictability(env, model, sampler, stats,
                                      args.n_episodes, args.cfg_lambda_pred,
                                      beh_dir, seeds[200:])
        elif beh == 'safety':
            res = eval_safety(env, model, sampler, stats, args.n_episodes,
                              beh_dir, seeds[400:])
        elif beh == 'grounding':
            res = eval_grounding(env, model, sampler, stats, args.n_episodes,
                                 args.cfg_lambda_ground, beh_dir, seeds[600:])

        # Summary
        n_ok = sum(1 for r in res if r["success"])
        print(f"\n  {beh}: {n_ok}/{len(res)} success ({100*n_ok/max(len(res),1):.0f}%)")

        if beh == 'legibility':
            avg_le = np.mean([r["L_early"] for r in res])
            print(f"  Avg L_early: {avg_le:.3f}")
        elif beh == 'predictability':
            avg_eff = np.mean([r["path_efficiency"] for r in res])
            print(f"  Avg path efficiency: {avg_eff:.3f}")
        elif beh == 'safety':
            n_clear = sum(1 for r in res if not r["collision"])
            print(f"  Obstacle clearance: {n_clear}/{len(res)}")
        elif beh == 'grounding':
            n_hov = sum(1 for r in res if r["hovered"])
            print(f"  Waypoint hover: {n_hov}/{len(res)}")

        all_results[beh] = res

    env.close()

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
