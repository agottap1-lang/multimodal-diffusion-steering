#!/usr/bin/env python3
"""
Proper VLM-guided trajectory selection with 5-second candidate evaluation.

Key Insight: VLM must see FULL reaching motion (5 seconds) to judge legibility,
not just first action chunk. Generate long trajectory candidates, score them,
then execute the most legible high-arc option.
"""

import argparse
import io
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler
from scripts.vlm_client import LegibilityScorer


def stabilize_gripper(actions: np.ndarray) -> np.ndarray:
    """Lock gripper to first action's value."""
    out = actions.copy()
    if out.shape[1] >= 5:
        fixed = float(np.clip(out[0, 4], -1.0, 1.0))
        out[:, 4] = fixed
    return out


def measure_arc(obs_trajectory: np.ndarray) -> float:
    """Measure arc from executed trajectory observations.
    
    Args:
        obs_trajectory: (T, obs_dim) array where obs[1] is EE Y position
    
    Returns:
        Maximum lateral |Y| position reached during trajectory
    """
    if len(obs_trajectory) == 0:
        return 0.0
    ee_y_positions = np.abs(obs_trajectory[:, 1])  # EE Y coordinate
    return float(np.max(ee_y_positions))


def arc_class(arc: float) -> str:
    """Classify arc into legibility ranges based on max lateral position."""
    if arc < 0.08:
        return "00-05"
    if arc < 0.12:
        return "10-14"
    return "15-19"


def infer_block_from_action(actions: np.ndarray) -> str:
    """Infer target block from initial motion direction."""
    avg_dy = float(np.mean(actions[:20, 1]))  # First 20 steps
    return "LEFT" if avg_dy > 0 else "RIGHT"


def enforce_block_direction(actions: np.ndarray, target_block: str) -> np.ndarray:
    """Enforce initial motion toward target block."""
    out = actions.copy()
    sign = 1.0 if target_block == "LEFT" else -1.0
    lock_steps = 15  # First 0.5 seconds
    out[:lock_steps, 1] = sign * np.abs(out[:lock_steps, 1])
    return out


def generate_long_trajectory(
    model,
    sampler,
    obs,
    obs_mean,
    obs_std,
    act_mean,
    act_std,
    device,
    n_steps: int = 150,
    temperature: float = 1.0,
    noise_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a long trajectory by chaining multiple diffusion samples.
    
    Strategy: Sample horizon-length chunks and concatenate them to form
    a longer trajectory that VLM can evaluate.
    """
    obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Generate initial noise with optional seed for reproducibility/diversity
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    
    with torch.no_grad():
        # Sample first chunk
        initial_noise = torch.randn(1, model.horizon, model.act_dim, device=device) * temperature
        seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=1.0, initial_noise=initial_noise)[0].cpu().numpy()
    
    seq = seq * act_std + act_mean
    seq = stabilize_gripper(seq)
    
    # If we need more steps, repeat sampling to extend trajectory
    trajectory = [seq]
    while len(np.concatenate(trajectory, axis=0)) < n_steps:
        with torch.no_grad():
            noise = torch.randn(1, model.horizon, model.act_dim, device=device) * temperature
            seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=1.0, initial_noise=noise)[0].cpu().numpy()
        seq = seq * act_std + act_mean
        seq = stabilize_gripper(seq)
        trajectory.append(seq)
    
    full_traj = np.concatenate(trajectory, axis=0)[:n_steps]
    return full_traj


def execute_and_capture_trajectory(
    env: TwoBlockPickEnv,
    actions: np.ndarray,
    capture_times: List[int] = [0, 30, 60, 90, 120, 150],
) -> Dict:
    """
    Execute trajectory and capture frames at specified timesteps.
    
    Returns dict with frames, success status, and picked block.
    """
    env.reset()
    frames = []
    result = None
    
    for step in range(len(actions)):
        if step in capture_times:
            frame = env.render(width=480, height=480)
            frames.append(Image.fromarray(frame))
        
        action = actions[step]
        result = env.step(action)
        
        if result.done:
            break
    
    # Pad with black frames if trajectory ended early
    while len(frames) < len(capture_times):
        frames.append(Image.new("RGB", (480, 480), "black"))
    
    # Extract success info
    info = result.info if result is not None else {}
    success = info.get('success', False)
    picked_left = info.get('picked_left', False)
    picked_right = info.get('picked_right', False)
    
    if picked_left and not picked_right:
        picked = "LEFT"
    elif picked_right and not picked_left:
        picked = "RIGHT"
    elif picked_left and picked_right:
        picked = "BOTH"
    else:
        picked = "NONE"
    
    return {
        "frames": frames,
        "success": success,
        "picked": picked,
        "steps": step + 1 if result else len(actions),
    }


def vlm_score_trajectory(
    scorer: LegibilityScorer,
    frames: List[Image.Image],
    target_block: str,
    video_id: str,
) -> Dict:
    """Score trajectory using VLM prefix_frames mode."""
    goal_A = f"pick {target_block} block"
    goal_B = "pick RIGHT block" if target_block == "LEFT" else "pick LEFT block"
    
    # Convert frames to bytes
    frames_bytes = []
    for frame in frames:
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=90)
        frames_bytes.append(buf.getvalue())
    
    # Score with VLM (prefix frames: t=0→1 frame, t=1→2 frames, etc.)
    result = scorer.score_trajectory(
        frames_bytes,
        goal_A=goal_A,
        goal_B=goal_B,
        mode='prefix_frames',
        video_id=video_id,
        t_sec=[0, 1, 2, 3, 4, 5],
    )
    
    return result


def rollout_baseline(
    checkpoint_path: str,
    episode_seed: int,
    rollout_seed: int,
    device: torch.device,
    max_steps: int = 400,
) -> Dict:
    """Run baseline policy with replanning to get successful trajectory."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    
    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3)
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    sampler = DDIMSampler(
        n_steps=cfg['n_diffusion_steps'],
        beta_start=cfg['beta_start'],
        beta_end=cfg['beta_end'],
        device=device
    )
    
    obs_mean = torch.tensor(ckpt['obs_mean'], device=device)
    obs_std = torch.tensor(ckpt['obs_std'], device=device)
    act_mean = ckpt['act_mean']
    act_std = ckpt['act_std']
    
    # Set seeds
    np.random.seed(rollout_seed)
    torch.manual_seed(rollout_seed)
    
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset(seed=episode_seed)
    
    action_queue = deque(maxlen=model.horizon)
    action_history = []
    observations = []
    done = False
    steps = 0
    result = None
    
    while not done and steps < max_steps:
        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=1.0)[0].cpu().numpy()
            
            seq = seq * act_std + act_mean
            seq = stabilize_gripper(seq)
            
            for action in seq:
                action_queue.append(action)
        
        action = action_queue.popleft()
        result = env.step(action)
        action_history.append(action)
        observations.append(result.obs)
        obs = result.obs
        done = bool(result.done)
        steps += 1
    
    env.close()
    
    # Extract result
    info = result.info if result is not None else {}
    success = info.get('success', False)
    picked_left = info.get('picked_left', False)
    picked_right = info.get('picked_right', False)
    
    if picked_left and not picked_right:
        picked = "LEFT"
    elif picked_right and not picked_left:
        picked = "RIGHT"
    elif picked_left and picked_right:
        picked = "BOTH"
    else:
        picked = "NONE"
    
    # Measure arc from first 150 steps of observations
    first_observations = np.array(observations[:150])
    arc = measure_arc(first_observations)
    
    return {
        "success": success,
        "picked": picked,
        "steps": steps,
        "arc": float(arc),
        "arc_class": arc_class(arc),
    }


def rollout_vlm_guided(
    checkpoint_path: str,
    episode_seed: int,
    rollout_seed: int,
    target_block: str,
    device: torch.device,
    n_candidates: int = 10,
    legibility_threshold: float = 0.70,
    arc15_threshold: float = 0.15,
    max_arc: float = 0.85,
    max_steps: int = 400,
) -> Dict:
    """Run VLM-guided policy with 5-second trajectory candidate selection."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    
    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3)
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    sampler = DDIMSampler(
        n_steps=cfg['n_diffusion_steps'],
        beta_start=cfg['beta_start'],
        beta_end=cfg['beta_end'],
        device=device
    )
    
    obs_mean = torch.tensor(ckpt['obs_mean'], device=device)
    obs_std = torch.tensor(ckpt['obs_std'], device=device)
    act_mean = ckpt['act_mean']
    act_std = ckpt['act_std']
    
    scorer = LegibilityScorer(model="gemini-2.5-flash")
    
    # Set seeds
    np.random.seed(rollout_seed)
    torch.manual_seed(rollout_seed)
    
    env_initial = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env_initial.reset(seed=episode_seed)
    
    # ========================================================================
    # PHASE 1: Generate and score 5-second trajectory candidates
    # ========================================================================
    print(f"  Generating {n_candidates} 5-second trajectory candidates...")
    
    candidates = []
    for idx in range(n_candidates):
        # Generate long trajectory with diversity
        temperature = np.linspace(0.8, 2.0, n_candidates)[idx]
        noise_seed = rollout_seed + idx + 1000
        
        trajectory = generate_long_trajectory(
            model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device,
            n_steps=150,  # 5 seconds @ 30Hz
            temperature=temperature,
            noise_seed=noise_seed,
        )
        
        # Enforce block direction
        trajectory = enforce_block_direction(trajectory, target_block)
        
        # Execute in separate env and capture frames
        env_sim = TwoBlockPickEnv(render=False, episode_length=200, cube_jitter=0.0)
        env_sim.reset(seed=episode_seed)
        
        exec_result = execute_and_capture_trajectory(
            env_sim,
            trajectory,
            capture_times=[0, 30, 60, 90, 120, 150],  # t=0,1,2,3,4,5 sec
        )
        
        env_sim.close()
        
        # Measure arc from executed observations
        arc = measure_arc(exec_result["observations"])
        
        # Score with VLM
        vlm_result = vlm_score_trajectory(
            scorer,
            exec_result["frames"],
            target_block,
            f"ep{episode_seed}_cand{idx}",
        )
        
        candidates.append({
            "idx": idx,
            "trajectory": trajectory,
            "arc": arc,
            "arc_class": arc_class(arc),
            "legibility": float(vlm_result.get('legibility_score', 0.5)),
            "vlm_choice": vlm_result.get('choice', 'NONE'),
            "temperature": temperature,
        })
        
        print(f"    Cand {idx}: arc={arc:.4f} ({arc_class(arc)}), leg={vlm_result.get('legibility_score', 0.5):.3f}, temp={temperature:.2f}")
    
    # ========================================================================
    # PHASE 2: Select best trajectory
    # ========================================================================
    
    # Filter: legible + arc-15 + not too extreme
    arc15_legible = [
        c for c in candidates
        if c["legibility"] >= legibility_threshold
        and c["arc"] >= arc15_threshold
        and c["arc"] <= max_arc
    ]
    
    if arc15_legible:
        chosen = max(arc15_legible, key=lambda c: c["arc"])
        selection_method = "arc15_legible"
    else:
        # Fallback: legible with moderate arc
        legible = [
            c for c in candidates
            if c["legibility"] >= legibility_threshold
            and c["arc"] <= max_arc
        ]
        if legible:
            chosen = max(legible, key=lambda c: c["arc"])
            selection_method = "legible_moderate"
        else:
            # Last resort: highest legibility
            chosen = max(candidates, key=lambda c: c["legibility"])
            selection_method = "highest_legibility"
    
    print(f"  -> Selected cand {chosen['idx']}: arc={chosen['arc']:.4f}, leg={chosen['legibility']:.3f}, method={selection_method}")
    
    # ========================================================================
    # PHASE 3: Execute selected trajectory with replanning safety net
    # ========================================================================
    
    env_exec = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env_exec.reset(seed=episode_seed)
    
    # Use selected trajectory as initial plan
    action_queue = deque(chosen["trajectory"], maxlen=150)
    action_history = []
    done = False
    steps = 0
    result = None
    
    while not done and steps < max_steps:
        if len(action_queue) == 0:
            # Replan if we run out of actions
            obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=1.0)[0].cpu().numpy()
            
            seq = seq * act_std + act_mean
            seq = stabilize_gripper(seq)
            seq = enforce_block_direction(seq, target_block)
            
            for action in seq:
                action_queue.append(action)
        
        action = action_queue.popleft()
        result = env_exec.step(action)
        action_history.append(action)
        obs = result.obs
        done = bool(result.done)
        steps += 1
    
    env_exec.close()
    env_initial.close()
    
    # Extract result
    info = result.info if result is not None else {}
    success = info.get('success', False)
    picked_left = info.get('picked_left', False)
    picked_right = info.get('picked_right', False)
    
    if picked_left and not picked_right:
        picked = "LEFT"
    elif picked_right and not picked_left:
        picked = "RIGHT"
    elif picked_left and picked_right:
        picked = "BOTH"
    else:
        picked = "NONE"
    
    return {
        "success": success,
        "picked": picked,
        "steps": steps,
        "arc": float(chosen["arc"]),
        "arc_class": chosen["arc_class"],
        "legibility": float(chosen["legibility"]),
        "selection_method": selection_method,
        "n_candidates": n_candidates,
        "all_candidates": [
            {
                "idx": c["idx"],
                "arc": float(c["arc"]),
                "legibility": float(c["legibility"]),
            }
            for c in candidates
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/paired_proper_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PROPER VLM-GUIDED TRAJECTORY SELECTION")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print()
    
    accepted_pairs = []
    attempt = 0
    
    while len(accepted_pairs) < args.episodes and attempt < args.max_attempts:
        attempt += 1
        episode_seed = args.seed + attempt
        rollout_seed = args.seed + attempt + 10000
        
        print("-" * 80)
        print(f"Attempt {attempt}/{args.max_attempts} | episode_seed={episode_seed}")
        
        # Baseline rollout
        baseline = rollout_baseline(args.checkpoint, episode_seed, rollout_seed, device)
        
        print(f"  Baseline: success={baseline['success']}, picked={baseline['picked']}, arc={baseline['arc']:.4f} ({baseline['arc_class']})")
        
        if not baseline["success"] or baseline["picked"] not in ["LEFT", "RIGHT"]:
            print("  -> reject (baseline not successful single-block pick)")
            continue
        
        # VLM-guided rollout
        guided = rollout_vlm_guided(
            args.checkpoint,
            episode_seed,
            rollout_seed,
            baseline["picked"],
            device,
            n_candidates=args.n_candidates,
        )
        
        print(f"  Guided: success={guided['success']}, picked={guided['picked']}, arc={guided['arc']:.4f}, leg={guided['legibility']:.3f}, method={guided['selection_method']}")
        
        if not guided["success"] or guided["picked"] != baseline["picked"]:
            print("  -> reject (guided not successful same-block pick)")
            continue
        
        print(f"  -> ACCEPTED pair {len(accepted_pairs) + 1}")
        
        # Save pair record
        pair_idx = len(accepted_pairs) + 1
        pair_record = {
            "pair_idx": pair_idx,
            "episode_seed": episode_seed,
            "baseline": baseline,
            "guided": guided,
            "accepted": True,
        }
        
        accepted_pairs.append(pair_record)
        
        pair_file = output_dir / f"pair_{pair_idx:02d}.json"
        with open(pair_file, "w") as f:
            json.dump(pair_record, f, indent=2)
    
    # Create summary
    if accepted_pairs:
        same_block_rate = 1.0  # By construction
        arc15_rate = sum(1 for p in accepted_pairs if p["guided"]["arc"] >= 0.15) / len(accepted_pairs)
        mean_legibility = sum(p["guided"]["legibility"] for p in accepted_pairs) / len(accepted_pairs)
        baseline_arc_mean = sum(p["baseline"]["arc"] for p in accepted_pairs) / len(accepted_pairs)
        guided_arc_mean = sum(p["guided"]["arc"] for p in accepted_pairs) / len(accepted_pairs)
        
        summary = {
            "experiment": "proper_vlm_guided_trajectory_selection",
            "total_accepted": len(accepted_pairs),
            "total_attempts": attempt,
            "same_block_rate": same_block_rate,
            "guided_arc15_rate": arc15_rate,
            "baseline_arc_mean": baseline_arc_mean,
            "guided_arc_mean": guided_arc_mean,
            "guided_legibility_mean": mean_legibility,
            "pairs": accepted_pairs,
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print()
        print("=" * 80)
        print("FINISHED")
        print("=" * 80)
        print(f"Accepted pairs: {len(accepted_pairs)}/{args.episodes}")
        print(f"Same-block rate: {same_block_rate:.1%} | Guided arc15 rate: {arc15_rate:.1%} | Guided mean legibility: {mean_legibility:.3f}")
        print(f"Baseline arc mean: {baseline_arc_mean:.4f} | Guided arc mean: {guided_arc_mean:.4f}")
    else:
        print("\n❌ No pairs accepted")


if __name__ == "__main__":
    main()
