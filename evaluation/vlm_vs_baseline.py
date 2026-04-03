#!/usr/bin/env python3
"""
VLM vs Baseline – Unified Comparison

Flow:
  1) BASELINE: run policy ONCE (with replanning) → full trajectory + video
  2) Record which block baseline targeted (LEFT or RIGHT)
  3) VLM-GUIDED: same episode, same target block
     - Generate up to MAX_ATTEMPTS 5-second trajectories
     - Record a 5-sec VIDEO of each attempt
     - Capture 6 frames (t=0,1,2,3,4,5 s), send to VLM for legibility scoring
     - Save comparison PNG of ALL candidate frames + VLM scores
     - VLM picks the best (highest legibility) candidate
     - Execute that trajectory FULLY (with replanning after 5 s) → video

Outputs per episode (all in --output-dir):
  ep{seed}_baseline.mp4                  – baseline full video
  ep{seed}_attempt_{i:02d}.mp4           – 5-sec attempt videos
  ep{seed}_candidates_all.png            – composite of all attempts + VLM scores
  ep{seed}_vlm_selected.mp4              – full trajectory of VLM pick
  results.json                           – structured results
"""

import argparse
import io
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler
from scripts.vlm_client import LegibilityScorer

# ═══════════════════════════════════════════════════════════════════════
# ARC MEASUREMENT (Bézier-correct thresholds)
# ═══════════════════════════════════════════════════════════════════════

def measure_arc(obs_traj: np.ndarray) -> float:
    """Max lateral |Y| deviation during trajectory."""
    if len(obs_traj) == 0:
        return 0.0
    return float(np.max(np.abs(obs_traj[:, 1])))


def arc_class(arc: float) -> str:
    if arc < 0.0786:
        return "00-04"
    elif arc < 0.1047:
        return "05-09"
    elif arc < 0.1335:
        return "10-14"
    else:
        return "15-19"


def is_arc15_19(arc: float) -> bool:
    return arc >= 0.1335


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def stabilize_gripper(actions: np.ndarray) -> np.ndarray:
    out = actions.copy()
    if out.shape[1] >= 5:
        out[:, 4] = float(np.clip(out[0, 4], -1.0, 1.0))
    return out


def infer_block(actions: np.ndarray) -> str:
    """LEFT if avg initial dy > 0, else RIGHT."""
    avg_dy = float(np.mean(actions[:20, 1]))
    return "LEFT" if avg_dy > 0 else "RIGHT"


def enforce_block_direction(actions: np.ndarray, target: str) -> np.ndarray:
    out = actions.copy()
    sign = 1.0 if target == "LEFT" else -1.0
    out[:15, 1] = sign * np.abs(out[:15, 1])
    return out


def load_policy(ckpt_path: str, device: torch.device):
    """Return (model, sampler, obs_mean, obs_std, act_mean, act_std, cfg)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg.get("hidden_dim", 256),
        n_blocks=cfg.get("n_blocks", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sampler = DDIMSampler(
        n_steps=cfg["n_diffusion_steps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        device=device,
    )
    obs_mean = torch.tensor(ckpt["obs_mean"], device=device)
    obs_std  = torch.tensor(ckpt["obs_std"],  device=device)
    act_mean = ckpt["act_mean"]
    act_std  = ckpt["act_std"]
    return model, sampler, obs_mean, obs_std, act_mean, act_std, cfg


def sample_chunk(model, sampler, obs, obs_mean, obs_std, act_mean, act_std,
                 device, temperature=1.0, seed=None):
    """Sample one action chunk (horizon steps) and denormalize."""
    if seed is not None:
        torch.manual_seed(seed)
    obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
    obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        seq = sampler.sample(model, obs_t, n_sampling_steps=10,
                             temperature=temperature)[0].cpu().numpy()
    seq = seq * act_std + act_mean
    seq = stabilize_gripper(seq)
    return seq


def generate_5sec_trajectory(model, sampler, obs, obs_mean, obs_std,
                             act_mean, act_std, device,
                             temperature=1.0, noise_seed=None):
    """Chain action chunks to produce 150 steps (5 sec @ 30 Hz)."""
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
    obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    horizon = model.horizon
    n_chunks = (150 + horizon - 1) // horizon
    traj = []
    for _ in range(n_chunks):
        with torch.no_grad():
            seq = sampler.sample(model, obs_t, n_sampling_steps=10,
                                 temperature=temperature)[0].cpu().numpy()
        seq = seq * act_std + act_mean
        seq = stabilize_gripper(seq)
        for a in seq:
            if len(traj) < 150:
                traj.append(a)
    return np.array(traj)


# ═══════════════════════════════════════════════════════════════════════
# 1. BASELINE – run once, record full video
# ═══════════════════════════════════════════════════════════════════════

def run_baseline(ckpt_path, episode_seed, rollout_seed, device,
                 output_dir, max_steps=400):
    print(f"\n{'='*78}")
    print(f" BASELINE  (no VLM, single run)")
    print(f"{'='*78}")

    model, sampler, obs_mean, obs_std, act_mean, act_std, cfg = \
        load_policy(ckpt_path, device)

    np.random.seed(rollout_seed)
    torch.manual_seed(rollout_seed)

    video_path = str(output_dir / f"ep{episode_seed}_baseline.mp4")
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    env.record_video(video_path, width=640, height=480, fps=30)
    obs = env.reset(seed=episode_seed)

    action_queue = deque(maxlen=model.horizon)
    observations = []
    first_actions = None
    done = False
    steps = 0
    result = None

    print("  Running baseline with replanning ...")

    while not done and steps < max_steps:
        if len(action_queue) == 0:
            seq = sample_chunk(model, sampler, obs, obs_mean, obs_std,
                               act_mean, act_std, device, temperature=1.0)
            if first_actions is None:
                first_actions = seq.copy()
            for a in seq:
                action_queue.append(a)

        action = action_queue.popleft()
        result = env.step(action)
        observations.append(result.obs)
        obs = result.obs
        done = bool(result.done)
        steps += 1

    env.close()  # flushes video

    info = result.info if result else {}
    picked_left = info.get("picked_left", 0) > 0.5
    picked_right = info.get("picked_right", 0) > 0.5
    success = info.get("success_left", 0) > 0.5 or info.get("success_right", 0) > 0.5

    picked = "LEFT" if (picked_left and not picked_right) else \
             "RIGHT" if (picked_right and not picked_left) else \
             "BOTH" if (picked_left and picked_right) else "NONE"

    target_block = infer_block(first_actions)
    obs_arr = np.array(observations[:150])
    arc = measure_arc(obs_arr)

    print(f"\n  Baseline result:")
    print(f"    Video   : {video_path}")
    print(f"    Success : {success}")
    print(f"    Picked  : {picked}")
    print(f"    Target  : {target_block}")
    print(f"    Arc(5s) : {arc:.4f} m  ({arc_class(arc)})")
    print(f"    Steps   : {steps}")

    return {
        "success": success,
        "picked": picked,
        "target_block": target_block,
        "arc": float(arc),
        "arc_class": arc_class(arc),
        "is_arc15_19": is_arc15_19(arc),
        "steps": steps,
        "video": video_path,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. VLM-GUIDED – generate attempts, score, pick, execute
# ═══════════════════════════════════════════════════════════════════════

def run_vlm_guided(ckpt_path, episode_seed, rollout_seed, target_block,
                   device, output_dir,
                   max_attempts=20, max_steps=400):
    print(f"\n{'='*78}")
    print(f" VLM-GUIDED  (target block = {target_block})")
    print(f"{'='*78}")
    print(f"  Max attempts : {max_attempts}")
    print(f"  Looking for  : arc 15-19 (≥ 0.1335 m)")

    model, sampler, obs_mean, obs_std, act_mean, act_std, cfg = \
        load_policy(ckpt_path, device)

    scorer = LegibilityScorer(model="gemini-2.5-flash")

    # ── Phase 1: generate 5-sec attempts (with replanning, like baseline) ──
    print(f"\n{'─'*78}")
    print(f" PHASE 1  Generate 5-second trajectory attempts (with replanning)")
    print(f"{'─'*78}\n")

    candidates = []
    arc15_count = 0

    for attempt_idx in range(max_attempts):
        noise_seed = rollout_seed + attempt_idx + 1000
        np.random.seed(noise_seed)
        torch.manual_seed(noise_seed)

        # Fresh env per attempt, 5-sec rollout WITH replanning + video
        vid_path = str(output_dir / f"ep{episode_seed}_attempt_{attempt_idx:02d}.mp4")
        env_sim = TwoBlockPickEnv(render=False, episode_length=200, cube_jitter=0.0)
        env_sim.record_video(vid_path, width=640, height=480, fps=30)
        obs = env_sim.reset(seed=episode_seed)

        action_queue = deque(maxlen=model.horizon)
        observations = []
        actions_taken = []
        frames = []
        capture_steps = {0, 30, 60, 90, 120, 149}  # t=0,1,2,3,4,5 sec
        done = False

        for step_idx in range(150):  # 5 seconds @ 30 Hz
            if len(action_queue) == 0:
                # Replan from CURRENT observation (natural diversity)
                seq = sample_chunk(model, sampler, obs, obs_mean, obs_std,
                                   act_mean, act_std, device, temperature=1.0)
                # Enforce target direction on first few steps of each chunk
                seq = enforce_block_direction(seq, target_block)
                for a in seq:
                    action_queue.append(a)

            action = action_queue.popleft()
            result = env_sim.step(action)
            observations.append(result.obs)
            actions_taken.append(action)
            obs = result.obs

            if step_idx in capture_steps:
                frame_rgb = env_sim.render(mode="rgb_array")
                frames.append(Image.fromarray(frame_rgb))

            if result.done:
                done = True
                break

        env_sim.close()  # flushes video

        obs_arr = np.array(observations)
        arc = measure_arc(obs_arr)
        arc_cls = arc_class(arc)
        is15 = is_arc15_19(arc)

        if is15:
            arc15_count += 1

        # Check which direction this trajectory actually goes (env-level)
        traj_arr = np.array(actions_taken)
        actual_dir = infer_block(traj_arr) if len(traj_arr) >= 20 else "UNKNOWN"
        dir_match = (actual_dir == target_block)

        dir_tag = "OK" if dir_match else f"wrong({actual_dir})"
        tag = f"ARC 15-19 !  [{dir_tag}]" if is15 else f"[{dir_tag}]"

        print(f"  Attempt {attempt_idx+1:2d}/{max_attempts}  "
              f"arc={arc:.4f} m ({arc_cls})  {tag}")

        candidates.append({
            "idx": attempt_idx,
            "trajectory": traj_arr,
            "arc": arc,
            "arc_class": arc_cls,
            "is_arc15_19": is15,
            "actual_direction": actual_dir,
            "direction_match": dir_match,
            "temperature": 1.0,
            "frames": frames,
            "video": vid_path,
        })

    print(f"\n  Generated {len(candidates)} attempts, "
          f"arc 15-19 found: {arc15_count}")

    # ── Phase 2: VLM scoring ─────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f" PHASE 2  VLM legibility scoring  ({len(candidates)} candidates)")
    print(f"{'─'*78}\n")

    goal_A = "pick the left block"
    goal_B = "pick the right block"

    for cand in candidates:
        idx = cand["idx"]
        # Encode frames as PNG bytes for VLM
        frames_bytes = []
        for fr in cand["frames"]:
            buf = io.BytesIO()
            fr.save(buf, format="PNG")
            frames_bytes.append(buf.getvalue())

        print(f"  Scoring attempt {idx:2d} ...", end="", flush=True)
        t0 = time.time()

        vlm_result = None
        for retry in range(3):
            try:
                vlm_result = scorer.score_trajectory(
                    image_bytes=frames_bytes,
                    goal_A=goal_A,
                    goal_B=goal_B,
                    mode="prefix_frames",
                    video_id=f"ep{episode_seed}_att{idx}",
                    t_sec=5.0,
                )
                break
            except Exception as exc:
                if retry < 2:
                    wait = (retry + 1) * 10
                    print(f" [retry {retry+1}, wait {wait}s: {exc}]",
                          end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" [FAILED after 3 retries: {exc}]")
                    vlm_result = {
                        "legibility_score": 0.5,
                        "choice": "?",
                        "cue": "VLM_ERROR",
                    }

        dt = time.time() - t0
        leg = float(vlm_result.get("legibility_score", 0.5))
        choice = vlm_result.get("choice", "?")
        cue = vlm_result.get("cue", "")

        pA = float(vlm_result.get("pA", 0.5))
        pB = float(vlm_result.get("pB", 0.5))
        cand["legibility"] = leg
        cand["pA"] = pA  # probability of LEFT
        cand["pB"] = pB  # probability of RIGHT
        cand["vlm_choice"] = choice
        cand["vlm_cue"] = cue
        cand["vlm_latency_ms"] = vlm_result.get("latency_ms", dt * 1000)

        mark = " *" if is_arc15_19(cand["arc"]) else ""
        dir_ok = "OK" if cand["direction_match"] else f"wrong({cand['actual_direction']})"
        print(f"  leg={leg:.3f}  choice={choice}  "
              f"pA={pA:.2f} pB={pB:.2f}  dir={dir_ok}  "
              f"arc={cand['arc']:.4f} ({cand['arc_class']}){mark}  "
              f"[{dt:.1f}s]")

    # ── Phase 3: VLM selection ───────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f" PHASE 3  VLM trajectory selection")
    print(f"{'─'*78}\n")

    # Filter by env-level direction (action Y-axis), NOT VLM label
    correct_dir = [c for c in candidates if c["direction_match"]]
    wrong_dir_count = len(candidates) - len(correct_dir)
    print(f"  Candidates heading toward {target_block} (env-level): "
          f"{len(correct_dir)}/{len(candidates)}")
    if wrong_dir_count > 0:
        print(f"  Rejected {wrong_dir_count} heading wrong direction")

    pool = correct_dir if correct_dir else candidates  # fallback
    # Rank by VLM legibility score (higher = clearer direction visible)
    ranked = sorted(pool, key=lambda c: c["legibility"], reverse=True)

    chosen = ranked[0]
    chosen["selected"] = True

    print(f"\n  VLM PICK  →  attempt {chosen['idx']}")
    print(f"    Legibility : {chosen['legibility']:.3f}")
    print(f"    Arc        : {chosen['arc']:.4f} m ({chosen['arc_class']})")
    print(f"    Is arc15-19: {chosen['is_arc15_19']}")
    print(f"    Direction  : {chosen['actual_direction']} (target: {target_block})")
    print(f"    VLM choice : {chosen['vlm_choice']}  cue: {chosen['vlm_cue']}")

    # ── Phase 3b: Comparison PNG ─────────────────────────────────────
    png_path = output_dir / f"ep{episode_seed}_candidates_all.png"
    _save_comparison_png(candidates, chosen["idx"], episode_seed,
                         target_block, png_path)

    # ── Phase 4: Execute selected trajectory FULLY ───────────────────
    print(f"\n{'─'*78}")
    print(f" PHASE 4  Execute VLM-selected trajectory to completion")
    print(f"{'─'*78}\n")

    vid_final = str(output_dir / f"ep{episode_seed}_vlm_selected.mp4")
    env_exec = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    env_exec.record_video(vid_final, width=640, height=480, fps=30)
    obs = env_exec.reset(seed=episode_seed)

    # First 150 steps: play selected trajectory
    action_queue = deque(chosen["trajectory"], maxlen=150)
    done = False
    steps = 0
    result = None
    replan_count = 0

    print(f"  Playing selected 5-sec trajectory ({len(chosen['trajectory'])} steps) ...")

    while not done and steps < max_steps:
        if len(action_queue) == 0:
            # Replan after initial trajectory exhausted
            replan_count += 1
            seq = sample_chunk(model, sampler, obs, obs_mean, obs_std,
                               act_mean, act_std, device, temperature=1.0)
            seq = enforce_block_direction(seq, target_block)
            for a in seq:
                action_queue.append(a)
            if replan_count <= 3:
                print(f"    Replan #{replan_count} at step {steps}")

        action = action_queue.popleft()
        result = env_exec.step(action)
        obs = result.obs
        done = bool(result.done)
        steps += 1

    env_exec.close()  # flushes video

    info = result.info if result else {}
    picked_left = info.get("picked_left", 0) > 0.5
    picked_right = info.get("picked_right", 0) > 0.5
    success = info.get("success_left", 0) > 0.5 or info.get("success_right", 0) > 0.5

    picked = "LEFT" if (picked_left and not picked_right) else \
             "RIGHT" if (picked_right and not picked_left) else \
             "BOTH" if (picked_left and picked_right) else "NONE"

    print(f"\n  VLM-guided result:")
    print(f"    Video   : {vid_final}")
    print(f"    Success : {success}")
    print(f"    Picked  : {picked}  (target was {target_block})")
    print(f"    Arc     : {chosen['arc']:.4f} m ({chosen['arc_class']})")
    print(f"    Legib.  : {chosen['legibility']:.3f}")
    print(f"    Steps   : {steps}")
    print(f"    Replans : {replan_count}")

    return {
        "success": success,
        "picked": picked,
        "arc": float(chosen["arc"]),
        "arc_class": chosen["arc_class"],
        "is_arc15_19": chosen["is_arc15_19"],
        "legibility": float(chosen["legibility"]),
        "vlm_choice": chosen["vlm_choice"],
        "vlm_cue": chosen["vlm_cue"],
        "selected_attempt": int(chosen["idx"]),
        "total_attempts": len(candidates),
        "arc15_count": arc15_count,
        "steps": steps,
        "replans": replan_count,
        "video": vid_final,
        "all_attempts": [
            {
                "idx": c["idx"],
                "arc": float(c["arc"]),
                "arc_class": c["arc_class"],
                "is_arc15_19": c["is_arc15_19"],
                "legibility": float(c.get("legibility", 0)),
                "vlm_choice": c.get("vlm_choice", ""),
                "temperature": float(c["temperature"]),
                "video": c["video"],
            }
            for c in candidates
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON PNG
# ═══════════════════════════════════════════════════════════════════════

def _save_comparison_png(candidates, selected_idx, episode_seed,
                         target_block, output_path: Path):
    """Grid of all candidates (rows) × 6 frames (cols) + VLM scores."""
    n = len(candidates)
    n_frames = 6
    fig, axes = plt.subplots(n, n_frames, figsize=(20, 2.8 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Episode {episode_seed} – VLM Candidate Comparison  "
        f"(target: {target_block})\n"
        f"Selected: attempt {selected_idx}  |  "
        f"arc 15-19 threshold ≥ 0.1335 m",
        fontsize=14, fontweight="bold", y=1.0,
    )

    for row, cand in enumerate(candidates):
        is_sel = cand["idx"] == selected_idx
        arc_v = cand["arc"]
        arc_c = cand["arc_class"]
        leg = cand.get("legibility", 0)
        frames = cand["frames"]

        for col in range(n_frames):
            ax = axes[row, col]
            if col < len(frames):
                ax.imshow(np.array(frames[col]))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={col}s", fontsize=10)

        # Row label
        colour = "green" if is_sel else ("orange" if cand["is_arc15_19"] else "black")
        weight = "bold" if is_sel else "normal"
        prefix = ">> SELECTED\n" if is_sel else ""
        label = (f"{prefix}Att {cand['idx']}\n"
                 f"arc {arc_v:.3f}m\n({arc_c})\n"
                 f"leg {leg:.2f}")
        axes[row, 0].set_ylabel(label, fontsize=9, color=colour,
                                fontweight=weight, labelpad=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Comparison PNG → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM vs Baseline – unified comparison")
    parser.add_argument("--checkpoint", type=str,
                        default="runs/diffusion_20260222_195530/ckpt_ep100.pt")
    parser.add_argument("--episode-seed", type=int, default=100)
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=20,
                        help="VLM attempts to generate candidates")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Max steps for full trajectory execution")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/vlm_vs_baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'█'*78}")
    print(f"  VLM vs BASELINE COMPARISON")
    print(f"{'█'*78}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Episode seed : {args.episode_seed}")
    print(f"  Rollout seed : {args.rollout_seed}")
    print(f"  Max attempts : {args.max_attempts}")
    print(f"  Max steps    : {args.max_steps}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Device       : {device}")
    print(f"  Timestamp    : {ts}")
    print(f"{'█'*78}\n")

    # ── Step 1: Baseline ──────────────────────────────────────────────
    baseline = run_baseline(
        args.checkpoint, args.episode_seed, args.rollout_seed,
        device, output_dir, max_steps=args.max_steps,
    )

    target_block = baseline["target_block"]
    print(f"\n  >> Both policies will target: {target_block} block\n")

    # ── Step 2: VLM-guided ────────────────────────────────────────────
    guided = run_vlm_guided(
        args.checkpoint, args.episode_seed,
        args.rollout_seed + 5000,   # different seeds from baseline
        target_block, device, output_dir,
        max_attempts=args.max_attempts,
        max_steps=args.max_steps,
    )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'█'*78}")
    print(f"  FINAL SUMMARY  (episode {args.episode_seed})")
    print(f"{'█'*78}")
    print(f"  Target block : {target_block}")
    print()
    print(f"  BASELINE")
    print(f"    Success    : {baseline['success']}")
    print(f"    Picked     : {baseline['picked']}")
    print(f"    Arc        : {baseline['arc']:.4f} m ({baseline['arc_class']})")
    print(f"    Is arc15-19: {baseline['is_arc15_19']}")
    print(f"    Video      : {baseline['video']}")
    print()
    print(f"  VLM-GUIDED")
    print(f"    Success    : {guided['success']}")
    print(f"    Picked     : {guided['picked']}")
    print(f"    Arc        : {guided['arc']:.4f} m ({guided['arc_class']})")
    print(f"    Is arc15-19: {guided['is_arc15_19']}")
    print(f"    Legibility : {guided['legibility']:.3f}")
    print(f"    VLM choice : {guided['vlm_choice']}")
    print(f"    Selected   : attempt #{guided['selected_attempt']}")
    print(f"    Arc15 found: {guided['arc15_count']}/{guided['total_attempts']}")
    print(f"    Video      : {guided['video']}")
    print()

    # Did VLM pick arc 15-19?
    if guided["is_arc15_19"]:
        print(f"  >>> VLM PICKED an arc 15-19 trajectory!")
    else:
        print(f"  >>> VLM did NOT pick arc 15-19  "
              f"(picked {guided['arc_class']})")
        # Show if any arc 15-19 was available
        if guided["arc15_count"] > 0:
            print(f"      (arc 15-19 WAS available but VLM chose higher legibility elsewhere)")
        else:
            print(f"      (no arc 15-19 candidates were generated)")
    print(f"{'█'*78}\n")

    # ── Save JSON results ─────────────────────────────────────────────
    results = {
        "timestamp": ts,
        "episode_seed": args.episode_seed,
        "rollout_seed": args.rollout_seed,
        "target_block": target_block,
        "checkpoint": args.checkpoint,
        "max_attempts": args.max_attempts,
        "baseline": {k: v for k, v in baseline.items() if k != "video"},
        "vlm_guided": {k: v for k, v in guided.items()},
    }
    json_path = output_dir / f"results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results JSON → {json_path}")


if __name__ == "__main__":
    main()
