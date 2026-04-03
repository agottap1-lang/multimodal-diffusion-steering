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
    out = actions.copy()
    if out.shape[1] >= 5:
        fixed = float(np.clip(out[0, 4], -1.0, 1.0))
        out[:, 4] = fixed
    return out


def measure_arc(obs_trajectory: np.ndarray) -> float:
    """Measure arc from executed trajectory observations."""
    if len(obs_trajectory) == 0:
        return 0.0
    ee_y_positions = np.abs(obs_trajectory[:, 1])
    return float(np.max(ee_y_positions))


def arc_class(arc: float) -> str:
    """Classify arc based on max lateral Y position."""
    if arc < 0.08:
        return "00-05"
    if arc < 0.12:
        return "10-14"
    return "15-19"


def infer_block_from_action(actions: np.ndarray, horizon: int) -> str:
    avg_dy = float(np.mean(actions[: max(1, horizon // 4), 1]))
    return "LEFT" if avg_dy > 0 else "RIGHT"


def enforce_block_direction(actions: np.ndarray, target_block: str, horizon: int) -> np.ndarray:
    out = actions.copy()
    sign = 1.0 if target_block == "LEFT" else -1.0
    lock_h = max(1, horizon // 4)
    out[:lock_h, 1] = sign * np.abs(out[:lock_h, 1])
    return out


def extract_status(info: Dict) -> Dict:
    success_left = bool(info.get("success_left", 0.0) > 0.5)
    success_right = bool(info.get("success_right", 0.0) > 0.5)
    success = success_left or success_right
    if success_left and not success_right:
        picked = "LEFT"
    elif success_right and not success_left:
        picked = "RIGHT"
    elif success_left and success_right:
        picked = "BOTH"
    else:
        picked_left = bool(info.get("picked_left", 0.0) > 0.5)
        picked_right = bool(info.get("picked_right", 0.0) > 0.5)
        if picked_left and not picked_right:
            picked = "LEFT"
        elif picked_right and not picked_left:
            picked = "RIGHT"
        elif picked_left and picked_right:
            picked = "BOTH"
        else:
            picked = "NONE"
    return {"success": success, "picked_block": picked}


def sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=1.0):
    obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=temperature)[0].cpu().numpy()
    seq = seq * act_std + act_mean
    return stabilize_gripper(seq)


def collect_candidate_frames(episode_seed: int, actions: np.ndarray, model, sampler, obs_mean, obs_std, act_mean, act_std, device, target_block: str, candidate_temp: float) -> tuple:
    """Simulate full 150-step trajectory WITH REPLANNING to show VLM realistic motion.
    
    Args:
        actions: Initial 8-action sequence to use for first replan
        candidate_temp: Temperature used to generate initial sequence
    
    Returns:
        (frames, observations) tuple
    """
    env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
    obs = env.reset(seed=episode_seed)
    capture_steps = [0, 30, 60, 90, 120, 150]
    frames = []
    observations = []
    
    action_queue = deque(maxlen=model.horizon)
    
    # Add initial candidate sequence to queue
    for action in actions:
        action_queue.append(action)
    
    for step in range(capture_steps[-1] + 1):
        # Replan when queue is empty (use baseline temperature=1.0 for subsequent replans)
        if len(action_queue) == 0:
            seq = sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=1.0)
            seq = enforce_block_direction(seq, target_block, model.horizon)
            for action in seq:
                action_queue.append(action)
        
        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs  # Update observation for next replan
        observations.append(result.obs)
        
        if step in capture_steps:
            frame = env.render(width=480, height=480)
            frames.append(Image.fromarray(frame))
        
        if result.done and step >= 120:
            break

    while len(frames) < 6:
        frames.append(Image.new("RGB", (480, 480), "black"))

    env.close()
    return frames[:6], np.array(observations)


def vlm_score_prefix(vlm_scorer: LegibilityScorer, frames: List[Image.Image], target_block: str, video_id: str) -> Dict:
    prefix_scores = []
    for t_sec in [0, 1, 3, 5]:
        imgs = frames[: t_sec + 1]
        bytes_list = []
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            bytes_list.append(buf.getvalue())

        score = vlm_scorer.score_trajectory(
            bytes_list,
            goal_A=f"pick the {target_block} block",
            goal_B=f"pick the {'RIGHT' if target_block == 'LEFT' else 'LEFT'} block",
            mode="prefix_frames",
            video_id=video_id,
            t_sec=float(t_sec),
        )
        prefix_scores.append(
            {
                "t_sec": t_sec,
                "legibility_score": float(score["legibility_score"]),
                "choice": str(score["choice"]),
                "pA": float(score["pA"]),
                "pB": float(score["pB"]),
            }
        )

    return {
        "legibility_score": prefix_scores[-1]["legibility_score"],
        "choice": prefix_scores[-1]["choice"],
        "pA": prefix_scores[-1]["pA"],
        "pB": prefix_scores[-1]["pB"],
        "prefix_scores": prefix_scores,
    }


def rollout_baseline(
    model,
    sampler,
    obs_mean,
    obs_std,
    act_mean,
    act_std,
    device,
    episode_seed: int,
    rollout_seed: int,
    max_steps: int = 400,
    record_path: Optional[Path] = None,
):
    np.random.seed(rollout_seed)
    torch.manual_seed(rollout_seed)

    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset(seed=episode_seed)
    if record_path is not None:
        env.record_video(str(record_path), width=640, height=480, fps=30)

    action_queue = deque(maxlen=model.horizon)
    observations = []
    done = False
    steps = 0
    result = None
    first_seq = None

    while not done and steps < max_steps:
        if len(action_queue) == 0:
            seq = sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=1.0)
            if first_seq is None:
                first_seq = seq.copy()
            for action in seq:
                action_queue.append(action)

        action = action_queue.popleft()
        result = env.step(action)
        observations.append(result.obs)
        obs = result.obs
        done = bool(result.done)
        steps += 1

    if record_path is not None:
        env.stop_video()
    env.close()

    info = result.info if result is not None else {}
    status = extract_status(info)

    if first_seq is None:
        first_seq = np.zeros((model.horizon, model.act_dim), dtype=np.float32)
    
    # Measure arc from executed observations (first 150 steps)
    obs_array = np.array(observations[:150]) if observations else np.array([])
    first_arc = measure_arc(obs_array)

    return {
        "success": bool(status["success"]),
        "picked_block": status["picked_block"],
        "steps": int(steps),
        "target_block": infer_block_from_action(first_seq, model.horizon),
        "first_arc": first_arc,
        "first_arc_class": arc_class(first_arc),
    }


def rollout_guided(
    model,
    sampler,
    vlm_scorer,
    obs_mean,
    obs_std,
    act_mean,
    act_std,
    device,
    episode_seed: int,
    rollout_seed: int,
    target_block: str,
    n_candidates: int,
    legibility_threshold: float = 0.70,
    arc15_threshold: float = 0.12,  # Updated threshold (max lateral Y >= 0.12m)
    max_steps: int = 400,
    record_path: Optional[Path] = None,
):
    np.random.seed(rollout_seed)
    torch.manual_seed(rollout_seed)

    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset(seed=episode_seed)
    if record_path is not None:
        env.record_video(str(record_path), width=640, height=480, fps=30)

    action_queue = deque(maxlen=model.horizon)
    done = False
    steps = 0
    result = None
    first_arc = 0.0
    first_leg = 0.5
    selection_method = ""

    while not done and steps < max_steps:
        if len(action_queue) == 0:
            if steps == 0:
                # First replan: Generate diverse candidates with VLM scoring
                candidates = []
                for idx, temp in enumerate(np.linspace(0.5, 3.0, n_candidates)):
                    seq = sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=float(temp))
                    seq = enforce_block_direction(seq, target_block, model.horizon)
                    frames, obs_traj = collect_candidate_frames(episode_seed, seq, model, sampler, obs_mean, obs_std, act_mean, act_std, device, target_block, float(temp))
                    score = vlm_score_prefix(vlm_scorer, frames, target_block, f"ep{episode_seed}_cand{idx}")
                    candidates.append(
                        {
                            "idx": idx,
                            "seq": seq,
                            "arc": measure_arc(obs_traj),  # Measure from executed trajectory
                            "score": score,
                        }
                    )

                # Debug: Show candidate arcs and legibility
                arcs_str = ", ".join([f"{c['arc']:.4f}({arc_class(c['arc'])}) leg={c['score']['legibility_score']:.3f}" for c in candidates])
                print(f"    Step {steps}: Candidates: [{arcs_str}]")
                
                # Selection with three-tier fallback (cap arc at 0.6m to prevent extreme motions that break task)
                arc15_legible = [c for c in candidates if c["arc"] >= arc15_threshold and c["arc"] <= 0.6 and c["score"]["legibility_score"] >= legibility_threshold]
                if arc15_legible:
                    chosen = max(arc15_legible, key=lambda c: c["arc"])
                    selection_method = "arc15_legible"
                else:
                    # Fallback: select moderate arc with good legibility
                    legible = [c for c in candidates if c["score"]["legibility_score"] >= legibility_threshold and c["arc"] <= 0.6]
                    if legible:
                        chosen = max(legible, key=lambda c: c["arc"])
                        selection_method = "legible_fallback"
                    else:
                        # Last resort: any candidate with reasonable arc
                        moderate = [c for c in candidates if c["arc"] <= 0.6]
                        if moderate:
                            chosen = max(moderate, key=lambda c: c["score"]["legibility_score"])
                            selection_method = "moderate_arc"
                        else:
                            chosen = min(candidates, key=lambda c: c["arc"])
                            selection_method = "lowest_arc"

                print(f"    -> Selected: arc={chosen['arc']:.4f} ({arc_class(chosen['arc'])}) leg={chosen['score']['legibility_score']:.3f} method={selection_method}")
                
                first_arc = float(chosen["arc"])
                first_leg = float(chosen["score"]["legibility_score"])
                seq = chosen["seq"]
            else:
                # Subsequent replans: Use baseline temperature=1.0 for stability (match baseline policy)
                seq = sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=1.0)
                seq = enforce_block_direction(seq, target_block, model.horizon)

            for action in seq:
                action_queue.append(action)

        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs
        done = bool(result.done)
        steps += 1

    if record_path is not None:
        env.stop_video()
    env.close()

    info = result.info if result is not None else {}
    status = extract_status(info)

    return {
        "success": bool(status["success"]),
        "picked_block": status["picked_block"],
        "steps": int(steps),
        "first_arc": first_arc,
        "first_arc_class": arc_class(first_arc),
        "legibility": first_leg,
        "selection_method": selection_method,
        "arc15": bool(first_arc >= arc15_threshold),
    }


def main(checkpoint: str, episodes: int, n_candidates: int, seed: int, max_attempts: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
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
        beta_start=cfg.get("beta_start", 0.0001),
        beta_end=cfg.get("beta_end", 0.02),
        device=device,
    )

    obs_mean = torch.tensor(ckpt["obs_mean"], device=device)
    obs_std = torch.tensor(ckpt["obs_std"], device=device)
    act_mean = ckpt["act_mean"]
    act_std = ckpt["act_std"]

    vlm_scorer = LegibilityScorer()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/paired_replanning_v2_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PAIRED ROLLOUTS V2 (BASELINE vs VLM-GUIDED)")
    print("=" * 90)
    print(f"Output: {out_dir}")

    accepted = []
    attempt = 0

    while len(accepted) < episodes and attempt < max_attempts:
        ep_seed = seed + attempt
        print("\n" + "-" * 90)
        print(f"Attempt {attempt + 1}/{max_attempts} | episode_seed={ep_seed}")

        base_eval_seed = 10000 + ep_seed
        guided_eval_seed = 20000 + ep_seed

        baseline_eval = rollout_baseline(
            model, sampler, obs_mean, obs_std, act_mean, act_std, device,
            episode_seed=ep_seed, rollout_seed=base_eval_seed, record_path=None
        )
        print(
            f"  Baseline: success={baseline_eval['success']}, picked={baseline_eval['picked_block']}, "
            f"arc={baseline_eval['first_arc']:.4f} ({baseline_eval['first_arc_class']})"
        )

        if not baseline_eval["success"] or baseline_eval["picked_block"] not in {"LEFT", "RIGHT"}:
            print("  -> reject (baseline not successful single-block pick)")
            attempt += 1
            continue

        guided_eval = rollout_guided(
            model, sampler, vlm_scorer, obs_mean, obs_std, act_mean, act_std, device,
            episode_seed=ep_seed,
            rollout_seed=guided_eval_seed,
            target_block=baseline_eval["picked_block"],
            n_candidates=n_candidates,
            record_path=None,
        )
        print(
            f"  Guided: success={guided_eval['success']}, picked={guided_eval['picked_block']}, "
            f"arc={guided_eval['first_arc']:.4f} ({guided_eval['first_arc_class']}), "
            f"leg={guided_eval['legibility']:.3f}, method={guided_eval['selection_method']}"
        )

        if (not guided_eval["success"]) or (guided_eval["picked_block"] != baseline_eval["picked_block"]):
            print("  -> reject (guided not successful same-block pick)")
            attempt += 1
            continue

        pair_idx = len(accepted) + 1
        pair_dir = out_dir / f"pair_{pair_idx:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        baseline_video = pair_dir / f"baseline_{baseline_eval['picked_block']}_arc_{baseline_eval['first_arc']:.4f}m.mp4"
        guided_video = pair_dir / f"vlm_guided_{guided_eval['picked_block']}_arc_{guided_eval['first_arc']:.4f}m.mp4"

        baseline_record = rollout_baseline(
            model, sampler, obs_mean, obs_std, act_mean, act_std, device,
            episode_seed=ep_seed, rollout_seed=base_eval_seed, record_path=baseline_video
        )
        guided_record = rollout_guided(
            model, sampler, vlm_scorer, obs_mean, obs_std, act_mean, act_std, device,
            episode_seed=ep_seed,
            rollout_seed=guided_eval_seed,
            target_block=baseline_eval["picked_block"],
            n_candidates=n_candidates,
            record_path=guided_video,
        )

        pair = {
            "pair": pair_idx,
            "episode_seed": ep_seed,
            "baseline": {
                "success": baseline_record["success"],
                "picked_block": baseline_record["picked_block"],
                "first_arc": baseline_record["first_arc"],
                "first_arc_class": baseline_record["first_arc_class"],
                "video": str(baseline_video.relative_to(out_dir)),
            },
            "vlm_guided": {
                "success": guided_record["success"],
                "picked_block": guided_record["picked_block"],
                "first_arc": guided_record["first_arc"],
                "first_arc_class": guided_record["first_arc_class"],
                "legibility": guided_record["legibility"],
                "selection_method": guided_record["selection_method"],
                "arc15": guided_record["arc15"],
                "video": str(guided_video.relative_to(out_dir)),
            },
            "same_block": guided_record["picked_block"] == baseline_record["picked_block"],
        }
        accepted.append(pair)

        print(f"  -> ACCEPTED pair {pair_idx} (saved in {pair_dir.name})")
        attempt += 1

    aggregate = {
        "accepted_pairs": len(accepted),
        "requested_pairs": episodes,
        "attempts_used": attempt,
        "same_block_rate": float(np.mean([p["same_block"] for p in accepted])) if accepted else 0.0,
        "guided_arc15_rate": float(np.mean([p["vlm_guided"]["arc15"] for p in accepted])) if accepted else 0.0,
        "guided_legibility_mean": float(np.mean([p["vlm_guided"]["legibility"] for p in accepted])) if accepted else 0.0,
    }

    summary = {
        "checkpoint": checkpoint,
        "n_candidates": n_candidates,
        "seed": seed,
        "pairs": accepted,
        "aggregate": aggregate,
    }

    summary_path = out_dir / "paired_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("FINISHED")
    print("=" * 90)
    print(f"Accepted pairs: {len(accepted)}/{episodes}")
    print(f"Summary: {summary_path}")
    print(
        f"Same-block rate: {aggregate['same_block_rate']:.2%} | "
        f"Guided arc15 rate: {aggregate['guided_arc15_rate']:.2%} | "
        f"Guided mean legibility: {aggregate['guided_legibility_mean']:.3f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paired baseline vs VLM-guided replanning rollouts (v2)")
    parser.add_argument("--checkpoint", type=str, default="runs/diffusion_20260222_195530/ckpt_ep100.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--n_candidates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_attempts", type=int, default=25)
    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        n_candidates=args.n_candidates,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
