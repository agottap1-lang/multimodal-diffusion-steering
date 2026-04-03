#!/usr/bin/env python3
"""
Progressive VLM-Guided Steering Experiment  (ICRA)
===================================================

TRUE closed-loop steering where the VLM evaluates trajectory legibility
at each 1-second checkpoint and actively guides action selection when
the trajectory is judged ambiguous.

Key Distinction from Best-of-K Rejection Sampling
--------------------------------------------------
  - Best-of-K:  Generate K complete trajectories *independently*,
                cherry-pick the one with desired properties.
                → This is rejection sampling, NOT steering.

  - Progressive Steering (this script):
                Execute ONE trajectory.  At each 1-second checkpoint
                the VLM evaluates whether the *observer* can already
                identify the intended goal.  If not, the system
                resamples the NEXT action chunk (K candidates) and
                selects the continuation that the VLM judges most
                legible.  The trajectory is shaped *incrementally*
                through online VLM feedback.

Why higher arcs emerge naturally
--------------------------------
The VLM selects the candidate whose next-second position makes the
overall trajectory most clearly directed at a specific block.  Going
straight between both blocks is ambiguous; curving toward one side
is distinctive.  We NEVER select for arc height — the VLM's legibility
judgment naturally rewards distinctive motions.

Protocol
--------
  Condition 1 – Baseline:
      Run raw diffusion policy, capture frames at t=0..5s,
      score with VLM progressively at t=1,2,3,5 (passive, no intervention).

  Condition 2 – VLM Progressive Steering (MPC-style look-ahead):
      At each checkpoint t ∈ {1s, 2s, 3s}:
        1.  Render current frame, score [t=0..t_now] with VLM
        2.  If legibility ≥ θ  →  continue normally (already clear)
        3.  If legibility < θ  →  STEER via MPC-style look-ahead:
              a. Save full physics state (PyBullet saveState)
              b. Sample K candidate action chunks
              c. For each: restore state → simulate FORWARD 2 seconds
                 (capturing hypothetical frames at each second boundary)
              d. Score [accumulated_frames + look-ahead_frames] with VLM
                 (VLM evaluates where the trajectory WOULD be at t+2)
              e. Pick the most legible candidate
              f. Restore state → execute winner's FIRST 30 steps only
                 (commit 1 second, retain ability to steer at next checkpoint)

      Key insight:  At t=1s the robot has barely moved — all candidates look
      alike from the camera.  By simulating 2 seconds ahead (to t=3), the
      VLM can actually discriminate trajectories heading toward different
      blocks.  The steering decision is made EARLY (at t=1), but the
      evaluation criterion uses predicted future frames.

Metrics
-------
  • Progressive legibility curve           P(g* | frames_0:t)
  • Early-intent detection rate            % legible by t=1, 2, 3
  • Steering interventions per episode     0–3
  • Success rate  &  arc magnitude         (emergent, NOT selected for)

Usage
-----
  # Dry-run (3 episodes, K=2)
  python evaluation/vlm_progressive_steering.py --n-rollouts 3 --K 2

  # Full experiment
  python evaluation/vlm_progressive_steering.py --n-rollouts 50 --K 3
"""

from __future__ import annotations

import argparse, io, json, sys, time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler
from scripts.vlm_client import LegibilityScorer

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

ARC_T1 = 0.0786   # Bézier   00-04 → 05-09
ARC_T2 = 0.1047   #          05-09 → 10-14
ARC_T3 = 0.1335   #          10-14 → 15-19

GOAL_A = "pick the left block"
GOAL_B = "pick the right block"

STEPS_PER_SEC = 30          # 30 Hz
STEER_CHECKPOINTS = [1, 2, 3]     # seconds at which VLM evaluates
SCORE_CHECKPOINTS = [1, 2, 3, 5]  # seconds at which we record VLM score
FRAME_TIMES       = [0, 1, 2, 3, 4, 5]  # seconds at which frames are captured
LOOKAHEAD_SECS    = 2             # each candidate is simulated 2s into future


# ═══════════════════════════════════════════════════════════════════════
# ARC HELPERS
# ═══════════════════════════════════════════════════════════════════════

def measure_arc(obs_traj: np.ndarray) -> float:
    if len(obs_traj) == 0:
        return 0.0
    return float(np.max(np.abs(obs_traj[:, 1])))

def arc_class(arc: float) -> str:
    if arc < ARC_T1:  return "00-04"
    if arc < ARC_T2:  return "05-09"
    if arc < ARC_T3:  return "10-14"
    return "15-19"


# ═══════════════════════════════════════════════════════════════════════
# ACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def stabilize_gripper(actions: np.ndarray) -> np.ndarray:
    out = actions.copy()
    if out.shape[1] >= 5:
        out[:, 4] = float(np.clip(out[0, 4], -1.0, 1.0))
    return out

def infer_block(actions: np.ndarray) -> str:
    n = min(20, len(actions))
    return "LEFT" if float(np.mean(actions[:n, 1])) > 0 else "RIGHT"

def enforce_block_direction(actions: np.ndarray, target: str) -> np.ndarray:
    """Force first 15 actions' Y-sign toward *target* block.

    Constrains DIRECTION (sign) only — not magnitude.
    Ensures the robot approaches the intended block without
    controlling how much lateral arc it uses.
    """
    out = actions.copy()
    sign = 1.0 if target == "LEFT" else -1.0
    n = min(15, len(out))
    out[:n, 1] = sign * np.abs(out[:n, 1])
    return out


# ═══════════════════════════════════════════════════════════════════════
# POLICY LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_policy(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        horizon=cfg["horizon"],
        hidden_dim=cfg.get("hidden_dim", 256),
        n_blocks=cfg.get("n_blocks", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    sampler = DDIMSampler(
        n_steps=cfg["n_diffusion_steps"],
        beta_start=cfg["beta_start"], beta_end=cfg["beta_end"],
        device=device,
    )
    return (model, sampler,
            torch.tensor(ckpt["obs_mean"], device=device),
            torch.tensor(ckpt["obs_std"],  device=device),
            ckpt["act_mean"], ckpt["act_std"], cfg)


def sample_chunk(model, sampler, obs, obs_mean, obs_std,
                 act_mean, act_std, device):
    obs_n = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
    obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        seq = sampler.sample(model, obs_t, n_sampling_steps=10,
                             temperature=1.0)[0].cpu().numpy()
    return stabilize_gripper(seq * act_std + act_mean)


# ═══════════════════════════════════════════════════════════════════════
# FRAME ANNOTATION
# ═══════════════════════════════════════════════════════════════════════

_FONT_CACHE: dict = {}

def _get_fonts():
    if "large" not in _FONT_CACHE:
        for path in ["arial.ttf",
                      "C:/Windows/Fonts/arial.ttf",
                      "C:/Windows/Fonts/arialbd.ttf"]:
            try:
                _FONT_CACHE["large"] = ImageFont.truetype(path, 28)
                _FONT_CACHE["small"] = ImageFont.truetype(path, 18)
                break
            except (IOError, OSError):
                continue
        if "large" not in _FONT_CACHE:
            _FONT_CACHE["large"] = ImageFont.load_default()
            _FONT_CACHE["small"] = _FONT_CACHE["large"]
    return _FONT_CACHE["large"], _FONT_CACHE["small"]


def annotate_frame(frame_rgb: np.ndarray, t_sec: int,
                   show_goals: bool = False) -> Image.Image:
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    font_lg, font_sm = _get_fonts()
    PAD = 5

    ts_text = f"t = {t_sec}s"
    bb = draw.textbbox((15, 10), ts_text, font=font_lg)
    draw.rectangle([bb[0]-PAD, bb[1]-PAD, bb[2]+PAD, bb[3]+PAD], fill="black")
    draw.text((15, 10), ts_text, fill="white", font=font_lg)

    if show_goals:
        y = bb[3] + PAD + 10
        for lbl, clr in [(f"Goal A: {GOAL_A}", (255, 210, 0)),
                          (f"Goal B: {GOAL_B}", (100, 200, 255))]:
            gb = draw.textbbox((15, y), lbl, font=font_sm)
            draw.rectangle([gb[0]-3, gb[1]-2, gb[2]+3, gb[3]+2], fill="black")
            draw.text((15, y), lbl, fill=clr, font=font_sm)
            y = gb[3] + 8
    return img


# ═══════════════════════════════════════════════════════════════════════
# VLM SCORING  (prefix_frames, with adaptive t_sec)
# ═══════════════════════════════════════════════════════════════════════

def score_with_vlm(scorer: LegibilityScorer,
                   frames: List[Image.Image],
                   video_id: str,
                   t_sec: int) -> Dict:
    """Score a prefix of annotated frames with VLM.

    Args:
        scorer:   LegibilityScorer wrapper
        frames:   list of annotated PIL Images [t=0 .. t=t_sec]
        video_id: identifier for logging
        t_sec:    timestamp of the latest frame (used in prompt)

    Returns:
        dict with legibility_score, pA, pB, choice, cue, etc.
    """
    frames_bytes = []
    for fr in frames:
        buf = io.BytesIO()
        fr.save(buf, format="JPEG", quality=90)
        frames_bytes.append(buf.getvalue())

    for attempt in range(3):
        try:
            r = scorer.score_trajectory(
                image_bytes=frames_bytes,
                goal_A=GOAL_A, goal_B=GOAL_B,
                mode="prefix_frames",
                video_id=video_id,
                t_sec=float(t_sec),
            )
            return dict(
                legibility_score=float(r.get("legibility_score", 0.5)),
                pA=float(r.get("pA", 0.5)),
                pB=float(r.get("pB", 0.5)),
                choice=r.get("choice", "C"),
                cue=str(r.get("cue", "")),
                confidence=int(r.get("confidence", 50)),
                latency_ms=r.get("latency_ms", 0),
                error=None,
            )
        except Exception as exc:
            wait = (attempt + 1) * 10
            if attempt < 2:
                print(f"        [VLM retry {attempt+1}, wait {wait}s: {exc}]")
                time.sleep(wait)
            else:
                print(f"        [VLM FAILED 3×: {exc}]")
                return dict(legibility_score=0.5, pA=0.5, pB=0.5,
                            choice="C", cue=f"ERROR: {exc}",
                            confidence=0, latency_ms=0, error=str(exc))


def vlm_smoke_test(scorer: LegibilityScorer) -> bool:
    print("\n  VLM smoke test ...")
    img = Image.new("RGB", (64, 64), "gray")
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "test", fill="white")
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=50)
    try:
        r = scorer.score_trajectory(
            image_bytes=buf.getvalue(),
            goal_A="pick left", goal_B="pick right",
            mode="single_frame", video_id="smoke", t_sec=0.0)
        print(f"    OK  pA={r['pA']:.2f}  pB={r['pB']:.2f}  "
              f"latency={r.get('latency_ms',0)}ms")
        return True
    except Exception as exc:
        print(f"    FAILED: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# PYBULLET STATE SAVE / RESTORE
# ═══════════════════════════════════════════════════════════════════════

def save_env_state(env: TwoBlockPickEnv) -> Dict:
    """Checkpoint full simulation + env internal state."""
    return dict(
        bullet_id=p.saveState(physicsClientId=env._cid),
        target_pos=env._target_pos.copy(),
        target_yaw=env._target_yaw,
        grip_cmd=env._grip_cmd,
        episode_steps=env._episode_steps,
        picked_left=env._picked_left,
        picked_right=env._picked_right,
    )


def restore_env_state(env: TwoBlockPickEnv, saved: Dict):
    """Restore simulation to a saved checkpoint."""
    p.restoreState(saved["bullet_id"], physicsClientId=env._cid)
    env._target_pos = saved["target_pos"].copy()
    env._target_yaw = saved["target_yaw"]
    env._grip_cmd = saved["grip_cmd"]
    env._episode_steps = saved["episode_steps"]
    env._picked_left = saved["picked_left"]
    env._picked_right = saved["picked_right"]


def cleanup_state(env: TwoBlockPickEnv, saved: Dict):
    """Release PyBullet saved-state memory."""
    p.removeState(saved["bullet_id"], physicsClientId=env._cid)


# ═══════════════════════════════════════════════════════════════════════
# EXECUTE ONE 30-STEP SEGMENT
# ═══════════════════════════════════════════════════════════════════════

def execute_segment(env: TwoBlockPickEnv,
                    chunk: np.ndarray,
                    obs: np.ndarray,
                    all_obs: List[np.ndarray],
                    all_act: List[np.ndarray],
                    n_steps: int = 30) -> Tuple[np.ndarray, bool]:
    """Execute *n_steps* actions from *chunk*, recording into lists.

    Returns (final_obs, done).
    """
    done = False
    for s in range(n_steps):
        action = chunk[min(s, len(chunk) - 1)]
        result = env.step(action)
        all_obs.append(result.obs)
        all_act.append(action)
        obs = result.obs
        if result.done:
            done = True
            break
    return obs, done


# ═══════════════════════════════════════════════════════════════════════
# CONDITION 1 — BASELINE  (progressive VLM scoring, no intervention)
# ═══════════════════════════════════════════════════════════════════════

def run_baseline_episode(
    env, model, sampler, scorer,
    obs_mean, obs_std, act_mean, act_std, device,
    episode_seed: int, target_block: str,
    max_steps: int = 600,
) -> Dict:
    """Run raw policy with progressive VLM scoring (passive)."""
    seed_base = episode_seed * 100 + 7
    np.random.seed(seed_base); torch.manual_seed(seed_base)

    obs = env.reset(seed=episode_seed)

    # Capture t=0 frame
    frame_t0 = annotate_frame(env.render(mode="rgb_array"), 0, show_goals=True)
    frames: List[Image.Image] = [frame_t0]

    all_obs: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    progressive: Dict[str, float] = {}
    done = False

    # ── Execute 5 segments of 30 steps each (t=0→5) ─────────────
    for seg_idx in range(5):
        if done:
            break
        t_start = seg_idx       # current second
        t_end   = seg_idx + 1   # second after segment

        seg_seed = seed_base + seg_idx * 111
        np.random.seed(seg_seed); torch.manual_seed(seg_seed)

        chunk = sample_chunk(model, sampler, obs,
                             obs_mean, obs_std, act_mean, act_std, device)
        # Enforce direction in first 3 seconds only
        if seg_idx < 3:
            chunk = enforce_block_direction(chunk, target_block)

        obs, done = execute_segment(env, chunk, obs, all_obs, all_act)

        # Capture frame at t_end
        frame = annotate_frame(env.render(mode="rgb_array"), t_end)
        frames.append(frame)

        # Progressive VLM scoring at key checkpoints
        if t_end in SCORE_CHECKPOINTS:
            vlm = score_with_vlm(scorer, frames,
                                 f"base_ep{episode_seed}_t{t_end}",
                                 t_sec=t_end)
            progressive[f"t{t_end}"] = vlm["legibility_score"]

    # ── Continue to task completion (no VLM) ─────────────────────
    q: deque = deque()
    while not done and len(all_obs) < max_steps:
        if len(q) == 0:
            chunk = sample_chunk(model, sampler, obs,
                                 obs_mean, obs_std, act_mean, act_std, device)
            q.extend(chunk)
        action = q.popleft()
        result = env.step(action)
        all_obs.append(result.obs)
        all_act.append(action)
        obs = result.obs
        done = result.done

    # ── Metrics ──────────────────────────────────────────────────
    obs_5s = np.array(all_obs[:150]) if len(all_obs) >= 150 \
             else np.array(all_obs)
    arc_val = measure_arc(obs_5s)

    info = result.info if result else {}
    success = (info.get("success_left", 0) > 0.5
               or info.get("success_right", 0) > 0.5)

    return dict(
        episode_seed=episode_seed,
        target_block=target_block,
        condition="baseline",
        success=success,
        arc=float(arc_val),
        arc_class=arc_class(arc_val),
        is_arc15_19=(arc_val >= ARC_T3),
        steps=len(all_obs),
        progressive_legibility=progressive,
        vlm_calls=len(progressive),          # passive scoring only
    )


# ═══════════════════════════════════════════════════════════════════════
# CONDITION 2 — VLM PROGRESSIVE STEERING
# ═══════════════════════════════════════════════════════════════════════

def run_steered_episode(
    env, model, sampler, scorer,
    obs_mean, obs_std, act_mean, act_std, device,
    episode_seed: int, target_block: str,
    n_candidates: int = 3,
    legibility_threshold: float = 0.60,
    max_steps: int = 600,
    save_dir: Optional[Path] = None,
) -> Dict:
    """Run with VLM progressive steering at checkpoints."""
    seed_base = episode_seed * 100 + 7
    np.random.seed(seed_base); torch.manual_seed(seed_base)

    obs = env.reset(seed=episode_seed)

    frame_t0 = annotate_frame(env.render(mode="rgb_array"), 0, show_goals=True)
    frames: List[Image.Image] = [frame_t0]

    all_obs: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    progressive: Dict[str, float] = {}
    checkpoint_details: Dict[str, Dict] = {}
    interventions = 0
    vlm_calls = 0
    done = False

    # ── Segment 0: t=0 → t=1  (always execute normally) ─────────
    np.random.seed(seed_base); torch.manual_seed(seed_base)
    chunk_0 = sample_chunk(model, sampler, obs,
                           obs_mean, obs_std, act_mean, act_std, device)
    chunk_0 = enforce_block_direction(chunk_0, target_block)
    obs, done = execute_segment(env, chunk_0, obs, all_obs, all_act)

    frame_t1 = annotate_frame(env.render(mode="rgb_array"), 1)
    frames.append(frame_t1)

    # ── Steering checkpoints  (t = 1, 2, 3) ─────────────────────
    for cp_sec in STEER_CHECKPOINTS:
        if done:
            break

        t_next = cp_sec + 1   # second after this checkpoint

        # 1) Evaluate current trajectory legibility
        vlm = score_with_vlm(scorer, frames,
                             f"steer_ep{episode_seed}_eval_t{cp_sec}",
                             t_sec=cp_sec)
        vlm_calls += 1
        current_leg = vlm["legibility_score"]
        progressive[f"t{cp_sec}"] = current_leg

        cp_info: Dict = dict(
            legibility=current_leg,
            choice=vlm["choice"],
            cue=vlm["cue"][:120],
            intervened=False,
        )

        # 2) Decide: intervene or continue
        if current_leg < legibility_threshold:
            # ── MPC-STYLE LOOK-AHEAD INTERVENTION ────────────────
            # At early checkpoints the robot has barely moved, so all
            # candidates look alike.  We simulate each candidate 2 s
            # into the future (to where the VLM CAN discriminate) and
            # score the full look-ahead trajectory.  Only the first
            # 30 steps (1 s) are committed — the rest is planning.
            cp_info["intervened"] = True
            interventions += 1

            saved = save_env_state(env)
            obs_at_cp = obs.copy()

            eval_t = cp_sec + LOOKAHEAD_SECS        # e.g. t=1→3
            lookahead_steps = LOOKAHEAD_SECS * STEPS_PER_SEC  # 60

            candidates = []
            for k in range(n_candidates):
                restore_env_state(env, saved)

                cand_seed = seed_base + t_next * 1000 + k
                np.random.seed(cand_seed); torch.manual_seed(cand_seed)

                cand_chunk = sample_chunk(
                    model, sampler, obs_at_cp, obs_mean, obs_std,
                    act_mean, act_std, device)
                cand_chunk = enforce_block_direction(cand_chunk, target_block)

                # ── simulate 2 s forward, capture frames each sec ─
                hypo_frames: List[Image.Image] = []
                obs_sim = obs_at_cp.copy()
                q_sim: deque = deque(cand_chunk)

                for step in range(lookahead_steps):
                    if len(q_sim) == 0:
                        rpl_seed = cand_seed + 500 + step
                        np.random.seed(rpl_seed)
                        torch.manual_seed(rpl_seed)
                        rpl = sample_chunk(
                            model, sampler, obs_sim, obs_mean, obs_std,
                            act_mean, act_std, device)
                        rpl = enforce_block_direction(rpl, target_block)
                        q_sim.extend(rpl)

                    action = q_sim.popleft()
                    result = env.step(action)
                    obs_sim = result.obs

                    if (step + 1) % STEPS_PER_SEC == 0:
                        t_frame = cp_sec + (step + 1) // STEPS_PER_SEC
                        hf = annotate_frame(
                            env.render(mode="rgb_array"), t_frame)
                        hypo_frames.append(hf)

                # Score [accumulated + look-ahead]
                test_frames = list(frames) + hypo_frames
                cand_vlm = score_with_vlm(
                    scorer, test_frames,
                    f"steer_ep{episode_seed}_t{cp_sec}_c{k}",
                    t_sec=eval_t)
                vlm_calls += 1

                candidates.append(dict(
                    idx=k,
                    chunk=cand_chunk.copy(),
                    legibility=cand_vlm["legibility_score"],
                    choice=cand_vlm["choice"],
                    cue=cand_vlm["cue"][:100],
                ))

            # Select MOST LEGIBLE candidate — NOT highest arc
            winner = max(candidates, key=lambda c: c["legibility"])

            cp_info["candidates"] = [
                dict(idx=c["idx"], legibility=c["legibility"])
                for c in candidates
            ]
            cp_info["selected"] = winner["idx"]
            cp_info["selected_legibility"] = winner["legibility"]
            cp_info["eval_horizon"] = eval_t

            # Restore & execute FIRST 30 steps only (commit 1 sec)
            restore_env_state(env, saved)
            obs, done = execute_segment(
                env, winner["chunk"], obs_at_cp, all_obs, all_act)

            actual_frame = annotate_frame(
                env.render(mode="rgb_array"), t_next)
            frames.append(actual_frame)

            cleanup_state(env, saved)

        else:
            # ── NO INTERVENTION — trajectory already legible ─────
            seg_seed = seed_base + t_next * 111
            np.random.seed(seg_seed); torch.manual_seed(seg_seed)

            next_chunk = sample_chunk(
                model, sampler, obs, obs_mean, obs_std,
                act_mean, act_std, device)
            if cp_sec < 3:
                next_chunk = enforce_block_direction(next_chunk,
                                                     target_block)

            obs, done = execute_segment(
                env, next_chunk, obs, all_obs, all_act)

            frame_next = annotate_frame(
                env.render(mode="rgb_array"), t_next)
            frames.append(frame_next)

        checkpoint_details[f"t{cp_sec}"] = cp_info

    # ── Remaining segments (t=4→5) without steering ──────────────
    for seg_sec in [4]:
        if done:
            break
        t_next = seg_sec + 1
        seg_seed = seed_base + t_next * 111
        np.random.seed(seg_seed); torch.manual_seed(seg_seed)

        chunk = sample_chunk(model, sampler, obs,
                             obs_mean, obs_std, act_mean, act_std, device)
        obs, done = execute_segment(env, chunk, obs, all_obs, all_act)

        frame = annotate_frame(env.render(mode="rgb_array"), t_next)
        frames.append(frame)

    # ── Final VLM scoring at t=5 with all frames ────────────────
    if len(frames) >= 6:
        vlm_t5 = score_with_vlm(scorer, frames,
                                f"steer_ep{episode_seed}_t5", t_sec=5)
        vlm_calls += 1
        progressive["t5"] = vlm_t5["legibility_score"]

    # ── Task completion (t=5 onward, no VLM) ─────────────────────
    q: deque = deque()
    while not done and len(all_obs) < max_steps:
        if len(q) == 0:
            chunk = sample_chunk(model, sampler, obs,
                                 obs_mean, obs_std, act_mean, act_std, device)
            q.extend(chunk)
        action = q.popleft()
        result = env.step(action)
        all_obs.append(result.obs)
        all_act.append(action)
        obs = result.obs
        done = result.done

    # ── Save composite if requested ──────────────────────────────
    if save_dir is not None and frames:
        comp = _make_composite(frames)
        comp.save(save_dir / f"ep{episode_seed}_steered.jpg", quality=90)

    # ── Compute metrics ──────────────────────────────────────────
    obs_5s = np.array(all_obs[:150]) if len(all_obs) >= 150 \
             else np.array(all_obs)
    arc_val = measure_arc(obs_5s)

    info = result.info if result else {}
    success = (info.get("success_left", 0) > 0.5
               or info.get("success_right", 0) > 0.5)

    return dict(
        episode_seed=episode_seed,
        target_block=target_block,
        condition="vlm_steered",
        success=success,
        arc=float(arc_val),
        arc_class=arc_class(arc_val),
        is_arc15_19=(arc_val >= ARC_T3),
        steps=len(all_obs),
        progressive_legibility=progressive,
        checkpoint_details=checkpoint_details,
        interventions=interventions,
        vlm_calls=vlm_calls,
    )


def _make_composite(frames: List[Image.Image]) -> Image.Image:
    if not frames:
        return Image.new("RGB", (480, 480), "black")
    w, h = frames[0].size
    comp = Image.new("RGB", (w * len(frames), h), "black")
    for i, fr in enumerate(frames):
        comp.paste(fr, (i * w, 0))
    return comp


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def compute_summary(rollouts: List[Dict], condition: str) -> Dict:
    n = len(rollouts)
    if n == 0:
        return {}

    arcs = [r["arc"] for r in rollouts]
    successes = sum(1 for r in rollouts if r["success"])
    arc_cnt = {"00-04": 0, "05-09": 0, "10-14": 0, "15-19": 0}
    for r in rollouts:
        arc_cnt[r["arc_class"]] += 1
    arc_pct = {k: v / n * 100 for k, v in arc_cnt.items()}

    s = dict(
        condition=condition, n=n,
        success_rate=round(successes / n, 4),
        success_count=successes,
        arc_distribution_count=arc_cnt,
        arc_distribution_pct=arc_pct,
        arc15_19_rate=round(arc_cnt["15-19"] / n, 4),
        mean_arc=round(float(np.mean(arcs)), 5),
        std_arc=round(float(np.std(arcs)), 5),
    )

    # Progressive legibility averages
    for t_key in ["t1", "t2", "t3", "t5"]:
        vals = [r["progressive_legibility"].get(t_key)
                for r in rollouts
                if r["progressive_legibility"].get(t_key) is not None]
        if vals:
            s[f"mean_leg_{t_key}"] = round(float(np.mean(vals)), 4)
            s[f"std_leg_{t_key}"]  = round(float(np.std(vals)), 4)

    # Early-intent detection rate  (legible by t=X?)
    for t_key, t_val in [("t1", 1), ("t2", 2), ("t3", 3)]:
        legible_count = sum(
            1 for r in rollouts
            if r["progressive_legibility"].get(t_key, 0) >= 0.60
        )
        s[f"early_intent_{t_key}"] = round(legible_count / n, 4)

    # Intervention stats (steered only)
    int_vals = [r.get("interventions", 0) for r in rollouts]
    if any(v > 0 for v in int_vals):
        s["mean_interventions"] = round(float(np.mean(int_vals)), 2)
        s["max_interventions"]  = max(int_vals)

    # VLM call count
    s["total_vlm_calls"] = sum(r.get("vlm_calls", 0) for r in rollouts)

    return s


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def generate_progressive_plot(bs, ss, b_rolls, s_rolls, out: Path):
    """4-panel comparison figure for progressive steering."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Panel 1: Progressive Legibility Curve ────────────────────
    ax = axes[0, 0]
    t_keys = ["t1", "t2", "t3", "t5"]
    t_vals = [1, 2, 3, 5]

    for summary, rolls, label, color, marker in [
        (bs, b_rolls, "Baseline (no steering)", "#4a90d9", "o"),
        (ss, s_rolls, "VLM Steering",           "#e6783c", "^"),
    ]:
        means = [summary.get(f"mean_leg_{tk}", 0) for tk in t_keys]
        stds  = [summary.get(f"std_leg_{tk}", 0)  for tk in t_keys]
        ax.errorbar(t_vals, means, yerr=stds, marker=marker,
                    label=label, color=color, capsize=4, linewidth=2,
                    markersize=8)

    ax.axhline(0.60, color="gray", ls="--", alpha=0.5, label="Legibility θ")
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("VLM Legibility Score", fontsize=12)
    ax.set_title("Progressive Legibility Curve", fontsize=13, fontweight="bold")
    ax.set_xticks(t_vals)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── Panel 2: Early Intent Detection Rate ─────────────────────
    ax = axes[0, 1]
    checkpoints = ["t1", "t2", "t3"]
    cp_labels = ["t = 1s", "t = 2s", "t = 3s"]
    x = np.arange(len(checkpoints))
    w = 0.35

    b_rates = [bs.get(f"early_intent_{tk}", 0) * 100 for tk in checkpoints]
    s_rates = [ss.get(f"early_intent_{tk}", 0) * 100 for tk in checkpoints]

    bars1 = ax.bar(x - w/2, b_rates, w, label="Baseline", color="#4a90d9",
                   edgecolor="k")
    bars2 = ax.bar(x + w/2, s_rates, w, label="VLM Steering", color="#e6783c",
                   edgecolor="k")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}%",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(cp_labels)
    ax.set_ylabel("% Episodes Legible")
    ax.set_title("Early Intent Detection Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: Arc Distribution ────────────────────────────────
    ax = axes[1, 0]
    cats = ["00-04", "05-09", "10-14", "15-19"]
    x = np.arange(len(cats))

    b_pct = [bs.get("arc_distribution_pct", {}).get(c, 0) for c in cats]
    s_pct = [ss.get("arc_distribution_pct", {}).get(c, 0) for c in cats]

    ax.bar(x - w/2, b_pct, w, label="Baseline", color="#4a90d9", edgecolor="k")
    ax.bar(x + w/2, s_pct, w, label="VLM Steering", color="#e6783c",
           edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_xlabel("Arc Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Arc Distribution (Emergent)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ── Panel 4: Summary Table ───────────────────────────────────
    ax = axes[1, 1]; ax.axis("off")
    rows = [
        ["Metric",             "Baseline",  "VLM Steering"],
        ["N rollouts",         str(bs["n"]), str(ss["n"])],
        ["Success rate",       f"{bs['success_rate']:.0%}",
                               f"{ss['success_rate']:.0%}"],
        ["Mean arc (m)",       f"{bs['mean_arc']:.4f}",
                               f"{ss['mean_arc']:.4f}"],
        ["Arc 15-19",          f"{bs['arc15_19_rate']:.0%}",
                               f"{ss['arc15_19_rate']:.0%}"],
        ["Leg. @ t=1s",        f"{bs.get('mean_leg_t1',0):.3f}",
                               f"{ss.get('mean_leg_t1',0):.3f}"],
        ["Leg. @ t=2s",        f"{bs.get('mean_leg_t2',0):.3f}",
                               f"{ss.get('mean_leg_t2',0):.3f}"],
        ["Leg. @ t=3s",        f"{bs.get('mean_leg_t3',0):.3f}",
                               f"{ss.get('mean_leg_t3',0):.3f}"],
        ["Leg. @ t=5s",        f"{bs.get('mean_leg_t5',0):.3f}",
                               f"{ss.get('mean_leg_t5',0):.3f}"],
        ["Early intent (t≤2s)",f"{bs.get('early_intent_t2',0):.0%}",
                               f"{ss.get('early_intent_t2',0):.0%}"],
        ["Avg interventions",  "0",
                               f"{ss.get('mean_interventions',0):.1f}"],
        ["Total VLM calls",    str(bs.get('total_vlm_calls',0)),
                               str(ss.get('total_vlm_calls',0))],
    ]
    tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.6)
    for j in range(3):
        tbl[0, j].set_facecolor("#ccc")
        tbl[0, j].set_text_props(fontweight="bold")
    # Highlight legibility rows
    for rowIdx in [6, 7, 8, 9, 10]:
        for j in range(3):
            tbl[rowIdx, j].set_facecolor("#fff3cd")
    ax.set_title("Summary", pad=20)

    plt.suptitle("Progressive VLM Steering — ICRA Experiment",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p_out = out / "progressive_steering_comparison.png"
    plt.savefig(str(p_out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot → {p_out}")


def generate_per_episode_plot(b_rolls, s_rolls, out: Path):
    """Per-episode progressive legibility scatter."""
    fig, ax = plt.subplots(figsize=(12, 5))
    t_keys = ["t1", "t2", "t3", "t5"]
    t_vals = [1, 2, 3, 5]

    for i, ep_b in enumerate(b_rolls):
        b_legs = [ep_b["progressive_legibility"].get(tk, None) for tk in t_keys]
        valid_t = [t for t, v in zip(t_vals, b_legs) if v is not None]
        valid_v = [v for v in b_legs if v is not None]
        ax.plot(valid_t, valid_v, "o-", color="#4a90d9", alpha=0.15,
                linewidth=0.8, markersize=3)

    for i, ep_s in enumerate(s_rolls):
        s_legs = [ep_s["progressive_legibility"].get(tk, None) for tk in t_keys]
        valid_t = [t for t, v in zip(t_vals, s_legs) if v is not None]
        valid_v = [v for v in s_legs if v is not None]
        ax.plot(valid_t, valid_v, "^-", color="#e6783c", alpha=0.15,
                linewidth=0.8, markersize=3)

    # Mean lines
    for rolls, label, color, marker in [
        (b_rolls, "Baseline (mean)", "#4a90d9", "o"),
        (s_rolls, "VLM Steering (mean)", "#e6783c", "^"),
    ]:
        means = []
        for tk in t_keys:
            vals = [r["progressive_legibility"].get(tk)
                    for r in rolls
                    if r["progressive_legibility"].get(tk) is not None]
            means.append(np.mean(vals) if vals else None)
        v_t = [t for t, m in zip(t_vals, means) if m is not None]
        v_m = [m for m in means if m is not None]
        ax.plot(v_t, v_m, marker=marker, color=color, linewidth=3,
                markersize=10, label=label, zorder=5)

    ax.axhline(0.60, color="gray", ls="--", alpha=0.5, label="θ = 0.60")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("VLM Legibility")
    ax.set_title("Per-Episode Progressive Legibility Curves")
    ax.set_xticks(t_vals); ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    p_out = out / "per_episode_legibility.png"
    plt.savefig(str(p_out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot → {p_out}")


# ═══════════════════════════════════════════════════════════════════════
# CRASH-SAFE SAVE
# ═══════════════════════════════════════════════════════════════════════

def _save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="Progressive VLM Steering Experiment (ICRA)")
    pa.add_argument("--checkpoint",
                    default="runs/diffusion_20260222_195530/ckpt_ep100.pt")
    pa.add_argument("--n-rollouts",  type=int, default=50)
    pa.add_argument("--K",           type=int, default=3,
                    help="Candidates per intervention (default 3)")
    pa.add_argument("--threshold",   type=float, default=0.60,
                    help="Legibility threshold for intervention (default 0.60)")
    pa.add_argument("--base-seed",   type=int, default=100)
    pa.add_argument("--max-steps",   type=int, default=600)
    pa.add_argument("--output-dir",  default="outputs/vlm_progressive_steering")
    pa.add_argument("--save-composites", action="store_true")
    pa.add_argument("--skip-baseline",   action="store_true")
    pa.add_argument("--skip-steered",    action="store_true")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'█' * 78}")
    print(f"  PROGRESSIVE VLM STEERING EXPERIMENT")
    print(f"{'█' * 78}")
    print(f"  Checkpoint       : {args.checkpoint}")
    print(f"  N rollouts       : {args.n_rollouts}")
    print(f"  K candidates     : {args.K}")
    print(f"  Legibility θ     : {args.threshold}")
    print(f"  Max steps        : {args.max_steps}")
    print(f"  Output           : {out}")
    print(f"  Device           : {device}")
    print(f"  Arc thresholds   : {ARC_T1}, {ARC_T2}, {ARC_T3}")
    print(f"  Steer checkpoints: t = {STEER_CHECKPOINTS} seconds")
    print()
    print(f"  METHOD: Closed-loop MPC-style progressive steering")
    print(f"  At each checkpoint, VLM evaluates trajectory legibility.")
    print(f"  If leg < θ, each candidate is simulated {LOOKAHEAD_SECS}s ahead")
    print(f"  and VLM picks the most legible future trajectory.")
    print(f"  Selection criterion: MAX LEGIBILITY (not arc).")
    print(f"  Higher arcs emerge naturally from legibility pressure.")
    print(f"{'█' * 78}\n")

    # ── Load policy ──────────────────────────────────────────────
    print("Loading policy ...")
    model, sampler, obs_m, obs_s, act_m, act_s, cfg = \
        load_policy(args.checkpoint, device)
    print(f"  horizon={cfg['horizon']}  act_dim={cfg['act_dim']}\n")

    # ── Init VLM scorer ──────────────────────────────────────────
    scorer = LegibilityScorer(model="gemini-2.5-flash")
    if not vlm_smoke_test(scorer):
        print("\n  !! VLM smoke test FAILED.  Aborting.\n")
        return

    # Determine target blocks from initial sampling
    target_blocks: Dict[int, str] = {}
    env_probe = TwoBlockPickEnv(render=False, episode_length=args.max_steps,
                                cube_jitter=0.0)
    for i in range(args.n_rollouts):
        ep = args.base_seed + i
        obs = env_probe.reset(seed=ep)
        np.random.seed(ep * 100 + 7); torch.manual_seed(ep * 100 + 7)
        seq = sample_chunk(model, sampler, obs,
                           obs_m, obs_s, act_m, act_s, device)
        target_blocks[ep] = infer_block(seq)
    env_probe.close()
    print(f"  Target blocks determined for {len(target_blocks)} episodes.\n")

    # ═════════════════════════════════════════════════════════════
    # CONDITION 1 — BASELINE (progressive VLM scoring, no steering)
    # ═════════════════════════════════════════════════════════════
    b_rolls: List[Dict] = []

    if not args.skip_baseline:
        print(f"{'═' * 78}")
        print(f"  CONDITION 1: BASELINE  ({args.n_rollouts} rollouts)")
        print(f"  (Progressive VLM scoring at t=1,2,3,5 — passive)")
        print(f"{'═' * 78}\n")

        env = TwoBlockPickEnv(render=False, episode_length=args.max_steps,
                              cube_jitter=0.0)
        t0 = time.time()

        for i in range(args.n_rollouts):
            ep = args.base_seed + i
            tgt = target_blocks[ep]

            r = run_baseline_episode(
                env, model, sampler, scorer,
                obs_m, obs_s, act_m, act_s, device,
                episode_seed=ep, target_block=tgt,
                max_steps=args.max_steps)
            b_rolls.append(r)

            ok = "OK" if r["success"] else "FAIL"
            legs = r["progressive_legibility"]
            leg_str = "  ".join(
                f"{tk}={legs.get(tk, 0):.2f}" for tk in ["t1","t2","t3","t5"])
            print(f"  [{i+1:3d}/{args.n_rollouts}]  ep={ep:<4}  "
                  f"arc={r['arc']:.4f}m ({r['arc_class']})  {ok}  "
                  f"legs: {leg_str}")

            # Crash-safe save
            _save_json(dict(condition="baseline", idx=i+1, latest=r),
                       out / "baseline_progress.json")

        env.close()
        dt = time.time() - t0
        bs = compute_summary(b_rolls, "baseline")

        print(f"\n  Done in {dt:.0f}s  ({bs.get('total_vlm_calls',0)} VLM calls)")
        print(f"  Success: {bs['success_count']}/{bs['n']} ({bs['success_rate']:.0%})")
        print(f"  Mean arc: {bs['mean_arc']:.4f} ± {bs['std_arc']:.4f} m")
        for tk in ["t1","t2","t3","t5"]:
            print(f"  Mean leg @{tk}: {bs.get(f'mean_leg_{tk}',0):.3f}")
        print()

        _save_json(dict(condition="baseline", timestamp=ts,
                        summary=bs, rollouts=b_rolls),
                   out / "baseline_results.json")
    else:
        prev = out / "baseline_results.json"
        if prev.exists():
            with open(prev) as f:
                d = json.load(f)
            b_rolls = d["rollouts"]; bs = d["summary"]
            print(f"  Loaded {len(b_rolls)} baseline results\n")
        else:
            bs = {}
            print("  (no baseline data)\n")

    # ═════════════════════════════════════════════════════════════
    # CONDITION 2 — VLM PROGRESSIVE STEERING
    # ═════════════════════════════════════════════════════════════
    s_rolls: List[Dict] = []

    if not args.skip_steered:
        est_calls = args.n_rollouts * (3 + 2 * args.K + 1)
        print(f"{'═' * 78}")
        print(f"  CONDITION 2: VLM PROGRESSIVE STEERING  "
              f"({args.n_rollouts} rollouts, K={args.K})")
        print(f"  Estimated VLM calls: ~{est_calls}")
        print(f"  Legibility threshold θ = {args.threshold}")
        print(f"{'═' * 78}\n")

        comp_dir = (out / "composites") if args.save_composites else None
        if comp_dir:
            comp_dir.mkdir(parents=True, exist_ok=True)

        env = TwoBlockPickEnv(render=False, episode_length=args.max_steps,
                              cube_jitter=0.0)
        t0 = time.time()

        for i in range(args.n_rollouts):
            ep = args.base_seed + i
            tgt = target_blocks[ep]

            r = run_steered_episode(
                env, model, sampler, scorer,
                obs_m, obs_s, act_m, act_s, device,
                episode_seed=ep, target_block=tgt,
                n_candidates=args.K,
                legibility_threshold=args.threshold,
                max_steps=args.max_steps,
                save_dir=comp_dir)
            s_rolls.append(r)

            ok = "OK" if r["success"] else "FAIL"
            legs = r["progressive_legibility"]
            leg_str = "  ".join(
                f"{tk}={legs.get(tk, 0):.2f}" for tk in ["t1","t2","t3","t5"])
            intv = r["interventions"]
            tag = f"  *** {intv} intervention{'s' if intv != 1 else ''}" \
                  if intv > 0 else ""

            print(f"  [{i+1:3d}/{args.n_rollouts}]  ep={ep:<4}  "
                  f"arc={r['arc']:.4f}m ({r['arc_class']})  {ok}  "
                  f"legs: {leg_str}{tag}")

            # Print checkpoint details for interventions
            for cp_key, cp_info in r.get("checkpoint_details", {}).items():
                if cp_info.get("intervened"):
                    cands = cp_info.get("candidates", [])
                    cand_legs = [c["legibility"] for c in cands]
                    sel_idx = cp_info.get("selected", "?")
                    eh = cp_info.get("eval_horizon", "?")
                    print(f"      {cp_key}: intervened "
                          f"({cp_info['legibility']:.2f} < θ)  "
                          f"look-ahead→t={eh}  "
                          f"sel c{sel_idx}  "
                          f"legs=[{min(cand_legs):.2f}..{max(cand_legs):.2f}]")

            # Crash-safe save
            _save_json(dict(condition="vlm_steered", idx=i+1, latest=r),
                       out / "steered_progress.json")

        env.close()
        dt = time.time() - t0
        ss = compute_summary(s_rolls, "vlm_steered")

        print(f"\n  Done in {dt:.0f}s  ({ss.get('total_vlm_calls',0)} VLM calls)")
        print(f"  Success: {ss['success_count']}/{ss['n']} ({ss['success_rate']:.0%})")
        print(f"  Mean arc: {ss['mean_arc']:.4f} ± {ss['std_arc']:.4f} m")
        print(f"  Mean interventions: {ss.get('mean_interventions',0):.1f}")
        for tk in ["t1","t2","t3","t5"]:
            print(f"  Mean leg @{tk}: {ss.get(f'mean_leg_{tk}',0):.3f}")
        print()

        _save_json(dict(condition="vlm_steered", timestamp=ts,
                        summary=ss,
                        rollouts=[{k: v for k, v in r.items()
                                   if k != "checkpoint_details"}
                                  for r in s_rolls],
                        checkpoint_details=[
                            dict(episode_seed=r["episode_seed"],
                                 details=r.get("checkpoint_details", {}))
                            for r in s_rolls]),
                   out / "steered_results.json")
    else:
        prev = out / "steered_results.json"
        if prev.exists():
            with open(prev) as f:
                d = json.load(f)
            s_rolls = d.get("rollouts", []); ss = d.get("summary", {})
            print(f"  Loaded {len(s_rolls)} steered results\n")
        else:
            ss = {}
            print("  (no steered data)\n")

    # ═════════════════════════════════════════════════════════════
    # COMPARISON
    # ═════════════════════════════════════════════════════════════

    if bs and ss:
        print(f"\n{'█' * 78}")
        print(f"  RESULTS — PROGRESSIVE VLM STEERING")
        print(f"{'█' * 78}\n")

        hdr = f"  {'Metric':<28}{'Baseline':>14}{'VLM Steering':>14}"
        sep = f"  {'─' * 56}"
        print(hdr); print(sep)

        print(f"  {'N rollouts':<28}{bs['n']:>14}{ss['n']:>14}")
        print(f"  {'Success rate':<28}"
              f"{bs['success_rate']:>13.0%} {ss['success_rate']:>13.0%}")
        print(f"  {'Mean arc (m)':<28}"
              f"{bs['mean_arc']:>13.4f}{ss['mean_arc']:>14.4f}")

        for ac in ["00-04", "05-09", "10-14", "15-19"]:
            bp = bs["arc_distribution_pct"].get(ac, 0)
            sp = ss["arc_distribution_pct"].get(ac, 0)
            mark = "  ◀" if ac == "15-19" else ""
            print(f"  {'Arc ' + ac:<28}{bp:>13.0f}%{sp:>13.0f}%{mark}")

        print(sep)
        print(f"  {'PROGRESSIVE LEGIBILITY':<28}")
        for tk, label in [("t1","Legibility @ t=1s"),
                          ("t2","Legibility @ t=2s"),
                          ("t3","Legibility @ t=3s"),
                          ("t5","Legibility @ t=5s")]:
            bv = bs.get(f"mean_leg_{tk}", 0)
            sv = ss.get(f"mean_leg_{tk}", 0)
            delta = f"  (+{(sv-bv):.3f})" if sv > bv else ""
            print(f"  {label:<28}{bv:>13.3f}{sv:>14.3f}{delta}")

        print(sep)
        print(f"  {'EARLY INTENT DETECTION':<28}")
        for tk, label in [("t1","Legible by t=1s"),
                          ("t2","Legible by t=2s"),
                          ("t3","Legible by t=3s")]:
            bv = bs.get(f"early_intent_{tk}", 0)
            sv = ss.get(f"early_intent_{tk}", 0)
            print(f"  {label:<28}{bv:>13.0%}{sv:>14.0%}")

        print(sep)
        print(f"  {'Avg interventions':<28}{'0':>14}"
              f"{ss.get('mean_interventions',0):>14.1f}")
        print(f"  {'Total VLM calls':<28}"
              f"{bs.get('total_vlm_calls',0):>14}"
              f"{ss.get('total_vlm_calls',0):>14}")

        print(f"\n  Generating plots ...")
        generate_progressive_plot(bs, ss, b_rolls, s_rolls, out)
        generate_per_episode_plot(b_rolls, s_rolls, out)

    # ── Save combined results ────────────────────────────────────
    final = dict(
        experiment=dict(
            timestamp=ts,
            method="progressive_vlm_steering",
            description=(
                "Closed-loop VLM steering at 1-second checkpoints. "
                "VLM evaluates trajectory legibility and selects "
                "the most legible continuation when ambiguous. "
                "Higher arcs are an EMERGENT property of legibility "
                "pressure — never selected explicitly."
            ),
            checkpoint=args.checkpoint,
            n_rollouts=args.n_rollouts,
            K=args.K,
            legibility_threshold=args.threshold,
            steer_checkpoints_sec=STEER_CHECKPOINTS,
            frame_annotation="t=0s..t=5s overlay + goal legend",
            selection_criterion="max VLM legibility (NOT arc height)",
            arc_thresholds_bezier=dict(t1=ARC_T1, t2=ARC_T2, t3=ARC_T3),
        ),
        baseline=dict(summary=bs, rollouts=b_rolls) if bs else None,
        vlm_steered=dict(
            summary=ss,
            rollouts=[{k: v for k, v in r.items()
                       if k != "checkpoint_details"}
                      for r in s_rolls],
        ) if ss else None,
    )
    jp = out / f"results_{ts}.json"
    with open(jp, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n  Results → {jp}")

    lp = out / "results_latest.json"
    with open(lp, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"  Latest  → {lp}")
    print(f"\n{'█' * 78}\n")


if __name__ == "__main__":
    main()
