#!/usr/bin/env python3
"""
VLM Steering Experiment  (ICRA Evaluation)
===========================================

Runs two experimental conditions as a **paired** design:

  Condition 1 (No Steering):  N rollouts with raw diffusion policy
  Condition 2 (VLM Steering): N rollouts with VLM best-of-K re-ranking

Both conditions share the SAME episode seed (block positions) and the
SAME target block per episode.  Only the trajectory *selection* differs.

Legibility Metric
-----------------
  - **Arc magnitude**: max lateral |Y| of end-effector in first 5 s
      Bézier-based thresholds: 0.0786, 0.1047, 0.1335
      Arc ≥ 0.1335 m  ⇒  arc class 15-19 (most legible)
  - **VLM legibility score**: max(pA, pB) from Gemini 2.5 Flash

VLM Frame Annotation  (the fix)
-------------------------------
  Every frame sent to VLM is annotated with:
    • Timestamp overlay   "t = 0s" … "t = 5s"
    • Goal legend on first frame ("Goal A: …", "Goal B: …")
  Frames are sent as `prefix_frames` for full temporal context.

Usage
-----
  # Dry-run (5 episodes) to verify everything works
  python evaluation/vlm_steering_experiment.py --n-rollouts 5 --n-candidates 5

  # Full experiment
  python evaluation/vlm_steering_experiment.py --n-rollouts 50 --n-candidates 10

  # Re-do only VLM condition (reuses saved no-steering data)
  python evaluation/vlm_steering_experiment.py --skip-no-steering
"""

import argparse, io, json, sys, time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.twoblockpick_env import TwoBlockPickEnv
from scripts.eval_with_videos import DiffusionPolicy, DDIMSampler
from scripts.vlm_client import LegibilityScorer
from evaluation.legibility_metrics import compute_legibility_from_obs

# ═════════════════════════════════════════════════════════════════════════
# CONSTANTS – Bézier-based arc thresholds
# ═════════════════════════════════════════════════════════════════════════

ARC_T1   = 0.0786   # 00-04 / 05-09 boundary
ARC_T2   = 0.1047   # 05-09 / 10-14 boundary
ARC_T3   = 0.1335   # 10-14 / 15-19 boundary  (legibility threshold)

GOAL_A   = "pick the left block"
GOAL_B   = "pick the right block"

# Steps at which we capture frames (30 Hz → 1 s intervals)
CAPTURE_STEPS = {0: 0, 30: 1, 60: 2, 90: 3, 120: 4, 149: 5}

# Block positions (world coords) — used for trajectory overlays
_BLOCK_LEFT_POS  = np.array([0.50, 0.07])   # X, Y
_BLOCK_RIGHT_POS = np.array([0.50, -0.07])
_EE_START_POS    = np.array([0.40, 0.0])


# ═════════════════════════════════════════════════════════════════════════
# ARC MEASUREMENT & CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════

def measure_arc(obs_traj: np.ndarray) -> float:
    """Max lateral |Y| deviation of end-effector (obs[:, 1])."""
    if len(obs_traj) == 0:
        return 0.0
    return float(np.max(np.abs(obs_traj[:, 1])))

def arc_class(arc: float) -> str:
    if arc < ARC_T1:   return "00-04"
    if arc < ARC_T2:   return "05-09"
    if arc < ARC_T3:   return "10-14"
    return "15-19"

def is_arc15_19(arc: float) -> bool:
    return arc >= ARC_T3


# ═════════════════════════════════════════════════════════════════════════
# ACTION HELPERS
# ═════════════════════════════════════════════════════════════════════════

def stabilize_gripper(actions: np.ndarray) -> np.ndarray:
    out = actions.copy()
    if out.shape[1] >= 5:
        out[:, 4] = float(np.clip(out[0, 4], -1.0, 1.0))
    return out

def infer_block(actions: np.ndarray) -> str:
    """LEFT if average initial dy > 0, else RIGHT."""
    n = min(20, len(actions))
    return "LEFT" if float(np.mean(actions[:n, 1])) > 0 else "RIGHT"

def enforce_block_direction(actions: np.ndarray, target: str) -> np.ndarray:
    """Force first 15 steps toward *target* block."""
    out = actions.copy()
    sign = 1.0 if target == "LEFT" else -1.0
    n = min(15, len(out))
    out[:n, 1] = sign * np.abs(out[:n, 1])
    return out


# ═════════════════════════════════════════════════════════════════════════
# POLICY LOADING & SAMPLING
# ═════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════
# FRAME ANNOTATION  (timestamp + goal labels)
# ═════════════════════════════════════════════════════════════════════════

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
    """
    Overlay timestamp (and optional goal legend) on a rendered frame.

    Args
    ----
    frame_rgb : (H, W, 3) uint8 array from ``env.render(mode='rgb_array')``
    t_sec     : integer seconds 0–5
    show_goals: if True, draw Goal A / Goal B legend (first frame only)

    Returns
    -------
    PIL.Image.Image with overlays
    """
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    font_lg, font_sm = _get_fonts()
    PAD = 5

    # ── timestamp ────────────────────────────────────────────────────
    ts_text = f"t = {t_sec}s"
    bb = draw.textbbox((15, 10), ts_text, font=font_lg)
    draw.rectangle([bb[0]-PAD, bb[1]-PAD, bb[2]+PAD, bb[3]+PAD], fill="black")
    draw.text((15, 10), ts_text, fill="white", font=font_lg)

    # ── goal legend (first frame) ────────────────────────────────────
    if show_goals:
        y = bb[3] + PAD + 10
        for lbl, clr in [(f"Goal A: {GOAL_A}", (255, 210, 0)),
                          (f"Goal B: {GOAL_B}", (100, 200, 255))]:
            gb = draw.textbbox((15, y), lbl, font=font_sm)
            draw.rectangle([gb[0]-3, gb[1]-2, gb[2]+3, gb[3]+2], fill="black")
            draw.text((15, y), lbl, fill=clr, font=font_sm)
            y = gb[3] + 8

    return img


def make_trajectory_plot(obs_traj: np.ndarray,
                         target_block: str = "RIGHT") -> Image.Image:
    """
    Create a top-down (bird's-eye) trajectory plot showing the EE path
    and block positions.  Sent to VLM as final frame.

    NOTE: Does NOT show which block is the target — VLM must infer
    direction purely from the trajectory shape.

    Args
    ----
    obs_traj : (T, 22) observations from rollout — obs[:, 0:2] = EE (X, Y)
    target_block : "LEFT" or "RIGHT" (unused — kept for API compat)

    Returns
    -------
    PIL.Image.Image (640×480)
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=100)
    fig.patch.set_facecolor('white')

    ee_x = obs_traj[:, 0]
    ee_y = obs_traj[:, 1]

    # Draw EE trajectory with time-color gradient
    n = len(ee_x)
    for i in range(n - 1):
        frac = i / max(n - 1, 1)
        color = (0.2, 0.2 + 0.6 * frac, 1.0 - 0.7 * frac)  # blue→green
        ax.plot(ee_y[i:i+2], ee_x[i:i+2], color=color, linewidth=2.5)

    # Start marker
    ax.plot(ee_y[0], ee_x[0], 'ko', markersize=10, label='Start')

    # End marker with arrow showing direction of motion
    if n > 2:
        dx = ee_x[-1] - ee_x[-3]
        dy = ee_y[-1] - ee_y[-3]
        ax.annotate('', xy=(ee_y[-1] + dy * 0.3, ee_x[-1] + dx * 0.3),
                    xytext=(ee_y[-1], ee_x[-1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.plot(ee_y[-1], ee_x[-1], 'r^', markersize=12, label='Current pos')

    # Block positions — draw as labeled squares
    bA = _BLOCK_LEFT_POS   # Goal A = left block (Y=+0.07)
    bB = _BLOCK_RIGHT_POS  # Goal B = right block (Y=-0.07)

    sq_sz = 0.02
    for bpos, label, color in [(bA, 'A (left)', '#FFD200'),
                                (bB, 'B (right)', '#64C8FF')]:
        rect = plt.Rectangle((bpos[1] - sq_sz, bpos[0] - sq_sz),
                              2*sq_sz, 2*sq_sz,
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(bpos[1], bpos[0] + sq_sz + 0.015, label,
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

    ax.set_xlabel('Y (lateral: + = left, - = right)', fontsize=11)
    ax.set_ylabel('X (forward)', fontsize=11)
    ax.set_title('Top-Down View: Gripper Trajectory (5 sec prefix)',
                 fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    # Set axis limits to show full workspace
    ax.set_xlim(-0.20, 0.20)
    ax.set_ylim(0.25, 0.60)

    fig.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# ═════════════════════════════════════════════════════════════════════════
# SINGLE 5-SEC ROLLOUT  (150 steps @ 30 Hz, with replanning)
# ═════════════════════════════════════════════════════════════════════════

def run_5sec_rollout(env, model, sampler, obs,
                     obs_mean, obs_std, act_mean, act_std, device,
                     target_block: str, *,
                     capture_frames: bool = False):
    """
    Execute 150 steps with closed-loop replanning.

    Returns
    -------
    dict with keys: observations, actions, arc, arc_class,
                    is_arc15_19, direction_match, frames (list[Image])
    
    When capture_frames=True, frames includes:
      - 6 annotated camera frames (t=0..5s)
      - 1 top-down trajectory plot (bird's-eye showing arc + block positions)
    """
    q: deque = deque()
    observations, actions_taken, frames = [], [], []

    for step in range(150):
        if len(q) == 0:
            seq = sample_chunk(model, sampler, obs,
                               obs_mean, obs_std, act_mean, act_std, device)
            seq = enforce_block_direction(seq, target_block)
            q.extend(seq)

        action = q.popleft()
        result = env.step(action)
        observations.append(result.obs)
        actions_taken.append(action)
        obs = result.obs

        if capture_frames and step in CAPTURE_STEPS:
            t_sec = CAPTURE_STEPS[step]
            fr = env.render(mode="rgb_array")
            frames.append(annotate_frame(fr, t_sec, show_goals=(t_sec == 0)))

        if result.done:
            # pad to 150 with last obs for consistent arc measurement
            while len(observations) < 150:
                observations.append(result.obs)
                actions_taken.append(action)
            break

    obs_arr = np.array(observations)
    act_arr = np.array(actions_taken)
    arc_val = measure_arc(obs_arr)
    actual_dir = infer_block(act_arr) if len(act_arr) >= 20 else "UNKNOWN"

    # ── trajectory plot for VLM (top-down view of EE path) ────────
    if capture_frames:
        traj_img = make_trajectory_plot(obs_arr, target_block=target_block)
        frames.append(traj_img)

    # ── task-agnostic legibility metrics ──────────────────────────
    try:
        leg_result = compute_legibility_from_obs(
            obs_arr, target_block, model="both", return_curves=False)
        leg_scores = leg_result.as_dict()
    except Exception:
        leg_scores = {}

    return dict(
        observations=obs_arr, actions=act_arr,
        arc=arc_val, arc_class=arc_class(arc_val),
        is_arc15_19=is_arc15_19(arc_val),
        actual_direction=actual_dir,
        direction_match=(actual_dir == target_block),
        frames=frames, final_obs=obs,
        legibility_metrics=leg_scores,
    )


# ═════════════════════════════════════════════════════════════════════════
# VLM SCORING  (with retry logic)
# ═════════════════════════════════════════════════════════════════════════

def score_candidate_vlm(scorer: LegibilityScorer,
                        frames: List[Image.Image],
                        video_id: str,
                        target_block: str = "RIGHT") -> Dict:
    """
    Send *annotated* frames to VLM via ``prefix_frames`` mode.

    Returns dict with direction-aware ``legibility_score`` =
    P(correct goal | frames), plus raw ``pA``, ``pB``.

    Args:
        target_block: "LEFT" or "RIGHT" — needed to compute
                      direction-correct legibility (pA if LEFT, pB if RIGHT).
    """
    # Map target_block to goal label for direction-aware scoring
    target_goal = "A" if target_block == "LEFT" else "B"

    # encode as JPEG bytes
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
                video_id=video_id, t_sec=5.0,
                target_goal=target_goal,
            )
            return dict(
                legibility_score=float(r.get("legibility_score", 0.5)),
                undirected_legibility=float(r.get("undirected_legibility", 0.5)),
                clarity=float(r.get("clarity", 0.5)),
                pA=float(r.get("pA", 0.5)),
                pB=float(r.get("pB", 0.5)),
                target_goal=target_goal,
                choice=r.get("choice", "C"),
                cue=str(r.get("cue", "")),
                confidence=int(r.get("confidence", 50)),
                latency_ms=r.get("latency_ms", 0),
                vlm_error=bool(r.get("vlm_error", False)),
                error=None,
            )
        except Exception as exc:
            if attempt < 2:
                wait = (attempt + 1) * 10
                print(f"      [VLM retry {attempt+1}, wait {wait}s: {exc}]")
                time.sleep(wait)
            else:
                print(f"      [VLM FAILED 3×: {exc}]")
                return dict(
                    legibility_score=0.5, undirected_legibility=0.5,
                    clarity=0.5,
                    pA=0.5, pB=0.5, target_goal=target_goal,
                    choice="C", cue=f"ERROR: {exc}",
                    confidence=0, latency_ms=0,
                    vlm_error=True, error=str(exc),
                )


# ═════════════════════════════════════════════════════════════════════════
# VLM SMOKE TEST  (run before spending money)
# ═════════════════════════════════════════════════════════════════════════

def vlm_smoke_test(scorer: LegibilityScorer) -> bool:
    """Send a single tiny test image to Gemini.  Returns True on success."""
    print("  Running VLM smoke test (1 cheap API call) ...")
    img = Image.new("RGB", (64, 64), color="gray")
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "test", fill="white")
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=50)
    try:
        r = scorer.score_trajectory(
            image_bytes=buf.getvalue(),
            goal_A="pick left", goal_B="pick right",
            mode="single_frame", video_id="smoke_test", t_sec=0.0,
        )
        print(f"    VLM OK  pA={r.get('pA',0):.2f}  "
              f"pB={r.get('pB',0):.2f}  latency={r.get('latency_ms',0)}ms")
        return True
    except Exception as exc:
        print(f"    VLM FAILED: {exc}")
        return False


# ═════════════════════════════════════════════════════════════════════════
# CONDITION 1  –  NO STEERING
# ═════════════════════════════════════════════════════════════════════════

def run_no_steering_episode(env, model, sampler,
                            obs_mean, obs_std, act_mean, act_std,
                            device, episode_seed: int,
                            max_steps: int = 400) -> Dict:
    """Run raw policy with replanning.  Returns episode result dict."""
    seed = episode_seed * 100 + 7
    np.random.seed(seed); torch.manual_seed(seed)

    obs = env.reset(seed=episode_seed)
    q: deque = deque(); observations = []; first_actions = None
    done = False; steps = 0; result = None

    while not done and steps < max_steps:
        if len(q) == 0:
            seq = sample_chunk(model, sampler, obs,
                               obs_mean, obs_std, act_mean, act_std, device)
            if first_actions is None:
                first_actions = seq.copy()
            q.extend(seq)
        action = q.popleft()
        result = env.step(action)
        observations.append(result.obs)
        obs = result.obs
        done = bool(result.done); steps += 1

    info = result.info if result else {}
    pl = info.get("picked_left", 0) > 0.5
    pr = info.get("picked_right", 0) > 0.5
    success = info.get("success_left", 0) > 0.5 or info.get("success_right", 0) > 0.5
    picked = ("LEFT" if (pl and not pr) else
              "RIGHT" if (pr and not pl) else
              "BOTH" if (pl and pr) else "NONE")

    target = infer_block(first_actions) if first_actions is not None else "UNKNOWN"
    obs_5s = np.array(observations[:150]) if len(observations) >= 150 \
             else np.array(observations)
    arc_val = measure_arc(obs_5s)

    # ── task-agnostic legibility metrics ──────────────────────────
    try:
        leg_result = compute_legibility_from_obs(
            obs_5s, target, model="both", return_curves=False)
        leg_scores = leg_result.as_dict()
    except Exception:
        leg_scores = {}

    return dict(
        episode_seed=episode_seed, success=success, picked=picked,
        target_block=target,
        arc=float(arc_val), arc_class=arc_class(arc_val),
        is_arc15_19=is_arc15_19(arc_val), steps=steps,
        legibility_metrics=leg_scores,
    )


# ═════════════════════════════════════════════════════════════════════════
# BASELINE VLM SCORING  (optional — re-run 5s with frame capture)
# ═════════════════════════════════════════════════════════════════════════

def vlm_score_baseline_episode(env, model, sampler, scorer,
                               obs_mean, obs_std, act_mean, act_std,
                               device, episode_seed: int,
                               target_block: str) -> Dict:
    """
    Re-run baseline 5-second trajectory with frame capture, then VLM-score.
    Uses the SAME seed as the no-steering episode for identical trajectory.
    """
    seed = episode_seed * 100 + 7
    np.random.seed(seed); torch.manual_seed(seed)
    obs = env.reset(seed=episode_seed)

    ro = run_5sec_rollout(env, model, sampler, obs,
                          obs_mean, obs_std, act_mean, act_std, device,
                          target_block, capture_frames=True)

    if ro["frames"]:
        vlm = score_candidate_vlm(scorer, ro["frames"],
                                  f"ep{episode_seed}_baseline")
    else:
        vlm = dict(legibility_score=0.5, pA=0.5, pB=0.5,
                   choice="C", cue="no_frames", confidence=0, error="no frames")

    return dict(legibility=float(vlm["legibility_score"]),
                vlm_choice=vlm["choice"],
                vlm_cue=vlm["cue"])


# ═════════════════════════════════════════════════════════════════════════
# CONDITION 2  –  VLM BEST-OF-K STEERING
# ═════════════════════════════════════════════════════════════════════════

def run_vlm_steering_episode(env, model, sampler, scorer,
                             obs_mean, obs_std, act_mean, act_std,
                             device, episode_seed: int,
                             target_block: str, n_candidates: int = 10,
                             max_steps: int = 400,
                             save_frames_dir: Optional[Path] = None,
                             selection_mode: str = "legible_max_arc",
                             leg_threshold: float = 0.55,
                             max_arc_cap: float = 0.18) -> Dict:
    """
    Generate K candidate 5-sec trajectories, VLM-score each,
    select the most legible, and execute to completion.
    """
    candidates = []

    # ── Phase 1: generate K candidates ──────────────────────────────
    for j in range(n_candidates):
        cseed = episode_seed * 100 + 1000 + j
        np.random.seed(cseed); torch.manual_seed(cseed)

        obs = env.reset(seed=episode_seed)
        ro = run_5sec_rollout(env, model, sampler, obs,
                              obs_mean, obs_std, act_mean, act_std, device,
                              target_block,
                              capture_frames=True)

        candidates.append(dict(
            idx=j, seed=cseed,
            arc=ro["arc"], arc_class=ro["arc_class"],
            is_arc15_19=ro["is_arc15_19"],
            direction_match=ro["direction_match"],
            actual_direction=ro["actual_direction"],
            frames=ro["frames"],        # annotated PIL images
            actions=ro["actions"],       # np array for replay
            legibility_metrics=ro.get("legibility_metrics", {}),
        ))

    # ── Phase 2: VLM scoring ────────────────────────────────────────
    for c in candidates:
        vid = f"ep{episode_seed}_c{c['idx']}"

        if not c["frames"]:
            c["vlm"] = dict(legibility_score=0.5, undirected_legibility=0.5,
                            pA=0.5, pB=0.5, target_goal="A" if target_block == "LEFT" else "B",
                            choice="C", cue="no_frames", confidence=0,
                            vlm_error=True, error="no frames")
        else:
            c["vlm"] = score_candidate_vlm(scorer, c["frames"], vid,
                                            target_block=target_block)

        # ── optionally save annotated composites to disk ─────────
        if save_frames_dir is not None and c["frames"]:
            comp = _make_composite(c["frames"])
            comp.save(save_frames_dir / f"ep{episode_seed}_c{c['idx']}.jpg",
                      quality=90)

    # ── Phase 3: select best candidate ───────────────────────────────
    # Filter out candidates where VLM failed (don't let noise into selection)
    pool = [c for c in candidates if c["direction_match"]]
    if not pool:
        pool = candidates  # fallback

    # Separate VLM-valid from VLM-failed candidates
    vlm_valid = [c for c in pool if not c["vlm"].get("vlm_error", False)]
    vlm_pool = vlm_valid if vlm_valid else pool  # fall back to all if all failed

    if selection_mode == "legible_max_arc":
        # VLM as binary gate → geometry argmax within safe arc range.
        # NOTE: This mode uses VLM only as pass/fail filter; arc magnitude
        # determines the actual winner. See "max_legibility" for true VLM ranking.
        legible = [c for c in vlm_pool
                   if c["vlm"]["legibility_score"] >= leg_threshold]
        if legible:
            safe = [c for c in legible if c["arc"] <= max_arc_cap]
            if safe:
                sel = max(safe, key=lambda c: c["arc"])
            else:
                sel = min(legible, key=lambda c: c["arc"])
        else:
            # No candidate is legible → pick highest arc as best-effort
            safe_all = [c for c in vlm_pool if c["arc"] <= max_arc_cap]
            if safe_all:
                sel = max(safe_all, key=lambda c: c["arc"])
            else:
                sel = max(vlm_pool, key=lambda c: c["arc"])

    elif selection_mode == "max_legibility":
        # True VLM ranking: pick candidate with highest direction-aware
        # legibility score (P(correct goal | frames)).
        sel = max(vlm_pool, key=lambda c: c["vlm"]["legibility_score"])

    elif selection_mode == "vlm_weighted":
        # Hybrid: rank by VLM legibility * arc_bonus.
        # This balances VLM signal with geometric distinctiveness.
        def hybrid_score(c):
            leg = c["vlm"]["legibility_score"]
            arc = c["arc"]
            arc_bonus = min(arc / max_arc_cap, 1.0)  # 0-1 scale
            return leg * (1.0 + 0.5 * arc_bonus)  # VLM dominates, arc is tiebreaker
        sel = max(vlm_pool, key=hybrid_score)

    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    sel_idx = sel["idx"]

    # ── Phase 4: replay selected + replan to completion ─────────────
    # Deterministic seed for reproducible replanning after the 5-sec window
    replay_seed = sel["seed"] + 9999
    np.random.seed(replay_seed); torch.manual_seed(replay_seed)

    obs = env.reset(seed=episode_seed)
    q = deque(sel["actions"])
    done = False; steps = 0; result = None; replans = 0

    while not done and steps < max_steps:
        if len(q) == 0:
            replans += 1
            seq = sample_chunk(model, sampler, obs,
                               obs_mean, obs_std, act_mean, act_std, device)
            # No direction enforcement during free-run replanning;
            # the robot is already near the target block at this point.
            q.extend(seq)
        action = q.popleft()
        result = env.step(action)
        obs = result.obs; done = bool(result.done); steps += 1

    info = result.info if result else {}
    pl = info.get("picked_left", 0) > 0.5
    pr = info.get("picked_right", 0) > 0.5
    success = info.get("success_left", 0) > 0.5 or info.get("success_right", 0) > 0.5
    picked = ("LEFT" if (pl and not pr) else
              "RIGHT" if (pr and not pl) else
              "BOTH" if (pl and pr) else "NONE")

    arc15_count = sum(1 for c in candidates if c["is_arc15_19"])

    return dict(
        episode_seed=episode_seed, target_block=target_block,
        success=success, picked=picked,
        arc=float(sel["arc"]), arc_class=sel["arc_class"],
        is_arc15_19=sel["is_arc15_19"],
        legibility=float(sel["vlm"]["legibility_score"]),
        vlm_choice=sel["vlm"]["choice"],
        vlm_cue=sel["vlm"]["cue"],
        selected_candidate=sel_idx,
        n_candidates=n_candidates,
        arc15_count=arc15_count,
        steps=steps, replans=replans,
        legibility_metrics=sel.get("legibility_metrics", {}),
        candidates=[
            dict(idx=c["idx"],
                 arc=float(c["arc"]),
                 arc_class=c["arc_class"],
                 is_arc15_19=c["is_arc15_19"],
                 direction_match=c["direction_match"],
                 legibility=float(c["vlm"]["legibility_score"]),
                 choice=c["vlm"]["choice"],
                 cue=(c["vlm"]["cue"] or "")[:120],
                 error=c["vlm"].get("error"),
                 L_composite=c.get("legibility_metrics", {}).get("L_composite"),
                 L_posterior=c.get("legibility_metrics", {}).get("L_posterior"),
                 L_early_intent=c.get("legibility_metrics", {}).get("L_early_intent"),
                 )
            for c in candidates
        ],
    )


def _make_composite(frames: List[Image.Image]) -> Image.Image:
    """Stitch annotated frames into a single wide composite."""
    if not frames:
        return Image.new("RGB", (480, 480), "black")
    w, h = frames[0].size
    comp = Image.new("RGB", (w * len(frames), h), "black")
    for i, fr in enumerate(frames):
        comp.paste(fr, (i * w, 0))
    return comp


# ═════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════

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
        arc15_19_count=arc_cnt["15-19"],
        mean_arc=round(float(np.mean(arcs)), 5),
        std_arc=round(float(np.std(arcs)), 5),
        median_arc=round(float(np.median(arcs)), 5),
        min_arc=round(float(np.min(arcs)), 5),
        max_arc=round(float(np.max(arcs)), 5),
    )
    if "legibility" in rollouts[0]:
        legs = [r["legibility"] for r in rollouts]
        s["mean_legibility"] = round(float(np.mean(legs)), 4)
        s["std_legibility"]  = round(float(np.std(legs)), 4)
        s["vlm_calls"] = sum(r.get("n_candidates", 0) for r in rollouts)

    # ── task-agnostic legibility metrics (aggregate) ─────────────
    # NOTE: L_geometric excluded (saturates at 1.0 for all arcs — broken metric)
    metric_keys = ["L_composite", "L_posterior", "L_early_intent",
                   "L_commitment", "L_entropy_auc"]
    for mk in metric_keys:
        vals = [r.get("legibility_metrics", {}).get(mk)
                for r in rollouts
                if r.get("legibility_metrics", {}).get(mk) is not None]
        if vals:
            s[f"mean_{mk}"] = round(float(np.mean(vals)), 4)
            s[f"std_{mk}"]  = round(float(np.std(vals)), 4)
    return s


# ═════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════

def generate_comparison_plot(ns_sum, vs_sum, ns_rolls, vs_rolls, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── bar chart ────────────────────────────────────────────────────
    cats = ["00-04", "05-09", "10-14", "15-19"]
    x = np.arange(len(cats)); w = 0.35
    ns_p = [ns_sum["arc_distribution_pct"][c] for c in cats]
    vs_p = [vs_sum["arc_distribution_pct"][c] for c in cats]

    ax = axes[0]
    b1 = ax.bar(x - w/2, ns_p, w, label="No Steering",  color="#4a90d9", edgecolor="k")
    b2 = ax.bar(x + w/2, vs_p, w, label="VLM Steering", color="#e6783c", edgecolor="k")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}%",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Arc Class"); ax.set_ylabel("Percentage (%)")
    ax.set_title("Arc Distribution Comparison")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.legend(); ax.grid(axis="y", alpha=.3)

    # ── histogram ────────────────────────────────────────────────────
    ax = axes[1]
    na = [r["arc"] for r in ns_rolls]
    va = [r["arc"] for r in vs_rolls]
    hi = max(max(na), max(va)) + 0.02
    bins = np.linspace(0, hi, 22)
    ax.hist(na, bins=bins, alpha=.55, label="No Steering",  color="#4a90d9", edgecolor="k")
    ax.hist(va, bins=bins, alpha=.55, label="VLM Steering", color="#e6783c", edgecolor="k")
    for t, lbl in [(ARC_T1,"05-09"), (ARC_T2,"10-14"), (ARC_T3,"15-19")]:
        ax.axvline(t, color="gray", ls="--", alpha=.7)
        ax.text(t+.002, ax.get_ylim()[1]*.92, lbl, fontsize=8, color="gray")
    ax.set_xlabel("Arc Magnitude (m)"); ax.set_ylabel("Count")
    ax.set_title("Arc Histogram"); ax.legend(); ax.grid(axis="y", alpha=.3)

    # ── summary table ────────────────────────────────────────────────
    ax = axes[2]; ax.axis("off")
    rows = [
        ["Metric", "No Steering", "VLM Steering"],
        ["N rollouts",   str(ns_sum["n"]),  str(vs_sum["n"])],
        ["Success rate", f"{ns_sum['success_rate']:.0%}",
                         f"{vs_sum['success_rate']:.0%}"],
        ["Arc 15-19",    f"{ns_sum['arc15_19_count']} ({ns_sum['arc15_19_rate']:.0%})",
                         f"{vs_sum['arc15_19_count']} ({vs_sum['arc15_19_rate']:.0%})"],
        ["Mean arc",     f"{ns_sum['mean_arc']:.4f} m",
                         f"{vs_sum['mean_arc']:.4f} m"],
        ["Std arc",      f"{ns_sum['std_arc']:.4f} m",
                         f"{vs_sum['std_arc']:.4f} m"],
    ]
    if "mean_legibility" in vs_sum:
        ns_leg = ns_sum.get("mean_legibility")
        ns_leg_str = f"{ns_leg:.3f}" if ns_leg is not None else "N/A"
        rows.append(["Mean VLM leg.", ns_leg_str,
                      f"{vs_sum['mean_legibility']:.3f}"])
    tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 1.8)
    for j in range(3):
        tbl[0, j].set_facecolor("#ccc")
        tbl[0, j].set_text_props(fontweight="bold")
    for j in range(3):
        tbl[4, j].set_facecolor("#fff3cd")  # highlight arc 15-19 row
    ax.set_title("Summary", pad=20)

    plt.tight_layout()
    p = out / "arc_distribution_comparison.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot → {p}")


# ═════════════════════════════════════════════════════════════════════════
# TASK-AGNOSTIC LEGIBILITY COMPARISON PLOT
# ═════════════════════════════════════════════════════════════════════════

def generate_legibility_comparison_plot(ns_sum, vs_sum, ns_rolls, vs_rolls, out: Path):
    """Generate a multi-panel comparison plot for the Bayesian legibility metrics.

    L_geometric is excluded (saturation bug).  Reported metrics in order:
    1. L_early_intent  (PRIMARY)
    2. L_posterior     (ROBUSTNESS)
    3. L_composite     (AGGREGATE)
    4. L_commitment    (SUPPLEMENTARY)
    5. L_entropy_auc   (SUPPLEMENTARY)
    """
    metric_keys = ["L_composite", "L_posterior", "L_early_intent",
                   "L_commitment", "L_entropy_auc"]
    labels = ["Composite", "Posterior", "Early\nIntent",
              "Commitment", "Entropy\nAUC"]

    def _extract(rolls, key):
        return [r.get("legibility_metrics", {}).get(key)
                for r in rolls
                if r.get("legibility_metrics", {}).get(key) is not None]

    ns_vals = {k: _extract(ns_rolls, k) for k in metric_keys}
    vs_vals = {k: _extract(vs_rolls, k) for k in metric_keys}

    # Only plot if we have data
    if not any(ns_vals[k] for k in metric_keys) and \
       not any(vs_vals[k] for k in metric_keys):
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── Panel 1: Grouped bar chart of mean scores ────────────────────
    ax = axes[0]
    x = np.arange(len(metric_keys))
    w = 0.35
    ns_means = [np.mean(ns_vals[k]) if ns_vals[k] else 0 for k in metric_keys]
    vs_means = [np.mean(vs_vals[k]) if vs_vals[k] else 0 for k in metric_keys]
    ns_stds  = [np.std(ns_vals[k])  if ns_vals[k] else 0 for k in metric_keys]
    vs_stds  = [np.std(vs_vals[k])  if vs_vals[k] else 0 for k in metric_keys]

    b1 = ax.bar(x - w/2, ns_means, w, yerr=ns_stds, capsize=3,
                label="No Steering", color="#4a90d9", edgecolor="k", alpha=0.85)
    b2 = ax.bar(x + w/2, vs_means, w, yerr=vs_stds, capsize=3,
                label="VLM Steering", color="#e6783c", edgecolor="k", alpha=0.85)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score [0-1]"); ax.set_ylim(0, 1.15)
    ax.set_title("Task-Agnostic Legibility Scores")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: L_composite histogram ────────────────────────────────
    ax = axes[1]
    ns_comp = ns_vals.get("L_composite", [])
    vs_comp = vs_vals.get("L_composite", [])
    if ns_comp or vs_comp:
        bins = np.linspace(0, 1, 22)
        if ns_comp:
            ax.hist(ns_comp, bins=bins, alpha=0.55, label="No Steering",
                    color="#4a90d9", edgecolor="k")
        if vs_comp:
            ax.hist(vs_comp, bins=bins, alpha=0.55, label="VLM Steering",
                    color="#e6783c", edgecolor="k")
        ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("L_composite"); ax.set_ylabel("Count")
        ax.set_title("Composite Legibility Distribution")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: L_composite vs arc scatter ──────────────────────────
    ax = axes[2]
    for rolls, label, color, marker in [
        (ns_rolls, "No Steering", "#4a90d9", "o"),
        (vs_rolls, "VLM Steering", "#e6783c", "^"),
    ]:
        arcs = [r["arc"] for r in rolls
                if r.get("legibility_metrics", {}).get("L_composite") is not None]
        comps = [r["legibility_metrics"]["L_composite"] for r in rolls
                 if r.get("legibility_metrics", {}).get("L_composite") is not None]
        if arcs and comps:
            ax.scatter(arcs, comps, c=color, marker=marker, alpha=0.6,
                       edgecolors="k", linewidths=0.5, s=50, label=label)

    ax.set_xlabel("Arc Magnitude (m)"); ax.set_ylabel("L_composite")
    ax.set_title("Arc vs Bayesian Legibility")
    ax.legend(); ax.grid(alpha=0.3)
    for t in [ARC_T1, ARC_T2, ARC_T3]:
        ax.axvline(t, color="gray", ls=":", alpha=0.4)

    plt.tight_layout()
    p = out / "legibility_comparison.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot → {p}")


# ═════════════════════════════════════════════════════════════════════════
# SAVE / LOAD INTERMEDIATE RESULTS  (crash-safety)
# ═════════════════════════════════════════════════════════════════════════

def _save_intermediate(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="VLM Steering Experiment (ICRA evaluation)")
    pa.add_argument("--checkpoint", default="runs/diffusion_20260222_195530/ckpt_ep100.pt")
    pa.add_argument("--n-rollouts",   type=int, default=50)
    pa.add_argument("--n-candidates", type=int, default=10)
    pa.add_argument("--base-seed",    type=int, default=100)
    pa.add_argument("--max-steps",    type=int, default=400)
    pa.add_argument("--output-dir",   default="outputs/vlm_steering_experiment")
    pa.add_argument("--skip-no-steering",  action="store_true")
    pa.add_argument("--skip-vlm-steering", action="store_true")
    pa.add_argument("--save-composites",   action="store_true",
                    help="Save annotated VLM composite images to disk")
    pa.add_argument("--score-baseline",    action="store_true",
                    help="Also VLM-score baseline trajectories for comparison")
    pa.add_argument("--selection-mode",
                    choices=["max_legibility", "legible_max_arc"],
                    default="legible_max_arc",
                    help="VLM selection strategy: "
                         "max_legibility = pick highest VLM score; "
                         "legible_max_arc = among legible (≥threshold), pick max arc")
    pa.add_argument("--legibility-threshold", type=float, default=0.55,
                    help="Legibility threshold for legible_max_arc mode (default 0.55)")
    pa.add_argument("--max-arc-cap", type=float, default=0.18,
                    help="Max arc cap (m) to avoid extreme arcs that hurt success (default 0.18)")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'█'*78}")
    print(f"  VLM STEERING EXPERIMENT")
    print(f"{'█'*78}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  N rollouts     : {args.n_rollouts} per condition")
    print(f"  N candidates K : {args.n_candidates}")
    print(f"  Base seed      : {args.base_seed}")
    print(f"  Max steps      : {args.max_steps}")
    print(f"  Output dir     : {out}")
    print(f"  Device         : {device}")
    print(f"  Arc thresholds : {ARC_T1}, {ARC_T2}, {ARC_T3}  (Bézier)")
    print(f"  Selection mode : {args.selection_mode}")
    if args.selection_mode == "legible_max_arc":
        print(f"  Legibility θ   : {args.legibility_threshold}")
    print(f"  Score baseline : {args.score_baseline}")
    print(f"{'█'*78}\n")

    # ── load policy ──────────────────────────────────────────────────
    print("Loading policy ...")
    model, sampler, obs_m, obs_s, act_m, act_s, cfg = \
        load_policy(args.checkpoint, device)
    print(f"  horizon={cfg['horizon']}  act_dim={cfg['act_dim']}\n")

    target_blocks: Dict[int, str] = {}   # episode_seed → target

    # ═════════════════════════════════════════════════════════════════
    # CONDITION 1:  NO STEERING
    # ═════════════════════════════════════════════════════════════════
    ns_rolls: List[Dict] = []

    if not args.skip_no_steering:
        print(f"{'═'*78}")
        print(f"  CONDITION 1: NO STEERING  ({args.n_rollouts} rollouts)")
        print(f"{'═'*78}\n")

        env = TwoBlockPickEnv(render=False, episode_length=args.max_steps,
                              cube_jitter=0.0)
        t0 = time.time()

        for i in range(args.n_rollouts):
            ep = args.base_seed + i
            r = run_no_steering_episode(
                env, model, sampler, obs_m, obs_s, act_m, act_s,
                device, ep, args.max_steps)
            ns_rolls.append(r)
            target_blocks[ep] = r["target_block"]

            tag = " *** ARC 15-19" if r["is_arc15_19"] else ""
            ok  = "OK" if r["success"] else "FAIL"
            print(f"  [{i+1:3d}/{args.n_rollouts}]  ep={ep}  "
                  f"arc={r['arc']:.4f}m ({r['arc_class']})  "
                  f"{ok}  tgt={r['target_block']}{tag}")

        env.close()
        dt = time.time() - t0

        ns_sum = compute_summary(ns_rolls, "no_steering")
        print(f"\n  Done in {dt:.0f}s")
        print(f"  Success : {ns_sum['success_count']}/{ns_sum['n']}  "
              f"({ns_sum['success_rate']:.0%})")
        print(f"  Arc 15-19 : {ns_sum['arc15_19_count']}/{ns_sum['n']}  "
              f"({ns_sum['arc15_19_rate']:.0%})")
        print(f"  Mean arc  : {ns_sum['mean_arc']:.4f} ± {ns_sum['std_arc']:.4f} m\n")

        # save intermediate
        _save_intermediate(dict(
            condition="no_steering", timestamp=ts,
            summary=ns_sum, rollouts=ns_rolls,
        ), out / "no_steering_results.json")
    else:
        # try to load previous results
        prev = out / "no_steering_results.json"
        if prev.exists():
            with open(prev) as f:
                d = json.load(f)
            ns_rolls = d["rollouts"]
            ns_sum   = d["summary"]
            for r in ns_rolls:
                target_blocks[r["episode_seed"]] = r["target_block"]
            print(f"  Loaded {len(ns_rolls)} previous no-steering results\n")
        else:
            ns_sum = {}
            print("  (no previous no-steering data found)\n")

    # ═════════════════════════════════════════════════════════════════
    # OPTIONAL:  VLM-SCORE BASELINE TRAJECTORIES
    # ═════════════════════════════════════════════════════════════════
    baseline_scorer = None

    if args.score_baseline and ns_rolls and not args.skip_vlm_steering:
        print(f"{'═'*78}")
        print(f"  SCORING BASELINE WITH VLM  ({len(ns_rolls)} rollouts)")
        print(f"  (Re-running 5-sec trajectories with frame capture + VLM)")
        print(f"{'═'*78}\n")

        try:
            baseline_scorer = LegibilityScorer(model="gemini-2.5-flash")
        except Exception as e:
            print(f"  Cannot init VLM for baseline scoring: {e}\n")

        if baseline_scorer is not None:
            env = TwoBlockPickEnv(render=False, episode_length=args.max_steps,
                                  cube_jitter=0.0)
            t0 = time.time()
            for i, r in enumerate(ns_rolls):
                ep  = r["episode_seed"]
                tgt = r["target_block"]
                vlm_info = vlm_score_baseline_episode(
                    env, model, sampler, baseline_scorer,
                    obs_m, obs_s, act_m, act_s, device, ep, tgt)
                r["legibility"] = vlm_info["legibility"]
                r["vlm_choice"] = vlm_info["vlm_choice"]
                r["vlm_cue"]    = vlm_info["vlm_cue"]
                print(f"  [{i+1:3d}/{len(ns_rolls)}]  ep={ep}  "
                      f"arc={r['arc']:.4f}m  leg={r['legibility']:.3f}")
            env.close()
            dt = time.time() - t0

            # recompute summary with legibility info
            ns_sum = compute_summary(ns_rolls, "no_steering")
            print(f"\n  Baseline VLM scoring done in {dt:.0f}s")
            if "mean_legibility" in ns_sum:
                print(f"  Mean baseline legibility: {ns_sum['mean_legibility']:.3f}\n")

            _save_intermediate(dict(
                condition="no_steering", timestamp=ts,
                summary=ns_sum, rollouts=ns_rolls,
            ), out / "no_steering_results.json")

    # ═════════════════════════════════════════════════════════════════
    # CONDITION 2:  VLM STEERING
    # ═════════════════════════════════════════════════════════════════
    vs_rolls: List[Dict] = []

    if not args.skip_vlm_steering:
        total_calls = args.n_rollouts * args.n_candidates
        print(f"{'═'*78}")
        print(f"  CONDITION 2: VLM STEERING  ({args.n_rollouts} rollouts, "
              f"K={args.n_candidates})")
        print(f"  Estimated VLM API calls: {total_calls}")
        print(f"{'═'*78}\n")

        # init scorer + smoke test
        scorer = LegibilityScorer(model="gemini-2.5-flash")
        if not vlm_smoke_test(scorer):
            print("\n  !! VLM smoke test FAILED – aborting VLM condition.")
            print("     Check API key / network.  No-steering data is saved.\n")
            vs_sum = {}
        else:
            comp_dir = (out / "composites") if args.save_composites else None
            if comp_dir:
                comp_dir.mkdir(parents=True, exist_ok=True)

            env = TwoBlockPickEnv(render=False, episode_length=args.max_steps,
                                  cube_jitter=0.0)
            t0 = time.time(); vlm_calls = 0

            for i in range(args.n_rollouts):
                ep = args.base_seed + i

                # resolve target block (same as no-steering if available)
                tgt = target_blocks.get(ep)
                if tgt is None:
                    obs = env.reset(seed=ep)
                    np.random.seed(ep * 100 + 7)
                    torch.manual_seed(ep * 100 + 7)
                    seq = sample_chunk(model, sampler, obs,
                                       obs_m, obs_s, act_m, act_s, device)
                    tgt = infer_block(seq)
                    target_blocks[ep] = tgt

                print(f"\n  [{i+1:3d}/{args.n_rollouts}]  ep={ep}  tgt={tgt}  "
                      f"(K={args.n_candidates} candidates)")

                r = run_vlm_steering_episode(
                    env, model, sampler, scorer,
                    obs_m, obs_s, act_m, act_s, device,
                    episode_seed=ep, target_block=tgt,
                    n_candidates=args.n_candidates,
                    max_steps=args.max_steps,
                    save_frames_dir=comp_dir,
                    selection_mode=args.selection_mode,
                    leg_threshold=args.legibility_threshold,
                    max_arc_cap=args.max_arc_cap,
                )
                vs_rolls.append(r)
                vlm_calls += args.n_candidates

                tag = " *** ARC 15-19" if r["is_arc15_19"] else ""
                ok  = "OK" if r["success"] else "FAIL"
                L_comp = r.get('legibility_metrics', {}).get('L_composite', 'N/A')
                L_comp_s = f"{L_comp:.3f}" if isinstance(L_comp, (int, float)) else L_comp
                print(f"    → sel=c{r['selected_candidate']}  "
                      f"arc={r['arc']:.4f}m ({r['arc_class']})  "
                      f"leg={r['legibility']:.3f}  L={L_comp_s}  {ok}{tag}")

                ca = [c["arc"]        for c in r["candidates"]]
                cl = [c["legibility"] for c in r["candidates"]]
                print(f"      cands: arcs=[{min(ca):.3f}..{max(ca):.3f}]  "
                      f"legs=[{min(cl):.2f}..{max(cl):.2f}]  "
                      f"arc15={r['arc15_count']}/{args.n_candidates}")

                # save after every episode (crash safety)
                _save_intermediate(dict(
                    condition="vlm_steering", timestamp=ts,
                    rollouts_so_far=len(vs_rolls),
                    latest=r,
                ), out / "vlm_steering_progress.json")

            env.close()
            dt = time.time() - t0

            vs_sum = compute_summary(vs_rolls, "vlm_steering")
            print(f"\n  Done in {dt:.0f}s  ({vlm_calls} VLM calls)")
            print(f"  Success : {vs_sum['success_count']}/{vs_sum['n']}  "
                  f"({vs_sum['success_rate']:.0%})")
            print(f"  Arc 15-19 : {vs_sum['arc15_19_count']}/{vs_sum['n']}  "
                  f"({vs_sum['arc15_19_rate']:.0%})")
            print(f"  Mean arc  : {vs_sum['mean_arc']:.4f} ± "
                  f"{vs_sum['std_arc']:.4f} m")
            if "mean_legibility" in vs_sum:
                print(f"  Mean leg  : {vs_sum['mean_legibility']:.3f}\n")

            _save_intermediate(dict(
                condition="vlm_steering", timestamp=ts,
                summary=vs_sum,
                rollouts=[{k: v for k, v in r.items() if k != "candidates"}
                          for r in vs_rolls],
                candidate_details=[
                    dict(episode_seed=r["episode_seed"],
                         candidates=r["candidates"])
                    for r in vs_rolls
                ],
            ), out / "vlm_steering_results.json")
    else:
        vs_sum = {}
        print("  Skipping VLM steering condition\n")

    # ═════════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ═════════════════════════════════════════════════════════════════

    print(f"\n{'█'*78}")
    print(f"  EXPERIMENT RESULTS")
    print(f"{'█'*78}")

    if ns_sum and vs_sum:
        hdr = f"  {'Metric':<22}{'No Steering':>15}{'VLM Steering':>15}"
        sep = f"  {'─'*22}{'─'*15}{'─'*15}"
        print(f"\n{hdr}\n{sep}")
        print(f"  {'N rollouts':<22}{ns_sum['n']:>15}{vs_sum['n']:>15}")
        print(f"  {'Success rate':<22}"
              f"{ns_sum['success_rate']:>14.0%} "
              f"{vs_sum['success_rate']:>14.0%}")
        for ac in ["00-04", "05-09", "10-14", "15-19"]:
            ns_p = ns_sum["arc_distribution_pct"][ac]
            vs_p = vs_sum["arc_distribution_pct"][ac]
            delta = ""
            if ns_p > 0:
                delta = f"  ({(vs_p-ns_p)/ns_p*100:+.0f}%)"
            mark = "  <<<" if ac == "15-19" else ""
            print(f"  {'Arc '+ac:<22}{ns_p:>14.0f}%{vs_p:>14.0f}%{delta}{mark}")
        print(f"  {'Mean arc':<22}"
              f"{ns_sum['mean_arc']:>13.4f}m"
              f"{vs_sum['mean_arc']:>13.4f}m")
        print(f"  {'Std arc':<22}"
              f"{ns_sum['std_arc']:>13.4f}m"
              f"{vs_sum['std_arc']:>13.4f}m")
        if "mean_legibility" in vs_sum:
            ns_leg = ns_sum.get("mean_legibility")
            ns_leg_str = f"{ns_leg:.3f}" if ns_leg is not None else "N/A"
            print(f"  {'Mean VLM leg.':<22}"
                  f"{ns_leg_str:>14}"
                  f"{vs_sum['mean_legibility']:>14.3f}")

        # ── task-agnostic legibility metrics comparison ──────────
        task_agnostic_keys = [
            ("mean_L_early_intent", "L_early_intent  [PRIMARY]"),
            ("mean_L_posterior",    "L_posterior     [ROBUSTNESS]"),
            ("mean_L_composite",    "L_composite     [AGGREGATE]"),
            ("mean_L_commitment",   "L_commitment"),
        ]
        has_any = any(k in ns_sum or k in vs_sum for k, _ in task_agnostic_keys)
        if has_any:
            print(f"\n  {'─'*52}")
            print(f"  {'Task-Agnostic Metrics':<22}{'No Steering':>15}{'VLM Steering':>15}")
            print(f"  {'─'*52}")
            for key, label in task_agnostic_keys:
                ns_v = ns_sum.get(key)
                vs_v = vs_sum.get(key)
                ns_s = f"{ns_v:.3f}" if ns_v is not None else "N/A"
                vs_s = f"{vs_v:.3f}" if vs_v is not None else "N/A"
                print(f"  {label:<22}{ns_s:>15}{vs_s:>15}")

        # plots
        print(f"\n  Generating plots ...")
        generate_comparison_plot(ns_sum, vs_sum, ns_rolls, vs_rolls, out)
        generate_legibility_comparison_plot(ns_sum, vs_sum, ns_rolls, vs_rolls, out)

    # ── save combined JSON ───────────────────────────────────────────
    final = dict(
        experiment=dict(
            timestamp=ts, checkpoint=args.checkpoint,
            n_rollouts=args.n_rollouts, n_candidates=args.n_candidates,
            base_seed=args.base_seed, max_steps=args.max_steps,
            arc_thresholds_bezier=dict(
                boundary_05_09=ARC_T1, boundary_10_14=ARC_T2,
                boundary_15_19=ARC_T3,
            ),
            legibility_metric="max(pA, pB) via Gemini 2.5 Flash",
            steering_method="best-of-K VLM re-ranking with"
                            " timestamp-annotated prefix frames",
            frame_annotation="t=0s..t=5s overlay + goal legend on first frame",
        ),
        no_steering=dict(summary=ns_sum, rollouts=ns_rolls) if ns_sum else None,
        vlm_steering=dict(
            summary=vs_sum,
            rollouts=[{k: v for k, v in r.items() if k != "candidates"}
                      for r in vs_rolls],
            candidate_details=[
                dict(episode_seed=r["episode_seed"],
                     candidates=r["candidates"])
                for r in vs_rolls],
        ) if vs_sum else None,
    )
    jp = out / f"results_{ts}.json"
    with open(jp, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n  Results → {jp}")

    lp = out / "results_latest.json"
    with open(lp, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"  Latest  → {lp}")
    print(f"\n{'█'*78}\n")


if __name__ == "__main__":
    main()
