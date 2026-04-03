#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_combined_prefix.py - VLM prefix-mode + video-30% evaluation on demo_videos_combined.

Modes
-----
  prefix  – extract frames at 0.5 s intervals up to 30 % of trajectory,
             send cumulative prefix windows of sizes 3, 4, 5, 6 frames.
             VLM returns pA / pB / cue / legible at each window size.

  video30 – re-encode the first 30 % of the video as an MP4 clip and
             send it together with an annotated reference frame.
             VLM returns a single pA / pB / cue / legible estimate.

  both    – run prefix then video30 for every video.

Camera calibration  (yaw = 135 °, pitch = -30 °)
-------------------------------------------------
  Image-LEFT  = world-RIGHT  →  "Goal A" in prompt
  Image-RIGHT = world-LEFT   →  "Goal B" in prompt

  Correct VLM choice:
    side = left  (world-left = image-right)  →  correct choice = "B"
    side = right (world-right = image-left)  →  correct choice = "A"

Filename convention
-------------------
  cfgXX_TYPE_SIDE_vNN.mp4
    TYPE = dec | neu | leg
    SIDE = left | right

Usage
-----
  # Prefix + video30 on cfg00 only, deceptive + legible:
  py -3 evaluation/eval_combined_prefix.py \\
      --configs 0 --styles dec leg \\
      --window_sizes 3 4 5 6 \\
      --mode both \\
      --api_key YOUR_KEY

  # Full run (cfg00-02, all styles):
  py -3 evaluation/eval_combined_prefix.py --api_key YOUR_KEY
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

# Force UTF-8 output on Windows so Unicode in print() works
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── camera calibration ────────────────────────────────────────────────────────
# Image-Left  = world-Right  →  prompt label "A"
# Image-Right = world-Left   →  prompt label "B"
CHOICE_TO_WORLD_SIDE: dict[str, str] = {"A": "right", "B": "left"}
SIDE_CORRECT_CHOICE:  dict[str, str] = {"left": "B",   "right": "A"}


# ── filename parsing ──────────────────────────────────────────────────────────
def parse_combined_filename(stem: str) -> dict | None:
    """Parse 'cfgXX_TYPE_SIDE_vNN' → meta dict.  Returns None if no match."""
    m = re.match(r"cfg(\d+)_(dec|neu|leg)_(left|right)_v(\d+)$", stem)
    if not m:
        return None
    return {
        "cfg_id":  int(m.group(1)),
        "style":   m.group(2),    # dec / neu / leg
        "side":    m.group(3),    # left / right
        "variant": int(m.group(4)),
    }


# ── video utilities ───────────────────────────────────────────────────────────
def extract_frames_at_half_second(
    video_path: Path,
    max_fraction: float = 0.30,
) -> tuple[list[bytes], list[float]]:
    """Return (PNG_bytes_list, timestamps_sec) sampled every 0.5 s up to
    max_fraction * total_duration (inclusive of the endpoint)."""
    cap = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec    = total_frames / fps
    max_sec      = total_sec * max_fraction

    frames: list[bytes] = []
    timestamps: list[float] = []
    t = 0.0
    while t <= max_sec + 1e-6:          # include the endpoint within tolerance
        idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        _, buf = cv2.imencode(".png", frame)
        frames.append(bytes(buf))
        timestamps.append(round(t, 2))
        t += 0.5
    cap.release()
    return frames, timestamps


def extract_video_clip_bytes(
    video_path: Path,
    max_fraction: float = 0.30,
) -> bytes:
    """Re-encode the first *max_fraction* of a video to MP4 and return the bytes."""
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_fr       = int(total_frames * max_fraction)
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(max_fr):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    out.release()
    cap.release()

    video_bytes = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return video_bytes


def annotate_first_frame(video_path: Path) -> bytes:
    """
    Extract first frame, draw colored circles + labels around the two red blocks.
      Block A (blue circle)  = image-LEFT  block
      Block B (green circle) = image-RIGHT block
    Returns PNG bytes.
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, np.array([0,   80, 80]), np.array([10,  255, 255]))
    m2  = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_mask          = m1 | m2
    cnts, _           = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]

    ann = frame.copy()
    if len(cnts) == 2:
        centers = []
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        if len(centers) == 2:
            centers.sort(key=lambda p: p[0])   # sort by x: index-0 = leftmost
            cv2.circle(ann, centers[0], 30, (255, 0, 0), 3)
            cv2.putText(ann, "A", (centers[0][0] - 8, centers[0][1] - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.circle(ann, centers[1], 30, (0, 255, 0), 3)
            cv2.putText(ann, "B", (centers[1][0] - 8, centers[1][1] - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(ann, "A = left block (blue)",  (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1)
            cv2.putText(ann, "B = right block (green)", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    _, buf = cv2.imencode(".png", ann)
    return bytes(buf)


# ── prompts ───────────────────────────────────────────────────────────────────
def build_prefix_prompt(video_id: str, n_frames: int, t_sec: float) -> str:
    return (
        f'You are evaluating LEGIBILITY: how easily a typical human observer can '
        f'infer the actor\'s intended goal from what they have observed.\n\n'
        f'You are given {n_frames} image(s) showing frames from t=0 to t={t_sec:.1f} '
        f'seconds from video_id = "{video_id}".\n'
        f'Frames are ordered from earliest to latest; you have observed the motion '
        f'up to time t={t_sec:.1f}s.\n'
        f'Use ALL frames provided to estimate the goal probabilities.\n\n'
        f'There are exactly two candidate goals:\n'
        f'Goal A: pick the left block\n'
        f'Goal B: pick the right block\n\n'
        f'Estimate probabilities using the provided frame(s):\n'
        f'- pA = P(Goal A | frames)\n'
        f'- pB = P(Goal B | frames)\n'
        f'Constraints:\n'
        f'- 0 <= pA,pB <= 1\n'
        f'- pA + pB = 1 (within rounding)\n\n'
        f'Provide EXACTLY ONE short visual cue from the frame(s) that supports '
        f'your probabilities.\n'
        f'Also output legibility:\n'
        f'- "legible_now" if a typical human could infer the goal now, '
        f'else "not_legible_yet".\n\n'
        f'Output ONLY valid JSON with keys: pA, pB, cue, legible.\n'
        f'No markdown. No extra text. No code fences.\n'
        f'Example format:\n'
        f'{{"pA": 0.62, "pB": 0.38, "cue": "gripper aligned with left block", '
        f'"legible": "legible_now"}}'
    )


def build_video30_prompt(video_id: str) -> str:
    return (
        f'You are analyzing a robot trajectory video.\n'
        f'The clip shows the FIRST 30 % of a manipulation sequence '
        f'(video_id = "{video_id}").\n\n'
        f'REFERENCE IMAGE: The first image has the blocks annotated:\n'
        f'  Block A (blue circle) = left block in the image\n'
        f'  Block B (green circle) = right block in the image\n\n'
        f'The robot gripper starts above the table and moves toward one of the two '
        f'small red blocks.\n\n'
        f'There are exactly two candidate goals:\n'
        f'Goal A: pick the left block\n'
        f'Goal B: pick the right block\n\n'
        f'Based ONLY on the early motion visible in the clip, estimate:\n'
        f'- pA = P(Goal A | early trajectory)\n'
        f'- pB = P(Goal B | early trajectory)\n'
        f'- ONE visual cue supporting your estimate\n'
        f'- legibility: "legible_now" if goal is clear, "not_legible_yet" otherwise\n\n'
        f'Output ONLY valid JSON with keys: pA, pB, cue, legible.\n'
        f'No markdown. No extra text. No code fences.\n'
        f'Example:\n'
        f'{{"pA": 0.7, "pB": 0.3, "cue": "gripper moves toward left", '
        f'"legible": "legible_now"}}'
    )


# ── VLM call ──────────────────────────────────────────────────────────────────
def call_vlm(
    client: genai.Client,
    parts: list,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    thinking_budget: int = 1024,
) -> dict:
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        response_mime_type="application/json",
        temperature=temperature,
    )
    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=model, contents=parts, config=config
        )
        latency_ms = int((time.time() - t0) * 1000)
        parsed = json.loads(resp.text)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        parsed["latency_ms"] = latency_ms
        parsed["model"] = model
        return parsed
    except json.JSONDecodeError:
        raw = ""
        try:
            raw = resp.text[:400]
        except Exception:
            pass
        return {
            "error": "json_decode",
            "raw": raw,
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {"error": str(e), "latency_ms": int((time.time() - t0) * 1000)}


def _parse_choice(result: dict) -> str:
    """Return 'A', 'B', or 'C' (ambiguous) based on pA vs pB."""
    pA = float(result.get("pA", 0.5))
    pB = float(result.get("pB", 0.5))
    if abs(pA - pB) < 0.05:
        return "C"
    return "A" if pA > pB else "B"


# ── prefix evaluation (per video) ────────────────────────────────────────────
def eval_prefix_video(
    video_path: Path,
    meta: dict,
    client: genai.Client,
    model: str,
    window_sizes: list[int],
    traj_fraction: float,
    inter_call_sleep: float = 0.5,
) -> dict:
    """Run the prefix protocol on a single video."""
    video_id   = video_path.stem
    true_side  = meta["side"]
    correct_ch = SIDE_CORRECT_CHOICE[true_side]

    print(f"\n  [PREFIX] {video_id}  (style={meta['style']}, side={true_side}, "
          f"correct={correct_ch})")

    frames, timestamps = extract_frames_at_half_second(video_path, traj_fraction)
    n_avail = len(frames)
    print(f"    {n_avail} frames @ 0.5s intervals: {timestamps}")

    ann_png = annotate_first_frame(video_path)

    vlm_results = []
    for n in window_sizes:
        if n > n_avail:
            print(f"    [skip n={n}: only {n_avail} frames available]")
            continue
        t_sec = timestamps[n - 1]
        print(f"    n={n} (t<={t_sec:.1f}s) ...", end="", flush=True)

        # Build multimodal parts:  [annotated_reference, frame_0, ..., frame_{n-1}, prompt]
        parts: list = [types.Part.from_bytes(data=ann_png, mime_type="image/png")]
        for i in range(n):
            parts.append(types.Part.from_bytes(data=frames[i], mime_type="image/png"))
        parts.append(build_prefix_prompt(video_id, n, t_sec))

        result = call_vlm(client, parts, model=model)
        result["t_sec"]    = t_sec
        result["n_frames"] = n

        choice = _parse_choice(result)
        result["choice"]     = choice
        result["correct"]    = (choice == correct_ch)
        result["world_pred"] = CHOICE_TO_WORLD_SIDE.get(choice, "unknown")

        pA  = result.get("pA", 0.5)
        pB  = result.get("pB", 0.5)
        leg = result.get("legible", "?")
        ok  = "OK" if result["correct"] else "WRONG"
        print(
            f"pA={pA:.2f}  pB={pB:.2f}  -> {choice} {ok}  "
            f"legible={leg}  ({result.get('latency_ms', 0)//1000}s)"
        )
        vlm_results.append(result)
        time.sleep(inter_call_sleep)

    return {
        "video_id":           video_id,
        "meta":               meta,
        "true_side":          true_side,
        "correct_choice":     correct_ch,
        "n_frames_available": n_avail,
        "timestamps_sec":     timestamps,
        "vlm_results":        vlm_results,
    }


# ── video-30% evaluation (per video) ─────────────────────────────────────────
def eval_video30_video(
    video_path: Path,
    meta: dict,
    client: genai.Client,
    model: str,
    traj_fraction: float,
) -> dict:
    """Send first 30 % of video as MP4 clip to VLM."""
    video_id   = video_path.stem
    true_side  = meta["side"]
    correct_ch = SIDE_CORRECT_CHOICE[true_side]

    print(f"\n  [VIDEO30] {video_id}  (style={meta['style']}, side={true_side}, "
          f"correct={correct_ch})", end="  ", flush=True)

    ann_png    = annotate_first_frame(video_path)
    clip_bytes = extract_video_clip_bytes(video_path, max_fraction=traj_fraction)
    print(f"clip={len(clip_bytes)//1024} KB … ", end="", flush=True)

    parts = [
        types.Part.from_bytes(data=ann_png,    mime_type="image/png"),
        types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4"),
        build_video30_prompt(video_id),
    ]
    result = call_vlm(client, parts, model=model)

    choice = _parse_choice(result)
    result["choice"]     = choice
    result["correct"]    = (choice == correct_ch)
    result["world_pred"] = CHOICE_TO_WORLD_SIDE.get(choice, "unknown")

    pA  = result.get("pA", 0.5)
    pB  = result.get("pB", 0.5)
    leg = result.get("legible", "?")
    ok  = "OK" if result["correct"] else "WRONG"
    print(
        f"pA={pA:.2f}  pB={pB:.2f}  -> {choice} {ok}  legible={leg}  "
        f"({result.get('latency_ms', 0)//1000}s)"
    )

    return {
        "video_id":       video_id,
        "meta":           meta,
        "true_side":      true_side,
        "correct_choice": correct_ch,
        "vlm_result":     result,
    }


# ── summary statistics ────────────────────────────────────────────────────────
def compute_prefix_summary(all_results: list) -> dict:
    """Per-style, per-window accuracy / pA / legibility / VLO statistics."""
    by_style:   dict[str, list] = defaultdict(list)
    by_style_w: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        style = r["meta"]["style"]
        for vr in r.get("vlm_results", []):
            n = vr["n_frames"]
            by_style[style].append(vr)
            by_style_w[style][n].append(vr)

    summary: dict = {}
    for style, items in by_style.items():
        acc      = float(np.mean([v["correct"] for v in items])) if items else 0.0
        mean_pA  = float(np.mean([v.get("pA", 0.5) for v in items])) if items else 0.5
        mean_pB  = float(np.mean([v.get("pB", 0.5) for v in items])) if items else 0.5
        frac_leg = float(np.mean(
            [1 if v.get("legible") == "legible_now" else 0 for v in items]
        )) if items else 0.0

        # VLO = first window where legible_now AND correct, per video
        vlo_list: list[int] = []
        for r in all_results:
            if r["meta"]["style"] != style:
                continue
            for vr in r.get("vlm_results", []):
                if vr.get("legible") == "legible_now" and vr.get("correct"):
                    vlo_list.append(vr["n_frames"])
                    break

        per_window: dict = {}
        for n, wlist in sorted(by_style_w[style].items()):
            per_window[n] = {
                "accuracy":     round(float(np.mean([v["correct"]             for v in wlist])), 3),
                "mean_pA":      round(float(np.mean([v.get("pA", 0.5)        for v in wlist])), 3),
                "mean_pB":      round(float(np.mean([v.get("pB", 0.5)        for v in wlist])), 3),
                "frac_legible": round(float(np.mean(
                    [1 if v.get("legible") == "legible_now" else 0 for v in wlist]
                )), 3),
                "n_samples": len(wlist),
            }

        summary[style] = {
            "overall_accuracy": round(acc, 3),
            "mean_pA":          round(mean_pA, 3),
            "mean_pB":          round(mean_pB, 3),
            "frac_legible":     round(frac_leg, 3),
            "n_videos":         len(all_results),
            "mean_vlo_frames":  (
                round(float(np.mean(vlo_list)), 2) if vlo_list else None
            ),
            "vlo_n":            len(vlo_list),
            "per_window":       per_window,
        }
    return summary


def compute_video30_summary(all_results: list) -> dict:
    by_style: dict[str, list] = defaultdict(list)
    for r in all_results:
        style = r["meta"]["style"]
        by_style[style].append(r.get("vlm_result", {}))

    summary: dict = {}
    for style, items in by_style.items():
        items_ok = [v for v in items if "error" not in v]
        acc      = float(np.mean([v.get("correct", False) for v in items_ok])) if items_ok else 0.0
        mean_pA  = float(np.mean([v.get("pA", 0.5) for v in items_ok])) if items_ok else 0.5
        frac_leg = float(np.mean(
            [1 if v.get("legible") == "legible_now" else 0 for v in items_ok]
        )) if items_ok else 0.0

        summary[style] = {
            "accuracy":     round(acc, 3),
            "mean_pA":      round(mean_pA, 3),
            "frac_legible": round(frac_leg, 3),
            "n_videos":     len(items),
            "n_ok":         len(items_ok),
        }
    return summary


def print_summary_table(prefix_summary: dict | None, video30_summary: dict | None) -> None:
    STYLES = ["dec", "neu", "leg"]

    if prefix_summary:
        print("\n" + "=" * 72)
        print("PREFIX MODE SUMMARY")
        print("=" * 72)
        print(f"{'style':<6}  {'acc':>6}  {'pA':>5}  {'pB':>5}  "
              f"{'legible%':>9}  {'VLO(n)':>8}")
        print("-" * 72)
        for s in STYLES:
            if s not in prefix_summary:
                continue
            d = prefix_summary[s]
            vlo = f"{d['mean_vlo_frames']:.1f}" if d["mean_vlo_frames"] else "—"
            print(
                f"{s:<6}  {d['overall_accuracy']:>6.1%}  {d['mean_pA']:>5.2f}  "
                f"{d['mean_pB']:>5.2f}  {d['frac_legible']:>9.1%}  {vlo:>8}"
            )
        print()
        print("Per-window breakdown (accuracy | frac_legible):")
        all_ns = sorted({n for d in prefix_summary.values() for n in d["per_window"]})
        header = f"{'style':<6}  " + "  ".join(f"n={n:<4}" for n in all_ns)
        print(header)
        print("-" * len(header))
        for s in STYLES:
            if s not in prefix_summary:
                continue
            pw = prefix_summary[s]["per_window"]
            row = f"{s:<6}  " + "  ".join(
                f"{pw[n]['accuracy']:.0%}/{pw[n]['frac_legible']:.0%}"
                if n in pw else "—     "
                for n in all_ns
            )
            print(row)

    if video30_summary:
        print("\n" + "=" * 72)
        print("VIDEO-30 % MODE SUMMARY")
        print("=" * 72)
        print(f"{'style':<6}  {'acc':>6}  {'pA':>5}  {'legible%':>9}  {'n':>4}")
        print("-" * 72)
        for s in STYLES:
            if s not in video30_summary:
                continue
            d = video30_summary[s]
            print(
                f"{s:<6}  {d['accuracy']:>6.1%}  {d['mean_pA']:>5.2f}  "
                f"{d['frac_legible']:>9.1%}  {d['n_videos']:>4}"
            )


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM prefix-mode + video-30% evaluation on demo_videos_combined",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video_dir", type=Path,
        default="data/demos/demo_videos_combined",
        help="Folder containing cfgXX_TYPE_SIDE_vNN.mp4 files",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default="outputs/combined_prefix_test",
        help="Where to write results",
    )
    parser.add_argument(
        "--mode", choices=["prefix", "video30", "both"],
        default="both",
        help="Evaluation mode (default: both)",
    )
    parser.add_argument(
        "--configs", type=int, nargs="+", default=[0, 1, 2],
        help="cfg IDs to include (default: 0 1 2)",
    )
    parser.add_argument(
        "--styles", nargs="+", choices=["dec", "neu", "leg"],
        default=["dec", "neu", "leg"],
        help="Trajectory styles to include (default: all three)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max videos per (cfg, style, side) group",
    )
    parser.add_argument(
        "--window_sizes", type=int, nargs="+", default=[3, 4, 5, 6],
        help="Prefix window sizes in frames (default: 3 4 5 6)",
    )
    parser.add_argument(
        "--traj_fraction", type=float, default=0.30,
        help="Fraction of trajectory to use (default: 0.30)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument("--temperature",     type=float, default=0.1)
    parser.add_argument("--thinking_budget", type=int,   default=1024)
    parser.add_argument(
        "--sleep", type=float, default=2.0,
        help="Seconds between video-level API calls (default: 2.0)",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Google API key (overrides GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List selected videos + frame counts without calling the VLM",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip videos that already have output JSON files (resume interrupted run)",
    )
    args = parser.parse_args()

    # ── API key ────────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        raise ValueError(
            "Google API key required.  "
            "Set --api_key KEY  or  $env:GOOGLE_API_KEY = 'KEY'"
        )
    client = genai.Client(api_key=api_key) if api_key else None

    # ── collect & filter videos ───────────────────────────────────────────────
    all_videos = sorted(args.video_dir.glob("*.mp4"))
    selected: list[tuple[Path, dict]] = []
    for vp in all_videos:
        meta = parse_combined_filename(vp.stem)
        if meta is None:
            continue
        if meta["cfg_id"] not in args.configs:
            continue
        if meta["style"] not in args.styles:
            continue
        selected.append((vp, meta))

    if args.limit:
        counts: dict = defaultdict(int)
        filtered: list = []
        for vp, meta in selected:
            key = (meta["cfg_id"], meta["style"], meta["side"])
            if counts[key] < args.limit:
                counts[key] += 1
                filtered.append((vp, meta))
        selected = filtered

    print(f"Selected {len(selected)} videos")
    print(f"  configs={args.configs}  styles={args.styles}  "
          f"mode={args.mode}  traj_fraction={args.traj_fraction:.0%}")
    print(f"  window_sizes={args.window_sizes}  model={args.model}")
    if args.dry_run:
        for vp, meta in selected:
            frames, ts = extract_frames_at_half_second(vp, args.traj_fraction)
            print(f"  {vp.name}: {len(frames)} frames  {ts}")
        return

    # ── run evaluations ────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix_dir  = args.output_dir / "prefix"
    video30_dir = args.output_dir / "video30"
    if args.mode in ("prefix", "both"):
        prefix_dir.mkdir(exist_ok=True)
    if args.mode in ("video30", "both"):
        video30_dir.mkdir(exist_ok=True)

    # ── load existing results when resuming ─────────────────────────────────
    prefix_results:  list[dict] = []
    video30_results: list[dict] = []

    if args.resume:
        for jf in sorted(prefix_dir.glob("*.json")):
            try:
                prefix_results.append(json.loads(jf.read_text()))
            except Exception:
                pass
        for jf in sorted(video30_dir.glob("*.json")):
            try:
                video30_results.append(json.loads(jf.read_text()))
            except Exception:
                pass
        print(f"  Resuming: loaded {len(prefix_results)} prefix, "
              f"{len(video30_results)} video30 existing results")

    already_prefix  = {r["video_id"] for r in prefix_results}
    already_video30 = {r["video_id"] for r in video30_results}

    for i, (vp, meta) in enumerate(selected, 1):
        need_prefix  = args.mode in ("prefix",  "both") and vp.stem not in already_prefix
        need_video30 = args.mode in ("video30", "both") and vp.stem not in already_video30

        if not need_prefix and not need_video30:
            print(f"[{i}/{len(selected)}] {vp.name}  (skip – already done)")
            continue

        print(f"\n[{i}/{len(selected)}] {vp.name}")

        if need_prefix:
            r = eval_prefix_video(
                vp, meta, client,
                model=args.model,
                window_sizes=args.window_sizes,
                traj_fraction=args.traj_fraction,
                inter_call_sleep=0.5,
            )
            prefix_results.append(r)
            (prefix_dir / f"{vp.stem}.json").write_text(json.dumps(r, indent=2))

        if need_video30:
            r2 = eval_video30_video(
                vp, meta, client,
                model=args.model,
                traj_fraction=args.traj_fraction,
            )
            video30_results.append(r2)
            (video30_dir / f"{vp.stem}.json").write_text(json.dumps(r2, indent=2))

        if i < len(selected):
            time.sleep(args.sleep)

    # ── save combined results ──────────────────────────────────────────────────
    prefix_summary  = None
    video30_summary = None

    if prefix_results:
        (args.output_dir / "results_prefix.json").write_text(
            json.dumps(prefix_results, indent=2)
        )
        prefix_summary = compute_prefix_summary(prefix_results)
        (args.output_dir / "summary_prefix.json").write_text(
            json.dumps(prefix_summary, indent=2)
        )

    if video30_results:
        (args.output_dir / "results_video30.json").write_text(
            json.dumps(video30_results, indent=2)
        )
        video30_summary = compute_video30_summary(video30_results)
        (args.output_dir / "summary_video30.json").write_text(
            json.dumps(video30_summary, indent=2)
        )

    print_summary_table(prefix_summary, video30_summary)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
