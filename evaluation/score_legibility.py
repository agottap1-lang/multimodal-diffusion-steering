#!/usr/bin/env python
"""
score_legibility.py - Score each policy rollout video for legibility using Gemini.

Takes:
  --eval_dir   : directory produced by eval_with_videos.py (contains results.json
                 and videos_success/ + videos_failure/)
  --api_key    : Gemini API key (or set GOOGLE_API_KEY env var)
  --k          : number of prefix timesteps to evaluate (default: 6)
  --fps_sample : how often to sample frames within each timestep window (default: 0.5 s)
  --sleep      : seconds to wait between API calls (default: 2.0)

Legibility Score per trajectory:
  VLO (VLM Onset) = first timestep at which the VLM correctly predicts the goal
  Lower VLO = more legible (goal evident earlier)
  If VLM never correctly predicts before k: VLO = k (worst case)

Camera convention (yaw=135, pitch=-30):
  Image-LEFT  = world-RIGHT block  → Goal A in prompt
  Image-RIGHT = world-LEFT  block  → Goal B in prompt
  picked_side='left'  → correct VLM answer = 'B'
  picked_side='right' → correct VLM answer = 'A'

Output:
  <eval_dir>/legibility_scores.json  - per-episode VLO + raw VLM responses
  <eval_dir>/legibility_summary.txt  - human-readable summary
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Camera calibration: picked_side → correct VLM choice
SIDE_TO_CORRECT_CHOICE = {'left': 'B', 'right': 'A'}


# ── frame extraction ──────────────────────────────────────────────────────────

def extract_prefix_frames(video_path: Path, n_frames: int) -> list[tuple[bytes, float]]:
    """
    Extract n_frames evenly spaced over the first 30% of the video.
    Returns list of (png_bytes, timestamp_sec).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = int(total_frames * 0.30)

    indices = [int(i * max_frame / max(n_frames - 1, 1)) for i in range(n_frames)]
    indices = sorted(set(indices))

    result = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        _, buf = cv2.imencode('.png', frame)
        result.append((bytes(buf), round(idx / fps, 2)))
    cap.release()
    return result


def extract_cumulative_prefix(video_path: Path, k: int, fps_sample: float = 0.5) -> list[list[tuple[bytes, float]]]:
    """
    Return k windows: window[i] = frames from t=0 to t=i*(total*0.30/k).
    Each window adds one more second of context (cumulative).
    Returns list of k lists, each list is [(png_bytes, t_sec), ...].
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps
    cap.release()

    # k cutoff times, evenly spaced over first 30%
    cutoffs = [total_sec * 0.30 * (i + 1) / k for i in range(k)]

    windows = []
    for cutoff_t in cutoffs:
        frames = []
        t = 0.0
        cap2 = cv2.VideoCapture(str(video_path))
        while t <= cutoff_t + 1e-6:
            idx = int(round(t * fps))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap2.read()
            if not ok:
                break
            _, buf = cv2.imencode('.png', frame)
            frames.append((bytes(buf), round(t, 2)))
            t += fps_sample
        cap2.release()
        if frames:
            windows.append(frames)
    return windows


# ── prompt ────────────────────────────────────────────────────────────────────

def build_prompt(n_frames: int, t_sec: float) -> str:
    return (
        f'You are evaluating LEGIBILITY: how easily a human observer can infer the '
        f'robot\'s intended goal from motion observed so far.\n\n'
        f'You are given {n_frames} frame(s) from t=0 to t={t_sec:.1f}s of a robot '
        f'manipulation task.\n'
        f'There are exactly two candidate goals:\n'
        f'  Goal A: pick the LEFT block (as seen in the image)\n'
        f'  Goal B: pick the RIGHT block (as seen in the image)\n\n'
        f'Estimate:\n'
        f'  pA = P(Goal A | frames)   [0..1]\n'
        f'  pB = P(Goal B | frames)   [0..1]   (pA + pB = 1)\n'
        f'  cue: ONE short phrase describing the visual cue supporting your estimate\n'
        f'  legible: "legible_now" if goal is clear to a human now, else "not_legible_yet"\n\n'
        f'Output ONLY valid JSON. No markdown. No extra text.\n'
        f'Example: {{"pA": 0.75, "pB": 0.25, "cue": "gripper moving left", "legible": "legible_now"}}'
    )


# ── VLM call ──────────────────────────────────────────────────────────────────

def call_vlm(client, frames_bytes_ts: list[tuple[bytes, float]]) -> dict:
    """Call Gemini with a list of (png_bytes, t_sec) frames."""
    from google import genai
    from google.genai import types

    t_max = frames_bytes_ts[-1][1] if frames_bytes_ts else 0.0
    prompt_text = build_prompt(len(frames_bytes_ts), t_max)

    parts = []
    for png_bytes, _ in frames_bytes_ts:
        parts.append(types.Part.from_bytes(data=png_bytes, mime_type='image/png'))
    parts.append(prompt_text)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=512),
        response_mime_type='application/json',
        temperature=0.1,
    )

    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash', contents=parts, config=config
        )
        latency_ms = int((time.time() - t0) * 1000)
        parsed = json.loads(resp.text)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        parsed['latency_ms'] = latency_ms
        return parsed
    except json.JSONDecodeError:
        raw = getattr(resp, 'text', '')[:300]
        return {'error': 'json_decode', 'raw': raw, 'latency_ms': int((time.time() - t0) * 1000)}
    except Exception as e:
        return {'error': str(e), 'latency_ms': int((time.time() - t0) * 1000)}


def parse_choice(result: dict) -> str:
    """Return 'A', 'B', or 'C' (ambiguous)."""
    pA = float(result.get('pA', 0.5))
    pB = float(result.get('pB', 0.5))
    if abs(pA - pB) < 0.05:
        return 'C'
    return 'A' if pA > pB else 'B'


# ── per-episode scoring ───────────────────────────────────────────────────────

def score_episode(ep_meta: dict, client, k: int, fps_sample: float, sleep: float) -> dict:
    """
    Score one episode's video for legibility.
    Returns dict with VLO and per-timestep VLM results.
    """
    video_path = ep_meta.get('video_path')
    picked_side = ep_meta.get('picked_side')

    if not video_path or not Path(video_path).exists():
        return {'episode': ep_meta['episode'], 'error': 'no_video', 'vlo': k}

    if not picked_side:
        return {'episode': ep_meta['episode'], 'error': 'no_picked_side', 'vlo': k}

    correct_choice = SIDE_TO_CORRECT_CHOICE[picked_side]
    windows = extract_cumulative_prefix(Path(video_path), k=k, fps_sample=fps_sample)

    timestep_results = []
    vlo = k  # worst case: never identified correctly

    for ts_idx, frames in enumerate(windows):
        if not frames:
            continue
        t_sec = frames[-1][1]
        vlm_out = call_vlm(client, frames)
        choice = parse_choice(vlm_out)
        correct = (choice == correct_choice)

        ts_result = {
            'timestep': ts_idx,
            't_sec': t_sec,
            'n_frames': len(frames),
            'pA': vlm_out.get('pA'),
            'pB': vlm_out.get('pB'),
            'cue': vlm_out.get('cue'),
            'legible': vlm_out.get('legible'),
            'choice': choice,
            'correct_choice': correct_choice,
            'correct': correct,
            'latency_ms': vlm_out.get('latency_ms'),
            'error': vlm_out.get('error'),
        }
        timestep_results.append(ts_result)

        if correct and vlo == k:
            vlo = ts_idx  # first correct prediction

        time.sleep(sleep)

    return {
        'episode': ep_meta['episode'],
        'success': ep_meta['success'],
        'picked_side': picked_side,
        'steps': ep_meta['steps'],
        'video_path': video_path,
        'correct_choice': correct_choice,
        'vlo': vlo,  # VLM Onset - lower is more legible
        'timesteps': timestep_results,
    }


# ── summary ───────────────────────────────────────────────────────────────────

def print_and_save_summary(scores: list[dict], eval_dir: Path, k: int):
    success_scores = [s for s in scores if s.get('success') and 'error' not in s]
    all_vlos = [s['vlo'] for s in success_scores]

    lines = []
    lines.append('=' * 65)
    lines.append('BASELINE LEGIBILITY SCORES  (VLM Onset = VLO)')
    lines.append('=' * 65)
    lines.append(f'  k (timestep windows)  : {k}')
    lines.append(f'  Episodes scored       : {len(success_scores)} successful')
    lines.append(f'  Mean VLO              : {np.mean(all_vlos):.2f}  (lower = more legible)')
    lines.append(f'  Median VLO            : {np.median(all_vlos):.1f}')
    lines.append(f'  VLO distribution      : {dict(sorted((v, all_vlos.count(v)) for v in set(all_vlos)))}')
    lines.append(f'  % legible by ts=0     : {100*sum(v==0 for v in all_vlos)/max(len(all_vlos),1):.1f}%')
    lines.append(f'  % legible by ts=1     : {100*sum(v<=1 for v in all_vlos)/max(len(all_vlos),1):.1f}%')
    lines.append(f'  % legible by ts=2     : {100*sum(v<=2 for v in all_vlos)/max(len(all_vlos),1):.1f}%')
    lines.append(f'  % never identified    : {100*sum(v==k for v in all_vlos)/max(len(all_vlos),1):.1f}%')
    lines.append('')
    lines.append('Per-episode breakdown:')
    lines.append(f'  {"Ep":>4}  {"Side":>6}  {"Steps":>6}  {"VLO":>4}  Correct-at-ts')
    lines.append('  ' + '-' * 50)
    for s in success_scores:
        ts_correct = [ts['timestep'] for ts in s.get('timesteps', []) if ts.get('correct')]
        lines.append(f'  {s["episode"]:>4}  {s["picked_side"]:>6}  {s["steps"]:>6}  {s["vlo"]:>4}  {ts_correct}')
    lines.append('=' * 65)

    summary = '\n'.join(lines)
    print(summary)

    out_path = eval_dir / 'legibility_summary.txt'
    out_path.write_text(summary, encoding='utf-8')
    print(f'\nSummary saved to: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='Directory from eval_with_videos.py (contains results.json)')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--k', type=int, default=6,
                        help='Number of prefix timestep windows (default: 6)')
    parser.add_argument('--fps_sample', type=float, default=0.5,
                        help='Seconds between sampled frames per window (default: 0.5)')
    parser.add_argument('--sleep', type=float, default=2.0,
                        help='Sleep between API calls in seconds (default: 2.0)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip episodes already scored in legibility_scores.json')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    results_path = eval_dir / 'results.json'
    if not results_path.exists():
        sys.exit(f'ERROR: {results_path} not found. Run eval_with_videos.py first.')

    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        sys.exit('ERROR: set --api_key or GOOGLE_API_KEY env var')

    from google import genai
    client = genai.Client(api_key=api_key)

    with open(results_path) as f:
        eval_results = json.load(f)

    episodes = eval_results.get('episodes', [])
    # Only score successful episodes with a known picked_side
    to_score = [e for e in episodes if e.get('success') and e.get('picked_side')]
    print(f'Eval dir     : {eval_dir}')
    print(f'Total episodes: {len(episodes)} | Successful+scorable: {len(to_score)}')
    print(f'k={args.k} windows, fps_sample={args.fps_sample}s, sleep={args.sleep}s\n')

    # Load existing scores for resume
    scores_path = eval_dir / 'legibility_scores.json'
    done_episodes = set()
    existing_scores = []
    if args.resume and scores_path.exists():
        with open(scores_path) as f:
            existing_scores = json.load(f)
        done_episodes = {s['episode'] for s in existing_scores if 'error' not in s}
        print(f'Resuming: {len(done_episodes)} episodes already scored.\n')

    all_scores = list(existing_scores)

    for i, ep_meta in enumerate(to_score):
        ep_num = ep_meta['episode']
        if ep_num in done_episodes:
            continue

        print(f'[{i+1}/{len(to_score)}] ep={ep_num:03d} '
              f'side={ep_meta["picked_side"]} steps={ep_meta["steps"]} '
              f'video={Path(ep_meta["video_path"]).name}')

        score = score_episode(ep_meta, client, k=args.k,
                              fps_sample=args.fps_sample, sleep=args.sleep)
        all_scores.append(score)

        vlo_str = f'VLO={score["vlo"]}' if 'error' not in score else f'ERROR={score["error"]}'
        print(f'  → {vlo_str}\n')

        # Save after each episode
        with open(scores_path, 'w') as f:
            json.dump(all_scores, f, indent=2)

    print_and_save_summary(all_scores, eval_dir, k=args.k)


if __name__ == '__main__':
    main()
