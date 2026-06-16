#!/usr/bin/env python3
"""
Re-evaluate a small pilot set with the new arc-aware prompt.

This runs LIVE API calls (approx 5 videos * 3 timepoints = 15 calls ~ $0.03).
"""
import os, sys, json
from pathlib import Path

# Insert gemini_vlm_eval onto path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gemini_vlm_eval" / "src"))

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.schema import ManifestEntry
from gemini_vlm_eval.video import extract_and_cache_frames

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("Set GEMINI_API_KEY in your environment / .env before running.")

MANIFEST_PATH = Path("data/manifest_combined_cfg00.jsonl")
VIDEO_ROOT    = Path("data/demos/demo_videos_combined")
OUTPUT_DIR    = Path("outputs/pilot_arc_prompt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pilot: high-arc-index videos (vi=7-9) where arc curvature is large and visible.
# v00 = smallest arc (5cm sweep) — essentially invisible to VLM.
# v07-v09 = largest arcs (20-28cm sweep) — visually obvious.
PILOT_IDS = [
    "cfg00_leg_left_v07",
    "cfg00_leg_left_v09",
    "cfg00_leg_right_v07",
    "cfg00_leg_right_v09",
    "cfg00_dec_left_v04",   # strongest deceptive feint
    "cfg00_dec_right_v04",
    "cfg00_neu_left_v02",
    "cfg00_neu_right_v02",
]
EVAL_TIMEPOINTS = [1, 2, 3, 5]

def load_manifest(path):
    entries = {}
    with open(path) as f:
        for line in f:
            e = ManifestEntry(**json.loads(line))
            entries[e.video_id] = e
    return entries

def find_video(video_root, video_id):
    """Search recursively for the video file."""
    for ext in [".mp4", ".avi", ".mov"]:
        for p in video_root.rglob(f"{video_id}{ext}"):
            return str(p)
    return None

def main():
    manifest = load_manifest(MANIFEST_PATH)
    client = GeminiClient(model="gemini-2.5-flash")
    results = []

    for vid_id in PILOT_IDS:
        if vid_id not in manifest:
            print(f"WARN: {vid_id} not in manifest, skipping")
            continue
        entry = manifest[vid_id]

        video_path = find_video(VIDEO_ROOT, vid_id)
        if not video_path:
            # Try the path stored in manifest
            video_path = entry.video_path if Path(entry.video_path).exists() else None
        if not video_path:
            print(f"WARN: video file not found for {vid_id}")
            continue

        entry = ManifestEntry(
            video_id=entry.video_id,
            video_path=video_path,
            goal_gt=entry.goal_gt,
            goal_A=entry.goal_A,
            goal_B=entry.goal_B,
            scene_id=entry.scene_id,
            task_family=entry.task_family,
            traj_type=entry.traj_type,
            notes=entry.notes,
        )

        print(f"\nEvaluating {vid_id}  (gt={entry.goal_gt}, type={entry.traj_type})")
        frames_dict = extract_and_cache_frames(
            video_path=video_path,
            video_id=vid_id,
            sample_rate_seconds=1.0,
            max_frames=6,
            save_frames=False,
        )
        timestamps = sorted(frames_dict.keys())

        for t_sec in EVAL_TIMEPOINTS:
            if t_sec not in timestamps:
                continue
            prefix_frames = [frames_dict[t]["jpeg_bytes"] for t in range(0, t_sec + 1) if t in frames_dict]
            result = client.evaluate_frame(
                image_bytes=prefix_frames,
                manifest_entry=entry,
                t_sec=t_sec,
                frame_idx=frames_dict[t_sec]["frame_idx"],
                mode="prefix_frames",
            )
            row = {
                "video_id": vid_id,
                "traj_type": entry.traj_type,
                "goal_gt": entry.goal_gt,
                "t_sec": t_sec,
                "pA": result.pA,
                "pB": result.pB,
                "choice": result.choice,
                "confidence": result.confidence,
                "cue": result.cue,
                "legible": result.legible,
            }
            results.append(row)
            correct = "CORRECT" if result.choice == entry.goal_gt else ("WRONG" if result.choice != "C" else "C")
            print(f"  t={t_sec}s: pA={result.pA:.2f} pB={result.pB:.2f} choice={result.choice} [{correct}]")
            print(f"    cue: {result.cue[:80]}")

    # Save results
    out_path = OUTPUT_DIR / "pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n=== SUMMARY (new arc-aware prompt) ===")
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r['traj_type']].append(r)

    for tt in ['legible', 'deceptive', 'neutral']:
        rows = by_type.get(tt, [])
        if not rows:
            continue
        c_rate = sum(1 for r in rows if r['choice'] == 'C') / len(rows) * 100
        correct = sum(1 for r in rows if r['choice'] == r['goal_gt'])
        wrong = sum(1 for r in rows if r['choice'] not in ('C', r['goal_gt']))
        expected_behavior = {
            'legible': 'high correct early',
            'deceptive': 'high WRONG early (showing deception)',
            'neutral': 'high C (ambiguous)',
        }
        print(f"  {tt:12s}: n={len(rows)} C={c_rate:.0f}% correct={correct} wrong={wrong}  (expected: {expected_behavior[tt]})")

if __name__ == "__main__":
    main()
