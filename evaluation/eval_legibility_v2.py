"""
Trajectory Legibility Evaluation v2 - Grounded Visual Analysis

Key improvements over v1:
1. MUCH shorter prompt (~1K chars vs 26K) - focused, not generic
2. Annotated reference frame - marks blocks with colored circles for spatial grounding
3. Image-space reasoning - asks VLM about image directions, maps to world post-hoc
4. Lower temperature (0.1) - more deterministic perceptual judgments
5. Multi-modal input: annotated first frame + full video
6. Camera perspective explicitly described

Architecture:
  VLM sees: [annotated_first_frame, video, short_prompt]
  VLM returns: image-space observations (endpoint, direction, curvature)
  Eval harness: maps image-space -> world-space using known camera calibration
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from google import genai
from google.genai import types


# ============================================================================
# Camera calibration: yaw=135 causes left/right reversal
# World left (Y=+0.07) -> appears on image RIGHT
# World right (Y=-0.07) -> appears on image LEFT
# ============================================================================
IMAGE_LEFT_IS_WORLD = "right"   # image-left block = world-right block
IMAGE_RIGHT_IS_WORLD = "left"   # image-right block = world-left block


def extract_first_frame(video_path: Path) -> np.ndarray:
    """Extract the first frame from a video file"""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    return frame


def annotate_frame_with_block_markers(frame: np.ndarray) -> bytes:
    """
    Annotate a frame with colored circles around the two blocks.
    
    Detects red blocks by color segmentation, then draws:
    - BLUE circle + "A" label around the image-LEFT block
    - GREEN circle + "B" label around the image-RIGHT block
    
    Returns PNG bytes of annotated frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color detection in HSV (red wraps around hue=0/180)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2
    
    # Find contours of red regions
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        # Fallback: try broader red range
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the two largest contours (the blocks)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    annotated = frame.copy()
    
    if len(contours) >= 2:
        # Get centroids
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        
        if len(centers) == 2:
            # Sort by x coordinate: left=0, right=1
            centers.sort(key=lambda c: c[0])
            
            # Block A = image-LEFT block (BLUE circle)
            cv2.circle(annotated, centers[0], 30, (255, 0, 0), 3)  # Blue
            cv2.putText(annotated, "A", (centers[0][0] - 8, centers[0][1] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Block B = image-RIGHT block (GREEN circle)
            cv2.circle(annotated, centers[1], 30, (0, 255, 0), 3)  # Green
            cv2.putText(annotated, "B", (centers[1][0] - 8, centers[1][1] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add legend
            cv2.putText(annotated, "A=left block (blue)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(annotated, "B=right block (green)", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Encode as PNG bytes
    _, png_data = cv2.imencode('.png', annotated)
    return png_data.tobytes()


def build_prompt(task_context: Optional[str] = None) -> str:
    """
    Build a short, focused prompt for trajectory analysis.
    ~800 chars instead of 26K - focused on what matters.
    """
    prompt = """Analyze this robot manipulation video.

SCENE: A table with two small red blocks. A robotic gripper starts above them and moves to reach one block.

REFERENCE IMAGE: The first image shows the starting scene with blocks marked:
- Block A (blue circle) is on the LEFT side of the image
- Block B (green circle) is on the RIGHT side of the image

VIDEO: Watch the complete trajectory of the gripper.

ANALYZE AND RETURN JSON:
{
  "observation": {
    "gripper_moves_toward": "A or B - which block does the gripper move toward?",
    "image_direction": "left or right - which direction does the gripper move in the image?",
    "path_shape": "direct / slight_curve / strong_curve / S_curve",
    "curvature_direction": "left / right / none - which way does the path curve?",
    "endpoint_block": "A or B - which block is the gripper closest to at the end?"
  },
  "trajectory_legibility": {
    "score": 0.0 to 1.0 (0.0 = completely ambiguous, 1.0 = intent obvious from the start),
    "early_commitment_fraction": 0.0 to 1.0 (fraction of trajectory where intent becomes clear),
    "reasoning": "Brief explanation of how readable the intent is"
  },
  "confidence": 0.0 to 1.0
}

SCORING GUIDE for trajectory_legibility.score:
- 0.9-1.0: Intent obvious from the very first movement
- 0.7-0.8: Intent clear by ~30% of trajectory
- 0.4-0.6: Ambiguous until halfway or later
- 0.0-0.3: Intent unclear until the very end

IMPORTANT: Focus on which block the gripper ACTUALLY reaches, not which direction you think it should go. Look at the final frames carefully."""
    
    if task_context:
        prompt = f"CONTEXT: {task_context}\n\n{prompt}"
    
    return prompt


def analyze_video_v2(
    video_path: Path,
    client: genai.Client,
    model: str = "gemini-3.1-pro-preview",
    thinking_budget: int = 4096,
    temperature: float = 0.1,
    task_context: Optional[str] = None,
    save_annotated_frame: Optional[Path] = None
) -> Dict:
    """
    Analyze a trajectory video with grounded visual reasoning.
    
    Sends: [annotated_first_frame, video, short_prompt]
    Returns: VLM analysis in image-space coordinates
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {video_path.name}")
    print(f"{'='*70}")
    
    # 1. Extract and annotate first frame
    frame = extract_first_frame(video_path)
    annotated_png = annotate_frame_with_block_markers(frame)
    print(f"  First frame annotated ({len(annotated_png)/1024:.1f} KB)")
    
    if save_annotated_frame:
        save_annotated_frame.parent.mkdir(parents=True, exist_ok=True)
        save_annotated_frame.write_bytes(annotated_png)
    
    # 2. Load video
    video_bytes = video_path.read_bytes()
    print(f"  Video loaded ({len(video_bytes)/1024:.1f} KB)")
    
    # 3. Build prompt
    prompt = build_prompt(task_context)
    
    # 4. Create multimodal content: [annotated_frame, video, prompt]
    frame_part = types.Part.from_bytes(data=annotated_png, mime_type="image/png")
    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        response_mime_type="application/json",
        temperature=temperature
    )
    
    # 5. Generate
    print(f"  Querying {model} (temp={temperature}, think={thinking_budget})...")
    start_time = time.time()
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=[frame_part, video_part, prompt],
            config=config
        )
        elapsed = time.time() - start_time
        print(f"  Response in {elapsed:.1f}s")
        
        parsed = json.loads(response.text)
        # Handle list-wrapped responses
        if isinstance(parsed, list):
            result = parsed[0] if parsed else {}
        else:
            result = parsed
        result["_metadata"] = {
            "video": video_path.name,
            "model": model,
            "temperature": temperature,
            "thinking_budget": thinking_budget,
            "elapsed_s": round(elapsed, 1)
        }
        
        # Print summary
        obs = result.get("observation", {})
        leg = result.get("trajectory_legibility", {})
        print(f"  -> Target: Block {obs.get('endpoint_block', '?')}")
        print(f"     Direction: {obs.get('image_direction', '?')}")
        print(f"     Path: {obs.get('path_shape', '?')}")
        print(f"     Legibility: {leg.get('score', '?')}")
        print(f"     Confidence: {result.get('confidence', '?')}")
        
        return result
        
    except json.JSONDecodeError as e:
        elapsed = time.time() - start_time
        print(f"  JSON ERROR: {e}")
        print(f"  Raw: {response.text[:300]}")
        return {"error": "json_decode", "raw": response.text[:500],
                "_metadata": {"video": video_path.name, "elapsed_s": round(elapsed, 1)}}
    except Exception as e:
        print(f"  ERROR: {e}")
        raw = ""
        try:
            raw = response.text[:500]
        except Exception:
            pass
        return {"error": str(e), "raw": raw, "_metadata": {"video": video_path.name}}


def map_prediction_to_world(analysis: Dict) -> Dict:
    """
    Map VLM's image-space prediction to world coordinates.
    
    With camera yaw=135:
      Block A (image-left) = world RIGHT block
      Block B (image-right) = world LEFT block
    """
    obs = analysis.get("observation", {})
    endpoint = obs.get("endpoint_block", "").upper()
    
    mapping = {
        "A": "right_block",   # Image-left = world-right
        "B": "left_block",    # Image-right = world-left
    }
    
    world_target = mapping.get(endpoint, "unknown")
    
    # Also map image_direction to world direction
    img_dir = obs.get("image_direction", "").lower()
    world_dir_map = {"left": "right", "right": "left"}
    world_direction = world_dir_map.get(img_dir, "unknown")
    
    return {
        "image_block": endpoint,
        "world_target": world_target,
        "world_side": world_target.replace("_block", "") if "_block" in world_target else "unknown",
        "image_direction": img_dir,
        "world_direction": world_direction
    }


def extract_ground_truth(video_name: str) -> Dict:
    """Extract ground truth from TwoBlockPick video filename"""
    match = re.match(r'cfg(\d+)_(left|right)_arc(\d+)', video_name)
    if match:
        side = match.group(2)
        arc_id = int(match.group(3))
        return {
            "target_side": side,
            "target_block": f"{side}_block",
            "arc_id": arc_id
        }
    return {}


def evaluate_accuracy(world_pred: Dict, ground_truth: Dict) -> Dict:
    """Compare world-mapped prediction to ground truth"""
    if not ground_truth:
        return {}
    
    gt_side = ground_truth["target_side"]
    pred_side = world_pred.get("world_side", "unknown")
    
    return {
        "side_correct": pred_side == gt_side,
        "gt_side": gt_side,
        "pred_side": pred_side,
        "image_block": world_pred.get("image_block", "?"),
    }


def main():
    parser = argparse.ArgumentParser(description="Trajectory Legibility Eval v2")
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.mp4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--thinking_budget", type=int, default=4096)
    parser.add_argument("--task_context", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_frames", action="store_true",
                        help="Save annotated first frames for debugging")
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set --api_key or GOOGLE_API_KEY")
    
    client = genai.Client(api_key=api_key)
    args.output.mkdir(parents=True, exist_ok=True)
    
    videos = sorted(args.video_dir.glob(args.pattern))
    if not videos:
        print(f"No videos matching {args.pattern} in {args.video_dir}")
        return
    if args.limit:
        videos = videos[:args.limit]
    
    print(f"Processing {len(videos)} videos -> {args.output}")
    print(f"Model: {args.model}, temp={args.temperature}, think={args.thinking_budget}")
    
    results = []
    correct = 0
    total_with_gt = 0
    
    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]", end="")
        
        # Annotated frame save path
        frame_path = None
        if args.save_frames:
            frame_path = args.output / "annotated_frames" / f"{vp.stem}_annotated.png"
        
        # Analyze
        analysis = analyze_video_v2(
            vp, client,
            model=args.model,
            temperature=args.temperature,
            thinking_budget=args.thinking_budget,
            task_context=args.task_context,
            save_annotated_frame=frame_path
        )
        
        # Map to world coordinates
        world_pred = map_prediction_to_world(analysis)
        analysis["world_prediction"] = world_pred
        
        # Ground truth & accuracy
        gt = extract_ground_truth(vp.stem)
        if gt:
            accuracy = evaluate_accuracy(world_pred, gt)
            analysis["ground_truth"] = gt
            analysis["accuracy"] = accuracy
            total_with_gt += 1
            if accuracy.get("side_correct"):
                correct += 1
            tag = "CORRECT" if accuracy.get("side_correct") else "WRONG"
            print(f"  >> {tag} (pred={world_pred['world_side']}, gt={gt['target_side']})")
        
        results.append(analysis)
        
        # Save individual
        out_file = args.output / f"{vp.stem}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        # Running accuracy
        if total_with_gt > 0:
            print(f"  Running accuracy: {correct}/{total_with_gt} = {correct/total_with_gt:.0%}")
        
        # Rate limit
        if i < len(videos):
            time.sleep(1.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if total_with_gt > 0:
        print(f"Side Accuracy: {correct}/{total_with_gt} = {correct/total_with_gt:.1%}")
        
        # Per-side breakdown
        left_correct = sum(1 for r in results 
                          if r.get("ground_truth", {}).get("target_side") == "left"
                          and r.get("accuracy", {}).get("side_correct"))
        left_total = sum(1 for r in results 
                        if r.get("ground_truth", {}).get("target_side") == "left")
        right_correct = sum(1 for r in results 
                           if r.get("ground_truth", {}).get("target_side") == "right"
                           and r.get("accuracy", {}).get("side_correct"))
        right_total = sum(1 for r in results 
                         if r.get("ground_truth", {}).get("target_side") == "right")
        
        if left_total > 0:
            print(f"  Left arcs:  {left_correct}/{left_total} = {left_correct/left_total:.1%}")
        if right_total > 0:
            print(f"  Right arcs: {right_correct}/{right_total} = {right_correct/right_total:.1%}")
    
    # Legibility stats
    leg_scores = [r.get("trajectory_legibility", {}).get("score", None) 
                  for r in results if "error" not in r]
    leg_scores = [s for s in leg_scores if s is not None]
    if leg_scores:
        print(f"\nLegibility: {np.mean(leg_scores):.3f} +/- {np.std(leg_scores):.3f}")
    
    # Save summary
    summary = {
        "total_videos": len(results),
        "accuracy": {
            "side_accuracy": correct / total_with_gt if total_with_gt else None,
            "correct": correct,
            "total": total_with_gt,
            "left": {"correct": left_correct, "total": left_total} if total_with_gt else None,
            "right": {"correct": right_correct, "total": right_total} if total_with_gt else None
        },
        "legibility": {
            "mean": float(np.mean(leg_scores)) if leg_scores else None,
            "std": float(np.std(leg_scores)) if leg_scores else None,
        },
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "thinking_budget": args.thinking_budget,
            "prompt_length": len(build_prompt()),
        }
    }
    
    summary_file = args.output / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    # Save all results
    all_file = args.output / "all_results.jsonl"
    with open(all_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"All results saved to {all_file}")


if __name__ == "__main__":
    main()
