#!/usr/bin/env python3
"""
Stage 1b: Visual VLM Code Synthesis
====================================

Addresses HONEST_ASSESSMENT issues:
  - "VLM used text-only prompt with no visual data"
  - "400 collected demos never shown to VLM"

This script sends Gemini actual trajectory **videos** from collected demos
alongside the code synthesis prompt. The VLM SEES what legible vs ambiguous
trajectories look like, grounding its scoring function in visual evidence.

Pipeline:
  1. Select contrasting demo videos (straight arc00 vs large arc15/19)
  2. Upload videos to Gemini's multimodal API
  3. Ask Gemini to generate a scoring function grounded in visual examples
  4. Validate + evaluate (reuses existing validation infrastructure)

Usage:
  python experiments/visual_vlm_synthesis.py --api_key YOUR_KEY
  python experiments/visual_vlm_synthesis.py  # uses GOOGLE_API_KEY env var
"""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.stage1_vlm_guidance import (
    extract_python_code,
    validate_generated_code,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEMO_VIDEO_DIR = Path('data/demos/demo_videos')

# ─── Curate contrasting examples ────────────────────────────────────────

def select_demo_videos() -> dict:
    """Pick a small set of contrasting demo videos:
    - 2 highly legible (large arc toward goal)
    - 2 ambiguous/straight (arc00, minimal lateral movement)
    - Both left and right goals represented
    """
    selections = {
        "legible_left_large_arc": DEMO_VIDEO_DIR / "cfg00_left_arc19.mp4",
        "legible_right_large_arc": DEMO_VIDEO_DIR / "cfg00_right_arc19.mp4",
        "straight_left_no_arc": DEMO_VIDEO_DIR / "cfg00_left_arc00.mp4",
        "straight_right_no_arc": DEMO_VIDEO_DIR / "cfg00_right_arc00.mp4",
    }

    # Verify files exist
    for label, path in selections.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, trying cfg01 fallback")
            fallback = Path(str(path).replace("cfg00", "cfg01"))
            if fallback.exists():
                selections[label] = fallback
            else:
                raise FileNotFoundError(f"Demo video not found: {path}")

    return selections


# ─── Visual VLM Prompt ──────────────────────────────────────────────────

VISUAL_PROMPT = textwrap.dedent("""\
You are an expert robotics researcher specializing in trajectory legibility.

I'm showing you 4 videos of a Franka Panda robot arm picking up one of two
red blocks on a table:

**Video 1** (legible_left_large_arc): The robot picks up the LEFT block using
  a wide arcing trajectory that clearly communicates its intent early.
**Video 2** (legible_right_large_arc): The robot picks up the RIGHT block using
  a wide arcing trajectory.
**Video 3** (straight_left_no_arc): The robot picks up the LEFT block using
  a straight/direct path — harder to tell early which block it's going for.
**Video 4** (straight_right_no_arc): The robot picks up the RIGHT block using
  a straight/direct path.

## Task Setup
- Left block at approximately (0.50, -0.07, 0.42) in world coordinates
- Right block at approximately (0.50, +0.07, 0.42) in world coordinates
- Robot end-effector starts at approximately (0.40, 0.0, 0.55)
- The blocks are ~14cm apart (y-axis separation)

## What Makes Videos 1 & 2 More Legible?
These trajectories are easier for a human observer to decode because:
- The robot CURVES toward the intended block early in the motion
- The lateral movement (y-axis) creates a clear visual signal
- The trajectory separates from the ambiguous center line quickly

## Your Task
Write a differentiable PyTorch scoring function that captures these
visual properties. The function should score Videos 1 & 2 style trajectories
HIGHER than Videos 3 & 4 style trajectories.

## Function Signature

```python
def vlm_legibility_score(
    ee_traj: torch.Tensor,      # (H, 3) predicted EE positions (x,y,z)
    goals: torch.Tensor,        # (K, 3) goal positions (K=2 blocks)
    true_goal_idx: int = 0,     # index of committed goal in goals
    early_frac: float = 0.30,   # fraction of trajectory considered "early"
) -> torch.Tensor:
    \"\"\"Return a differentiable scalar score in [0, 1]. Higher = more legible.
    Must support torch.autograd.grad() backpropagation through ee_traj.\"\"\"
```

## Requirements

1. **Differentiable**: No argmax, no if/else on tensor values, no .item(),
   no numpy. Use only PyTorch operations that support autograd.
2. **Multi-criteria** — combine these geometric cues you can SEE in the videos:
   - **Bayesian posterior** P(g*|x) using Gaussian proximity (how close to goal)
   - **Directional commitment** (velocity alignment with goal direction — visible
     as the trajectory curving toward one block)
   - **Lateral deviation** (early y-movement toward the goal — the key visual
     signal in Videos 1 & 2)
   - **Curvature profile** (smooth arc vs straight line — the main visual
     difference between legible and ambiguous demos)
3. **Auto-calibrated**: Derive scale parameters from inter-goal distance
4. **Numerically stable**: Use log-sum-exp, epsilon floors, torch.clamp
5. **Return scalar tensor**: Higher = more legible, range [0, 1]

## Reference Function (Baseline)
This function uses ONLY Gaussian proximity and achieves L_early=0.952:

```python
def l_early_intent_torch(ee_traj, goals, true_goal_idx=0, early_frac=0.30):
    H = ee_traj.shape[0]
    early_end = max(1, int(H * early_frac))
    early_traj = ee_traj[:early_end]
    dists = torch.cdist(goals, goals)
    mask = dists > 1e-6
    d_min = dists[mask].min() if mask.any() else torch.tensor(0.14)
    sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))
    diff = early_traj.unsqueeze(1) - goals.unsqueeze(0)
    sq_dist = (diff ** 2).sum(-1)
    log_like = -sq_dist / (2.0 * sigma ** 2)
    posteriors = torch.softmax(log_like, dim=-1)
    return posteriors[:, true_goal_idx].mean()
```

Your function should improve on this by capturing the VISUAL cues you observe
in the demo videos — particularly the early lateral arc that distinguishes
legible from straight trajectories.

Output ONLY the Python function code, starting with `def vlm_legibility_score(`.
No markdown fences, no imports (torch and math are available), no extra text.
""")


# ─── Call Gemini with Videos ────────────────────────────────────────────

def call_gemini_visual(api_key: str, videos: dict,
                       model: str = "gemini-2.5-flash",
                       max_retries: int = 3) -> str:
    """Send demo videos + prompt to Gemini for visual code synthesis."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Upload video files
    print("\n  Uploading demo videos to Gemini...")
    uploaded_files = {}
    for label, path in videos.items():
        print(f"    Uploading {label}: {path.name}")
        uploaded = client.files.upload(file=str(path))
        # Wait for processing
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"Video upload failed for {label}: {uploaded.state}")
        uploaded_files[label] = uploaded
        print(f"    -> {uploaded.state.name}")

    # Build multimodal content
    contents = []
    for label, uf in uploaded_files.items():
        contents.append(types.Part.from_uri(
            file_uri=uf.uri,
            mime_type="video/mp4",
        ))
        contents.append(types.Part.from_text(text=f"\n[This is: {label}]\n"))
    contents.append(types.Part.from_text(text=VISUAL_PROMPT))

    for attempt in range(max_retries):
        print(f"\n  [Gemini Visual] Attempt {attempt+1}/{max_retries} ...")
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=8192),
            temperature=0.2 + attempt * 0.2,
        )
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            raw_text = resp.text
            code = extract_python_code(raw_text)
            if code and "def vlm_legibility_score(" in code:
                print(f"  [Gemini Visual] Got valid function ({len(code)} chars)")
                return code
            else:
                print(f"  [Gemini Visual] No valid function in response")
                print(f"  Preview: {raw_text[:300]}...")
        except Exception as e:
            print(f"  [Gemini Visual] API error: {e}")
            time.sleep(2 ** attempt)

    # Cleanup uploaded files
    for uf in uploaded_files.values():
        try:
            client.files.delete(name=uf.name)
        except Exception:
            pass

    raise RuntimeError("Failed to get valid code from Gemini after all retries")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--api_key', default=os.environ.get('GOOGLE_API_KEY', ''))
    ap.add_argument('--model', default='gemini-2.5-flash')
    ap.add_argument('--output', default='outputs/stage1/vlm_score_fn_visual.py')
    ap.add_argument('--skip_eval', action='store_true')
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set GOOGLE_API_KEY or use --api_key")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print("  VISUAL VLM CODE SYNTHESIS")
    print(f"  Sending demo trajectory videos to Gemini for grounded synthesis")
    print(f"{'='*72}")

    # 1. Select contrasting demo videos
    print(f"\n  Step 1: Selecting demo videos...")
    videos = select_demo_videos()
    for label, path in videos.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"    {label}: {path.name} ({size_mb:.1f} MB)")

    # 2. Call Gemini with videos
    print(f"\n  Step 2: Generating visually-grounded scoring function...")
    code = call_gemini_visual(args.api_key, videos, model=args.model)

    # 3. Validate
    print(f"\n  Step 3: Validating generated code...")
    # Prepend necessary imports
    full_code = f"import torch\nimport math\nimport torch.nn as nn\n\n{code}"
    success, msg, fn = validate_generated_code(code, device)
    print(f"    Validation: {'PASS' if success else 'FAIL'} — {msg}")

    if not success:
        print(f"\n  Code that failed validation:")
        for i, line in enumerate(code.split('\n'), 1):
            print(f"    {i:3d} | {line}")
        print(f"\n  Saving raw code for inspection anyway...")

    # 4. Save
    with open(out_path, 'w') as f:
        f.write(f"# Visual VLM-generated scoring function\n")
        f.write(f"# Generated via Gemini {args.model} with demo video grounding\n")
        f.write(f"# Videos shown: {', '.join(v.name for v in videos.values())}\n")
        f.write(f"# Validated: {success}\n\n")
        f.write("import torch\nimport math\n\n")
        f.write(code)
        f.write("\n")
    print(f"\n  Saved to: {out_path}")

    # 5. Quick comparison if validation passed
    if success and not args.skip_eval:
        print(f"\n  Step 4: Quick discrimination test...")
        goals = torch.tensor([[0.50, -0.07, 0.42],
                               [0.50,  0.07, 0.42]], device=device)

        # Legible arc trajectory (like video 1)
        legible = torch.zeros(32, 3, device=device, requires_grad=True)
        for i in range(32):
            t = i / 31.0
            legible.data[i] = torch.tensor([
                0.40 + t * 0.10,
                0.0 - t * 0.07 - 0.04 * math.sin(math.pi * t),  # arc toward left
                0.55 - t * 0.13
            ])

        # Straight trajectory (like video 3)
        straight = torch.zeros(32, 3, device=device, requires_grad=True)
        for i in range(32):
            t = i / 31.0
            straight.data[i] = torch.tensor([
                0.40 + t * 0.10,
                0.0 - t * 0.07,  # straight line to left
                0.55 - t * 0.13
            ])

        # Ambiguous (goes to center)
        ambig = torch.zeros(32, 3, device=device, requires_grad=True)
        for i in range(32):
            t = i / 31.0
            ambig.data[i] = torch.tensor([
                0.40 + t * 0.10,
                0.0,
                0.55 - t * 0.13
            ])

        s_arc = fn(legible, goals, true_goal_idx=0).item()
        s_str = fn(straight, goals, true_goal_idx=0).item()
        s_amb = fn(ambig, goals, true_goal_idx=0).item()

        print(f"    Arc trajectory:      {s_arc:.4f}")
        print(f"    Straight-to-goal:    {s_str:.4f}")
        print(f"    Ambiguous center:    {s_amb:.4f}")

        if s_arc > s_str > s_amb:
            print(f"    EXCELLENT: arc > straight > ambiguous (ideal ordering)")
        elif s_arc > s_amb and s_str > s_amb:
            print(f"    GOOD: Both goal-directed > ambiguous")
        else:
            print(f"    WARNING: Unexpected ordering")

    # 6. Summary
    result = {
        "method": "visual_vlm_synthesis",
        "model": args.model,
        "videos_shown": {k: v.name for k, v in videos.items()},
        "validation_passed": success,
        "validation_message": msg,
        "output_path": str(out_path),
    }
    summary_path = out_path.parent / 'visual_vlm_synthesis_result.json'
    with open(summary_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  DONE — Visual VLM synthesis {'SUCCEEDED' if success else 'FAILED'}")
    print(f"{'='*72}\n")


if __name__ == '__main__':
    main()
