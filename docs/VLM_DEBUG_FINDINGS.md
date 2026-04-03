# VLM Selection Debug Results

**Date:** February 28, 2026  
**Debug Run:** vlm_debug_20260228_131811

---

## Problem Identified

### ❌ Issue: VLM Cannot See Trajectory Differences

**Symptoms:**
1. Generated 5 candidates with diverse arcs:
   - Candidate 0: 0.0739m (arc 10-14)
   - Candidate 1: 0.1610m (arc 15-19) ✓
   - Candidate 2: 0.1747m (arc 15-19) ✓ <- LARGEST ARC
   - Candidate 3: 0.1441m (arc 10-14)
   - Candidate 4: 0.1737m (arc 15-19) ✓

2. **All VLM scores were identical: 0.500**
3. VLM responses indicated frames were not useful:
   - Candidate 0: "gripper is centered between both blocks"
   - Candidate 1-4: "no objects or gripper visible"

4. **Selected candidate 0 (arc=0.0739m) instead of arc 15-19 candidates**
   - When all scores are 0.500, argmax picks first one

---

## Root Causes

### 1. Trajectory Overlay May Not Be Visible
- The `TrajectoryVisualizer._overlay_trajectory()` draws:
  - Orange arrows showing predicted path
  - Yellow circles at predicted positions
  - Magenta circle for current EE position
  - Green/Red circles for block positions

**Potential issues:**
- Projection from 3D world coords to 2D pixels might be off
- Camera angle in PyBullet might not show the overlays
- Colors might blend with background
- Trajectory might be drawn outside visible frame

### 2. VLM Prompt May Need Improvement
Current approach sends raw frame without context. VLM might need:
- Better prompt explaining what to lookfor
- Multiple frames (sequence) instead of single prefix frame  
- Higher contrast visualization
- Text annotations on the frame itself

### 3. PyBullet Rendering vs Overlay Timing
The visualizer:
1. Calls `env.render()` to get base frame
2. Overlays trajectory using OpenCV
3. Converts to JPEG

**Issue:** PyBullet might be rendering before the trajectory is simulated, or the overlay might not be composited correctly.

---

## Recommended Fixes

### Fix 1: Improve Trajectory Visualization
```python
# In trajectory_visualizer.py
def _overlay_trajectory(...):
    # Use brighter, more contrasting colors
    trajectory_color = (0, 255, 255)  # Cyan - highly visible
    marker_color = (255, 0, 255)  # Magenta
    
    # Thicker lines
    thickness = 6  # Was 4
    
    # Add trajectory label
    cv2.putText(img, "PREDICTED PATH", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Draw arc direction indicator
    if len(position_deltas) > 2:
        dy_total = sum(d[1] for d in position_deltas)
        direction = "LEFT" if dy_total > 0 else "RIGHT"
        cv2.putText(img, f"TARGET: {direction}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
```

### Fix 2: Better VLM Prompt
```python
# In vlm_client.py score_trajectory()
prompt = f"""
You are viewing a robot trajectory prediction visualization.

The image shows:
- Current gripper position (magenta circle)  
- Predicted trajectory path (orange/cyan arrows)
- Two goal options:
  * Goal A: {goal_A}
  * Goal B: {goal_B}

Look at the DIRECTION of the predicted arrow path.
Based on the early trajectory curve, which goal is the robot moving toward?

Respond with your assessment.
"""
```

### Fix 3: Add Debugging Visualization
Create annotated frames that explicitly show:
1. Baseline arc magnitude on the frame
2. Target direction (LEFT/RIGHT)
3. Predicted path with measurements

### Fix 4: Test with Simple Synthetic Data
Before using VLM, test visualization with known trajectories:
```python
# Create obvious test cases
straight_trajectory = np.zeros((8, 5))  # No curve
left_curve = np.zeros((8, 5))
left_curve[:, 1] = 0.02  # Clear leftward curve
right_curve = np.zeros((8, 5))
right_curve[:, 1] = -0.02  # Clear rightward curve

# Visualize and manually inspect
```

---

## Next Steps

1. **Immediate**: Check `runs/vlm_debug_20260228_131811/candidates_comparison.png`
   - Confirm trajectory overlays are visible
   - If not visible, fix projection or rendering

2. **Test Visualization Independently**:
   ```bash
   py test_trajectory_visualizer.py
   ```
   - Create simple test trajectories
   - Verify overlays appear correctly
   - Save annotated examples

3. **Improve Contrast**:
   - Use high-contrast colors (cyan, magenta, yellow)
   - Add text labels directly on frames
   - Increase line thickness

4. **Re-test VLM**:
   ```bash
   py debug_vlm_selection.py --n_samples 5 --seed 42
   ```
   - Check if VLM scores differ now
   - Verify arc 15-19 candidates get higher scores

5. **Full Evaluation**:
   Once VLM can discriminate, run full experiment:
   ```bash
   py scripts/eval_legibility_steering.py \
       --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt \
       --n_episodes 20 \
       --n_samples 5 \
       --save_videos
   ```
---

## Files for Investigation

**Debug outputs:**
- `runs/vlm_debug_20260228_131811/candidates_comparison.png` - Grid showing all 5 candidates
- `runs/vlm_debug_20260228_131811/vlm_input_frames/` - Individual frames sent to VLM
- `runs/vlm_debug_20260228_131811/debug_results.json` - Full VLM responses

**Code to check:**
- `scripts/trajectory_visualizer.py` lines 126-191 - Overlay logic
- `scripts/vlm_client.py` lines 70-110 - VLM prompt and API call  
- `debug_vlm_selection.py` lines 305-350 - Candidate generation

---

## Expected Behavior

**When working correctly:**
1. Generate 5 candidates with arc diversity (got this ✓)
2. VLM sees trajectory overlays on frames (FAILING ✗)
3. VLM scores differ based on arc magnitude:
   - Arc 15-19: legibility_score > 0.70
   - Arc 10-14: legibility_score ~ 0.55-0.70
   - Arc 00-05: legibility_score < 0.55
4. argmax selects highest scoring candidate (arc 15-19)
5. Executed trajectory shows large legible curve

**Currently:**
- Scores all 0.500 (VLM uncertain)
- Selects first candidate by default
- Results in less legible trajectory

---

## Status

- ✅ Arc diversity generation: WORKING (28% arc 15-19)
- ✅ Candidate sampling: WORKING (got 3 arc 15-19 in 5 samples)
- ❌ Trajectory visualization: NEEDS FIX (VLM can't see differences)
- ❌ VLM discrimination: BLOCKED (waiting for visualization fix)
- ⏸️ Selection logic: CORRECT (but getting uniform inputs)

**Priority: Fix trajectory visualization so VLM can see predicted paths**
