# Arc Measurement Correction Summary

**Date:** March 9, 2026
**Status:** ✅ Complete - All scripts updated

## Problem Identified

The arc measurement was using **cumsum of actions** which is fundamentally broken:
- `max(|cumsum(actions[:, 1])|)` measured cumulative displacement (0.79-2.30m range)
- This does NOT correlate with visual arc size
- High cumsum can result from oscillation/wobbling, not large sweeps
- User visual validation: 0.79m cumsum video showed large arc, 0.98m cumsum showed gentle arc

## Correct Measurement

**Method:** Measure maximum lateral Y position during executed trajectory
- `max(|ee_positions[:, 1]|)` where ee_positions come from observations
- Range: 0.06-0.15m (matches physical constraints)
- Directly correlated with visual arc size ✓

## Updated Thresholds

Based on demo Bézier control points → executed trajectory analysis:

| Arc Class | Old Threshold (cumsum) | New Threshold (max \|Y\|) | Physical Meaning |
|-----------|----------------------|--------------------------|------------------|
| 00-05 (gentle) | < 0.05m | < 0.08m | Nearly straight path |
| 10-14 (moderate) | 0.05-0.15m | 0.08-0.12m | Moderate lateral sweep |
| 15-19 (large/legible) | ≥ 0.15m | ≥ 0.12m | Large sweeping arc |

**Key Insight:** Demo control point cp_y_mag = 0.23-0.28m → executed max |Y| ≈ 0.12-0.15m

## Files Updated

### Core Evaluation Scripts ✅

1. **paired_rollouts_proper.py** (623 lines)
   - Updated `measure_arc()` to take observations instead of actions
   - Updated `arc_class()` thresholds to 0.08/0.12m
   - Modified `execute_and_capture_trajectory()` to collect observations
   - Modified `rollout_baseline()` to collect observations
   - Updated arc measurement callsites to use executed observations

2. **paired_replanning_rollouts_v2.py** (535 lines)
   - Updated `measure_arc()` signature
   - Updated `arc_class()` thresholds
   - Changed default `arc15_threshold` from 0.15 to 0.12m
   - Modified `collect_candidate_frames()` to return observations
   - Updated VLM candidate selection to use executed arc measurements
   - Modified `rollout_baseline()` to collect observations

3. **paired_iterative_vlm.py** (709 lines)
   - **Already correct!** Was using `max(|obs[:, 1]|)` ✓
   - Updated `arc_class()` thresholds from 0.11/0.17/0.23 to 0.08/0.12m
   - Updated `arc_min` threshold from 0.15 to 0.12m

### Video Generation Scripts ✅

4. **generate_arc15_policy_videos.py** (228 lines)
   - Updated `measure_arc()` to take observations
   - Updated `classify_arc()` thresholds
   - Modified arc filtering to do test execution (150 steps) before video recording
   - Changed arc threshold from 0.15m to 0.12m

### Diagnostic Scripts ✅

5. **debug_vlm_selection.py** (902 lines)
   - Updated `measure_arc()` to take observations
   - Added `measure_arc_from_actions()` helper for action-based approximation
   - Updated `classify_arc()` thresholds to 0.08/0.12m
   - Updated all callsites to use appropriate function

6. **verify_arc_diversity.py** (416 lines)
   - Updated docstring to note cumsum is approximation
   - Added warning that true arc requires execution

### Comparison Scripts ✅

7. **test_vlm_vs_no_vlm.py** (462 lines)
   - Added both `measure_arc()` (for observations) and `measure_arc_from_actions()` (for approximation)
   - Updated `classify_arc()` thresholds to 0.08/0.12m

8. **compare_policy_vs_vlm_videos.py** (472 lines)
   - Added both measurement functions
   - Updated thresholds

9. **rollout_policy_vs_vlm_guided.py** (526 lines)
   - Added both measurement functions
   - Updated `arc_class()` thresholds

## Verification Results

Generated 11 verification rollouts (seeds 100-401) with videos and trajectory plots:
- Videos show executed trajectories
- Plots display max |Y| measurement and arc classification
- Visual inspection confirmed: arc ≥ 0.12m shows large sweeping motion ✓

**User confirmation:** "yes this is correct way of measuring"

## Key Changes Summary

### Measurement Functions

**Before (BROKEN):**
```python
def measure_arc(actions: np.ndarray) -> float:
    dy_cumsum = np.cumsum(actions[:, 1])
    return float(np.max(np.abs(dy_cumsum)))
```

**After (CORRECT):**
```python
def measure_arc(obs_trajectory: np.ndarray) -> float:
    """Measure arc from executed trajectory observations."""
    if len(obs_trajectory) == 0:
        return 0.0
    ee_y_positions = np.abs(obs_trajectory[:, 1])  # EE Y coordinate
    return float(np.max(ee_y_positions))
```

**Helper for diagnostics:**
```python
def measure_arc_from_actions(actions: np.ndarray) -> float:
    """Estimate arc from actions (approximation only)."""
    dy_cumsum = np.cumsum(actions[:, 1])
    return float(np.max(np.abs(dy_cumsum)))
```

### Classification Thresholds

**Before:**
```python
def arc_class(arc: float) -> str:
    if arc < 0.05: return "00-05"
    if arc < 0.15: return "10-14"
    return "15-19"
```

**After:**
```python
def arc_class(arc: float) -> str:
    if arc < 0.08: return "00-05"
    if arc < 0.12: return "10-14"
    return "15-19"
```

## Implementation Pattern

For scripts that execute rollouts:
1. Collect observations during `env.step()` execution
2. Store: `observations.append(result.obs)`
3. Measure: `arc = measure_arc(np.array(observations))`
4. Use max |Y| position from observations[:, 1]

For scripts that only sample actions (rare):
- Use `measure_arc_from_actions()` approximation
- Add clear warning this is less accurate

## Impact on Evaluation

### Expected Changes

1. **Arc distribution will shift:**
   - Old: ~100% classified as arc 15-19 (due to cumsum >= 0.15m)
   - New: More realistic distribution across arc classes

2. **VLM guidance more effective:**
   - Can now correctly identify and select arc 15-19 trajectories
   - Threshold 0.12m matches visual observation

3. **Policy capability assessment:**
   - Can accurately measure what percentage of samples reach arc 15-19
   - Inform whether CFG steering or trajectory warping is needed

## Next Steps

1. ✅ All core scripts updated
2. ✅ Syntax validation passed
3. ⏳ Re-run `verify_arc_diversity.py` to assess policy's natural arc capability
4. ⏳ Complete final 4 verification rollouts (seeds 500-503)
5. ⏳ Run full evaluation with corrected measurement

## Technical Notes

### Why Cumsum Failed

- **Cumsum measures:** Total accumulated lateral displacement
- **Can be high from:** Multiple small back-and-forth movements (oscillation)
- **Can be low from:** Large smooth arc that returns near center
- **Not correlated with:** Visual arc size or maximum lateral deviation

### Why Max |Y| Works

- **Max |Y| measures:** Furthest lateral position reached
- **High value means:** Arm traveled far from center (large visual arc)
- **Low value means:** Arm stayed near center (gentle/straight path)
- **Directly correlated with:** Visual perception of arc size ✓

### Observation Array Structure

```python
obs = env.step(action).obs  # shape: (obs_dim,)
# obs[0] = EE X position
# obs[1] = EE Y position  ← Used for arc measurement
# obs[2] = EE Z position
# ... other features
```

## Validation

- ✅ User confirmed visual match between measurement and video appearance
- ✅ Thresholds validated:
  - Arc 15-19 videos show arm sweeping to Y ≈ 0.12-0.15m laterally
  - Arc 00-05 videos show gentle/straight paths with Y < 0.08m
- ✅ All scripts compile without errors
- ✅ 11 verification videos generated successfully

---

**Conclusion:** Arc measurement is now corrected across all scripts. The system can accurately identify and measure arc 15-19 trajectories for legibility-guided evaluation.
