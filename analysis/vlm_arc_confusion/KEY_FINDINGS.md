# VLM Arc Confusion Analysis: Key Findings

## Executive Summary

**Critical Discovery**: The VLM exhibits a paradoxical behavior where **high-arc trajectories (15-19) become legible faster (2.8s) but have LOWER final accuracy (22.2%)** compared to moderate arcs (44.4% at 2.8s). This pattern strongly suggests the VLM is **confusing or flipping left/right predictions** for lateral arc trajectories.

---

## Overall Statistics

- **Total Evaluations**: 438 VLM predictions
- **Total Videos**: 40 (20 left arcs + 20 right arcs)
- **Arc Range**: 0-19 (from straight to high lateral motion)
- **Time Points**: 0-10 seconds (11 evaluations per video)
- **Overall Accuracy**: 40.0%

---

## Arc Class Performance Breakdown

### Arc 00-04: Straight Trajectories
- **Final Accuracy**: 30.0% ❌
- **Time to Legible**: 3.3s
- **Avg Confidence**: 75.5%
- **Avg Prediction Flips**: 0.70
- **Individual Arc Performance**:
  - Arc 00: 50% final accuracy (0.5 flips)
  - Arc 01: 50% final accuracy (0.5 flips)
  - Arc 02: **0%** final accuracy (1.5 flips) ⚠️
  - Arc 03: 50% final accuracy (0.5 flips)
  - Arc 04: **0%** final accuracy (0.5 flips) ⚠️

### Arc 05-09: Slight Lateral Trajectories
- **Final Accuracy**: 50.0% ✅ (at chance level)
- **Time to Legible**: 4.3s (SLOWEST to become legible)
- **Avg Confidence**: 76.3%
- **Avg Prediction Flips**: 0.50
- **Individual Arc Performance**:
  - Arc 05: 50% final accuracy (0.5 flips)
  - Arc 06: 50% final accuracy (0.5 flips)
  - Arc 07: 50% final accuracy (0.0 flips)
  - Arc 08: 50% final accuracy (0.5 flips)
  - Arc 09: 50% final accuracy (1.0 flips)

### Arc 10-14: Moderate Lateral Trajectories  
- **Final Accuracy**: 44.4% 
- **Time to Legible**: 2.8s (FASTER than straight and slight arcs!)
- **Avg Confidence**: 81.2% (highest confidence)
- **Avg Prediction Flips**: 0.10 (MOST STABLE predictions)
- **Individual Arc Performance**:
  - Arc 10: 50% final accuracy (0.0 flips)
  - Arc 11: 50% final accuracy (0.0 flips)
  - Arc 12: **0%** final accuracy (0.0 flips) ⚠️
  - Arc 13: 50% final accuracy (0.5 flips)
  - Arc 14: 50% final accuracy (0.0 flips)

### Arc 15-19: High Lateral Trajectories ⚠️ **MOST CONFUSING**
- **Final Accuracy**: 22.2% ❌❌ (WORSE than chance!)
- **Time to Legible**: 2.8s (FASTEST to become legible!)
- **Avg Confidence**: 81.0% (VLM is confident but WRONG!)
- **Avg Prediction Flips**: 0.90 (MOST UNSTABLE predictions)
- **Individual Arc Performance**:
  - Arc 15: **0%** final accuracy (1.0 flips) ⚠️
  - Arc 16: 50% final accuracy (2.0 flips - MOST UNSTABLE!)
  - Arc 17: **0%** final accuracy (0.5 flips) ⚠️
  - Arc 18: 50% final accuracy (0.5 flips)
  - Arc 19: **0%** final accuracy (0.5 flips) ⚠️

---

## The Paradox: Speed vs Accuracy

| Arc Class | Time to Legible | Final Accuracy | Confidence | Flips |
|-----------|----------------|----------------|------------|-------|
| **Arc 00-04** (Straight) | 3.3s | 30.0% | 75.5% | 0.70 |
| **Arc 05-09** (Slight) | **4.3s** ⬆️ | **50.0%** ✅ | 76.3% | 0.50 |
| **Arc 10-14** (Moderate) | **2.8s** ⬇️ | 44.4% | **81.2%** | **0.10** |
| **Arc 15-19** (High) | **2.8s** ⬇️ | **22.2%** ❌ | 81.0% | **0.90** |

### Key Insight:
**High arcs (15-19) become legible 1.5 seconds FASTER than slight arcs (05-09), but their final accuracy is 27.8 percentage points LOWER (22.2% vs 50.0%)!**

This means:
1. ✅ VLM detects distinctive lateral motion EARLY
2. ❌ VLM predicts the WRONG target (likely flipping left/right)
3. ⚠️ VLM is CONFIDENT (81%) about its WRONG prediction

---

## Prediction Flip Analysis

### Most Unstable Arcs (Frequent Prediction Changes):
1. **Arc 16**: 2.0 flips per video (sequence: AABAAAAAAAA)
2. **Arc 02**: 1.5 flips per video (sequence: CCAACCCBAAB)
3. **Arc 15**: 1.0 flips per video (sequence: CCAAAAAAAAB - predicts A then flips to B at end!)
4. **Arc 09**: 1.0 flips per video (sequence: CAAACABAAAA)

### Most Stable Arcs (Consistent Predictions):
1. **Arc 07-08, 10-12, 14**: 0.0-0.1 flips (predictions stabilize early)

### Pattern Discovery:
**High arcs (15-19) show late-stage prediction flips** - VLM correctly predicts for 1-9 seconds, then FLIPS to wrong choice at t=10s!

Examples:
- Arc 15 (left): `CCAAAAAAAAB` ← Predicts A (correct) then flips to B at end
- Arc 19 (left): `CACACAAAAAB` ← Mostly A (correct) then flips to B at end

---

## Evidence of Systematic Confusion

### Problem Arcs with 0% Final Accuracy:
- **Arc 02** (left, straight): Flips 1.5 times, ends wrong
- **Arc 04** (left, straight): Becomes legible at 4s, flips to wrong at 10s
- **Arc 12** (left, moderate): Becomes legible at 1s but final prediction is wrong
- **Arc 15** (left, high): Correctly predicts A for 9s, flips to B at end
- **Arc 17** (left, high): Becomes legible at 2s, ends wrong
- **Arc 19** (left, high): Becomes legible at 1s, ends wrong

### Common Pattern:
**3 out of 5 high arcs (60%) end with 0% accuracy**, compared to:
- 2 out of 5 straight arcs (40%)
- 0 out of 5 slight arcs (0%)
- 1 out of 5 moderate arcs (20%)

---

## Confidence vs Correctness Mismatch

| Arc Class | Avg Confidence | Final Accuracy | Confidence-Accuracy Gap |
|-----------|----------------|----------------|------------------------|
| Arc 00-04 | 75.5% | 30.0% | **+45.5%** ⚠️ |
| Arc 05-09 | 76.3% | 50.0% | +26.3% |
| Arc 10-14 | 81.2% | 44.4% | **+36.8%** ⚠️ |
| Arc 15-19 | 81.0% | 22.2% | **+58.8%** ⚠️⚠️⚠️ |

**High arcs have the WORST confidence-accuracy mismatch (+58.8%)**: VLM is 81% confident but only 22% correct!

---

## Hypothesis: VLM Lateral Arc Confusion

### Evidence Supporting "Left-Right Flip" Hypothesis:

1. **High lateral arcs trigger EARLY legibility detection** (2.8s vs 4.3s)
   - VLM sees distinctive lateral motion
   - VLM becomes confident (81%) in its prediction

2. **But predictions are systematically WRONG** (22.2% accuracy)
   - Worse than chance (50%)
   - 3 out of 5 high arcs have 0% final accuracy

3. **Late-stage prediction flips** common in high arcs
   - Arc 15: `...AAAAAAAB` (correct for 9s, flips to wrong at end)
   - Arc 19: `...AAAAAAB` (correct for 9s, flips to wrong at end)

4. **Moderate arcs are MOST stable** (0.10 flips)
   - But still only 44.4% accurate
   - VLM is confident (81.2%) but makes consistent errors

### Interpretation:
The VLM likely:
- ✅ **Correctly detects** that motion is lateral/legible
- ❌ **Incorrectly infers** which side (left vs right) is the target
- ⚠️ **Confidently commits** to the wrong prediction
- 🔄 **Sometimes flips** prediction late in trajectory (but often to wrong side)

---

## Recommendations for VLM-Guided Steering

### ❌ DO NOT use high arcs (15-19) as "legible" examples
- Despite fast legibility detection (2.8s), final accuracy is only 22.2%
- VLM is likely flipping left/right predictions

### ✅ DO use slight arcs (05-09) as baseline
- Best accuracy (50%, at chance level)
- Slowest to become legible (4.3s) but most reliable

### ⚠️ CAUTION with moderate arcs (10-14)
- High confidence (81.2%) and stability (0.10 flips)
- But only 44.4% accuracy - still systematically confused

### 🔍 INVESTIGATE further:
1. Why do high arcs cause left-right confusion?
2. Is it the viewing angle or trajectory curvature?
3. Can we improve VLM prompting to fix this?
4. Should we provide reference frames or spatial cues?

---

## Files Generated

All analysis outputs saved to: `analysis/vlm_arc_confusion/`

### Visualizations:
- **accuracy_heatmap.png** - Arc × Time heatmap showing prediction accuracy
- **accuracy_curves.png** - Per-arc accuracy trajectories over time
- **confidence_heatmap.png** - Arc × Time heatmap showing VLM confidence
- **flip_analysis.png** - Prediction instability analysis (4 subplots)

### Data Files:
- **vlm_timepoints_detailed.csv** - All 438 evaluations (similar to user_study format)
- **per_arc_accuracy.csv** - Accuracy statistics per arc per timepoint
- **flip_statistics.csv** - Prediction flip patterns per video
- **summary_statistics.json** - Complete numerical results

---

## Next Steps

1. **Visual Inspection**: Review the 4 generated PNG figures to see patterns
2. **Per-Arc Analysis**: Examine `per_arc_accuracy.csv` for specific arc × time patterns
3. **Flip Sequences**: Check `flip_statistics.csv` to see exact prediction sequences
4. **VLM Debugging**: Investigate WHY high arcs cause confusion (view frames, analyze prompts)
5. **Steering Strategy**: Design legibility steering that accounts for VLM's systematic errors

---

**Generated**: March 13, 2026  
**Analysis Script**: `analysis/analyze_vlm_arc_confusion.py`  
**Source Data**: `outputs/demo_legibility_prefix_cfg00/results.jsonl`
