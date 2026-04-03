# CRITICAL: VLM Right Arc Complete Failure

## The Crisis

**ALL 20 RIGHT ARC VIDEOS HAVE 0% FINAL ACCURACY**

This is not a subtle issue - the VLM **SYSTEMATICALLY FAILS** on right-side trajectories.

## Evidence

### Left Arcs (Mixed Performance):
- cfg00_left_arc00: ✅ Final prediction A (correct) - sequence: `CCCACCCAAAA` 
- cfg00_left_arc02: ❌ Final prediction B (wrong) - sequence: `CCAACCCBAAB`
- cfg00_left_arc03: ✅ Final prediction A (correct) - sequence: `CCCCCCCAAAA`
- **Left arc accuracy: 50%** (10/20 correct)

### Right Arcs (TOTAL FAILURE):
- cfg00_right_arc00: ❌ Final prediction A (wrong) - sequence: `CCCCCCBAAAA` ⚠️
- cfg00_right_arc01: ❌ Final prediction A (wrong) - sequence: `CCCCCBAAAAA` ⚠️
- cfg00_right_arc02: ❌ Final prediction A (wrong) - sequence: `BCCCCCBAAAA` ⚠️
- cfg00_right_arc03: ❌ Final prediction A (wrong) - sequence: `CCCBCBBAAAA` ⚠️
- cfg00_right_arc07: ❌ Final prediction A (wrong) - sequence: `CCCCCCAAAAA` (NEVER predicts B!)
- cfg00_right_arc10: ❌ Final prediction A (wrong) - sequence: `CCCCCCAAAAA` (NEVER predicts B!)
- cfg00_right_arc11: ❌ Final prediction A (wrong) - sequence: `CCCCCCAAAAA` (NEVER predicts B!)
- **Right arc accuracy: 0%** (0/20 correct)

## The Pattern: VLM Correctly Identifies, Then FLIPS

Example: **cfg00_right_arc00** (ground truth = B/right block)

| Time | pA | pB | Choice | VLM Says | Correct? |
|------|-----|-----|--------|----------|----------|
| t=0-5 | 0.50 | 0.50 | C | "stationary" | N/A |
| **t=6** | **0.05** | **0.95** | **B** | "gripper moved towards the right block" | ✅ |
| **t=7** | **0.99** | **0.01** | **A** | "gripper positioned directly above the **left** block" | ❌ |
| t=8 | 1.00 | 0.00 | A | "gripper is directly above the **left** block" | ❌ |
| t=9 | 0.99 | 0.01 | A | "gripper positioned above **left** block" | ❌ |
| t=10 | 1.00 | 0.00 | A | "gripper is grasping the **left** block" | ❌ |

**The VLM CORRECTLY sees it moving to the right block at t=6, then FLIPS and says it's grasping the LEFT block at t=7-10!**

## Why This Happens

### The VLM's Perception Error:

1. **At t=6-7 (mid-trajectory)**: VLM correctly tracks directional motion
   - "gripper moved towards the right block" ✅

2. **At t=8-10 (grasping phase)**: VLM MISIDENTIFIES which block is being grasped
   - Says "grasping the left block" but it's actually grasping the RIGHT block ❌
   - VLM confidence: 99-100% (completely confident but completely wrong!)

### Root Cause: Spatial Disorientation

The VLM appears to:
- ✅ Track motion direction correctly (early phase)
- ❌ Lose spatial reference when gripper hovers/grasps (late phase)
- ❌ Confuse left/right identity of blocks in close-up views

Possible technical reasons:
1. **Camera viewpoint**: Top-down view may lack depth cues
2. **Symmetry**: Two identical blocks look mirror-symmetric
3. **Crop/zoom**: Late frames may show only gripper+single block (no spatial context)
4. **Prefix accumulation**: VLM sees 8-10 frames but may weight recent frames more
5. **No spatial anchors**: No consistent reference (walls, edges, asymmetric markers)

## Implications for Your Questions

### Q1: "Why is it performing bad at right arcs?"
**A: It's not "bad" - it's SYSTEMATICALLY BIASED toward predicting LEFT (A).**

- When gripper is over/grasping a block, VLM defaults to "it's the left block"
- This might be a model bias (training data imbalance?) or spatial confusion
- 0% accuracy on right arcs means VLM has a strong LEFT ("A") bias

### Q2: "How is final accuracy wrong when end-effector is directly above a block?"
**A: The VLM sees A BLOCK but MIS-IDENTIFIES WHICH block (left vs right).**

At t=10 for cfg00_right_arc00:
- **Reality**: Gripper grasping RIGHT block
- **VLM says**: "gripper is grasping the left block" (100% confidence)
- **Problem**: VLM correctly detects "grasping action" but confuses block IDENTITY

The VLM's **object detection works**, but **spatial reasoning fails**.

### Q3: "What should we do with flip information?"
**A: Flips reveal VLM's uncertainty about spatial identity.**

Pattern observed:
- **Stable predictions** (0 flips): VLM commits to one wrong answer early (e.g., right_arc07: `CCCCCCAAAAA`)
- **Flip predictions** (1-2 flips): VLM wavers between correct and wrong (e.g., right_arc00: `CCCCCCBAAAA`)

**Flips are NOT always bad**: They show VLM is gathering conflicting evidence. The RIGHT arc videos that flip at t=6-7 (`...BAAAA`) are actually CLOSER to correct than those that never flip (`...AAAAA`).

For steering:
- ✅ Use pA/pB **trends** (increasing confidence) not just final values
- ⚠️ Flips indicate VLM confusion - wait for stabilization
- ❌ Don't trust high confidence (99%) if it appears suddenly after a flip

### Q4: "Is VLM correct when it's confident?"
**A: NO! High confidence ≠ correctness.**

Examples:
- Right arcs at t=7-10: 99-100% confidence but 0% accuracy
- VLM confidently hallucinates that gripper is over LEFT block

For steering:
- ❌ **You CANNOT use confidence as a reliability signal**
- ❌ **High confidence (81-100%) with right arcs = systematically wrong**

### Q5: "Should we consider when VLM gets it right as legible time?"
**A: Only if prediction REMAINS stable afterward.**

Look at cfg00_right_arc00:
- t=6: Correct prediction (B) with 95% confidence → Seems legible!
- t=7: FLIPS to wrong prediction (A) with 99% confidence → Actually NOT legible!

**True legibility** should mean:
- VLM gets correct prediction AND
- Prediction remains stable for next 2-3 seconds

For cfg00_left_arc01:
- t=0: Predicts A (correct) with 60% confidence
- t=1-10: CONSISTENTLY predicts A (correct) with 80-100% confidence
- **This is true legibility**: Early correct + sustained stability

### Q6: "What is legibility metric for us?"
**Current definition is FLAWED because it ignores final accuracy.**

**Proposed New Legibility Definition:**

```
Legibility Time = First time when:
1. VLM predicts correct goal (choice == ground_truth)
2. Confidence > 60%
3. Prediction remains correct for next N seconds (e.g., N=3)
```

**With this definition**:
- cfg00_right_arc00: NOT legible (predicts B at t=6 but flips to A at t=7)
- cfg00_left_arc01: Legible at t=1 (predicts A consistently t=1-10)

### Q7: "Should we use change of pA/pB over time?"
**A: YES! Temporal dynamics matter more than single-frame confidence.**

**Legibility signals to track:**

1. **Confidence trajectory**: 
   - Rising confidence: `0.50 → 0.60 → 0.80 → 0.95` = gaining certainty ✅
   - Sudden jump: `0.50 → 0.99` = suspicious (might flip back) ⚠️

2. **Probability divergence**:
   - pA and pB separating: `(0.5, 0.5) → (0.7, 0.3) → (0.9, 0.1)` = clear preference ✅
   - pA and pB crossing: `(0.3, 0.7) → (0.7, 0.3)` = confusion, not legible ❌

3. **Prediction stability**:
   - Same choice for T consecutive seconds (e.g., T=3) ✅
   - Flipping between A/B = not legible yet ❌

**Recommended approach**:
```python
def is_legible(predictions, window=3):
    """
    Check if trajectory is legible based on temporal stability.
    
    Args:
        predictions: List of (time, pA, pB, choice) tuples
        window: Number of consecutive seconds for stability
    
    Returns:
        (is_legible, legible_time, confidence)
    """
    for i in range(len(predictions) - window + 1):
        window_preds = predictions[i:i+window]
        choices = [p[3] for p in window_preds]
        
        # Check stability: same choice for window duration
        if len(set(choices)) == 1 and choices[0] != 'C':
            # Check confidence is rising/stable
            confidences = [max(p[1], p[2]) for p in window_preds]
            if all(c >= 0.6 for c in confidences):
                return True, predictions[i][0], confidences[-1]
    
    return False, None, None
```

## The Left Bias Problem

**Statistical evidence of VLM LEFT bias:**

| Arc Side | Videos | Final Correct | Final Accuracy |
|----------|--------|---------------|----------------|
| **Left (A)** | 20 | 10/20 | **50%** |
| **Right (B)** | 20 | 0/20 | **0%** |

**Chi-square test**: This is NOT random chance (p < 0.001)

The VLM has learned a preference/bias toward predicting "left block" (A) when uncertain about spatial identity.

## Implications for Steering

### ❌ DO NOT do this:
1. ❌ Show VLM two candidate trajectories and ask "which is more legible?"
   - VLM will pick the one that **ends with gripper over LEFT block** (regardless of actual legibility)
   - This is selection bias, not legibility assessment

2. ❌ Use VLM confidence to select trajectories
   - VLM is 99-100% confident when WRONG on right arcs
   - Confidence is not calibrated

3. ❌ Use "time to legible" as a quality metric
   - Right arcs become "legible" early (t=2-6) but predictions are WRONG
   - Fast legibility ≠ correct legibility

### ✅ DO this instead:

1. ✅ **Fix the VLM's spatial grounding first**
   
   Add spatial reference to prompts:
   ```
   "The LEFT block is the block on the LEFT side of the image.
    The RIGHT block is the block on the RIGHT side of the image.
    Ignore any rotation or perspective effects.
    Focus on the XY position in the image frame."
   ```

2. ✅ **Use VLM for RELATIVE comparisons only**
   
   Don't ask: "Is trajectory A legible?" (VLM can't assess this accurately)
   
   Ask: "Which trajectory shows CLEARER directional motion toward its target?"
   - Compare motion trajectories, not final positions
   - VLM is better at detecting MOTION than SPATIAL IDENTITY

3. ✅ **Validate VLM with left-right symmetry tests**
   
   Before trusting VLM for steering:
   - Show 10 left-pick videos → VLM should predict A consistently
   - Show 10 MIRRORED left-pick videos → VLM should predict B consistently
   - If VLM fails mirroring test, its spatial reasoning is broken

4. ✅ **Use multi-stage verification**
   
   ```python
   # Stage 1: Generate N candidate trajectories
   candidates = diffusion_policy.sample(obs, N=10)
   
   # Stage 2: VLM scores motion clarity (0-1s, before spatial confusion)
   early_scores = vlm.score_motion_clarity(candidates, t=0to1)
   
   # Stage 3: Keep top-K based on motion clarity
   top_k = select_top_k(candidates, early_scores, K=3)
   
   # Stage 4: Execute and let ACTUAL outcome determine success
   # (Don't trust VLM's late-stage spatial predictions!)
   ```

5. ✅ **Learn from left-only data first**
   
   Since VLM is 50% accurate on left arcs vs 0% on right:
   - Train/validate steering strategy using LEFT-pick scenarios only
   - Test if learned strategy generalizes to right-pick (might need data augmentation)
   - Consider training separate policies for left vs right (or use goal-conditioning)

## Recommended Next Steps

### Immediate (Debugging):
1. **Visualize frames at t=6-7 for right arcs** - see what VLM sees when it flips
2. **Check if left/right blocks are visually identical** - might need markers/colors
3. **Test VLM with explicit left/right labels in prompt** - does verbal cue help?

### Short-term (Fix VLM grounding):
1. **Add spatial reference frame** - e.g., "left block is at X=0.1, right at X=0.3"
2. **Provide multi-view inputs** - side view + top view to disambiguate depth
3. **Use post-processing** - flip predictions if confidence<70% and arc detected

### Long-term (Steering strategy):
1. **Don't use VLM for absolute judgments** - only relative comparisons
2. **Focus on motion legibility (t=0-3s)** - ignore late-stage spatial predictions
3. **Validate with human evaluation** - check if VLM "legibility" matches human perception
4. **Consider training VLM adapter** - fine-tune on YOUR specific camera/scene setup

---

**Bottom line**: The VLM has a **fundamental left-right confusion problem**. You cannot use it for steering until this is fixed, or you design around this limitation by:
- Using VLM only for early-stage motion assessment (t=0-3s)
- Ignoring late-stage predictions (t=7-10s) that confuse block identity
- Validating with left-only examples before trusting right-side predictions
