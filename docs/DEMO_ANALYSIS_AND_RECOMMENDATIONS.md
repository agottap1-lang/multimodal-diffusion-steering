# Demo Quality Analysis & Training Recommendations

## Executive Summary

Based on comprehensive demo inspection and analysis, this document addresses:
1. Demo quality assessment
2. Classification metrics for continuous action prediction
3. Multimodal math explanation (how L/R are computed)
4. Demo balance verification results
5. Learning difficulty from high variation
6. Data splitting strategy recommendations

---

## 1. Demo Quality Assessment Results

### ✅ DEMO QUALITY: EXCELLENT (with minor caveats)

**Key Findings from Enhanced Inspection:**

| Metric | Result | Status |
|--------|--------|--------|
| Early commitment clarity | 0.058 separation | ✅ GOOD |
| Trajectory smoothness | 0.000995 avg delta | ✅ EXCELLENT |
| Config-level balance | 10/10 configs at 50/50 | ✅ PERFECT |
| Mode separability | 4.97 score | ✅ EXCELLENT |
| Action outliers | 1284 (1.05% of dy) | ⚠️ MINOR |

**What Makes These Demos GOOD for Multimodal Learning:**

1. **Clear Early Commitment:**
   - Left demos: mean_dy = +0.029 (positive lateral motion)
   - Right demos: mean_dy = -0.029 (negative lateral motion)
   - Mode separation of 0.058 is well above threshold (>0.015)
   - Robot clearly "commits" to left or right within first 20 steps

2. **Smooth Trajectories:**
   - Mean action delta 0.001 (very smooth)
   - No jittery/oscillatory behavior
   - Bézier curves create natural, legible motions

3. **Excellent Mode Separability:**
   - Separability score 4.97 → modes are clearly distinguishable
   - Between-mode distance >> within-mode variance
   - Policy can learn the two modes are different

4. **Perfect Balance:**
   - Every configuration: exactly 20 left + 20 right
   - Overall: 200L + 200R (50/50)
   - No systematic bias that could confuse learning

**Minor Issue: 1284 dy outliers (1.05%)**
- These are expected due to 20 different arc variations
- Outliers represent the most extreme arc sweeps
- NOT a quality problem - this is intentional diversity

---

## 2. Classification Metrics for Continuous Action Prediction

### Why Traditional Metrics Don't Apply

For continuous action prediction, we cannot compute traditional precision/recall/F1 **at the action level** because:
- Actions are continuous vectors (5-d: dx, dy, dz, dyaw, grip)
- No discrete "classes" to predict at each timestep
- MSE/MAE are the standard metrics for continuous outputs

### What We CAN Measure: Outcome Classification

**Convert continuous actions → discrete task outcomes:**
- `left_success`: Robot picked left cube
- `right_success`: Robot picked right cube
- `failure`: Robot failed to pick either cube

**Classification Metrics Implemented:**

1. **Success Rate** (binary: success vs failure)
   - `precision_success = (n_left + n_right) / total`
   - `precision_failure = n_failures / total`

2. **Mode Balance** (within successes)
   - `balance = 1.0 - 2 * |0.5 - (n_left / n_success)|`
   - 1.0 = perfect 50/50 split
   - 0.0 = total collapse to one mode

3. **Mode Consistency** (per environment seed)
   - Count seeds with both left AND right outcomes → "multimodal seeds"
   - Count seeds with only one mode → "collapsed seeds"
   - Ideal: >50% of seeds are multimodal

4. **Confusion Matrix**
   - Not traditional (no ground truth "intended" mode)
   - Instead: distribution of {left, right, failure} outcomes

### Implementation: `compute_classification_metrics.py`

Usage:
```bash
python scripts/compute_classification_metrics.py --results outputs/test/results.csv
```

Outputs:
- Outcome distribution (left/right/failure counts)
- Per-class metrics (success/failure rates)
- Mode balance score (0.0-1.0)
- Per-seed consistency analysis

---

## 3. Multimodal Math Explanation: How L and R Are Computed

### Question: "In multimodal math how is L or R obtained?"

**Answer: L and R come from the ENVIRONMENT, not the model.**

### Step-by-Step Process:

1. **During Rollout:** Policy generates actions → robot executes → cube is lifted

2. **Success Detection:** Environment checks which cube crossed z-threshold (0.52m)
   ```python
   # From envs/twoblockpick_env.py lines 415-433
   def _check_success(self):
       lz = left_cube_z_position
       rz = right_cube_z_position
       
       if lz > SUCCESS_Z and not picked_left and not picked_right:
           self._picked_left = True  # LEFT was lifted FIRST
       if rz > SUCCESS_Z and not picked_right and not picked_left:
           self._picked_right = True  # RIGHT was lifted FIRST
   ```

3. **Outcome Assignment:**
   - If `picked_left == True`: outcome = "left_success"
   - If `picked_right == True`: outcome = "right_success"
   - If neither: outcome = "failure"

4. **Aggregation Per Seed:** For K×M evaluation:
   ```
   env_seed = 100 (fixes cube placement)
   ├─ sample_seed = 0 → outcome = "left_success"   ┐
   ├─ sample_seed = 1 → outcome = "failure"        │
   ├─ sample_seed = 2 → outcome = "left_success"   ├─ Count: L=3, R=1, F=6
   ├─ sample_seed = 3 → outcome = "right_success"  │
   ├─ ...                                          │
   └─ sample_seed = 9 → outcome = "failure"        ┘
   
   Then: L = 3, R = 1 for seed 100
   ```

5. **Entropy Computation:**
   ```python
   # From eval_multimodality.py lines 880-916
   successes = L + R
   if successes >= 5:
       p_left = L / successes
       p_right = R / successes
       entropy = -p_left * log2(p_left) - p_right * log2(p_right)
   ```

### Key Point: NO LABELS DURING INFERENCE

The policy receives:
- ✅ Observation (22-d: ee pose + cube poses)
- ❌ NO goal label ("pick left" or "pick right")
- ❌ NO mode conditioning

The policy MUST learn to express both modes purely through stochastic diffusion noise.

---

## 4. Demo Balance Verification: ✅ PASSED

### Results from `inspect_demos_enhanced.py`:

**Overall Balance:**
- Left demos: 200 (50.0%)
- Right demos: 200 (50.0%)
- Perfect 50/50 split ✅

**Config-Level Balance (10 configurations):**

| Config ID | Left | Right | Total | Balance |
|-----------|------|-------|-------|---------|
| 0 | 20 | 20 | 40 | 0.50 ✓ |
| 1 | 20 | 20 | 40 | 0.50 ✓ |
| 2 | 20 | 20 | 40 | 0.50 ✓ |
| 3 | 20 | 20 | 40 | 0.50 ✓ |
| 4 | 20 | 20 | 40 | 0.50 ✓ |
| 5 | 20 | 20 | 40 | 0.50 ✓ |
| 6 | 20 | 20 | 40 | 0.50 ✓ |
| 7 | 20 | 20 | 40 | 0.50 ✓ |
| 8 | 20 | 20 | 40 | 0.50 ✓ |
| 9 | 20 | 20 | 40 | 0.50 ✓ |

**Arc Variation Balance:**
- 20 arc variations per mode
- Each arc variation appears in:
  - 10 left-pick demos (one per config)
  - 10 right-pick demos (one per config)
- Perfectly balanced across all variations ✅

**Conclusion:** Demo balance is PERFECT at all levels (overall, config-level, variation-level).

---

## 5. Is High Variation Difficult to Learn?

### Current Variation:

**Total Unique Trajectories per Mode:**
- 10 block configs × 20 arc variations = **200 unique trajectories per mode**
- Total dataset: 400 demos (200L + 200R)

**Variation Sources:**

1. **Block Position Configs (10 types):**
   - Both shifted: 4 configs
   - Left only shifted: 3 configs
   - Right only shifted: 3 configs
   - Offsets: ±5mm in x/y

2. **Arc Variations (20 types):**
   - Lateral sweep: 0.05m → 0.28m
   - Control point height: 0.56m → 0.68m
   - Control point x: 0.38m → 0.28m

### Analysis: Is This Too Much Variation?

**PROS (Good for Robustness):**
- ✅ Policy learns generalizable features (not overfitting to one trajectory)
- ✅ Robust to cube placement variation (±5mm)
- ✅ Multiple approach strategies → more flexible policy
- ✅ Real-world applicability (demos aren't all identical)

**CONS (Harder to Learn):**
- ⚠️ 200 unique trajectories per mode → sparse coverage of trajectory space
- ⚠️ Model must generalize across 20 arc types + 10 configs
- ⚠️ Each unique trajectory has only 1 demonstration
- ⚠️ May explain the 13% success rate (model confused by diversity)

### Evidence from Current Results:

1. **Training Loss: 0.003 (excellent)** → Model CAN fit the data
2. **Success Rate: 13%** → Model CANNOT execute reliably
3. **Failure Mode: Oscillation** → Model uncertain about which trajectory to follow

**Hypothesis:** The variation is TOO HIGH for the model capacity (1.1M params).

### Recommendation: **REDUCE VARIATION (smallest change)**

**Option A: Reduce Arc Variations (20 → 5)**
- Keep all 10 block configs (robustness to placement)
- Use only 5 arc variations instead of 20
- New dataset: 10 configs × 5 arcs = 50 unique trajectories per mode
- Each trajectory appears 4 times (20 left + 20 right per config, 5 arcs → 4 repeats)

**Option B: Reduce Block Configs (10 → 3)**
- Keep all 20 arc variations (diversity in approach)
- Use only 3 configs: one "both", one "left", one "right"
- New dataset: 3 configs × 20 arcs = 60 unique trajectories per mode

**Recommendation: Choose Option A (reduce arcs to 5)**

**Reasoning:**
1. Block position robustness is critical (eval uses cube_jitter)
2. Arc variations add trajectory diversity but 20 is excessive
3. Reducing 20→5 arcs gives 4× more samples per trajectory
4. Model can learn mode structure better with repeated examples

**Suggested Arc Variations (5 total):**
- Arc 1: cp_y = ±0.05m (gentlest)
- Arc 2: cp_y = ±0.12m
- Arc 3: cp_y = ±0.18m (medium)
- Arc 4: cp_y = ±0.24m
- Arc 5: cp_y = ±0.28m (most extreme)

**Expected Improvement:**
- Before: 1 demo per unique trajectory
- After: 4 demos per unique trajectory
- Better gradient signal for learning
- Less confusion during inference

---

## 6. Data Splitting Strategy for Training/Testing

### Current Approach: NO TEST SPLIT

**Problem:**
- All 400 demos used for training
- `test_demos.npz` exists but unclear if it's a separate held-out set
- No way to verify generalization to unseen configurations

### Recommended Strategy: **CONFIG-BASED SPLIT**

**Why Config-Based (Not Random Episode Split)?**

1. **Tests Generalization:** Hold out entire configs → tests if policy can handle novel cube placements
2. **Prevents Data Leakage:** Random split means test configs also in train → overestimates performance
3. **Matches Eval Protocol:** Eval uses env_seed (cube placement) → should test on unseen placements

### Proposed Split:

**Training Set: 8 configs (320 demos)**
- Config 0, 1, 2, 3, 4, 5, 7, 8
- Mix of "both", "left", "right" shift types
- 8 configs × 40 demos = 320 demos (160L + 160R)

**Test Set: 2 configs (80 demos)**
- Config 6, 9
- Held-out configs for generalization testing
- 2 configs × 40 demos = 80 demos (40L + 40R)

**Advantage:**
- Test set has SAME arc variations (20 types)
- Test set has DIFFERENT cube placements
- Tests if policy learned "pick a cube" vs "replay memorized trajectory"

### Alternative: **ARC-BASED SPLIT** (if reducing arcs to 5)

**Training Arcs: 0, 1, 2, 3 (4/5)**
**Test Arcs: 4 (1/5)**

- All 10 configs in both train and test
- Test if policy generalizes to slightly different approach curvatures
- Less critical than config-based split

### Implementation Steps:

1. **Modify `collect_demos_twoblockpick.py`:**
   - Add `--train_configs` flag: `--train_configs 0,1,2,3,4,5,7,8`
   - Add `--test_configs` flag: `--test_configs 6,9`
   - Save separate `train_demos.npz` and `test_demos.npz`

2. **Modify `train_diffusion_policy.py`:**
   - Load only `train_demos.npz` for training
   - Add validation split within training configs (e.g., config 7 for val)

3. **Modify Eval:**
   - Eval on both train configs (seen) and test configs (unseen)
   - Compare success rates: seen vs unseen
   - Good generalization: unseen ~80% of seen performance

### Validation Split (Within Training):

Even within 8 training configs, use episode-level split:
- Example: Use first 30 episodes of each config for train (240 demos)
- Use last 10 episodes of each config for val (80 demos)
- Prevents validation leakage while keeping config diversity

---

## 7. Final Recommendations: Path to Success

### Immediate Actions (Before Retraining):

1. ✅ **Demo Quality Check:** PASSED (already excellent)

2. ⚠️ **Reduce Arc Variations:** 20 → 5 (re-collect demos)
   - Keeps 10 block configs (robustness)
   - Gives 4× more samples per unique trajectory
   - Expected: better learning, less confusion

3. ✅ **Implement Config-Based Split:**
   - Train: configs 0-5, 7-8 (320 demos)
   - Test: configs 6, 9 (80 demos)
   - Validation: Last 25% of train demos (80 demos)

4. ✅ **Add Classification Metrics to Eval:**
   - Already implemented: `compute_classification_metrics.py`
   - Run after each eval to track mode balance

### Training Configuration Changes:

| Parameter | Current | Recommended | Reasoning |
|-----------|---------|-------------|-----------|
| `smooth_weight` | 0.01 | 0.05 | Stronger oscillation suppression |
| `execute_steps` | 8 | 8 (keep) | Already fixed to 8 (good) |
| `horizon` | 48 | 48 (keep) | Long-term planning is good |
| `epochs` | 400 | 500 | More epochs with reduced variation |
| Dataset | 400 (all) | 320 (train split) | Proper generalization test |

### Expected Outcomes After Changes:

**With Reduced Variation (5 arcs) + Config Split:**
- Training loss: Similar (0.003-0.005)
- **Success rate on TRAIN configs: 40-60%** (up from 13%)
- **Success rate on TEST configs: 30-50%** (unseen, should be lower)
- **Mode balance: 0.6-0.8** (up from 0.0)
- **Multimodal seeds: 3-6 out of 10** (up from 1/10)

### Long-Term Improvements (If Success Rate Still Low):

1. **Increase Model Capacity:** 1.1M → 5M params (deeper ResBlocks)
2. **DDIM Sampling:** Deterministic (but may reduce multimodality)
3. **Explicit Mode Conditioning:** Add left/right label during training, remove at inference
4. **Different Architecture:** Transformer instead of MLP (better sequence modeling)

---

## 8. Summary: Answers to Your Questions

### Q1: How to inspect demos and identify failure points?
**A:** Created `inspect_demos_enhanced.py` with 6 quality checks. Current demos are EXCELLENT (minor outliers expected from variation).

### Q2: Classification metrics for continuous actions?
**A:** Implemented `compute_classification_metrics.py`. Measures outcome classification (left/right/failure), mode balance, and per-seed consistency.

### Q3: How is L or R obtained in multimodal math?
**A:** L and R are **ENVIRONMENT outputs** (which cube was lifted first), NOT model predictions. Counted per env_seed across M rollouts with different noise.

### Q4: Demo balance verified?
**A:** ✅ PERFECT. 200L + 200R overall, 20L + 20R per config (all 10 configs).

### Q5: Is variation difficult to learn?
**A:** YES. 200 unique trajectories per mode with only 1 example each is TOO SPARSE. **Recommendation: Reduce arcs 20→5** for 4× more samples per trajectory.

### Q6: Data splitting strategy?
**A:** **Config-based split:** Train on 8 configs, test on 2 held-out configs. Tests generalization to new cube placements, matches eval protocol.

### Q7: Plan before large eval?
**A:** 
1. Re-collect demos with 5 arc variations (not 20)
2. Implement config-based train/test split
3. Train with smooth_weight=0.05, epochs=500
4. Run small eval (K=5, M=5) on both train and test configs
5. If success >40% on train, >30% on test → proceed to large eval
6. Otherwise, increase model capacity or try DDIM

---

**Next Step:** Re-collect demonstration dataset with 5 arc variations instead of 20, then retrain with config-based split.
