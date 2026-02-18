# Compositional Split Strategy for Multimodal Diffusion Policy

## Executive Summary: ICRA-Style Rigorous Evaluation Framework

**Problem:** Current evaluation uses random seeds without systematic generalization testing.

**Solution:** Compositional train/test split that tests:
1. **Scene generalization**: New cube configurations (unseen placements)
2. **Trajectory generalization**: New arc variations (unseen approach styles)
3. **Compositional generalization**: BOTH new scenes AND new trajectories

---

## 1. Demo Structure Analysis

### Compositional Organization (10 × 20 × 2 = 400)

```
10 block configs × 20 arc variations × 2 modes (left/right) = 400 demos

Config 0: [20 left demos (arc 0-19)] + [20 right demos (arc 0-19)] = 40
Config 1: [20 left demos (arc 0-19)] + [20 right demos (arc 0-19)] = 40
...
Config 9: [20 left demos (arc 0-19)] + [20 right demos (arc 0-19)] = 40
```

**Verified:**
- ✅ All 10 configs perfectly balanced (20L + 20R each)
- ✅ All 400 episodes accounted for
- ✅ Uniform structure across all configs

---

## 2. Proposed Splitting Strategy: COMPOSITIONAL-HOLDOUT

### Rationale

**Why not random split?**
- Random split tests IID generalization only
- Doesn't test if policy can handle NEW scenes or NEW trajectories
- Real deployment requires both types of generalization

**Why compositional split?**
- Tests TWO types of generalization independently AND jointly
- Enables granular failure analysis
- Matches robotics deployment scenarios

### Split Design

| Split | Configs | Arcs | Modes | Size | Purpose |
|-------|---------|------|-------|------|---------|
| **Train** | 0-6 (7) | 0-15 (16) | L+R | 224 | Training set |
| **Validation** | 7 (1) | 0-15 (16) | L+R | 32 | Early stopping |
| **Test-trajectory** | 0-7 (8) | 16-19 (4) | L+R | 64 | New arcs, seen configs |
| **Test-scene** | 8-9 (2) | 0-15 (16) | L+R | 64 | New configs, seen arcs |
| **Test-full** | 8-9 (2) | 16-19 (4) | L+R | 16 | BOTH new |

**Total:** 224 train + 32 val + 64 traj + 64 scene + 16 full = 400 ✓

---

## 3. Evaluation Protocol

### Training Procedure

1. **Train on 224 episodes** (configs 0-6, arcs 0-15, both modes)
2. **Validate on 32 episodes** (config 7, arcs 0-15, both modes)
3. **Early stopping** based on validation success rate
4. **Save best checkpoint** based on validation performance

### Evaluation Procedure

For each test split, run K×M evaluation:
- **K env seeds** mapped to configs in that split
- **M sample seeds** per env seed (different diffusion noise)
- Record: success rate, mode balance, multimodal seeds

**Test splits:**

1. **Validation** (sanity check)
   - Configs: 7
   - Arcs: 0-15 (SEEN)
   - Expected: Highest success rate (~40-60%)
   - Purpose: Verify training worked

2. **Test-trajectory** (trajectory generalization)
   - Configs: 0-7 (SEEN)
   - Arcs: 16-19 (NEW)
   - Expected: Moderate success (~30-50%)
   - Tests: Can policy generalize to slightly different approach curves?

3. **Test-scene** (scene generalization)
   - Configs: 8-9 (NEW)
   - Arcs: 0-15 (SEEN)
   - Expected: Lower success (~20-40%)
   - Tests: Can policy handle new cube placements?

4. **Test-full** (compositional generalization)
   - Configs: 8-9 (NEW)
   - Arcs: 16-19 (NEW)
   - Expected: Lowest success (~10-30%)
   - Tests: Can policy handle BOTH new scenes AND new trajectories?

---

## 4. Generalization Gap Metrics

### Definition

**Generalization gap** = (Validation performance) - (Test performance)

### Metrics to Compute

1. **Trajectory gap** = Val - Test-trajectory
   - Measures difficulty of generalizing to new approach styles
   - Small gap (<10%) = good trajectory generalization

2. **Scene gap** = Val - Test-scene
   - Measures difficulty of generalizing to new cube placements
   - Small gap (<15%) = good scene generalization

3. **Full gap** = Val - Test-full
   - Measures difficulty of full compositional generalization
   - Expected to be larger (20-30%)

4. **Composition gap** = Avg(Test-traj, Test-scene) - Test-full
   - Measures EMERGENT difficulty when both types of generalization required
   - Positive value = composition is harder than individual components
   - Large gap (>20%) suggests need for compositional inductive biases

---

## 5. Statistical Analysis

### Per-Split Metrics

For each split, compute:
- Success rate (with 95% confidence interval via bootstrap)
- Mode balance score (0.0-1.0)
- Number of multimodal seeds (both L+R observed)
- Entropy of mode distribution per seed

### Comparison Tests

- **Paired t-test**: Val vs Test-trajectory, Val vs Test-scene
- **ANOVA**: Differences across all 4 test sets
- **Effect size**: Cohen's d for generalization gaps

### Success Criteria

**Minimum acceptable:**
- Val success: >40%
- Test-trajectory: >30%
- Test-scene: >25%
- Test-full: >15%

**Good performance:**
- Val success: >50%
- Test-trajectory: >40%
- Test-scene: >35%
- Test-full: >25%

**Excellent performance:**
- Val success: >60%
- Test-trajectory: >50%
- Test-scene: >45%
- Test-full: >35%

---

## 6. Implementation

### Files Created

1. **`scripts/analyze_demo_structure.py`**
   - Analyzes compositional structure
   - Proposes multiple splitting strategies
   - Generates `data/demos/splits_compositional.json`

2. **`scripts/train_with_splits.py`**
   - Filters demos by split indices
   - Creates train/val splits
   - Saves split metadata

3. **`scripts/eval_compositional.py`**
   - Runs evaluation on all test splits
   - Computes generalization gaps
   - Generates comprehensive report

4. **`data/demos/splits_compositional.json`**
   - Contains episode indices for each split
   - Generated by `analyze_demo_structure.py`

### Usage

**Step 1: Generate splits**
```bash
python scripts/analyze_demo_structure.py --path data/demos/demos.npz
# Creates data/demos/splits_compositional.json
```

**Step 2: Prepare filtered datasets**
```bash
python scripts/train_with_splits.py \
    --split_file data/demos/splits_compositional.json \
    --val_config 7 \
    --output_dir runs/compositional_split
# Creates split_info.json with train/val/test indices
```

**Step 3: Train with splits**
```bash
# Modify train_diffusion_policy.py to load only train indices
# Or manually filter demos.npz to create train_demos.npz and val_demos.npz

python scripts/train_diffusion_policy.py \
    --demos data/demos/train_demos.npz \
    --val_demos data/demos/val_demos.npz \
    --output_dir runs/compositional_model
```

**Step 4: Evaluate on all splits**
```bash
# Test-trajectory
python scripts/eval_multimodality.py \
    --ckpt runs/compositional_model/ckpt_best.pt \
    --K 8 --M 10 --execute_steps 8 \
    --out_dir outputs/eval_test_trajectory

# Test-scene  
python scripts/eval_multimodality.py \
    --ckpt runs/compositional_model/ckpt_best.pt \
    --K 2 --M 32 --execute_steps 8 \
    --out_dir outputs/eval_test_scene

# Test-full
python scripts/eval_multimodality.py \
    --ckpt runs/compositional_model/ckpt_best.pt \
    --K 2 --M 8 --execute_steps 8 \
    --out_dir outputs/eval_test_full
```

**Step 5: Generate report**
```bash
python scripts/eval_compositional.py \
    --ckpt runs/compositional_model/ckpt_best.pt \
    --split_file data/demos/splits_compositional.json \
    --output_dir outputs/compositional_report
```

---

## 7. Expected Outcomes

### Hypothesis 1: Trajectory Generalization (Easy)

**Claim:** Policy should generalize well to new arc variations (arcs 16-19) on seen configs (0-7).

**Reasoning:** 
- Arc variations differ only in control point magnitude
- All arcs share the same Bézier structure
- Policy has seen 16 arc types → interpolation/extrapolation to 4 new types is feasible

**Expected:** Trajectory gap < 10%

### Hypothesis 2: Scene Generalization (Harder)

**Claim:** Policy will struggle more with new cube configs (8-9) even with seen arcs (0-15).

**Reasoning:**
- Cube positions affect observation space directly
- Small positional shifts can create OOD observations
- Only 7 configs seen during training → 2 new configs is significant

**Expected:** Scene gap = 15-20%

### Hypothesis 3: Compositional Difficulty (Hardest)

**Claim:** Combining BOTH new configs AND new arcs (Test-full) will be significantly harder than either alone.

**Reasoning:**
- Composition of two sources of distribution shift
- Policy must generalize on two dimensions simultaneously
- Fewer overlapping support regions in (config, arc) space

**Expected:** Full gap = 25-30%, Composition gap = 10-15%

---

## 8. Failure Analysis

### If ALL test sets fail (<15% success)

**Diagnosis:** Model hasn't learned the task at all
**Solution:**
- Check training loss convergence
- Verify demo quality
- Increase model capacity
- Try DDIM sampling

### If only Test-full fails (<15%)

**Diagnosis:** Compositional generalization failure
**Solution:**
- Add data augmentation (jitter both cube positions AND trajectories during training)
- Increase training data diversity
- Add compositional priors (e.g., modular architecture)

### If Test-scene fails but Test-trajectory succeeds

**Diagnosis:** Cube position is overfitted
**Solution:**
- Increase cube jitter during training
- Add more block configurations to training set
- Use observation dropout or masking

### If Test-trajectory fails but Test-scene succeeds

**Diagnosis:** Trajectory overfitting (unlikely given 16 arcs)
**Solution:**
- Add trajectory noise augmentation
- Simplify arc variations (use fewer, more distinct arcs)

---

## 9. Comparison to Current Approach

| Aspect | Current Approach | Compositional Split |
|--------|-----------------|---------------------|
| **Train/Test split** | No split (all 400 used) | 224 train / 144 test |
| **Generalization test** | None (eval on random seeds) | Scene + Trajectory + Compositional |
| **Validation** | Random episode split | 1 held-out config |
| **Eval protocol** | K=10, M=20 on random seeds | 4 test sets with specific purposes |
| **Metrics** | Success rate, entropy | + Generalization gaps |
| **Rigor** | Low (no OOD test) | High (ICRA-standard) |

---

## 10. Advantages of This Approach

### Scientific Rigor

- **Controls for distribution shift**: Tests specific types of generalization
- **Reproducible**: Fixed splits, not random seeds each time
- **Interpretable**: Can pinpoint failure modes (scene vs trajectory)

### Practical Value

- **Matches deployment**: Real robots encounter new scenes AND new trajectories
- **Informs design**: Know which type of generalization to focus on
- **Enables ablations**: Test data augmentation strategies per generalization type

### Publication Ready

- ICRA/RSS/CoRL standard evaluation
- Enables comparison with future work (fixed splits)
- Demonstrates thorough experimental design

---

## 11. Alternative Strategies Considered

### Strategy 1: Config-Holdout Only

**Pro:** Simpler, more training data (320 episodes)
**Con:** Doesn't test trajectory generalization

### Strategy 2: Arc-Holdout Only

**Pro:** Tests trajectory interpolation/extrapolation
**Con:** Doesn't test scene generalization

### Strategy 3: Random Stratified

**Pro:** Standard ML baseline
**Con:** Only tests IID generalization, not OOD

### Strategy 4: K-Fold Cross-Validation

**Pro:** Averages over multiple config splits
**Con:** Computationally expensive (5× training), less interpretable

**Selected:** Compositional-Holdout (Strategy from Section 2)
**Reason:** Tests BOTH types of generalization with most interpretable setup

---

## 12. Integration with Existing 20-Arc Design

### User's Constraint: Keep All 20 Arcs

**Reasoning:** Want to demonstrate policy can learn diverse left/right approaches

**Our Approach:** Still use all 20 arcs, but split 16 train / 4 test

**Benefits:**
- Training set still has high diversity (16 arcs × 7 configs = 112 unique trajectories per mode)
- Test set verifies generalization to slightly different arcs
- Maintains user's intent of diverse trajectory learning

**Alternative (if want more training data):**
- Use all 20 arcs in training, test on interpolated/extrapolated arcs via data augmentation
- But this doesn't test true OOD generalization

---

## 13. Next Steps

### Immediate (Before Retraining)

1. ✅ **Run structure analysis** → `analyze_demo_structure.py` (DONE)
2. ✅ **Generate split indices** → `splits_compositional.json` (DONE)
3. ✅ **Verify balance** → All configs 50/50 ✓ (DONE)

### Short-Term (Training)

4. **Filter demos by split indices** → Create `train_demos.npz`, `val_demos.npz`
5. **Train with splits** → Use filtered datasets, early stopping on val
6. **Track validation metrics** → Success rate, mode balance per epoch

### Medium-Term (Evaluation)

7. **Eval on validation** → Sanity check (should be highest success)
8. **Eval on Test-trajectory** → Measure trajectory generalization gap
9. **Eval on Test-scene** → Measure scene generalization gap
10. **Eval on Test-full** → Measure compositional generalization gap

### Long-Term (Analysis)

11. **Compute generalization gaps** → Quantify each type of generalization
12. **Statistical significance** → Bootstrap CIs, t-tests
13. **Generate ICRA-style report** → Comprehensive writeup
14. **Identify failure modes** → Which test set is hardest? Why?

---

## 14. Summary: Why This Approach is Rigorous

### 1. Systematic OOD Testing
- Not just "does it work?" but "where does it fail?"
- Tests interpolation (arcs 16-19) and extrapolation (configs 8-9)

### 2. Interpretable Failures
- If Test-scene fails: observation space overfitting
- If Test-trajectory fails: trajectory space overfitting
- If Test-full fails worse than expected: compositional brittleness

### 3. Actionable Insights
- Each failure mode suggests specific fixes
- Can prioritize improvements based on gap magnitudes

### 4. Publication Quality
- Matches top-tier robotics conference standards
- Reproducible (fixed splits, not random seeds)
- Enables fair comparison with future work

### 5. Aligns with Deployment
- Real robots must handle new scenes and new task variations
- This split directly measures those capabilities

---

**Status:** Framework implemented, ready for training and evaluation.

**Files:**
- ✅ `scripts/analyze_demo_structure.py`
- ✅ `scripts/train_with_splits.py`
- ✅ `scripts/eval_compositional.py`
- ✅ `data/demos/splits_compositional.json`

**Next:** Train model with compositional splits and evaluate on all 4 test sets.
