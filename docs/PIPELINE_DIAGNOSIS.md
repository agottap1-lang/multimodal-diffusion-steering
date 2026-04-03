# PIPELINE DIAGNOSIS & FIX PLAN
## Senior ML Engineer Review - February 14, 2026

---

## EXECUTIVE SUMMARY

**Critical Issues Found:**
1. ⚠️ **MULTIMODALITY IS IMPOSSIBLE TO TEST** - Using DDIM with eta=0 (deterministic)
2. ⚠️ **TRAIN/EVAL DISTRIBUTION MISMATCH** - Demo jitter=0, eval jitter=0.015
3. ⚠️ **GRIPPER TEMPORAL SMEARING** - Ensemble averages gripper commands
4. ⚠️ **COARSE REPLANNING NEAR TARGET** - execute_steps=8-16 too large for precision tasks

---

## ROOT CAUSE HYPOTHESES (RANKED BY IMPACT)

### 1. **MULTIMODALITY NOT TESTABLE (CRITICAL - BLOCKS GOAL #2)**

**Problem:**
- Goal: Prove multimodality (same env_seed → different sample_seed → Left vs Right)
- Current: DDIM with eta=0 is **DETERMINISTIC**
- Same obs + same model = same actions, period
- Different sample_seeds have NO EFFECT when eta=0

**Evidence:**
```python
# eval_multimodality.py line 71
sampling_method: str = 'ddim'  # Default

# train_diffusion_policy.py p_sample_ddim with eta=0
# No noise injection → deterministic
```

**Why this explains low multimodality:**
- For any env_seed, all M sample_seeds produce IDENTICAL trajectories
- You'll always pick the same cube (whichever the model biases toward)
- Entropy will be 0, collapse_seeds will be K/K

**Fix Priority:** **IMMEDIATE**

**Proposed Fix:**
```python
# Option A: Use DDPM (stochastic baseline)
--sampling_method ddpm  # Noise at every denoising step

# Option B: Use DDIM with eta > 0
--sampling_method ddim --ddim_eta 0.3  # Controlled stochasticity

# Option C: Temperature scaling (add later)
```

**Implementation:**
- Add `--ddim_eta` CLI arg (default 0.0)
- Pass eta through to `p_sample_ddim()`
- Add diagnostic: sample twice, measure L2 diff, warn if deterministic
- Document: "eta=0 → deterministic, cannot test multimodality"

---

### 2. **TRAIN/EVAL DISTRIBUTION SHIFT (HIGH IMPACT ON SUCCESS)**

**Problem:**
- Demos: cube_jitter=0 → cubes at EXACT positions (±5mm config offsets only)
- Eval: cube_jitter=0.015 → cubes at positions ± 15mm random jitter
- Policy trained on precise positions, tested on jittered positions
- **1.5cm shift is HUGE for precision grasping**

**Evidence:**
```python
# collect_demos_twoblockpick.py line 283
cube_jitter=0.0  # Exact positions

# twoblockpick_env.py default
_CUBE_JITTER = 0.015  # ±1.5cm

# eval uses default → mismatch!
```

**Why this explains low success:**
- Model learns to grasp at fixed (x=0.50, y=±0.07)
- Eval places cubes at (x=0.50±0.015, y=±0.07±0.015)
- Grasp misses by up to 2.1cm (diagonal)
- 4cm cube → 2.1cm error = 50% miss

**Expected Impact:** **+20-30% success if fixed**

**Proposed Fix:**
```python
# Option A: Train with augmented jitter (robust policy)
# Add noise to obs cube positions during training

# Option B: Eval with jitter=0 (match demos)
py scripts/eval_multimodality.py --cube_jitter 0.0

# Option C: Recollect demos with jitter=0.015
# Most realistic but expensive
```

**Recommendation:** **Option B immediately** (eval with jitter=0), then Option A for robustness.

---

### 3. **GRIPPER TEMPORAL SMEARING (MEDIUM-HIGH IMPACT)**

**Problem:**
- Temporal ensemble averages ALL action dims including gripper
- Gripper is DISCRETE: +1 (open) or -1 (closed)
- Averaging turns grasp into slow ramp: +1 → +0.5 → 0 → -0.5 → -1
- Delays grasp, object may slip

**Evidence:**
```python
# eval_multimodality.py lines 168-177
for j in range(overlap):
    blended[j] = (w_old * remaining[j] + blended[j]) / (1.0 + w_old)
# ALL dims averaged, including gripper (dim 4)
```

**Why this explains low success:**
- Demo: Sharp grip close over 40 steps (+1 → -1 linearly)
- Policy with ensemble: Gradual ramp, further delayed by averaging
- Object not grasped firmly → falls during lift

**Expected Impact:** **+5-10% success**

**Proposed Fix:**
```python
# Ensemble dx,dy,dz,dyaw (dims 0-3) only
# Use newest plan value for grip (dim 4)

if self.temporal_ensemble and self._pending_chunks:
    blended = chunk.copy()
    for j in range(overlap):
        blended[j, :4] = weighted_average  # pos/yaw only
        blended[j, 4] = chunk[j, 4]        # grip: newest plan
```

---

### 4. **COARSE REPLANNING NEAR TARGET (MEDIUM IMPACT)**

**Problem:**
- execute_steps=8-16 → replans every 0.4-0.8 seconds
- Open-loop execution for 8-16 steps
- Near cube (precision phase): ±0.05m/step × 16 steps = ±0.8m drift potential
- Demos use closed-loop scripted descent (30 waypoints, ~0.5cm steps)

**Why this explains low success:**
- Approach phase: execute_steps=16 OK (coarse movements)
- Descent phase: execute_steps=16 too coarse, drifts off target
- Demos don't have this open-loop execution

**Expected Impact:** **+10-15% success**

**Proposed Fix:**
```python
# Dynamic MPC: adjust execute_steps based on proximity
dist_to_cube = np.linalg.norm(ee_pos[:2] - target_cube_pos[:2])

if dist_to_cube > 0.15:      # Far: approach
    execute_steps = 16
elif dist_to_cube > 0.05:    # Near: descent
    execute_steps = 4
else:                        # Very near: precision grasp
    execute_steps = 1
```

---

### 5. **DDPM/DDIM SAMPLING INSTABILITY (ADDRESSED)**

**Status:** Already implemented DDIM, but eta=0 makes it deterministic.

**For multimodality:** Need eta > 0
**For success:** DDIM eta=0 should give smoother, more stable trajectories than DDPM

---

### 6. **OTHER POTENTIAL ISSUES (LOWER PRIORITY)**

**a) Action Scaling:**
- 0.05m/step × 16 steps = 0.8m cumulative
- Demos travel 0.42m total
- Max cumulative (if all +1) = 1.6m (4x demos)
- Likely OK, but could add action magnitude penalty

**b) Horizon Mismatch:**
- Trained: horizon=48
- Execute: 8-16 steps, replan
- Discards 32-40 predicted steps
- Standard MPC, likely OK

**c) Observation Normalization:**
- Validated: no critical issues
- Cube x/y positions floored at std=0.01
- This is intentional (low variance in demos) but reduces sensitivity

---

## PRIORITIZED FIX PLAN

### PHASE 1: MAKE MULTIMODALITY TESTABLE (30 min)
1. Add `--ddim_eta` parameter to eval
2. Add sampler determinism diagnostic
3. Document eta=0 caveat in help text
4. Test with eta=0.3 and eta=1.0

**Expected:** Multimodality observable with eta>0

---

### PHASE 2: FIX TRAIN/EVAL MISMATCH (15 min)
1. Add `--cube_jitter` to eval CLI
2. Save jitter metadata in demo .npz
3. Add warning when mismatch detected
4. Rerun eval with --cube_jitter 0.0

**Expected:** +20-30% success

---

### PHASE 3: FIX GRIPPER ENSEMBLE (20 min)
1. Add `--ensemble_grip` flag (default False)
2. Modify temporal ensemble to skip gripper when False
3. Rerun eval with --ensemble_grip False

**Expected:** +5-10% success

---

### PHASE 4: DYNAMIC MPC (45 min)
1. Add proximity-based execute_steps logic
2. Make thresholds configurable
3. Add `--dynamic_mpc` flag (default False)
4. Test on multiple seeds

**Expected:** +10-15% success

---

### PHASE 5: VLM SCAFFOLD (90 min)
1. Candidate generation loop
2. Scoring interface stub
3. Directory structure
4. Metadata JSON logging

**Expected:** Infrastructure for legibility optimization

---

### PHASE 6: TESTS (60 min)
1. test_ddim_determinism
2. test_ddim_stochasticity
3. test_jitter_warning
4. Smoke run mode (K=1, M=3)

---

## PREDICTED OUTCOMES

### After Phase 1 (Multimodality Testable):
- With eta=0.3: Expect 2-4 bimodal seeds (if base success >30%)
- With eta=1.0 (DDPM equivalent): More multimodality but lower success

### After Phase 2 (Jitter Fix):
- Success: 14% → 35-45%
- Multimodality: Still limited by low diversity in training

### After Phase 3 (Gripper Fix):
- Success: 35-45% → 45-55%
- Multimodality: Slightly improved (successful grasps on both sides)

### After Phase 4 (Dynamic MPC):
- Success: 45-55% → 60-70%
- Multimodality: Good (if success >50%)

### Overall Target Achievement:
- **Goal 1 (High Success):** 60-70% achievable with all fixes
- **Goal 2 (Multimodality):** Achievable with eta>0 + success >50%

---

## IMPLEMENTATION SEQUENCE

1. **Reasoning Document** ✓ (this file)
2. **Phase 1: Multimodality testability** (next)
3. **Phase 2: Jitter fix** (high ROI)
4. **Phase 3: Gripper ensemble** (quick win)
5. **Phase 4: Dynamic MPC** (precision improvement)
6. **Phase 5: VLM scaffold** (future extensibility)
7. **Phase 6: Tests** (validation)

---

## CODE CHANGE SUMMARY (PREVIEW)

### Files to Modify:
1. `scripts/eval_multimodality.py` - Add eta, jitter, ensemble_grip, dynamic MPC
2. `scripts/train_diffusion_policy.py` - Pass eta to DDIM sampler
3. `scripts/collect_demos_twoblockpick.py` - Save jitter metadata
4. `envs/twoblockpick_env.py` - Expose cube positions for proximity check
5. `tests/test_pipeline.py` - New test file

### New Files:
1. `scripts/test_multimodality_diagnostic.py` - Sampler stochasticity check
2. `scripts/eval_with_steering.py` - VLM scaffold (Phase 5)

---

## RISK ASSESSMENT

### Low Risk:
- Phases 1, 2, 3: Minor parameter changes, backward compatible
- Easy to rollback

### Medium Risk:
- Phase 4 (Dynamic MPC): Changes control logic, needs careful testing
- Could cause oscillation if thresholds wrong

### High Risk:
- None (all changes are additive with flags)

---

## SUCCESS METRICS

### Multimodality (Goal 2):
- **Target:** ≥5/10 seeds with both L+R picks (when eta>0)
- **Baseline:** 0/10 (currently deterministic)

### Success Rate (Goal 1):
- **Target:** ≥60% after all fixes
- **Baseline:** 14%

### Legibility (Future):
- **Target:** VLM score >0.7 for selected trajectories
- **Baseline:** N/A (not yet implemented)

---

## NEXT STEPS

Ready to implement. Starting with Phase 1 (multimodality testability).

Awaiting approval to proceed with code changes.

---

