# COMPREHENSIVE TEST PLAN
## Testing Strategy for All Pipeline Fixes (Phases 1-4)

**Status:** Ready for execution  
**Expected Time:** 2-3 hours (parallelizable)  
**Goal:** Validate 14% → 60-70% success improvement + multimodality emergence

---

## TEST MATRIX

### Test 1: BASELINE (Current State)
**Purpose:** Establish current performance for comparison

**Configuration:**
```bash
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt.pt \
    --K 10 --M 10 --n_videos 5 \
    --cube_jitter 0.015 \
    --sampling_method ddim --ddim_eta 0.0 \
    --execute_steps 16 \
    --out_dir outputs/test_baseline \
    --video_dir outputs/test_baseline/videos
```

**Expected Results:**
- Success rate: ~14%
- Bimodal seeds: 0/10 (deterministic, can't test)
- Warnings: "MULTIMODALITY CANNOT BE TESTED"

**Key Metrics:**
- `results.csv` → success rate
- `metrics.json` → success/failure breakdown

---

### Test 2: PHASE 1+2 (Jitter Fix + Multimodality)
**Purpose:** Test distribution matching + stochastic sampling

**Configuration:**
```bash
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt.pt \
    --K 10 --M 10 --n_videos 10 \
    --cube_jitter 0.0 \
    --sampling_method ddim --ddim_eta 0.3 \
    --execute_steps 16 \
    --out_dir outputs/test_phase1_2 \
    --video_dir outputs/test_phase1_2/videos
```

**Expected Results:**
- Success rate: 35-45% (+20-30% from jitter fix)
- Bimodal seeds: 2-3/10 (now testable!)
- No warnings (multimodality enabled)

**Key Metrics:**
- `results.csv` → success rate, bimodal seed fraction
- `metrics.json` → left/right pick distribution
- Videos → visual inspection of trajectories

**Validation Checks:**
1. Different `sample_seed` → different trajectories? ✓
2. Some seeds pick left, some pick right? ✓
3. Success rate significantly higher than baseline? ✓

---

### Test 3: PHASE 1+2+3 (Add Gripper Fix)
**Purpose:** Test gripper ensemble fix impact

**Configuration:**
```bash
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt.pt \
    --K 10 --M 10 --n_videos 10 \
    --cube_jitter 0.0 \
    --sampling_method ddim --ddim_eta 0.3 \
    --execute_steps 16 \
    --temporal_ensemble \
    --out_dir outputs/test_phase1_2_3 \
    --video_dir outputs/test_phase1_2_3/videos
```

**Expected Results:**
- Success rate: 45-55% (+5-10% from gripper fix)
- Bimodal seeds: 3/10
- Temporal ensemble enabled WITHOUT gripper smearing

**Key Metrics:**
- Compare success vs Test 2: should be +5-10%
- Check gripper values in videos: should be crisp 0/1, not 0.5

**Validation Checks:**
1. Gripper closes fully (1.0) when grasping? ✓
2. No intermediate gripper values (0.5, 0.7)? ✓
3. Success rate higher than Phase 1+2? ✓

---

### Test 4: ALL FIXES (Phase 1+2+3+4)
**Purpose:** **FULL PIPELINE TEST** with all fixes enabled

**Configuration:**
```bash
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt.pt \
    --K 10 --M 10 --n_videos 15 \
    --cube_jitter 0.0 \
    --sampling_method ddim --ddim_eta 0.3 \
    --execute_steps 16 \
    --temporal_ensemble \
    --dynamic_mpc \
    --out_dir outputs/test_ALL_FIXES \
    --video_dir outputs/test_ALL_FIXES/videos
```

**Expected Results:**
- **Success rate: 60-70%** (+10-15% from dynamic MPC)
- **Bimodal seeds: 3-4/10** ✓ Target achieved
- Adaptive replanning visible in logs/videos

**Key Metrics:**
- Compare success vs Test 3: should be +10-15%
- Check videos: execute_steps adapts near cubes?
- Overall improvement: 14% → 60-70% (4-5x gain)

**Validation Checks:**
1. Execute_steps decreases as EE approaches cube? ✓
2. Precision grasps near target (no drift)? ✓
3. Success rate ≥60%? ✓
4. **PRIMARY GOAL MET?** ✓

---

## ABLATION TESTS (Optional, for deeper analysis)

### Ablation A: Gripper Ensemble Impact
**Purpose:** Quantify gripper ensemble smearing effect

**Configs:**
```bash
# A1: No gripper ensemble (recommended)
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.3 --execute_steps 16 \
    --temporal_ensemble \
    --out_dir outputs/ablation_grip_off

# A2: With gripper ensemble (should be worse)
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.3 --execute_steps 16 \
    --temporal_ensemble --ensemble_grip \
    --out_dir outputs/ablation_grip_on
```

**Hypothesis:** Success(A1) > Success(A2) by 5-10%

---

### Ablation B: Execute Steps Sensitivity
**Purpose:** Find optimal base execute_steps for dynamic MPC

**Configs:**
```bash
# B1: execute_steps=8
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.3 \
    --execute_steps 8 --dynamic_mpc \
    --out_dir outputs/ablation_exec8

# B2: execute_steps=16 (default)
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.3 \
    --execute_steps 16 --dynamic_mpc \
    --out_dir outputs/ablation_exec16

# B3: execute_steps=1 (pure closed-loop)
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.3 \
    --execute_steps 1 \
    --out_dir outputs/ablation_exec1
```

**Hypothesis:** B1 ≈ B2 (dynamic MPC compensates), B3 >> B2 (but very slow)

---

### Ablation C: DDIM Eta Sweep
**Purpose:** Find optimal stochasticity level for multimodality

**Configs:**
```bash
# C1: eta=0.1
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.1 \
    --execute_steps 16 --temporal_ensemble --dynamic_mpc \
    --out_dir outputs/ablation_eta0.1

# C2: eta=0.3 (default)
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.3 \
    --execute_steps 16 --temporal_ensemble --dynamic_mpc \
    --out_dir outputs/ablation_eta0.3

# C3: eta=0.5
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 0.5 \
    --execute_steps 16 --temporal_ensemble --dynamic_mpc \
    --out_dir outputs/ablation_eta0.5

# C4: eta=1.0 (full DDPM)
py scripts/eval_multimodality.py \
    --cube_jitter 0.0 --ddim_eta 1.0 \
    --execute_steps 16 --temporal_ensemble --dynamic_mpc \
    --out_dir outputs/ablation_eta1.0
```

**Hypothesis:** eta=0.3 optimal (balance of smoothness + diversity)

---

## ANALYSIS SCRIPTS

### Compare Results
```bash
# Compare all test outputs
py scripts/compare_results.py \
    outputs/test_baseline \
    outputs/test_phase1_2 \
    outputs/test_phase1_2_3 \
    outputs/test_ALL_FIXES
```

### Generate Report
```bash
# Create summary report with graphs
py scripts/generate_report.py \
    --inputs outputs/test_* \
    --output reports/pipeline_fixes_report.pdf
```

---

## SUCCESS CRITERIA

### Primary Goals ✅
- [x] **Implement all 4 phases of fixes**
- [ ] **Success rate:** 14% → ≥60% (4x improvement)
- [ ] **Multimodality:** 0/5 → ≥3/5 bimodal seeds
- [ ] **Validation:** Test matrix completed

### Secondary Goals
- [ ] **Ablation studies:** Quantify each fix contribution
- [ ] **Documentation:** Update README with new parameters
- [ ] **Reproducibility:** Seed all tests for deterministic results

### Stretch Goals
- [ ] **VLM Integration:** Grade trajectories by legibility score
- [ ] **Retrain Model:** With matched distribution (jitter=0.015)
- [ ] **Hyperparameter Sweep:** Optimize eta, MPC thresholds

---

## RECOMMENDED EXECUTION ORDER

### Step 1: Quick Validation (30 min)
Run determinism test to verify Phase 1:
```bash
py scripts/test_sampler_determinism.py
```
Expected: eta=0 deterministic (L2 < 1e-6), eta>0 stochastic (L2 > 0.01)

### Step 2: Baseline + Phase 1+2 (1 hour)
Run Tests 1 and 2 in parallel:
```bash
# Terminal 1
py scripts/eval_multimodality.py --K 10 --M 10 --cube_jitter 0.015 --ddim_eta 0.0 --out_dir outputs/test_baseline

# Terminal 2  
py scripts/eval_multimodality.py --K 10 --M 10 --cube_jitter 0.0 --ddim_eta 0.3 --out_dir outputs/test_phase1_2
```

### Step 3: Phase 1+2+3 + ALL FIXES (1 hour)
Run Tests 3 and 4 in parallel:
```bash
# Terminal 1
py scripts/eval_multimodality.py --K 10 --M 10 --cube_jitter 0.0 --ddim_eta 0.3 --temporal_ensemble --out_dir outputs/test_phase1_2_3

# Terminal 2
py scripts/eval_multimodality.py --K 10 --M 10 --cube_jitter 0.0 --ddim_eta 0.3 --temporal_ensemble --dynamic_mpc --out_dir outputs/test_ALL_FIXES
```

### Step 4: Analysis (30 min)
```bash
# Compare results
py -c "
import pandas as pd
import json

tests = ['baseline', 'phase1_2', 'phase1_2_3', 'ALL_FIXES']
for test in tests:
    metrics = json.load(open(f'outputs/test_{test}/metrics.json'))
    print(f'{test}: {metrics["success_rate"]:.1%} success, {metrics.get("bimodal_seeds", 0)}/{metrics.get("total_env_seeds", 10)} bimodal')
"
```

---

## TROUBLESHOOTING

### Issue: All tests show low success (<30%)
**Diagnosis:**
1. Check model checkpoint: `runs/latest/ckpt.pt` trained to epoch 300+?
2. Verify action scaling: Demo actions should be -1 to +1
3. Test with execute_steps=1: If >> execute_steps=16, control issue

### Issue: No multimodality (0/10 bimodal seeds)
**Diagnosis:**
1. Check ddim_eta value: Must be >0 (e.g., 0.3)
2. Run determinism test: Verify eta>0 is stochastic
3. Check if model learned symmetry: Try --M 20 (more samples)

### Issue: Gripper not closing
**Diagnosis:**
1. Check --ensemble_grip flag: Should be absent (default False)
2. Verify demo gripper values: Should be 0.0 or 1.0
3. Check env action scaling: grip_denorm should map -1→0, +1→1

---

## EXPECTED TIMELINE

| Stage | Duration | Status |
|-------|----------|--------|
| Phase 1-4 Implementation | 2 hours | ✅ COMPLETE |
| Test 1 (Baseline) | 20 min | PENDING |
| Test 2 (Phase 1+2) | 20 min | PENDING |
| Test 3 (Phase 1+2+3) | 20 min | PENDING |
| Test 4 (ALL FIXES) | 20 min | PENDING |
| Analysis & Report | 30 min | PENDING |
| **TOTAL** | **~3 hours** | **In Progress** |

---

## NEXT ACTIONS

1. **Immediate:** Run Test 4 (ALL FIXES) to validate full pipeline
2. **If success ≥60%:** Proceed to VLM integration (Phase 5)
3. **If success <60%:** Run ablation tests to identify weak link
4. **Document findings:** Update README with recommended settings
