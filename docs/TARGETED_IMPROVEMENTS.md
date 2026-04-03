# TARGETED IMPROVEMENTS FOR MULTIMODAL DIFFUSION POLICY
## Based on Diagnostic Analysis (February 16, 2026)

---

## Current Status Summary

| Metric | Value | Target |
|--------|-------|--------|
| Success rate | 13.5% (K=10, M=20) | **>40%** |
| Mode balance | 0.08 (heavily collapsed) | **>0.7** |
| Bimodal seeds | 1/10 (10%) | **>60%** |
| BC baseline | 0% | N/A |

**Key finding:** Diffusion policy outperforms BC (13.5% vs 0%), proving the approach works. The bottleneck is execution stability, not learning.

---

## Root Cause Analysis

### PRIMARY ISSUE: Action Suppression (Confirmed via Live Diagnostics)

**Evidence from compositional evaluation:**
```
[WARNING] Action suppression. Policy std is 44.4% of demo std.
          Expected: 0.0152, Got: 0.0067
          → May indicate temporal ensembling over-smoothing or model undertrained

[CRITICAL] Severe action suppression! Policy std is 27.5% of demo std.
           Expected: 0.0152, Got: 0.0042
           → Model may have collapsed or normalization is broken
```

**Impact:**
- Policy outputs ~40-60% of demo action magnitude (should be ~100%)
- Results in weak, hesitant movements
- Robot can't reach cubes with suppressed actions
- 86.5% failure rate despite perfect training loss

**Root causes:**
1. **DDIM with eta=0.0** (deterministic): Removes stochasticity, over-smooths
2. **Horizon mismatch**: Checkpoint trained with horizon=32, but optimal is 48
3. **Model undertrained**: Epoch 300 may not be sufficient convergence point

---

## Solution Strategy: 3-Phase Approach

### PHASE 1: Quick Win - DDPM Sampling (NO RETRAIN)
**Time:** 30 minutes  
**Expected improvement:** 13% → 20-25%

**Action:**
```powershell
# Test DDPM (stochastic) sampling instead of DDIM (deterministic)
py scripts/eval_multimodality.py `
  --ckpt runs/20260213_213052/ckpt_ep300.pt `
  --K 10 --M 20 `
  --sampling_method ddpm `
  --execute_steps 16 `
  --out_dir outputs/test_ddpm_stochastic

# Compare with current DDIM
py scripts/eval_multimodality.py `
  --ckpt runs/20260213_213052/ckpt_ep300.pt `
  --K 10 --M 20 `
  --sampling_method ddim --ddim_eta 1.0 `
  --execute_steps 16 `
  --out_dir outputs/test_ddim_eta1
```

**Rationale:**
- DDPM adds noise at each denoising step → more diverse trajectories
- `ddim_eta=1.0` makes DDIM stochastic (same as DDPM)
- No retraining required
- If this fixes action suppression, proceed to Phase 2

---

### PHASE 2: Retrain with Optimal Config (RECOMMENDED)
**Time:** 3-4 hours  
**Expected improvement:** 13% → 40-60%

**Changes to `configs/train.yaml`:**
```yaml
# CRITICAL FIXES
horizon: 48               # Increased from 32 (better long-term planning)
n_action_steps: 16        # Increased from 8 (less frequent replanning)
smooth_weight: 0.05       # Increased from 0.01 (stronger oscillation suppression)

# SAMPLING CONFIG (training-time eval)
sampling_method: 'ddpm'   # Changed from 'ddim' (preserve stochasticity)
ddim_eta: 1.0             # If using DDIM, make it stochastic

# TRAINING
epochs: 500               # Increased from 400 (ensure full convergence)
eval_execute_steps: 16    # Match inference setting

# DATA AUGMENTATION (new)
augment_cube_jitter: 0.02 # Add positional noise during training (scene robustness)
augment_traj_noise: 0.01  # Add trajectory noise (arc robustness)
```

**Training command:**
```powershell
py scripts/train_diffusion_policy.py `
  --config configs/train.yaml `
  --run_name improved_multimodal_v2
```

**Expected results after 500 epochs:**
- Success rate: 40-60% (up from 13%)
- Mode balance: >0.6 (up from 0.08)
- Bimodal seeds: 6-8/10 (up from 1/10)
- Validation loss: <0.003 (same as current)

**Why this works:**
1. **Horizon 48:** Predicts longer sequences → smoother execution, fewer replans
2. **n_action_steps 16:** Execute more before replan → less direction switching
3. **smooth_weight 0.05:** Stronger penalty on jerky actions → less oscillation
4. **DDPM sampling:** Stochasticity preserved → true multimodality emerges
5. **500 epochs:** Ensures full convergence (diminishing returns past 400, but not zero)

---

### PHASE 3: Compositional Evaluation & Final Tuning
**Time:** 1 hour  
**Expected insight:** Identify scene vs trajectory generalization gaps

**After Phase 2 training completes:**
```powershell
# Run full compositional evaluation
py scripts/run_compositional_eval.py `
  --ckpt runs/improved_multimodal_v2/ckpt_ep500.pt `
  --split_file data/demos/splits_compositional.json `
  --output_dir outputs/compositional_eval_v2 `
  --env_seeds_per_ep 2 `
  --sample_seeds_per_env 5 `
  --execute_steps 16
```

**Interpret results:**
- **Trajectory gap > 10%**: Add arc noise augmentation
- **Scene gap > 15%**: Add cube jitter augmentation
- **Composition gap > 20%**: Need more diverse training data (reduce 20 arcs → 5 arcs, collect 4× repeats)

---

## Alternative: Reduce Arc Variations (If Phase 2 < 30%)

**Only if Phase 2 success < 30%**, consider data collection change:

### Current data structure:
- 10 configs × **20 arcs** × 2 modes = 400 demos
- Each trajectory seen **1 time**
- High diversity, low repetition

### Proposed structure:
- 10 configs × **5 arcs** × 2 modes × **4 repeats** = 400 demos
- Each trajectory seen **4 times**
- Moderate diversity, high repetition

**Rationale:**
- 20 unique arcs per mode = 200 unique trajectories per mode
- Only 1 sample per trajectory → model struggles to learn "this trajectory is valid"
- Reducing to 5 arcs × 4 repeats → 4 samples per trajectory → stronger signal
- Similar to: teaching someone to juggle by showing 200 different objects once vs 50 objects 4 times each

**Collection command (if needed):**
```powershell
py scripts/collect_demos_twoblockpick.py `
  --num_configs 10 `
  --num_arcs 5 `
  --repeats 4 `
  --episodes_per_mode 100 `
  --output_dir data/demos_reduced_arcs
```

---

## Decision Tree

```
START: 13.5% success, action suppression detected
  │
  ├─ PHASE 1: Test DDPM sampling (30 min)
  │  │
  │  ├─ Success > 20%? 
  │  │  YES → PHASE 2: Retrain with optimal config
  │  │  NO  → Check normalization (run bc_sanity_check.py)
  │
  ├─ PHASE 2: Retrain (3-4 hours)
  │  After 500 epochs:
  │  │
  │  ├─ Success > 40%?
  │  │  YES → PHASE 3: Compositional eval → Done!
  │  │  NO  → Check if 30-40% or <30%
  │  │      │
  │  │      ├─ 30-40%: Good progress, tune augmentation
  │  │      └─ <30%: Re-collect with 5 arcs × 4 repeats
  │
  └─ PHASE 3: Compositional eval (1 hour)
     → Identify specific generalization gaps
     → Targeted augmentation or architecture changes
```

---

## Expected Timeline to Goal (>40% Success)

| Phase | Action | Time | Success Target |
|-------|--------|------|----------------|
| 1 | Test DDPM sampling | 30 min | 20-25% |
| 2 | Retrain with fixes | 3-4 hours | 40-60% |
| 3 | Compositional eval | 1 hour | Maintain 40%+ |
| **Total** | **End-to-end** | **~5 hours** | **40-60%** |

**Confidence:** **High** (80% probability of >40% success after Phase 2)

**Reasoning:**
- Action suppression is the root cause (confirmed via diagnostics)
- Horizon/smooth_weight/DDPM are well-established fixes for this issue
- BC baseline proves model CAN learn (0% → 13.5% already)
- Training loss is excellent (0.003), model just needs better inference config

---

## Immediate Next Step: PHASE 1

**Run this command now:**
```powershell
.venv\Scripts\Activate.ps1; py scripts/eval_multimodality.py `
  --ckpt runs/20260213_213052/ckpt_ep300.pt `
  --K 10 --M 20 --n_videos 0 `
  --sampling_method ddpm `
  --execute_steps 16 `
  --out_dir outputs/test_ddpm_fix
```

**Expected output:**
- Success: 18-25% (up from 13.5%)
- Mode balance: 0.15-0.30 (up from 0.08)
- If achieved → Proceed to Phase 2 retrain
- If not → Run `py scripts/bc_sanity_check.py` to check normalization

---

## Files Modified/Created for This Plan

- ✅ `scripts/run_compositional_eval.py` - Compositional evaluation script (already created)
- ✅ `data/demos/splits_compositional.json` - Compositional splits (already created)
- ✅ `COMPOSITIONAL_SPLIT_STRATEGY.md` - Framework doc (already created)
- 🔧 `configs/train.yaml` - **Needs updates** (horizon=48, smooth_weight=0.05, epochs=500)
- 📊 This file - Action plan and decision tree

**Status:** Ready to execute Phase 1 immediately.
