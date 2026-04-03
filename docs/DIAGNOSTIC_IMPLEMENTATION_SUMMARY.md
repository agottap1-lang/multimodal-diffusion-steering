# Diagnostic Implementation Summary

## Overview

This document summarizes all changes made to implement comprehensive diagnostics for the multimodal diffusion policy, as requested in the COPILOT MASTER PROMPT. The goal: eliminate training/eval mismatch and identify root causes of "action collapse" (tiny action magnitudes, low success rates).

**Key constraint**: Do NOT change training loss/objective; MINIMAL safe changes only.

---

## Implementation Status

### ✅ Completed

1. **Training-time quick eval is now fully config-driven**
   - Added `eval_ddim_steps` parameter to support strided DDIM schedules
   - Added `eval_temporal_ensemble` parameter to match eval behavior
   - Updated `_quick_sim_eval()` to accept and use these parameters

2. **Enhanced action collapse diagnostics in eval script**
   - Two-tier severity warnings: CRITICAL (<0.005) vs WARNING (<0.1)
   - Clearer diagnostic messages with next-step suggestions
   - References to new diagnostic scripts

3. **Created BC sanity check utility** (`scripts/bc_sanity_check.py`)
   - Tests policy forward pass offline (no simulation)
   - Validates action normalization/unnormalization correctness
   - Detects scaling bugs and model collapse

4. **Created sampler sanity check utility** (`scripts/sampler_sanity_check.py`)
   - Tests DDPM vs DDIM with different eta values
   - Validates strided DDIM schedules
   - Checks for near-zero outputs and determinism

### ⏸️ Not Implemented (Already Exists or Not Needed)

- **EE displacement logging**: Already implemented in eval_multimodality.py (lines 554-558)
- **Action chunk logging**: Already implemented (lines 521-531)
- **Demo comparison**: Already implemented (lines 184-200)
- **Temporal ensemble control**: Configurable via `--temporal_ensemble` flag
- **Action scaling verification**: Can be added if needed, but existing diagnostics sufficient

---

## File-by-File Changes

### 1. `configs/train.yaml`

**Changes added:**
```yaml
# Line 43-44 (after eval_ddim_eta)
eval_ddim_steps: null  # null = use all n_diffusion_steps; set to 20-50 for strided DDIM schedule
eval_temporal_ensemble: true  # average overlapping action chunks (smoother but may over-smooth)
```

**Purpose**: Make training-time quick eval sampling fully config-driven to match eval script behavior.

**Default behavior**: Uses all diffusion steps (no striding), temporal ensemble enabled.

---

### 2. `scripts/train_diffusion_policy.py`

**Changes:**

#### a) Updated `_quick_sim_eval()` signature (line 773)
```python
# Added two new parameters:
ddim_steps: int | None = None,
temporal_ensemble: bool = True
```

#### b) Updated DiffusionPolicyRunner instantiation (line 799)
```python
# Changed from hardcoded temporal_ensemble=True to:
temporal_ensemble=temporal_ensemble,
```

#### c) Updated both call sites (lines 692, 702)
```python
# Added to both _quick_sim_eval calls:
ddim_steps=cfg.get("eval_ddim_steps", None),
temporal_ensemble=cfg.get("eval_temporal_ensemble", True)
```

**Purpose**: Training-time quick eval now respects eval_ddim_steps and eval_temporal_ensemble from config.

**Impact**: Can now test different sampling configurations during training without code changes.

---

### 3. `scripts/eval_multimodality.py`

**Changes:**

#### a) Enhanced action collapse warnings (lines 264-277)
```python
# OLD (single-tier warning):
if pos_std < 0.1:
    print(f"  [WARNING] Action collapse detected!")
    if self._demo_std is not None and pos_std < self._demo_std * 0.3:
        print(f"  [WARNING] Policy std is {pos_std/self._demo_std:.1%} of demo std!")
        print(f"            Likely causes: (1) action scaling bug, (2) DDIM implementation bug,")
        print(f"            (3) model collapse, or (4) normalization mismatch.")

# NEW (two-tier severity with next-step suggestions):
if pos_std < 0.005:
    print(f"  [CRITICAL] Severe action collapse! Policy outputs near-zero (std < 0.005).")
    print(f"            This likely indicates MODEL COLLAPSE or catastrophic scaling bug.")
    print(f"            → Run scripts/bc_sanity_check.py to test offline forward pass")
    print(f"            → Run scripts/sampler_sanity_check.py to verify sampler outputs")
elif pos_std < 0.1:
    print(f"  [WARNING] Action collapse detected! Policy outputs small actions (std < 0.1).")
    if self._demo_std is not None and pos_std < self._demo_std * 0.3:
        print(f"  [WARNING] Policy std is {pos_std/self._demo_std:.1%} of demo std!")
        print(f"            Likely causes: (1) action scaling mismatch, (2) normalization bug,")
        print(f"            (3) temporal ensembling over-smoothing, or (4) model undertrained.")
        print(f"            → Try without --temporal_ensemble flag to rule out ensembling")
        print(f"            → Try eval_sampling_method='ddpm' in config for comparison")
```

**Purpose**: Clearer diagnostic messages with severity levels and actionable next steps.

**Impact**: Faster root cause identification during debugging.

---

### 4. `scripts/bc_sanity_check.py` (NEW FILE - 250 lines)

**Purpose**: Test policy forward pass offline without simulation.

**What it does**:
1. Loads checkpoint and demo data
2. Samples random (obs, action) pairs from demos
3. Runs policy forward pass (obs → predicted action)
4. Compares predicted vs demo actions (motion dims and yaw separately)
5. Prints ratios and diagnostics

**Key outputs**:
- `std_ratio`: predicted_std / demo_std (should be ~1.0 if no bugs)
- `abs_mean_ratio`: predicted_abs_mean / demo_abs_mean
- Diagnostic interpretation: PASS/WARNING/FAIL with likely causes

**When to use**: When action collapse is suspected, run this FIRST to rule out offline bugs before testing in simulation.

**Example usage**:
```powershell
python scripts/bc_sanity_check.py --ckpt runs/20260213_213052/ckpt_ep300.pt
python scripts/bc_sanity_check.py --ckpt runs/latest/ckpt_ep500.pt --n_samples 100
```

---

### 5. `scripts/sampler_sanity_check.py` (NEW FILE - 280 lines)

**Purpose**: Test different sampling methods and compare outputs.

**What it does**:
1. Loads checkpoint
2. Creates fixed observation (zero vector, normalized)
3. Samples with 4 configurations:
   - DDPM (full T-step denoising)
   - DDIM eta=1.0 (stochastic, DDPM-like)
   - DDIM eta=0.0 (deterministic, ODE solver)
   - DDIM eta=0.0, 20 steps (strided schedule)
4. Compares mean/std/min/max for motion dims and yaw
5. Checks for near-zero outputs, determinism, and cross-method ratios

**Key checks**:
- **Model collapse**: DDPM std < 0.005 → model outputs near-zero
- **Action suppression**: DDPM std < 0.1 → outputs too small
- **DDIM correctness**: DDIM/DDPM ratio close to 1.0
- **Strided schedule**: 20-step DDIM produces valid outputs
- **Determinism**: DDIM eta=0.0 same seed → same output

**When to use**: When action collapse is suspected AND BC sanity check passes, run this to test sampler behavior.

**Example usage**:
```powershell
python scripts/sampler_sanity_check.py --ckpt runs/20260213_213052/ckpt_ep300.pt
python scripts/sampler_sanity_check.py --ckpt runs/latest/ckpt_ep500.pt --n_samples 20
```

---

## Test Command Sequences

### Scenario 1: New checkpoint shows action collapse in eval

```powershell
# Step 1: Test offline forward pass (fastest, rules out model collapse)
python scripts/bc_sanity_check.py --ckpt runs/20260213_213052/ckpt_ep300.pt

# If BC check FAILS (ratio far from 1.0):
#   → Model collapse or normalization bug
#   → Check training logs, verify demo stats
#   → Re-train with different hyperparameters

# If BC check PASSES (ratio ~1.0):
#   → Continue to Step 2

# Step 2: Test sampler outputs
python scripts/sampler_sanity_check.py --ckpt runs/20260213_213052/ckpt_ep300.pt

# If sampler check shows near-zero outputs:
#   → Model collapse (all samplers fail)
#   → Check training logs

# If sampler check shows reasonable outputs:
#   → Continue to Step 3

# Step 3: Test in simulation with different configs
# 3a) WITHOUT temporal ensemble (faster execution)
python scripts/eval_multimodality.py --K 5 --M 10 --ckpt runs/20260213_213052/ckpt_ep300.pt \
    --out_dir outputs/test_no_ensemble

# 3b) WITH temporal ensemble (matches training default)
python scripts/eval_multimodality.py --K 5 --M 10 --ckpt runs/20260213_213052/ckpt_ep300.pt \
    --temporal_ensemble --out_dir outputs/test_with_ensemble

# 3c) With DDPM instead of DDIM
python scripts/eval_multimodality.py --K 5 --M 10 --ckpt runs/20260213_213052/ckpt_ep300.pt \
    --sampling_method ddpm --out_dir outputs/test_ddpm

# 3d) With strided DDIM (20 steps)
python scripts/eval_multimodality.py --K 5 --M 10 --ckpt runs/20260213_213052/ckpt_ep300.pt \
    --ddim_steps 20 --out_dir outputs/test_ddim_20steps

# Compare results to identify what configuration reduces action collapse
```

### Scenario 2: Training shows low eval success during checkpoints

```powershell
# Modify config BEFORE training:
# configs/train.yaml:
#   eval_temporal_ensemble: false  # Try without ensembling
#   eval_sampling_method: "ddpm"   # Try DDPM instead of DDIM
#   eval_ddim_steps: 20            # Try strided schedule (faster eval)

# Train with modified config
python scripts/train_diffusion_policy.py --config configs/train.yaml

# During training, watch for action collapse warnings in console
# If eval success improves with different sampling config, use that for full eval
```

### Scenario 3: Comparing two checkpoints

```powershell
# Test both checkpoints offline
python scripts/bc_sanity_check.py --ckpt runs/20260213_213052/ckpt_ep300.pt > bc_ep300.txt
python scripts/bc_sanity_check.py --ckpt runs/latest/ckpt_ep500.pt > bc_ep500.txt

# Test samplers
python scripts/sampler_sanity_check.py --ckpt runs/20260213_213052/ckpt_ep300.pt > sampler_ep300.txt
python scripts/sampler_sanity_check.py --ckpt runs/latest/ckpt_ep500.pt > sampler_ep500.txt

# Full eval comparison
python scripts/eval_multimodality.py --K 10 --M 20 --ckpt runs/20260213_213052/ckpt_ep300.pt \
    --out_dir outputs/compare_ep300 --temporal_ensemble

python scripts/eval_multimodality.py --K 10 --M 20 --ckpt runs/latest/ckpt_ep500.pt \
    --out_dir outputs/compare_ep500 --temporal_ensemble

# Compare outputs:
# - bc_ep300.txt vs bc_ep500.txt: Which has better std_ratio?
# - sampler_ep300.txt vs sampler_ep500.txt: Which has higher DDPM std?
# - outputs/compare_*: Which has higher success rate?
```

---

## Root Cause Status Table

| Hypothesis | Status | Evidence | Next Steps |
|------------|--------|----------|------------|
| **DDIM sampler bug** | ✅ **RULED OUT** | Single-step test: DDIM/DDPM ratio = 1.03 (scripts/debug_ddim_step_compare.py) | None needed; coefficients mathematically correct |
| **Model collapse** | ⚠️ **TEST NEEDED** | Training loss ~0.004 but sim success ~0-8%; action std ~0.01-0.09 (expected ~0.3-0.5) | Run `scripts/bc_sanity_check.py` and `scripts/sampler_sanity_check.py` |
| **Action scaling mismatch** | ⚠️ **TEST NEEDED** | Train vs eval environment may have different action_scale_pos | Check env.action_scale_pos in training and eval; verify demo metadata |
| **Normalization bug** | ⚠️ **TEST NEEDED** | Unnormalization in eval may differ from training | BC sanity check will catch this (predicted != demo actions offline) |
| **Temporal ensemble over-smoothing** | ⚠️ **TEST NEEDED** | Training uses ensemble by default; averaging may suppress multimodality | Test WITHOUT --temporal_ensemble flag; compare action std |
| **DDIM deterministic suppression** | ⚠️ **INVESTIGATE** | DDIM eta=0.0 is deterministic ODE solver; may suppress variance vs stochastic DDPM | Test with eval_sampling_method='ddpm' or eval_ddim_eta=0.3 (stochastic) |
| **Strided schedule under-sampling** | ⚠️ **TEST NEEDED** | If eval_ddim_steps is too small, may skip important timesteps | Keep eval_ddim_steps=null (or >=50) during initial debugging |
| **Obs0 vs ObsT mismatch** | ⚠️ **INVESTIGATE** | Policy may be trained on obs_t but eval uses obs_{t-1} or vice versa | Review DiffusionPolicyRunner._plan() and rollout() obs feeding |

### How to Update This Table

After running diagnostic scripts, update the Status column:

- ✅ **RULED OUT**: Definitive evidence shows this is NOT the problem
- ⚠️ **INVESTIGATE**: Some evidence suggests this may be an issue (continue testing)
- ❌ **CONFIRMED**: Definitive evidence shows this IS the problem (fix required)
- ⏳ **IN PROGRESS**: Currently testing this hypothesis

---

## Configuration Quick Reference

### Training Config (`configs/train.yaml`)

```yaml
# Sampling configuration for training-time quick eval
eval_sampling_method: "ddim"         # "ddpm" or "ddim"
eval_ddim_eta: 0.0                   # 0.0=deterministic, 1.0=stochastic (DDPM-like)
eval_ddim_steps: null                # null=all steps, 20-50=strided schedule
eval_temporal_ensemble: true         # true=ensemble (smoother), false=no ensemble (more stochastic)
eval_execute_steps: 16               # how many plan steps to execute before replanning
```

**Recommended for debugging action collapse:**
- Start with `eval_sampling_method: "ddpm"` (more stochastic, closer to training diffusion process)
- Set `eval_temporal_ensemble: false` (rule out over-smoothing)
- Keep `eval_ddim_steps: null` (don't use strided schedule until DDPM works)

### Eval Script Flags (`scripts/eval_multimodality.py`)

```powershell
# Sampling configuration
--sampling_method ddpm              # Use DDPM instead of DDIM
--ddim_eta 0.0                      # DDIM stochasticity (0.0=deterministic)
--ddim_steps 20                     # Use strided schedule (faster but may lose quality)

# Execution configuration
--temporal_ensemble                 # Enable temporal ensembling (smoother actions)
--execute_steps 16                  # Steps to execute before replanning (higher=faster but less adaptive)

# Debugging flags
--verbose                           # Print action chunk stats for first 3 plans
--log_chunks                        # Save all action chunks to CSV
--cube_jitter 0.0                   # Disable cube jitter for deterministic eval
```

**Recommended for debugging:**
```powershell
python scripts/eval_multimodality.py \
    --ckpt runs/20260213_213052/ckpt_ep300.pt \
    --K 5 --M 10 \
    --sampling_method ddpm \
    --execute_steps 1 \
    --verbose \
    --log_chunks \
    --out_dir outputs/debug_ddpm_exec1
```

---

## Diagnostic Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  ACTION COLLAPSE DIAGNOSED IN EVAL                          │
│  (Training loss low, sim success low, action std very small)│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Run BC Sanity Check   │
         │ (offline forward pass)│
         └───────────┬───────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    ┌─────────┐          ┌──────────┐
    │ PASS    │          │ FAIL     │
    │ (~1.0)  │          │ (!=1.0)  │
    └────┬────┘          └────┬─────┘
         │                    │
         │                    ▼
         │         ┌──────────────────────┐
         │         │ MODEL COLLAPSE or    │
         │         │ NORMALIZATION BUG    │
         │         │ → Check training     │
         │         │ → Verify demo stats  │
         │         └──────────────────────┘
         │
         ▼
┌────────────────────┐
│ Run Sampler Check  │
│ (DDPM/DDIM compare)│
└────────┬───────────┘
         │
  ┌──────┴──────┐
  │             │
  ▼             ▼
┌─────────┐  ┌──────────────┐
│ Near-   │  │ Reasonable   │
│ zero    │  │ magnitudes   │
└────┬────┘  └──────┬───────┘
     │              │
     ▼              ▼
┌──────────┐  ┌─────────────────┐
│ MODEL    │  │ Test in SIM     │
│ COLLAPSE │  │ with variations:│
└──────────┘  │ - No ensemble   │
              │ - DDPM sampling │
              │ - execute_steps │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        ┌──────────┐      ┌──────────┐
        │ Collapse │      │ Success! │
        │ persists │      │ Found    │
        └────┬─────┘      │ config   │
             │            └──────────┘
             ▼
   ┌──────────────────────┐
   │ SCALING MISMATCH or  │
   │ EXECUTION BUG        │
   │ → Check action_scale │
   │ → Check obs feeding  │
   └──────────────────────┘
```

---

## Expected Outputs

### BC Sanity Check (PASS):
```
RESULTS (Position dims [:3] only)

Demo actions (ground truth):
  std:        0.183456
  abs_mean:   0.142378
  range:      [-0.892, +0.967]

Predicted actions (policy forward pass):
  std:        0.179832
  abs_mean:   0.139021
  range:      [-0.847, +0.921]

Ratios (predicted / demo):
  std_ratio:           0.980
  abs_mean_ratio:      0.976

✓ PASS: Policy predictions match demo magnitudes (ratio ~1.0)
  → Forward pass and (un)normalization are correct
  → No scaling bugs detected in offline test
```

### BC Sanity Check (FAIL - model collapse):
```
RESULTS (Position dims [:3] only)

Predicted actions (policy forward pass):
  std:        0.008123
  abs_mean:   0.005921
  range:      [-0.034, +0.029]

Ratios (predicted / demo):
  std_ratio:           0.044

✗ FAIL: Policy predictions significantly different from demos
  → Predicted std is 4.4% of demo std
  → Severe action suppression detected!

✗ CRITICAL: Predicted actions are near-zero!
  → Model may have collapsed (outputs near-zero regardless of input)
  → Check training logs for loss explosion or vanishing gradients
```

### Sampler Sanity Check (PASS):
```
DDPM (full T-step):
  mean:     +0.000834
  std:      0.152341
  abs_mean: 0.117234
  range:    [-0.781, +0.823]

DDIM eta=0.0 (deterministic):
  mean:     +0.000621
  std:      0.148762
  abs_mean: 0.114509
  range:    [-0.764, +0.809]

DDIM eta=0.0 vs DDPM std ratio: 0.976

✓ DDPM outputs reasonable magnitude (std >= 0.1)
✓ DDIM eta=0.0 close to DDPM (ratio=0.976)
  → DDIM coefficients appear correct
✓ DDIM eta=0.0 is deterministic (same seed → same output)

✓ All samplers working correctly
```

---

## Minimal Change Guarantee

All changes adhere to the constraint "Do NOT change training loss/objective; MINIMAL safe changes only":

✅ **No changes to**:
- Training loop (`train.py` lines 1-680)
- Loss computation (`NoiseNet`, `DDPMSchedule.get_loss()`)
- Model architecture (`NoiseNet.__init__()`, `.forward()`)
- Optimizer, scheduler, or hyperparameters
- Demo collection or preprocessing

✅ **Only changes made**:
- Added config parameters for eval-time sampling (eval_ddim_steps, eval_temporal_ensemble)
- Enhanced diagnostic print statements (no behavioral changes)
- Created NEW diagnostic scripts (bc_sanity_check.py, sampler_sanity_check.py)
- Updated existing eval script parameter passing (no logic changes)

---

## Next Steps After Running Diagnostics

1. **If BC check fails**: Fix action normalization or retrain model
2. **If sampler check shows near-zero**: Retrain model with different hyperparameters
3. **If both pass but eval still fails**: Test temporal_ensemble, sampling_method, execute_steps
4. **If eval succeeds with specific config**: Update train.yaml defaults to match
5. **If all tests pass and eval succeeds**: Proceed to LDSRL-style VLM steering experiments

---

## Document Version

- **Created**: 2026-02-13
- **Author**: GitHub Copilot (Claude Sonnet 4.5)
- **Purpose**: Implementation summary for COPILOT MASTER PROMPT diagnostic requirements
- **Status**: All core diagnostics implemented; ready for testing on checkpoint ep300
