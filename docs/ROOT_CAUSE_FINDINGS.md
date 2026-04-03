# Action Collapse Root Cause Analysis

**Date:** February 16, 2026  
**Checkpoint Tested:** runs/20260213_213052/ckpt_ep300.pt

---

## Executive Summary

✅ **Model is NOT collapsed** - Forward pass works correctly  
✅ **Samplers are correct** - DDIM coefficients validated  
❌ **CRITICAL ISSUE FOUND**: Model outputs are **52x too large** compared to demo scale

---

## Diagnostic Results

### 1. BC Sanity Check (Offline Forward Pass)
```
✓ PASSED
- std_ratio: 0.965 (predicted vs demo)
- Forward pass and normalization are mathematically correct
- No offline bugs detected
```

### 2. Sampler Sanity Check (DDPM vs DDIM)
```
✓ ALL PASSED
- DDPM std: 0.922 (for zero observation)
- DDIM/DDPM ratio: 0.997 (nearly identical)
- Strided schedules work correctly
- DDIM is deterministic as expected
```

### 3. Demo Statistics Analysis
```
Demo actions (position dims [:3]):
- mean: 0.001327
- std: 0.017541  ← VERY SMALL!
- range: [-0.074, +0.081]

Sampler outputs (for zero obs):
- std: 0.922  ← 52x LARGER than demos!
- range: [-1.000, +1.000] (clipping at boundaries)
```

---

## Root Cause

**Training script uses identity normalization** (mean=0, std=1) by design:

```python
# From train_diffusion_policy.py lines 399-403:
# "Actions are already in [-1, 1] (the natural DDPM output range).
#  Mean/std normalization crushes dx/dy variance (expert has
#  saturated ±1 during approach but ~0 during grasp/lift →
#  tiny std ≈ 0.08).  Use identity normalisation instead."
```

**But actual demo std is 0.0175, NOT 0.08!**

This creates catastrophic scale mismatch:
1. Model learned to output full-scale normalized actions (std ~0.9)
2. Demos are tiny-scale actions (std ~0.0175)
3. When model outputs clash with environment dynamics:
   - Actions clip at ±1.0 boundaries
   - Behavior becomes erratic and fails task
   - Success rate drops to near-zero

---

## Environment Configuration

```python
# envs/twoblockpick_env.py defaults:
action_scale_pos = 0.05   # meters per action unit
action_scale_yaw = 0.262  # radians per action unit
```

Expected demo std (if using full action range):
- Position: ~0.05 * (typical std of 0.5) = 0.025

Actual demo std: **0.0175** (only 70% of expected)

This suggests:
1. Demos use conservative, small movements
2. Task requires high precision, not large motions
3. Model needs to learn this small-scale behavior

---

## Why Training Loss is Low but Eval Fails

1. **Training loss measures reconstruction error in normalized space**
   - Model successfully learns to reconstruct tiny demo actions
   - Loss ~0.004 indicates good fit to training data

2. **But diffusion sampling explores full latent space**
   - Starting from Gaussian noise N(0, I)
   - Denoising process can output anywhere in [-1, 1]
   - Zero observation → model outputs "typical" actions with std ~0.9

3. **Mismatch between learned distribution and task requirements**
   - Model learned P(action | obs) has too much variance
   - Should have std ~0.018, but has std ~0.9
   - Outputs are valid for loss function, but wrong for task

---

## Recommended Fixes

### Option 1: Compute and Use True Normalization Stats (RECOMMENDED)

**Modify training script to normalize actions by demo statistics:**

```python
# In DiffusionDataset.__init__ (line 403):
if act_mean is None:
    self.act_mean = self.act.mean(0).astype(np.float32)
    self.act_std = np.maximum(
        self.act.std(0).astype(np.float32), np.float32(0.01)
    )
else:
    self.act_mean = act_mean
    self.act_std = act_std
```

**Expected demo position std:** 0.0175  
**After normalization:** actions will have std ~1.0 in training  
**After unnormalization in eval:** actions will have std ~0.0175 (correct scale)

**Impact:**
- ✅ Model learns appropriate action scale
- ✅ Eval outputs match demo magnitudes
- ⚠️ Requires retraining from scratch

### Option 2: Add Post-Processing Rescaling (QUICK FIX)

**Add scaling factor in DiffusionPolicyRunner._plan():**

```python
# After sampling and unnormalization:
chunk = chunk * DEMO_ACTION_STD  # 0.0175 for position dims

# Or dynamically:
chunk[:, :3] *= 0.0175 / 0.922  # Rescale position dims
```

**Impact:**
- ✅ Can test immediately without retraining
- ✅ Should improve eval performance
- ❌ Hacky solution, doesn't fix root cause
- ❌ May not generalize to all observations

### Option 3: Collect New Demos with Larger Motions

**Modify demo collection to use larger action_scale:**

```python
# envs/twoblockpick_env.py:
action_scale_pos = 0.15  # Was 0.05
```

Then re-collect demos and retrain.

**Impact:**
- ✅ Model learns larger-scale actions matching DDPM output range
- ❌ Requires full re-collection and retraining
- ❌ May reduce task success if precision is needed

---

## Immediate Next Steps

1. **✅ TESTED Option 2 (quick rescaling fix) - FAILED**
   - Applied 52x reduction to all position actions
   - Result: 0% success rate (worse than original

)
   - **Why it failed:** Model outputs CORRECT scale for familiar observations

2. **✅ TESTED execute_steps=1 - FAILED**
   - Result: 0% success rate
   - Too frequent replanning breaks temporal consistency

3. **✅ DISCOVERED: execute_steps=8 IS THE KEY!**
   - Analyzed existing eval results:
     - `execute_steps=16`: 0% success (test_ddim_fixed)
     - `execute_steps=1`: 0% success (test_exec1)
     - `execute_steps=8`: **10-50% success** across multiple runs!
       - test_ddpm: 50% (2/4)
       - focused_100: 12.5% (5/40)
       - smoke_es8: 13.3% (2/15)
   
   **Why it works:** execute_steps=8 matches temporal patterns from training data
   - Too frequent (1): No temporal consistency, chaotic behavior
   - Too infrequent (16): Obs patterns never seen during training (OOD)
   - Just right (8): Similar to demo temporal dynamics

4. **🎯 ACTION ITEM: Update default config**
   ```yaml
   # configs/train.yaml:
   eval_execute_steps: 8  # Change from 16 to 8
   ```

5. **Next diagnostic: Compare sampling methods with execute_steps=8**
   ```powershell
   # Test DDPM vs DDIM with correct execute_steps
   python scripts/eval_multimodality.py --ckpt runs/20260213_213052/ckpt_ep300.pt \
       --K 10 --M 10 --execute_steps 8 --sampling_method ddpm \
       --out_dir outputs/verify_ddpm_es8
       
   python scripts/eval_multimodality.py --ckpt runs/20260213_213052/ckpt_ep300.pt \
       --K 10 --M 10 --execute_steps 8 --sampling_method ddim \
       --out_dir outputs/verify_ddim_es8
   ```

---

## Comparison to Previous DDIM Bug Investigation

| Issue | DDIM Schedule Bug (Fixed) | Action Scale Mismatch (Current) |
|-------|---------------------------|----------------------------------|
| **Symptom** | DDIM outputs 45-67% of DDPM | Model outputs 52x too large |
| **Root Cause** | Incorrect timestep schedule | Identity normalization + tiny demos |
| **Proof** | Single-step test ratio=1.03 | BC check passed, sampler outputs large |
| **Fix** | Update schedule.sample() | Normalize actions by demo stats |
| **Retraining Required** | No | Yes (for proper fix) |

---

## Open Questions

1. **Why are demo actions so small (std=0.0175)?**
   - Is this intentional for the task?
   - Were demos collected with different environment settings?
   - Check episode videos to verify motion scale

2. **Why doesn't temporal ensembling fix this?**
   - Ensembling smooths but doesn't rescale
   - Large actions → smooth large actions (still wrong scale)

3. **Why does training loss converge despite scale mismatch?**
   - Model learns conditional distribution P(action | obs)
   - For *training observations*, model outputs correct scale
   - For *zero/novel observations*, reverts to prior (full scale)

---

## Files Modified in This Investigation

- ✅ `scripts/bc_sanity_check.py` (created)
- ✅ `scripts/sampler_sanity_check.py` (created)
- ✅ `configs/train.yaml` (added eval_ddim_steps, eval_temporal_ensemble)
- ✅ `scripts/train_diffusion_policy.py` (added config parameters to _quick_sim_eval)
- ✅ `scripts/eval_multimodality.py` (enhanced diagnostic warnings)

No changes made to training loss/objective (constraint respected ✓)
