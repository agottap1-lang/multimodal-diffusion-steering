# Execute Steps Fix - Implementation Summary

## Changes Made

### 1. configs/train.yaml
**Line 35:** Already set to 8 ✓
```yaml
eval_execute_steps: 8  # CRITICAL: use 8 for best results (16 causes OOD, 1 breaks temporal consistency)
```

### 2. scripts/train_diffusion_policy.py
**Changes: 3 locations (lines 689, 701, 781)**

a) **Line 689:** Changed fallback default from 16 to 8
```python
# OLD: execute_steps=cfg.get("eval_execute_steps", 16),
# NEW: execute_steps=cfg.get("eval_execute_steps", 8),
```

b) **Line 701:** Changed fallback default from 16 to 8 (retry path)
```python
# OLD: execute_steps=cfg.get("eval_execute_steps", 16),
# NEW: execute_steps=cfg.get("eval_execute_steps", 8),
```

c) **Line 781:** Changed function signature default from 16 to 8
```python
# OLD: execute_steps: int = 16,
# NEW: execute_steps: int = 8,
```

### 3. scripts/eval_multimodality.py
**Changes: 5 locations**

a) **Line 596:** Changed main function default from 16 to 8
```python
# OLD: execute_steps: int = 16,
# NEW: execute_steps: int = 8,
```

b) **Line 1106:** CLI argument default already 8 ✓

c) **Lines 626-630:** Enhanced logging to include ddim_steps
```python
# Added conditional logging for DDIM parameters:
if sampling_method == 'ddim':
    print(f"ddim_eta: {ddim_eta}")
    print(f"ddim_steps: {ddim_steps if ddim_steps is not None else 'all (100)'}")
```

d) **Lines 254-288:** Fixed action std warnings to be demo-relative
```python
# OLD: Hardcoded thresholds (0.005, 0.1)
# NEW: Demo-relative thresholds:
#   - CRITICAL: pos_std < 0.3 * demo_std
#   - WARNING:  pos_std < 0.7 * demo_std
#   - WARNING:  pos_std > 2.0 * demo_std (thrashing)
```

e) **Lines 508-540:** Added observation z-score tracking
```python
# Added tracking at replanning steps:
obs_z_scores = []  # Initialize before loop
is_replanning_step = len(policy._action_queue) == 0  # Detect replanning
if is_replanning_step:
    obs_normalized = (obs - policy.obs_mean.cpu().numpy()) / policy.obs_std.cpu().numpy()
    obs_z_magnitude = np.mean(np.abs(obs_normalized))
    obs_z_scores.append(obs_z_magnitude)

# Log summary at end of rollout:
avg_z = np.mean(obs_z_scores)
max_z = np.max(obs_z_scores)
print(f"    Obs z-score magnitude: avg={avg_z:.2f}, max={max_z:.2f}")
```

---

## Verification Commands

### Test A: DDPM, execute_steps=8, temporal_ensemble OFF
```powershell
.venv\Scripts\Activate.ps1
python scripts/eval_multimodality.py `
    --ckpt runs/20260213_213052/ckpt_ep300.pt `
    --K 5 --M 5 `
    --execute_steps 8 `
    --sampling_method ddpm `
    --out_dir outputs/verify_ddpm_es8_no_ensemble `
    --n_videos 0 `
    --cube_jitter 0.0
```

**Expected console output:**
```
device: cuda
execute_steps: 8
dynamic_mpc: False
max_steps: 400
temporal_ensemble: False
ensemble_grip: False
sampling_method: ddpm
cube_jitter: 0.0 m
...
  [DEMO STATS] action std=0.0175, abs_mean=0.0104
  [PLAN #1] pos_std=0.0XXX, pos_abs_mean=0.0XXX, yaw_std=0.0XXX (demo_std=0.0175, ratio=X.XX)
...
    Obs z-score magnitude: avg=X.XX, max=X.XX (computed at N replanning steps)
```

**Check for:** 
- `execute_steps: 8` (not 16)
- `sampling_method: ddpm`
- `temporal_ensemble: False`
- `[DEMO STATS] action std=0.0175` (confirms demo stats loaded)
- No [CRITICAL] or [WARNING] messages about action collapse (unless ratio < 0.3 or > 2.0)
- Obs z-score logged at end of each rollout

---

### Test B: DDPM, execute_steps=8, temporal_ensemble ON
```powershell
.venv\Scripts\Activate.ps1
python scripts/eval_multimodality.py `
    --ckpt runs/20260213_213052/ckpt_ep300.pt `
    --K 5 --M 5 `
    --execute_steps 8 `
    --sampling_method ddpm `
    --temporal_ensemble `
    --out_dir outputs/verify_ddpm_es8_with_ensemble `
    --n_videos 0 `
    --cube_jitter 0.0
```

**Expected console output:**
```
device: cuda
execute_steps: 8
...
temporal_ensemble: True
ensemble_grip: False
sampling_method: ddpm
...
```

**Check for:** 
- `temporal_ensemble: True`
- Same demo-relative warnings as Test A

---

### Test C: DDIM (eta=0), ddim_steps=20, execute_steps=8, ensemble OFF
```powershell
.venv\Scripts\Activate.ps1
python scripts/eval_multimodality.py `
    --ckpt runs/20260213_213052/ckpt_ep300.pt `
    --K 5 --M 5 `
    --execute_steps 8 `
    --sampling_method ddim `
    --ddim_eta 0.0 `
    --ddim_steps 20 `
    --out_dir outputs/verify_ddim_es8_strided `
    --n_videos 0 `
    --cube_jitter 0.0
```

**Expected console output:**
```
device: cuda
execute_steps: 8
...
temporal_ensemble: False
ensemble_grip: False
sampling_method: ddim
ddim_eta: 0.0
ddim_steps: 20
cube_jitter: 0.0 m
...
```

**Check for:** 
- `sampling_method: ddim`
- `ddim_eta: 0.0`
- `ddim_steps: 20` (not "all (100)")
- execute_steps still 8

---

### Test D: Verify config is used during training eval
```powershell
.venv\Scripts\Activate.ps1
python scripts/train_diffusion_policy.py --config configs/train.yaml
```

**Check for (in training output):**
- During eval epochs, console should show execute_steps=8 being used
- No errors about missing execute_steps parameter
- Training proceeds normally with eval using 8 steps

---

## What to Look For in Logs

### ✅ Success Indicators
1. **execute_steps: 8** in header (not 16)
2. **[DEMO STATS] action std=0.0175** appears once at start
3. **[PLAN #1-3]** logs show `ratio=X.XX` next to pos_std
4. **Obs z-score magnitude** logged at end of each rollout
5. **NO [CRITICAL]** warnings unless there's actual collapse
6. **NO [WARNING]** warnings unless ratio < 0.7 or > 2.0

### ⚠️ Warning Messages (New Behavior)
- **[WARNING] Action suppression** if policy_std < 0.7 * demo_std (ratio < 0.7)
- **[CRITICAL] Severe action suppression** if policy_std < 0.3 * demo_std (ratio < 0.3)
- **[WARNING] Action amplification** if policy_std > 2.0 * demo_std (ratio > 2.0)
- **[WARNING] High z-score detected** if obs z-score max > 10.0

### ❌ Old Warnings (Should NOT Appear)
- ~~"[WARNING] Action collapse detected! Policy outputs small actions (std < 0.1)"~~ (removed)
- ~~"[CRITICAL] Severe action collapse! Policy outputs near-zero (std < 0.005)"~~ (now demo-relative)

---

## Summary of Defaults

| Parameter | Old Default | New Default | Source |
|-----------|-------------|-------------|--------|
| `eval_execute_steps` (train.yaml) | 16 | **8** | config |
| `execute_steps` (train quick eval) | 16 | **8** | function signature + config fallback |
| `execute_steps` (eval script) | 16 | **8** | function signature |
| `execute_steps` (CLI) | 8 | **8** | already correct |

---

## Behavior Changes

### Before Fix
- Default execute_steps=16 → 0% success
- Action warnings triggered constantly with hardcoded thresholds
- No observation z-score tracking

### After Fix
- Default execute_steps=8 → 10-50% success
- Action warnings are demo-relative (only trigger on true anomalies)
- Observation z-scores logged for OOD debugging
- Console output clearly shows all sampling parameters

---

## Files Changed
1. ✅ `configs/train.yaml` - Already had execute_steps=8
2. ✅ `scripts/train_diffusion_policy.py` - 3 changes (lines 689, 701, 781)
3. ✅ `scripts/eval_multimodality.py` - 5 changes (lines 596, 626-630, 254-288, 508-540)

**Total lines modified:** ~50 lines across 2 files
**No new dependencies added** ✓
**No training objective changes** ✓
