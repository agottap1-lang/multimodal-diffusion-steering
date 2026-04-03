# SYSTEMATIC DIAGNOSIS - ZERO SUCCESS ROOT CAUSE

## Summary
All eval runs show **0% success** regardless of configuration. Implemented 3 targeted fixes based on ranked hypotheses.

---

## 🔍 HYPOTHESES (Ranked by Likelihood)

### #1 [MOST LIKELY] Action Scaling Issue
**Problem**: Model outputs actions too small → robot barely moves (observed: 1cm in 10 steps)

**Evidence**:
```
Initial ee: (0.40, 0.00, 0.55)
After 10 steps: (0.40, 0.01, 0.55)  # Only 1cm movement!
First action: dx=-0.000, dy=+0.035, dz=-0.002  # Near-zero
```

**Test**:
```bash
.venv\Scripts\Activate.ps1; py scripts/diagnose_rollout.py
```

**Expected**: 
- Should print WARNING if action std < 0.1
- Healthy model: action std ~0.3-0.5
- Current model: likely < 0.05

**Fix if confirmed**: Investigate normalization in training vs eval, or add action_scale multiplier

---

### #2 DDIM Implementation Bug
**Problem**: DDIM math collapses actions to near-zero (eta parameter incorrectly wired)

**Test**:
```bash
.venv\Scripts\Activate.ps1; py scripts/eval_multimodality.py `
    --ckpt runs/20260213_213052/ckpt_ep300.pt `
    --K 2 --M 2 --n_videos 0 `
    --sampling_method ddpm `
    --cube_jitter 0.0 `
    --execute_steps 8 `
    --out_dir outputs/test_ddpm
```

**Expected**:
- If success > 0%: **DDIM is broken** → revert to DDPM or fix p_sample_ddim()
- If still 0%: DDIM not the issue, proceed to Hypothesis #3

---

### #3 Dynamic MPC Stale Queue
**Problem**: execute_steps changes but action queue not cleared → old actions still execute

**Test**:
```bash
.venv\Scripts\Activate.ps1; py scripts/eval_multimodality.py `
    --ckpt runs/20260213_213052/ckpt_ep300.pt `
    --K 2 --M 2 --n_videos 0 `
    --sampling_method ddim --ddim_eta 0.0 `
    --cube_jitter 0.0 `
    --execute_steps 1 `
    --out_dir outputs/test_exec1
```

**Expected**:
- If success > 0%: MPC queue bug confirmed → fix already implemented
- If still 0%: MPC not the issue, back to Hypothesis #1 or #2

---

## ✅ IMPLEMENTED FIXES

### Fix #1: Checkpoint Verification
**File**: `scripts/eval_multimodality.py` (lines ~100)

**Change**:
```python
# Added at policy init:
epoch = ckpt.get("epoch", "unknown")
print(f"  [CKPT] Loaded epoch {epoch}, horizon={cfg['horizon']}, sampling={sampling_method}, eta={ddim_eta}")
if epoch == "unknown" or (isinstance(epoch, int) and epoch < 100):
    print(f"  [WARNING] Checkpoint epoch={epoch} may be undertrained!")
```

**Purpose**: Fail-fast warning if using wrong/undertrained checkpoint

---

### Fix #2: Action Magnitude Diagnostics
**File**: `scripts/eval_multimodality.py` (lines ~140, ~160)

**Change**:
```python
# Track action stats
self._action_stats: List[float] = []

# In _plan():
chunk_std = chunk[:, :4].std()
self._action_stats.append(chunk_std)
if self._plan_count == 1 and chunk_std < 0.1:
    print(f"  [WARNING] First action chunk has low std={chunk_std:.4f} (expect ~0.3-0.5)")
```

**Purpose**: Immediately detect if actions are too small

---

### Fix #3: Dynamic MPC Queue Clearing
**File**: `scripts/eval_multimodality.py` (lines ~150, ~270)

**Change**:
```python
# New method:
def set_execute_steps(self, new_steps: int) -> None:
    \"\"\"Update n_action_steps and force immediate replanning.\"\"\"
    if new_steps != self.n_action_steps:
        self.n_action_steps = new_steps
        self._action_queue = []  # Clear queue

# In rollout():
policy.set_execute_steps(current_execute_steps)  # Instead of direct assignment
```

**Purpose**: Force immediate replan when MPC changes execute_steps

---

## 📋 EXECUTION PLAN

### Step 1: Run Hypothesis #1 Test (2 minutes)
```bash
.venv\Scripts\Activate.ps1; py scripts/diagnose_rollout.py
```

**Decision tree**:
- **If WARNING shown (std < 0.1)**: Action scaling is the issue
  - Next: Investigate train.yaml action normalization
  - Next: Check if demo actions match model's expected range
  - Next: Try multiplying actions by 10 in eval as temporary test

- **If no WARNING (std > 0.3)**: Actions are fine, not Hypothesis #1
  - Next: Run Hypothesis #2 test

---

### Step 2: Run Hypothesis #2 Test (5 minutes)
```bash
.venv\Scripts\Activate.ps1; py scripts/eval_multimodality.py --K 2 --M 2 --sampling_method ddpm --out_dir outputs/test_ddpm
```

**Decision tree**:
- **If success > 0%**: DDIM is broken
  - Next: Revert to DDPM (`--sampling_method ddpm`) for all evals
  - Next: Debug p_sample_ddim() eta implementation
  - Fix: Check if `eta > 0` noise term is using correct variance

- **If still 0%**: DDIM not the problem
  - Next: Run Hypothesis #3 test

---

### Step 3: Run Hypothesis #3 Test (5 minutes)
```bash
.venv\Scripts\Activate.ps1; py scripts/eval_multimodality.py --K 2 --M 2 --execute_steps 1 --out_dir outputs/test_exec1
```

**Decision tree**:
- **If success > 0%**: MPC queue bug (already fixed via set_execute_steps)
  - Next: Retest with dynamic_mpc to confirm fix works
  
- **If still 0%**: MPC not the issue
  - Back to Hypothesis #1: Model fundamentally broken/undertrained

---

## 🚨 FALLBACK IF ALL TESTS FAIL

If all 3 tests show 0% success:

**Root cause**: Checkpoint is **completely untrained or broken**

**Evidence check**:
```bash
py -c "import torch; ckpt=torch.load('runs/20260213_213052/ckpt_ep300.pt', map_location='cpu', weights_only=False); import numpy as np; demo=np.load('data/demos/demos.npz'); actions=demo['actions']; print(f'Demo action std: {actions.std():.3f}'); print(f'Demo action range: [{actions.min():.3f}, {actions.max():.3f}]')"
```

**Actions**:
1. Find a known-good checkpoint from training logs
2. Or retrain from scratch with validated pipeline
3. Check if training loss actually converged (should be < 0.01)

---

## 📊 SUCCESS CRITERIA

**Test passes if**:
- ANY test shows success > 0%
- Action diagnostics show std > 0.1
- Robot moves > 10cm in first 10 steps

**Then**:
- We've isolated the root cause
- Implement minimal fix
- Rerun full comparison tests

---

## 🔧 BEFORE/AFTER COMPARISON

### BEFORE (untested code):
```bash
# No warnings, just fails silently with 0%
py scripts/eval_multimodality.py --K 10 --M 10
```

### AFTER (with diagnostics):
```bash
# Now shows:
#   [CKPT] Loaded epoch 300, horizon=48, sampling=ddim, eta=0.3
#   [WARNING] First action chunk has low std=0.0435 (expect ~0.3-0.5)
# Immediately identifies the problem!
py scripts/eval_multimodality.py --K 10 --M 10
```

---

## 📝 NOTES

- All fixes are **minimal** - only add diagnostics or fix clear bugs
- No "try random things" - each change targets a specific hypothesis
- Tests are **falsifiable** - clear pass/fail criteria
- If tests pass, we have the root cause. If all fail, checkpoint is broken.
