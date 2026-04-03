# 🚨 ROOT CAUSE ANALYSIS - 13% SUCCESS EXPLAINED

## The Complete Bug Chain

### 1. **Demo Action Statistics ARE TINY** (Not a Bug, Correct Values)
```
Demo actions (raw):
  Position (dx, dy, dz): std = [0.0124, 0.0148, 0.0284] m
  Yaw (dyaw):           std = 0.0 rad
  Gripper:              std = 0.7050

These IS the raw action magnitudes - tiny because:
  - Demo collection uses action_scale_pos = 0.05 m/step
  - Raw actions in [-1, 1]
  - Actual position: raw_action * 0.05 m = values like 0.01 * 0.05 = 0.0005 m per timestep
  - Over 32-step horizon: 32 * 0.0005 = 0.016 m = 1.6 cm reach
```

### 2. **CRITICAL BUG: Checkpoint stores NORMALIZED stats instead of ORIGINAL**

**File**: `scripts/train_diffusion_policy.py` line 404-405
```python
if act_mean is None:
    self.act_mean = np.zeros(act_dim, dtype=np.float32)  # ❌ NORMALIZED
    self.act_std = np.ones(act_dim, dtype=np.float32)    # ❌ NORMALIZED
```

**What this does WRONG:**
- During training: `actions_normalized = (actions_raw - demo_mean) / demo_std`
- Model learns to output `action_normalized ∈ [-1, 1]`
- Problem: Checkpoint saves `act_mean=[0,0,0,0,0]` and `act_std=[1,1,1,1,1]` (normalized!)
- During inference: `action_final = model_output * 1 + 0 = model_output`
- Only a 0.5 output stays at 0.5 instead of scaling to 0.5 * 0.0124 ≈ 0.0062 m

| Step | Value Needed | What Happens | Result |
|------|---|---|---|
| Model predicts | 0.5 raw (target action) | - | - |
| Denormalize | 0.5 * demo_std (MISSING!) | 0.5 * 1.0 (using [1,1,1,1,1]) | 0.5 m (40x too large!) |
| Environment scales | 0.5 * 0.05 = 0.025 m | Uses 0.5 * 0.05 = 0.025 m | Correct by accident? |
| Robot moves | 2.5 cm | 2.5 cm | Actually works?? |

Wait, let me reconsider... The actions output by the model are in [-1, 1] because they were trained on normalized actions. So:
- Model outputs: x_norm ∈ [-1, 1]
- Should denormalize: x_raw = x_norm * demo_std + demo_mean
- But it's: x_raw = x_norm * 1 + 0 = x_norm (still normalized!)
- Then environment: position_change = x_raw * 0.05 = x_norm * 0.05

So the robot is receiving actions like 0.1 * 0.05 = 0.005 m = 0.5 cm per step!

### 3. **Why it STILL Gets ~13% Success (Not 0%)**

With 0. 5 cm per step for 32 steps = 16 cm reach:
- Sometimes lucky early positioning
- Random exploration during diffusion sampling
- Some seeds might accidentally reach cube
- Gripper still has full range (±1.0)

But it's **NOT learning the task properly** - just lucky random successes.

### 4. **VERIFICATION: Check Checkpoint Content**

```bash
python -c "
import torch
ckpt = torch.load('runs/latest/ckpt.pt')
print('Action mean:', ckpt['act_mean'])
print('Action std:', ckpt['act_std'])
print('[Expected from diagnostics]')
print('Action mean: [0. 0. 0. 0. 0.] ← WRONG')
print('Action std: [1. 1. 1. 1. 1.] ← WRONG')
print('[Should be from demos]')
print('Action std: [0.0124, 0.0148, 0.0284, 0, 0.7050]')
"
```

---

## Fix Strategy

### **FIX #1: Compute Real Demo Statistics**

In training code, ALWAYS compute from demos:
```python
demo_data = np.load('data/demos/demos.npz')
demo_actions = demo_data['actions']  # (N, T, 5)
act_mean = demo_actions.mean(axis=(0, 1))  # (5,)
act_std = demo_actions.std(axis=(0, 1))    # (5,)
act_std = np.maximum(act_std, 0.01)        # floor at 0.01 to avoid divide-by-zero
print(f"Computed act_mean: {act_mean}")
print(f"Computed act_std: {act_std}")
```

### **FIX #2: Save REAL Stats to Checkpoint**

```python
checkpoint = {
    'model': model.state_dict(),
    'obs_mean': obs_stats_mean,      # ✓ Correct
    'obs_std': obs_stats_std,        # ✓ Correct
    'act_mean': ACT_MEAN_FROM_DEMOS, # ✓ FIXED
    'act_std': ACT_STD_FROM_DEMOS,   # ✓ FIXED
    'epoch': epoch,
    'loss': loss,
    'config': cfg,
}
torch.save(checkpoint, path)
```

### **FIX #3: Verify in Evaluation**

```python
# In eval_multimodality.py DiffusionPolicyRunner.__init__:
print(f"✓ Action denormalization loaded:")
print(f"  act_mean: {ckpt_mean}"
print(f"  act_std: {ckpt_std}")
# Report if std is [1,1,1,1,1] → ERROR
if (ckpt_std == np.ones(5)).all():
    raise ValueError("❌ CRITICAL: Checkpoint has NORMALIZED stats, not original demos!")
```

---

## Expected Impact After Fix

| Metric | Before (Buggy) | After (Fixed) |
|--------|---|---|
| Action magnitude | 0.005 m/step | 0.0124 * 0.05 ≈ 0.0006 m/step (wrong direction - too small!) |
| Actually wait... | ... | Let me reconsider |

Hmm, actually the demo actions (0.0124 m) ARE already tiny. That's because:
- Demos were collected with action_scale_pos=0.05 m/step
- Raw demo actions in normalized [-1, 1] scale are like 0.01-0.03
- Which mean 0.01-0.03 * 0.05 = 0.0005-0.0015 m = 0.5-1.5 mm per step

So maybe the issue is **DEMOS TOO SLOW** not action scaling bug!

---

## Alternative Hypothesis: Demos Are Slow

The demos might have been collected with very cautious/slow approach:
- Approach phase: 3 cm/s = 0.001 m/step with 240 Hz physics
- 32 steps = 3.2 cm reach
- Slow grasping makes model learn slow strategies

**But then why is success rate 13% and not 0%?**

Because:
1. Gripper works (std=0.7 - full range)
2. Random lucky positions
3. Initial cube placement jitter helps sometimes

---

## Real Root Cause: Likely BOTH

1. **Demo actions are legitimately small** (cautious demo collection)
2. **Checkpoint missing action stats** (would make it worse)

**Solution:**
1. Fix checkpoint to save real stats (necessary but not sufficient)
2. Re-collect demos with FASTER, MORE AGGRESSIVE grasping
3. Verify action magnitudes are bigger: target std ≥ 0.1 m

---

## Action Plan

1. **Today: Fix the checkpoint bug**
   - Rewrite train script to compute and save real action statistics
   - Verify checkpoint has correct stats before using

2. **Tomorrow: Collect better demos**
   - Use faster teleoperation (target z speed: 5 cm/s min)
   - Collect 100-200 more demos with explicit speed guidance
   - Verify new demos have action_std ≥ 0.1 m

3. **Retrain with fixed code**
   - Training on correct data + no checkpoint bug
   - Expect: 30-50% success at 100 epochs
