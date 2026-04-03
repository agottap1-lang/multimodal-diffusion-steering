# Debug-to-Success Pipeline Guide

## Overview

This guide provides a systematic approach to debug and improve a diffusion policy experiencing action collapse (low action std ~0.01–0.06 vs demo std ~0.3–0.5) and poor success rates (~0–16%).

The pipeline includes:
1. **Comprehensive diagnosis** (diagnose_policy.py)
2. **Detailed evaluation logging** (eval_multimodality.py enhancements)
3. **Action scaling verification**
4. **Multimodal selection mode** (only after >50% success)

---

## Phase 1: Diagnosis

### Step 1.1: Run Comprehensive Policy Diagnosis

**Purpose:** Identify the root cause of action collapse by comparing policy output to demo statistics across multiple sampling configurations.

```bash
# Basic diagnosis
py scripts/diagnose_policy.py --ckpt runs/latest/ckpt_ep300.pt

# Specify demo file explicitly
py scripts/diagnose_policy.py --ckpt runs/latest/ckpt_ep300.pt --demos data/demos/demos.npz

# More samples for robust statistics
py scripts/diagnose_policy.py --ckpt runs/latest/ckpt_ep300.pt --n_samples 20

# CPU-only for deterministic testing
py scripts/diagnose_policy.py --ckpt runs/latest/ckpt_ep300.pt --cpu_only
```

**What it tests:**
- Demo action statistics (baseline)
- DDPM sampling (no temporal ensemble)
- DDPM sampling (with temporal ensemble)
- DDIM eta=0.0 (deterministic, no temporal ensemble)
- DDIM eta=0.0 (deterministic, with temporal ensemble)
- DDIM eta=0.3 (stochastic, no temporal ensemble)
- DDIM eta=0.3 (stochastic, with temporal ensemble)

**Expected output:**
```
STEP 1: DEMO ACTION STATISTICS
  Overall: std=0.3421, abs_mean=0.2134, range=[-0.987, +0.989]

STEP 3: POLICY SAMPLING TESTS
Testing: DDPM, ensemble=OFF
  std=0.3198, abs_mean=0.2087
  policy/demo std ratio = 0.93  ✅ Healthy

Testing: DDIM eta=0.0, ensemble=OFF
  std=0.0127, abs_mean=0.0084
  policy/demo std ratio = 0.04  ❌ SEVERE ACTION COLLAPSE

STEP 5: DIAGNOSTIC RECOMMENDATIONS
  ❌ DIAGNOSIS: DDIM IMPLEMENTATION BUG
     - DDPM produces healthy actions but DDIM collapses
     - Action: Debug p_sample_ddim() math (check sigma_t, dir_coef)
     - File: scripts/train_diffusion_policy.py
```

**Common diagnoses:**
1. **DDIM bug**: DDPM works but DDIM collapses → Fix p_sample_ddim() math
2. **Model/normalization issue**: Both DDPM and DDIM collapse → Check action scaling or retrain
3. **Temporal ensemble too aggressive**: Works without ensemble but collapses with it → Reduce ensemble_decay or disable
4. **All healthy but low success**: Issue is MPC strategy, not sampling → Tune execute_steps or try dynamic_mpc

---

## Phase 2: Detailed Evaluation

### Step 2.1: Run Evaluation with Scaling Verification

**Purpose:** Verify action scaling matches between demo collection and evaluation to catch scaling bugs.

```bash
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 5 --M 5 --n_videos 0 \
    --sampling_method ddim --ddim_eta 0.0 \
    --cube_jitter 0.0 --execute_steps 16 \
    --verify_scaling \
    --out_dir outputs/test_scaling
```

**What it checks:**
- Demo collection `action_scale_pos` (from metadata)
- Eval environment `action_scale_pos`
- Warns if mismatch detected

**Expected output:**
```
ACTION SCALING VERIFICATION
  Demo collection scaling:
    action_scale_pos: 0.0500 m/step
    action_scale_yaw: 15.0 deg/step

  Eval environment scaling:
    action_scale_pos: 0.0500 m/step
    action_scale_yaw: 15.0 deg/step

  ✓ Action scaling matches between demo collection and eval
```

**If mismatch detected:**
```
  ⚠️  WARNING: ACTION SCALING MISMATCH DETECTED!
     Demo pos scale: 0.0500 m/step
     Eval pos scale: 0.1000 m/step
     Ratio: 2.00x
     This will cause the robot to move faster/slower than demos!
```

**Action:** Update `TwoBlockPickEnv.action_scale_pos` to match demo collection setting.

---

### Step 2.2: Run Evaluation with Detailed Logging

**Purpose:** Log first 5 planned chunks and EE displacement per 10 steps to diagnose execution issues.

```bash
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 2 --M 2 --n_videos 2 \
    --sampling_method ddim --ddim_eta 0.0 \
    --cube_jitter 0.0 --execute_steps 16 \
    --log_chunks \
    --log_ee_displacement \
    --out_dir outputs/test_logging
```

**What it logs (per rollout, first 2 only):**
```
  env_seed=100  sample= 5000
    [t=  0, plan #1] chunk: std=0.3421, abs_mean=0.2134, range=[-0.987, +0.989]
    [t= 16, plan #2] chunk: std=0.3198, abs_mean=0.2087, range=[-0.952, +0.943]
    [t= 32, plan #3] chunk: std=0.3087, abs_mean=0.2001, range=[-0.921, +0.918]
    [t= 48, plan #4] chunk: std=0.2951, abs_mean=0.1923, range=[-0.887, +0.891]
    [t= 64, plan #5] chunk: std=0.2812, abs_mean=0.1845, range=[-0.851, +0.863]
    
    EE displacement (meters) per 10-step window:
      steps   0- 10: 0.4123 m
      steps  10- 20: 0.3987 m
      steps  20- 30: 0.3654 m
      steps  30- 40: 0.2981 m
      ...
```

**Use cases:**
- **Chunk std decreasing**: Temporal ensemble or dynamic MPC causing smoothing
- **Low EE displacement**: Robot barely moving → action collapse or scaling bug
- **High EE displacement**: Robot moving too fast → check action_scale_pos

---

### Step 2.3: Test Temporal Ensemble Toggle

**Purpose:** Isolate whether temporal ensemble is causing action collapse or poor performance.

```bash
# Without temporal ensemble (default in updated code)
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 5 --M 5 --n_videos 0 \
    --sampling_method ddim --ddim_eta 0.0 \
    --cube_jitter 0.0 --execute_steps 16 \
    --out_dir outputs/no_ensemble

# With temporal ensemble (explicitly enable)
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 5 --M 5 --n_videos 0 \
    --sampling_method ddim --ddim_eta 0.0 \
    --cube_jitter 0.0 --execute_steps 16 \
    --temporal_ensemble \
    --out_dir outputs/with_ensemble
```

**Compare:**
- Success rates
- Action std in logged chunks (if using --log_chunks)
- Smoothness of trajectories in videos

**Decision:**
- If ensemble helps (higher success): Keep it enabled
- If ensemble hurts (lower success, action collapse): Disable or reduce ensemble_decay

---

## Phase 3: Iteration

### Step 3.1: Fix Identified Issues

Based on diagnosis results:

**Issue: DDIM implementation bug**
- Fix: Update `p_sample_ddim()` in `scripts/train_diffusion_policy.py`
- Verify: Rerun diagnosis script, check DDIM std matches DDPM
- Test: Full eval with fixed DDIM

**Issue: Action scaling mismatch**
- Fix: Update `TwoBlockPickEnv.action_scale_pos` or recollect demos
- Verify: Rerun with --verify_scaling
- Test: Full eval with matched scaling

**Issue: Normalization stats incorrect**
- Fix: Retrain model or recompute normalization from demos
- Verify: Check act_std in checkpoint diagnostics
- Test: Full eval with corrected normalization

**Issue: Temporal ensemble too aggressive**
- Fix: Set ensemble_decay=0.9 (from 0.7) or disable
- Verify: Rerun with different ensemble_decay values
- Test: Compare success rates

---

### Step 3.2: Baseline Deterministic Evaluation

**Goal:** Achieve >50% success in deterministic mode before attempting multimodal selection.

```bash
# Deterministic DDIM (eta=0.0)
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 10 --M 1 --n_videos 10 \
    --sampling_method ddim --ddim_eta 0.0 \
    --cube_jitter 0.0 --execute_steps 16 \
    --out_dir outputs/deterministic_baseline \
    --video_dir outputs/deterministic_baseline/videos
```

**Expected after fixes:**
- Success rate: 50-70%
- Action std: >0.25 (comparable to demos)
- No action collapse warnings
- Clean, smooth trajectories in videos

**If success <50%:**
1. Rerun diagnosis script to check for remaining issues
2. Try different execute_steps values (1, 2, 4, 8, 16)
3. Try dynamic_mpc mode
4. Check train/eval jitter mismatch with --cube_jitter

---

## Phase 4: Multimodal Selection (After >50% Success Only!)

### Step 4.1: Enable Multimodal Selection with Placeholder Value Function

**Purpose:** Sample multiple diverse strategies per replan and select best according to a value function.

**⚠️ WARNING:** Only use after achieving >50% success in deterministic mode! Otherwise, all candidates will have action collapse and selection won't help.

```bash
# Multimodal selection with DDIM eta=0.3 (stochastic)
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 10 --M 10 --n_videos 10 \
    --sampling_method ddim --ddim_eta 0.3 \
    --cube_jitter 0.0 --execute_steps 16 \
    --multimodal_selection \
    --n_candidates 5 \
    --out_dir outputs/multimodal_selection \
    --video_dir outputs/multimodal_selection/videos
```

**What happens:**
1. At each replan, sample 5 candidate chunks with different diffusion seeds
2. Score each with `_value_function()` (currently: prefers moderate action std)
3. Pick highest-scoring candidate
4. Execute first k steps, repeat

**Current placeholder value function:**
```python
def _value_function(self, obs: np.ndarray, chunk: np.ndarray) -> float:
    # Prefers chunks with action std ~0.35 (like demos)
    chunk_std = chunk[:, :4].std()
    target_std = 0.35
    score = -abs(chunk_std - target_std)
    return float(score)
```

**Expected output:**
```
⚠️  MULTIMODAL SELECTION MODE ENABLED
  This mode samples 5 candidate chunks per replan and selects
  the best according to a value function.
  
  CURRENT: Using placeholder value function (prefers moderate action std)
  TODO: Replace _value_function() with VLM or learned value model
  
  NOTE: Only use this after achieving >50% success in deterministic mode!

...

[MULTIMODAL SELECT] plan #1: sampled 5 candidates, picked #2 (score=-0.012)
  scores: ['-0.087', '-0.012', '-0.134', '-0.098', '-0.156']
```

---

### Step 4.2: Replace Placeholder with Real Value Function

**Where:** `scripts/eval_multimodality.py`, line ~320 in `DiffusionPolicyRunner._value_function()`

**Option 1: VLM-based scoring (e.g., CLIP similarity to goal description)**

```python
def _value_function(self, obs: np.ndarray, chunk: np.ndarray) -> float:
    # Extract current state
    ee_pos = obs[:3]
    left_cube = obs[8:11]
    right_cube = obs[15:18]
    
    # Simulate chunk effects (rough estimate)
    final_ee = ee_pos + chunk[:16, :3].sum(axis=0) * 0.05  # action_scale_pos
    
    # Score based on distance to both cubes (prefer closer approach)
    dist_left = np.linalg.norm(final_ee - left_cube)
    dist_right = np.linalg.norm(final_ee - right_cube)
    score = -min(dist_left, dist_right)  # Prefer chunk that brings EE closer
    
    return float(score)
```

**Option 2: Learned value function (trained Q-network)**

```python
def _value_function(self, obs: np.ndarray, chunk: np.ndarray) -> float:
    # Load pretrained value network
    if not hasattr(self, '_value_net'):
        self._value_net = torch.load('value_net.pt').to(self.device)
        self._value_net.eval()
    
    # Encode obs + chunk
    obs_t = torch.from_numpy(obs).float().to(self.device)
    chunk_t = torch.from_numpy(chunk).float().to(self.device)
    state_action = torch.cat([obs_t, chunk_t.flatten()])
    
    # Predict value
    with torch.no_grad():
        value = self._value_net(state_action)
    
    return float(value.item())
```

**Option 3: VLM integration (CLIP or similar)**

```python
def _value_function(self, obs: np.ndarray, chunk: np.ndarray) -> float:
    # Render trajectory from obs + chunk (requires env simulator)
    frames = self._simulate_chunk(obs, chunk)  # Returns list of RGB frames
    
    # Encode with CLIP vision encoder
    clip_features = self._clip_encoder(frames)
    
    # Compare to goal description embedding
    goal_text = "robot grasps the left cube smoothly"
    text_features = self._clip_text_encoder(goal_text)
    
    # Compute similarity
    score = torch.cosine_similarity(clip_features, text_features)
    
    return float(score.item())
```

---

## Usage Examples

### Example 1: Full Debug Workflow (From Scratch)

```bash
# 1. Diagnose the issue
py scripts/diagnose_policy.py --ckpt runs/latest/ckpt_ep300.pt --n_samples 10

# 2. Verify action scaling
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 5 --M 5 --n_videos 0 \
    --verify_scaling \
    --out_dir outputs/verify

# 3. Detailed logging to isolate issues
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 2 --M 2 --n_videos 2 \
    --log_chunks --log_ee_displacement \
    --out_dir outputs/detailed

# 4. Test without temporal ensemble
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 5 --M 5 --n_videos 0 \
    --out_dir outputs/no_ensemble

# 5. Fix identified issues, then baseline evaluation
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 10 --M 1 --n_videos 10 \
    --sampling_method ddim --ddim_eta 0.0 \
    --out_dir outputs/baseline

# 6. If success >50%, try multimodal selection
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 10 --M 10 --n_videos 10 \
    --sampling_method ddim --ddim_eta 0.3 \
    --multimodal_selection --n_candidates 5 \
    --out_dir outputs/multimodal
```

---

### Example 2: Quick Smoke Test (After Training)

```bash
# Quick diagnosis to check if model is healthy
py scripts/diagnose_policy.py --ckpt runs/latest/ckpt_ep300.pt --n_samples 5

# If diagnosis looks good, run quick eval
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 5 --M 5 --n_videos 5 \
    --out_dir outputs/smoke
```

---

### Example 3: Systematic Hyperparameter Sweep

```bash
# Test different execute_steps values
for steps in 1 2 4 8 16; do
    py scripts/eval_multimodality.py \
        --ckpt runs/latest/ckpt_ep300.pt \
        --K 5 --M 5 --n_videos 0 \
        --execute_steps $steps \
        --out_dir outputs/sweep_steps_$steps
done

# Test different ddim_eta values
for eta in 0.0 0.1 0.3 0.5 1.0; do
    py scripts/eval_multimodality.py \
        --ckpt runs/latest/ckpt_ep300.pt \
        --K 5 --M 5 --n_videos 0 \
        --sampling_method ddim --ddim_eta $eta \
        --out_dir outputs/sweep_eta_$eta
done
```

---

## Troubleshooting

### Issue: "Action collapse detected" warnings

**Cause:** Policy outputs tiny actions (std << demo std)

**Solutions:**
1. Run diagnosis script to identify root cause (DDIM bug vs model issue)
2. Verify action scaling matches (--verify_scaling)
3. Check normalization stats in checkpoint
4. If DDIM-specific, fix p_sample_ddim() math
5. If model-wide, retrain with correct configuration

---

### Issue: Low success rate despite healthy action magnitudes

**Cause:** MPC execution strategy suboptimal

**Solutions:**
1. Try different execute_steps values (1, 2, 4, 8, 16)
2. Enable dynamic_mpc for proximity-based replanning
3. Check train/eval jitter mismatch (--cube_jitter)
4. Increase training epochs (model undertrained)

---

### Issue: Multimodal selection doesn't improve performance

**Causes:**
1. Deterministic baseline <50% success (all candidates collapsed)
2. Value function not discriminative (placeholder is too simple)
3. n_candidates too low (not enough diversity)
4. ddim_eta too low (candidates not diverse enough)

**Solutions:**
1. Fix baseline performance first (must be >50%)
2. Replace placeholder value function with real VLM/learned model
3. Increase n_candidates to 8-10
4. Try ddim_eta=0.5 or sampling_method=ddpm

---

## Key Flags Reference

### diagnose_policy.py

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ckpt` | str | required | Path to checkpoint file |
| `--demos` | str | from config | Path to demo file |
| `--n_samples` | int | 10 | Number of samples per configuration |
| `--cpu_only` | flag | False | Force CPU-only execution |

### eval_multimodality.py (New Flags)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--log_chunks` | flag | False | Log first 5 planned chunks' stats |
| `--log_ee_displacement` | flag | False | Log EE displacement per 10-step window |
| `--verify_scaling` | flag | False | Verify action scaling matches demos |
| `--multimodal_selection` | flag | False | Enable multimodal selection mode |
| `--n_candidates` | int | 5 | Number of candidates when selection enabled |
| `--temporal_ensemble` | flag | False | Enable temporal ensemble (changed default) |

**Note:** `--temporal_ensemble` is now opt-in (was opt-out). This prevents accidental action smoothing during debugging.

---

## Summary of Changes

1. **diagnose_policy.py** (NEW): Comprehensive diagnosis script comparing policy vs demos across 6 configurations
2. **eval_multimodality.py**:
   - Added `--log_chunks` flag and logging in rollout()
   - Added `--log_ee_displacement` flag and EE tracking
   - Added `--verify_scaling` flag and action scaling verification
   - Added `--multimodal_selection` flag and candidate selection logic
   - Added `--n_candidates` parameter
   - Changed `--temporal_ensemble` from default True to default False (opt-in)
3. **Multimodal selection** infrastructure:
   - `_value_function()` placeholder (ready for VLM integration)
   - `_plan_with_selection()` samples N candidates, picks best
   - Properly gated behind flag with warnings

**Philosophy:**
- Minimal, surgical changes
- Config-driven (everything behind flags)
- Clear diagnostic output
- Systematic debug-to-success workflow
- Ready for future VLM integration (placeholder value function)