# Fix Summary: Training/Eval Sampling Mismatch & Action Collapse Diagnostics

## Overview
Added config-driven sampling for train-time eval, enhanced diagnostics for action collapse detection, and created a comprehensive DDIM vs DDPM sanity check script.

## Changes Made

### 1. Config-driven sampling for train-time quick eval

#### `configs/train.yaml` (lines 33-35, new):
```yaml
# Evaluation
eval_epochs: [100, 200, 300, 400, 500]
eval_max_steps: 400
eval_execute_steps: 16
eval_sampling_method: "ddim"   # "ddpm" or "ddim" for train-time quick eval
eval_ddim_eta: 0.0             # deterministic during training eval (0.0); use 0.3+ for multimodality testing

# Paths
demo_path: data/demos/demos.npz
ckpt_dir: runs
```

**Key points:**
- `eval_sampling_method`: "ddim" (default) or "ddpm"
- `eval_ddim_eta`: 0.0 for deterministic training eval, can be changed to 0.3+ for multimodality
- Easy fallback to DDPM by changing eval_sampling_method to "ddpm"

#### `scripts/train_diffusion_policy.py` - Updated `_quick_sim_eval()` signature (lines 729-750):
```python
def _quick_sim_eval(ckpt_path: str, device: torch.device,
                    K: int = 5, M: int = 10,
                    max_steps: int = 400,
                    video_dir: str | None = None,
                    n_videos: int = 3,
                    execute_steps: int = 16,
                    sampling_method: str = 'ddim',      # NEW
                    ddim_eta: float = 0.0) -> dict:     # NEW
    """Run a quick sim evaluation and return metrics dict.
    
    Args:
        sampling_method: 'ddpm' or 'ddim' (default: 'ddim')
        ddim_eta: DDIM stochasticity parameter (0.0=deterministic)
    """
    from scripts.eval_multimodality import DiffusionPolicyRunner, rollout
    from envs.twoblockpick_env import TwoBlockPickEnv

    policy = DiffusionPolicyRunner(ckpt_path, device, 
                                    temporal_ensemble=True,
                                    ensemble_decay=0.7,
                                    sampling_method=sampling_method,    # CHANGED
                                    ddim_eta=ddim_eta)                  # CHANGED
    env = TwoBlockPickEnv(render=False, episode_length=max_steps)
```

#### `scripts/train_diffusion_policy.py` - Updated calls in `train()` (lines 646 & 653):
```python
# First call (with videos)
metrics = _quick_sim_eval(str(ckpt_path), device, K=5, M=10,
                          max_steps=cfg.get("eval_max_steps", 400),
                          video_dir=str(video_dir),
                          n_videos=5,
                          execute_steps=cfg.get("eval_execute_steps", 16),
                          sampling_method=cfg.get("eval_sampling_method", "ddim"),  # NEW
                          ddim_eta=cfg.get("eval_ddim_eta", 0.0))                   # NEW

# Second call (without videos, after exception)
metrics = _quick_sim_eval(str(ckpt_path), device, K=5, M=10,
                          max_steps=cfg.get("eval_max_steps", 400),
                          video_dir=None,
                          n_videos=0,
                          execute_steps=cfg.get("eval_execute_steps", 16),
                          sampling_method=cfg.get("eval_sampling_method", "ddim"),  # NEW
                          ddim_eta=cfg.get("eval_ddim_eta", 0.0))                   # NEW
```

### 2. Enhanced action collapse diagnostics with demo comparison

#### `scripts/eval_multimodality.py` - DiffusionPolicyRunner.__init__() (lines 96-182):
```python
def __init__(self, ckpt_path: str, device: torch.device, ...):
    # ... existing code ...
    
    # FIX #2: Track action statistics for diagnosis
    self._action_stats: List[float] = []  # std of each planned chunk
    
    # FIX #3: Load demo action statistics for comparison  # NEW BLOCK
    self._demo_std: float | None = None
    self._demo_abs_mean: float | None = None
    demo_path = cfg.get("demo_path", "data/demos/demos.npz")
    try:
        if Path(demo_path).exists():
            demo_data = np.load(demo_path, allow_pickle=True)
            demo_actions = demo_data["actions"]  # (N, T, act_dim)
            # Compute statistics over position deltas (dx, dy, dz, dyaw)
            all_pos_actions = demo_actions[:, :, :4].reshape(-1, 4)
            self._demo_std = float(all_pos_actions.std())
            self._demo_abs_mean = float(np.abs(all_pos_actions).mean())
            print(f"  [DEMO STATS] action std={self._demo_std:.4f}, abs_mean={self._demo_abs_mean:.4f}")
    except Exception as e:
        print(f"  [WARNING] Could not load demo stats: {e}")
```

#### `scripts/eval_multimodality.py` - DiffusionPolicyRunner._plan() (lines 200-227):
```python
def _plan(self, obs: np.ndarray) -> None:
    # ... sampling code ...
    
    # FIX #2 & #3: Action magnitude diagnostics with demo comparison
    chunk_std = chunk[:, :4].std()  # std of position deltas
    chunk_abs_mean = np.abs(chunk[:, :4]).mean()
    self._action_stats.append(chunk_std)
    
    # Print diagnostics for first 3 plans or if action collapse detected
    should_print = (self._plan_count < 3) or (chunk_std < 0.1)
    
    if should_print:
        ratio_str = ""
        if self._demo_std is not None and self._demo_std > 0:
            ratio = chunk_std / self._demo_std
            ratio_str = f" (demo_std={self._demo_std:.4f}, ratio={ratio:.2f})"
        
        print(f"  [PLAN #{self._plan_count + 1}] chunk_std={chunk_std:.4f}, abs_mean={chunk_abs_mean:.4f}{ratio_str}")
        
        # Warning for action collapse
        if chunk_std < 0.1:
            print(f"  [WARNING] Action collapse detected! Policy outputs very small actions.")
            if self._demo_std is not None and chunk_std < self._demo_std * 0.3:
                print(f"  [WARNING] Policy std is {chunk_std/self._demo_std:.1%} of demo std!")
                print(f"            Likely causes: (1) action scaling bug, (2) DDIM implementation bug,")
                print(f"            (3) model collapse, or (4) normalization mismatch.")
            print(f"            Action range: [{chunk.min():.3f}, {chunk.max():.3f}]")
    
    self._plan_count += 1
    # ... rest of method ...
```

### 3. New comprehensive DDIM vs DDPM sanity check script

#### `scripts/test_sampling_methods.py` (NEW FILE, 266 lines):

**Purpose:** Unit-test-like script to diagnose sampling method issues without full rollouts.

**Tests performed:**
1. DDPM (baseline stochastic)
2. DDIM eta=1.0 (should match DDPM stochasticity)
3. DDIM eta=0.0 (deterministic, reproducibility check)
4. DDIM eta=0.3 (controlled stochasticity for multimodality)

**Checks:**
- ✅ Action collapse detection (std < 0.05 = severe, < 0.1 = mild)
- ✅ DDIM determinism (eta=0.0 should give identical samples)
- ✅ DDIM stochasticity (eta>0 should vary across seeds)
- ✅ Finite outputs (no NaN/inf)
- ✅ Comparative diagnostics (DDPM vs DDIM)

**Usage:**
```bash
# Standard test (uses GPU if available)
py scripts/test_sampling_methods.py --ckpt runs/latest/ckpt_ep300.pt

# CPU-only for deterministic testing
py scripts/test_sampling_methods.py --ckpt runs/latest/ckpt_ep300.pt --cpu_only
```

**Output format:**
```
Testing: DDPM (baseline)
  Sample 0: std=0.3421, abs_mean=0.2134
  ✅ Healthy action magnitude

Testing: DDIM eta=0.0 (deterministic)
  Determinism check: max_diff=1.23e-08
  ✅ PASS: Deterministic

SUMMARY COMPARISON
Method                          std  abs_mean  Status
--------------------------------------------------------------------------------
DDPM (baseline)              0.3421    0.2134  ✅ HEALTHY
DDIM eta=1.0 (like DDPM)     0.3198    0.2087  ✅ HEALTHY
DDIM eta=0.0 (deterministic) 0.3156    0.2043  ✅ HEALTHY
DDIM eta=0.3 (multimodal)    0.3287    0.2109  ✅ HEALTHY

DIAGNOSTIC RECOMMENDATIONS:
  ✅ All methods producing healthy action magnitudes
     Next: Run full evaluation with multimodality testing
```

## Key Design Decisions

### 1. Why NOT "must retrain to learn DDIM"?
**Reasoning:** The model predicts noise (epsilon), not actions directly. DDIM vs DDPM is a **sampling-time choice** that uses the same noise predictions differently. The model doesn't "learn" DDIM or DDPM — it learns to denoise. Only the inference trajectory differs.

**When retraining IS needed:**
- Changing data distribution
- Changing architecture
- Changing normalization scheme
- Changing horizon or action dim

**When retraining is NOT needed:**
- Switching DDIM ↔ DDPM at inference
- Changing ddim_eta parameter
- Changing execute_steps or other MPC parameters

### 2. Minimal, surgical changes
- ✅ No refactoring of unrelated code
- ✅ Backward compatible (defaults to deterministic DDIM)
- ✅ Easy fallback to DDPM via config
- ✅ No changes to training loss or objective

### 3. Diagnostic philosophy
Target the REAL failure modes:
- **Action collapse** (std << demo_std) → likely scaling/normalization bug
- **DDIM determinism failure** → DDIM implementation bug
- **DDPM works but DDIM fails** → DDIM-specific math error

## Testing Protocol

### Step 1: Sanity check existing checkpoint
```bash
py scripts/test_sampling_methods.py --ckpt runs/20260213_213052/ckpt_ep300.pt --cpu_only
```
Expected: Diagnose whether DDIM or model is the issue.

### Step 2: Start training with fixed config
```bash
py scripts/train_diffusion_policy.py --config configs/train.yaml
```
Training will now use DDIM eta=0.0 for deterministic train-time eval.

### Step 3: After training to epoch 300, test multimodality
```bash
# Deterministic baseline (should have good success rate but no multimodality)
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 10 --M 1 --n_videos 0 \
    --sampling_method ddim --ddim_eta 0.0 \
    --cube_jitter 0.0 --execute_steps 16 \
    --out_dir outputs/ep300_ddim_det

# Multimodal test (should show diverse L/R strategies)
py scripts/eval_multimodality.py \
    --ckpt runs/latest/ckpt_ep300.pt \
    --K 10 --M 10 --n_videos 10 \
    --sampling_method ddim --ddim_eta 0.3 \
    --cube_jitter 0.0 --execute_steps 16 \
    --temporal_ensemble --dynamic_mpc \
    --out_dir outputs/ep300_multimodal \
    --video_dir outputs/ep300_multimodal/videos
```

### Step 4: If action collapse persists, use diagnostics
The enhanced diagnostics will now print:
```
[DEMO STATS] action std=0.3421, abs_mean=0.2134
[PLAN #1] chunk_std=0.0127, abs_mean=0.0084 (demo_std=0.3421, ratio=0.04)
[WARNING] Action collapse detected! Policy outputs very small actions.
[WARNING] Policy std is 4% of demo std!
          Likely causes: (1) action scaling bug, (2) DDIM implementation bug,
          (3) model collapse, or (4) normalization mismatch.
```

## Files Modified

1. **configs/train.yaml** - Added eval_sampling_method and eval_ddim_eta
2. **scripts/train_diffusion_policy.py** - Updated _quick_sim_eval() to accept and use config params
3. **scripts/eval_multimodality.py** - Added demo stats loading and enhanced diagnostics in _plan()
4. **scripts/test_sampling_methods.py** - NEW comprehensive DDIM vs DDPM test script

## Summary

**Problem:** Training used wrong sampler (DDPM) while eval expected DDIM, causing distribution mismatch and making multimodality untestable.

**Solution:** Made train-time eval sampling configurable via train.yaml, added demo comparison diagnostics to catch action collapse early, and created unit-test-like script to isolate DDIM bugs from model issues.

**Philosophy:** Minimal changes, maximal diagnostics. Fix the config mismatch and add tools to catch the real failure mode (action collapse) quickly.
