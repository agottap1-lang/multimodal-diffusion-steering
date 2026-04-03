# Understanding VLM-Guided vs Baseline Diffusion Policy

## Problem Summary

**Observation**: 
- ✅ Baseline succeeds: `success=True, picked=RIGHT, arc=0.2365`
- ❌ VLM-Guided fails: `success=False, picked=NONE, arc=0.5748` (even with legibility=0.900)

**Question**: Why does VLM-guided rollout fail when it selects a trajectory with good legibility and arc-15-19?

---

## Part 1: How Baseline Diffusion Policy Works

### Code Location: `rollout_baseline()` (lines 140-203)

```python
# LINE 162-165: Initialize environment
env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
obs = env.reset(seed=episode_seed)  # Reset with SAME seed every time
```
**Purpose**: Create environment with specific seed for reproducibility.
**Key**: Same seed → same initial block positions

---

```python
# LINE 167: Create action queue
action_queue = deque(maxlen=model.horizon)  # horizon = 8 actions
```
**Purpose**: Buffer to store upcoming actions
**Why**: Policy generates 8 actions at once (called "horizon"), but executes 1 action per timestep

---

```python
# LINE 173-180: Main execution loop - REPLANNING
while not done and steps < max_steps:
    if len(action_queue) == 0:  # Queue is empty → time to replan
        seq = sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=1.0)
        if first_seq is None:
            first_seq = seq.copy()  # Save first sequence for arc measurement
        for action in seq:
            action_queue.append(action)
```

**Purpose**: Generate new 8-action sequence when queue empties
**Key Insight - REPLANNING**: 
- Every 8 steps, policy looks at current observation and generates NEW 8 actions
- This is CLOSED-LOOP control: policy adapts to actual robot state
- If robot drifts off course, next replan corrects it

**Why it works**:
```
Step 0-7:   Use actions from 1st sample (based on initial obs)
Step 8-15:  Use actions from 2nd sample (based on obs at step 8) ← CORRECTION HAPPENS HERE
Step 16-23: Use actions from 3rd sample (based on obs at step 16) ← AND HERE
...
```

---

```python
# LINE 182-186: Execute one action at a time
action = action_queue.popleft()  # Take first action from queue
result = env.step(action)        # Apply to environment
obs = result.obs                 # Get new observation
done = bool(result.done)
steps += 1
```

**Purpose**: Execute actions one-by-one, observing feedback after each
**Key**: Robot **responds** to each action → observation drifts from prediction

---

### Why Baseline Succeeds

1. **Replanning corrects errors**: Every 8 steps, policy sees actual robot state and generates corrective actions
2. **Conservative temperature (1.0)**: Actions are close to mean learned behavior → stable
3. **No modification**: Uses policy exactly as trained

---

## Part 2: How VLM-Guided Policy Works

### Code Location: `rollout_guided()` (lines 207-310)

```python
# LINE 241-256: FIRST REPLAN - Generate candidates with VLM scoring
if steps == 0:  # Only at the very beginning
    # First replan: Generate diverse candidates with VLM scoring
    candidates = []
    for idx, temp in enumerate(np.linspace(0.5, 3.0, n_candidates)):  # temperature range 0.5 to 3.0
        seq = sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=float(temp))
        seq = enforce_block_direction(seq, target_block, model.horizon)  # Force direction toward target
        frames = collect_candidate_frames(episode_seed, seq)  # Simulate trajectory
        score = vlm_score_prefix(vlm_scorer, frames, target_block, f"ep{episode_seed}_cand{idx}")
        candidates.append(...)
```

**Purpose**: Generate 8 different initial action sequences with diverse trajectories
**Key Steps**:
1. Sample 8 candidates at temperatures 0.5, 0.78, 1.06, ..., 3.0
2. For EACH candidate, **simulate it** in a separate environment to capture frames
3. Send frames to VLM to score legibility
4. Select candidate with highest arc among those with legibility ≥ 0.70

---

### Critical Problem 1: What `collect_candidate_frames()` Does

```python
# LINE 87-105: Simulate candidate trajectory
def collect_candidate_frames(episode_seed: int, actions: np.ndarray) -> List[Image.Image]:
    env = TwoBlockPickEnv(render=False, episode_length=400, cube_jitter=0.0)
    env.reset(seed=episode_seed)  # Same seed as baseline
    capture_steps = [0, 30, 60, 90, 120, 150]  # Capture at 0, 1, 2, 3, 4, 5 seconds
    frames = []

    for step in range(capture_steps[-1] + 1):  # 151 steps
        action = actions[step] if step < len(actions) else actions[-1]  # ⚠️ PROBLEM
        result = env.step(action)
        if step in capture_steps:
            frame = env.render(width=480, height=480)
            frames.append(Image.fromarray(frame))
```

**⚠️ CRITICAL BUG**: 
- `actions` is only **8 steps long** (one horizon)
- But we're simulating for **151 steps** (5 seconds @ 30Hz)
- After step 7, code repeats `actions[-1]` (last action) for 143 steps!

**What VLM Actually Sees**:
```
Steps 0-7:   Execute 8 actions from candidate sequence
Steps 8-150: Repeat actions[7] over and over (STUCK)
```

**Result**: 
- VLM sees a trajectory that executes 8 actions then **freezes** with repeated action
- This does NOT reflect what happens during actual rollout (which has replanning)
- VLM scores a **fake trajectory** that will never happen in reality

---

### Critical Problem 2: What Happens During Actual Execution

```python
# LINE 285-293: Subsequent replans use different logic
else:  # After first replan (steps > 0)
    # Subsequent replans: Use moderate temperature to maintain arc diversity without breaking task completion
    seq = sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=1.5)
    seq = enforce_block_direction(seq, target_block, model.horizon)
```

**What Happens in Reality**:
```
Step 0-7:    Use VLM-selected sequence (arc=0.5748, temperature=some high value)
Step 8-15:   Generate NEW sequence at temperature=1.5 (no VLM scoring)
Step 16-23:  Generate NEW sequence at temperature=1.5 (no VLM scoring)
...
```

**Problem**: 
- VLM only influences the **first 8 actions** (first 0.27 seconds)
- After that, policy generates actions at temperature=1.5 with no VLM guidance
- Temperature 1.5 is higher than baseline (1.0) → more exploratory/unstable

---

### Critical Problem 3: `enforce_block_direction()`

```python
# LINE 47-52
def enforce_block_direction(actions: np.ndarray, target_block: str, horizon: int) -> np.ndarray:
    out = actions.copy()
    sign = 1.0 if target_block == "LEFT" else -1.0
    lock_h = max(1, horizon // 4)  # Lock first 2 actions (8 // 4 = 2)
    out[:lock_h, 1] = sign * np.abs(out[:lock_h, 1])  # Force dy to correct sign
    return out
```

**Purpose**: Force early actions to move toward target block
**How**: 
- If target=LEFT, force dy (lateral motion) positive for first 2 actions
- If target=RIGHT, force dy negative for first 2 actions

**Problem**: 
- Overwrites learned action values
- May clash with policy's natural motion planning
- Only affects first 2 of 8 actions → rest can still go wrong direction

---

## Part 3: Why VLM-Guided Fails

### Summary of Issues

#### Issue 1: VLM Sees Fake Trajectory
```
VLM evaluation: 8 real actions + 143 repeated actions (frozen gripper)
Actual rollout:  8 actions → replan → 8 new actions → replan...
```
**Impact**: VLM's legibility score is **meaningless** because it's rating a trajectory that never executes

---

#### Issue 2: High Arc Selection Breaks Execution
From output:
```
Attempt 1: Selected arc=0.5748 (baseline was 0.2365) → Failed
Attempt 2: Selected arc=0.8186 (baseline was 0.6313) → Failed  
Attempt 3: Selected arc=0.6888 (baseline was 0.0326) → Failed
```

**Pattern**: VLM always selects **higher arc** than baseline
- Higher arc = more lateral motion = more exaggerated reaching
- These extreme motions might overshoot the block or collide with obstacles
- After first 8 steps, replanning at temperature=1.5 makes it worse

---

#### Issue 3: Temperature Mismatch
```
Baseline:     temperature=1.0 throughout (stable)
VLM-Guided:   temperature=0.5 to 3.0 for first sequence (extreme diversity)
              temperature=1.5 for subsequent replans (more unstable than baseline)
```

**Impact**: 
- High-temperature initial actions may put robot in unstable state
- Higher temperature replanning can't recover

---

#### Issue 4: Only First 8 Actions Are VLM-Guided
```
Total episode: ~400 steps (50 replanning cycles)
VLM-guided:    8 steps   (1 replanning cycle)
Not VLM-guided: 392 steps (49 replanning cycles)
```

**Impact**: 98% of the trajectory has NO legibility guarantee

---

## Part 4: How General Diffusion Policy Works

### Conceptual Model

**What is Diffusion Policy?**
Imagine you have a photo that's completely covered in noise (TV static). Diffusion models learn to gradually remove that noise to reveal a clear image. Diffusion policy does the same but for **action sequences**:

```
Start:     [random noise] → meaningless actions
Step 1:    [less noise]   → rough action trajectory  
Step 2:    [even less]    → refined trajectory
...
Step 10:   [clean]        → executable action sequence
```

### Technical Details from Your Code

```python
# LINE 80-84: Sample action sequence
def sample_action_seq(model, sampler, obs, obs_mean, obs_std, act_mean, act_std, device, temperature=1.0):
    obs_norm = (obs - obs_mean.cpu().numpy()) / obs_std.cpu().numpy()
    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        seq = sampler.sample(model, obs_tensor, n_sampling_steps=10, temperature=temperature)[0].cpu().numpy()
```

**Step-by-Step**:
1. **Normalize observation**: Scale robot state (joint angles, gripper position, etc.) to standard range
2. **Create initial noise**: `temperature` controls how random the starting point is
   - temperature=1.0: Standard random noise → typical behaviors
   - temperature=3.0: Very random noise → unusual, exploratory behaviors
3. **Denoise in 10 steps**: Model runs 10 iterations, each time predicting and removing noise
4. **Output**: 8-step action sequence (horizon=8)

**Key Hyperparameter: Temperature**
```
temperature=0.5:  Conservative, close to mean learned behavior
temperature=1.0:  Normal, trained distribution (this is what baseline uses)
temperature=2.0:  Exploratory, diverse behaviors
temperature=3.0:  Very exploratory, possibly unstable
```

### Why Diffusion Policy Needs Replanning

Diffusion generates actions based on **current observation**, but:
- Observation is just current state, not future trajectory
- Model can't perfectly predict how actions will affect environment
- Small errors compound over long horizons

**Solution: Replanning (Closed-Loop Control)**
```
Predict 8 actions → Execute 1 → Observe result → 
  → Execute 1 → Observe → ... → Queue empty →
    → Predict NEW 8 actions based on ACTUAL current state →
      → Repeat
```

This is why baseline works even though it sometimes generates imperfect initial sequences.

---

## Part 5: Root Cause Analysis

### Why Baseline Succeeds

1. ✅ **Consistent temperature**: Always 1.0 → stable, trained distribution
2. ✅ **Frequent replanning**: Corrects errors every 8 steps
3. ✅ **No action modification**: Uses policy as learned
4. ✅ **Closed-loop control**: Adapts to actual robot state

### Why VLM-Guided Fails

1. ❌ **VLM sees fake trajectory**: Evaluates 8 actions + 143 repeats of last action
2. ❌ **VLM only affects 2% of episode**: First 8 of ~400 steps
3. ❌ **High arc selection**: Selects extreme trajectories that may be unsafe
4. ❌ **Temperature mismatch**: Higher than baseline (1.5 vs 1.0) for 98% of steps
5. ❌ **Action enforcement breaks policy**: `enforce_block_direction()` overwrites learned values

### Critical Misconception

**What you thought VLM does**:
- VLM evaluates full 5-second (150-step) trajectory with replanning
- VLM guides entire episode execution
- Selected trajectory is what robot actually follows

**What VLM actually does**:
- VLM evaluates 8 actions repeated 18 times (frozen motion)
- VLM only influences first 0.27 seconds
- Selected trajectory is immediately discarded after 8 steps, then policy generates new actions without VLM

---

## Part 6: How to Fix This

### Option 1: VLM Evaluates REPLANNED Trajectories (Most Realistic)

**Problem**: VLM currently evaluates "open-loop" trajectory (8 actions repeated)
**Solution**: Simulate full episode WITH REPLANNING, capture frames, score with VLM

```python
def simulate_full_trajectory_with_replanning(model, sampler, obs_initial, target_block, episode_seed):
    """Simulate 5 seconds (150 steps) with proper replanning."""
    env = TwoBlockPickEnv(...)
    obs = env.reset(seed=episode_seed)
    
    frames = []
    capture_times = [0, 30, 60, 90, 120, 150]
    
    action_queue = deque(maxlen=8)
    for step in range(150):
        if len(action_queue) == 0:
            # Generate new 8 actions (THIS IS REPLANNING)
            seq = sample_action_seq(model, sampler, obs, ..., temperature=<CANDIDATE_TEMP>)
            seq = enforce_block_direction(seq, target_block, 8)
            for action in seq:
                action_queue.append(action)
        
        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs  # ← UPDATE OBS FOR NEXT REPLAN
        
        if step in capture_times:
            frames.append(env.render())
    
    return frames  # These frames show ACTUAL replanned trajectory
```

**Benefit**: VLM now sees what actually happens during execution

---

### Option 2: Don't Modify Actions After VLM Selection

**Problem**: `enforce_block_direction()` may break VLM-selected trajectory
**Solution**: Only enforce direction on candidates BEFORE VLM scoring, not during execution

---

### Option 3: Lower Subsequent Replan Temperature

**Problem**: temperature=1.5 for 98% of steps is higher than baseline
**Solution**: Use temperature=1.0 for subsequent replans (match baseline)

```python
else:  # Subsequent replans
    seq = sample_action_seq(..., temperature=1.0)  # ← Change from 1.5 to 1.0
```

---

### Option 4: Use Arc Cap More Conservatively

**Problem**: Allowing arc up to 0.9m is still too high
**Solution**: Cap at 0.6m or even match baseline's arc

```python
arc15_legible = [c for c in candidates 
                 if 0.15 <= c["arc"] <= 0.60  # ← Tighter bound
                 and c["legibility"] >= 0.70]
```

---

## Part 7: Recommended Next Steps

### Immediate Fix (Easiest)

Change line 289 to use baseline temperature:
```python
seq = sample_action_seq(..., temperature=1.0)  # Match baseline
```

This alone might fix most failures.

### Better Fix (Correct VLM Evaluation)

Rewrite `collect_candidate_frames()` to simulate WITH replanning at the candidate's temperature for first window only, then temperature=1.0 for remaining steps. This way VLM sees realistic trajectory.

### Best Fix (Full Pipeline)

1. Simulate full 5-second trajectory with replanning for each candidate
2. VLM scores the REPLANNED trajectory  
3. Select best candidate
4. During actual execution, use that candidate's temperature only for first replan, then temperature=1.0

This ensures:
- VLM sees realistic trajectory
- Selection is meaningful
- Execution matches evaluation
- Stability is maintained after first replan

---

## Summary

**The core issue**: You're asking VLM to score a 5-second trajectory, but only showing it 8 actions (0.27 seconds) that get repeated. Then during execution, you only use VLM's choice for those 8 actions and ignore it for the remaining 392 actions. This disconnect between what VLM scores and what actually executes is why guided rollouts fail despite "good" VLM scores.

**The fix**: Make VLM see and score what will actually happen - a full replanned trajectory. Then use that information throughout execution, not just for the first 8 actions.
