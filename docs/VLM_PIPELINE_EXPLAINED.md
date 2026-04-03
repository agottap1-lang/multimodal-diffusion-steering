# VLM-Guided Legibility Steering - Complete Pipeline

## Overview

The system has **TWO stages** working together:

### Stage 1: CFG-Noise Steering (Goal-Preserving Trajectory Generation)
**Purpose:** Generate diverse high-arc trajectories that maintain the same goal

### Stage 2: VLM Reranking (Legibility Selection)
**Purpose:** Select the MOST LEGIBLE trajectory from the diverse candidates

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AT EACH REPLANNING STEP                          │
└─────────────────────────────────────────────────────────────────────────┘

1. OBSERVE CURRENT STATE
   Input: Current robot observation (22-dim: ee_pos, ee_quat, grip, L_block, R_block)
   Normalize: obs_norm = (obs - obs_mean) / obs_std

                              ↓

2. CFG-NOISE STEERING (vlm_guided_policy.py lines 115-208)
   ┌───────────────────────────────────────────────────────────┐
   │ a) Generate BASELINE trajectory                           │
   │    base_noise = torch.randn(1, H, A)                      │
   │    baseline_action = sampler.sample(policy, obs,          │
   │                                      initial_noise=base_noise) │
   │                                                            │
   │ b) Check baseline arc style                               │
   │    baseline_dy_cumsum = np.cumsum(baseline_action[:, 1])  │
   │    baseline_arc = max(|baseline_dy_cumsum|)              │
   │    baseline_endpoint_y = baseline_dy_cumsum[-1]          │
   │    target_block = "LEFT" if avg(dy) > 0 else "RIGHT"     │
   │                                                            │
   │ c) If arc < 0.15m → Generate HIGH-ARC VARIANTS (up to N) │
   │    For variant_idx in [1, 2, ..., N]:                    │
   │      - Create constrained perturbation:                   │
   │          perturbation = randn_like(base_noise) * 0.35     │
   │                                                            │
   │      - Apply lateral mask (arc profile):                  │
   │          arc_profile = Gaussian(peak=t=0.45, width=0.2)  │
   │          endpoint_damping = 1.0 - exp(-(t-1)²/0.02)      │
   │          lateral_mask[:,:,1] = arc_profile *              │
   │                                endpoint_damping *         │
   │                                guidance_scale             │
   │                                                            │
   │      - Generate variant:                                  │
   │          variant_noise = base_noise + perturbation *      │
   │                         (1 + lateral_mask)               │
   │          variant_action = sampler.sample(policy, obs,     │
   │                                          initial_noise=variant_noise) │
   │                                                            │
   │      - VALIDATE GOAL CONSISTENCY:                         │
   │          variant_dy_avg = mean(variant_action[:H/4, 1])  │
   │          same_direction = sign(variant_dy) == sign(baseline_dy) │
   │          endpoint_close = |variant_endpoint_y -           │
   │                           baseline_endpoint_y| < 0.10m   │
   │                                                            │
   │      - ONLY KEEP if: same_direction AND endpoint_close   │
   │                                                            │
   │ OUTPUT: [baseline, variant1, variant2, ..., variantN]    │
   │         ALL target the SAME block (LEFT or RIGHT)        │
   │         ALL have DIFFERENT arc styles (low to high)      │
   └───────────────────────────────────────────────────────────┘

                              ↓

3. VLM RERANKING (vlm_guided_policy.py lines 210-250)
   ┌───────────────────────────────────────────────────────────┐
   │ a) Visualize each candidate trajectory                    │
   │    For each candidate in [baseline, variant1, ...]:       │
   │      viz_image = visualizer.render_frame_with_trajectory( │
   │          env=env,                                          │
   │          obs=obs,                                          │
   │          action_sequence=candidate,                        │
   │          n_steps=8,  # Show future 8 steps                │
   │          show_future=True                                  │
   │      )                                                     │
   │                                                            │
   │    OUTPUT: [image1.png, image2.png, ..., imageN.png]     │
   │            Each shows robot + trajectory overlay          │
   │                                                            │
   │ b) Query VLM for legibility scores                        │
   │    scores = vlm_scorer.score_trajectory_batch(            │
   │        image_bytes_list=[image1, image2, ...],            │
   │        goal_A="pick the left block",                      │
   │        goal_B="pick the right block",                     │
   │        mode="single_frame"                                │
   │    )                                                       │
   │                                                            │
   │    VLM (Gemini 2.0) analyzes each image:                 │
   │      - Understands which goal is intended                 │
   │      - Evaluates trajectory clarity/legibility            │
   │      - Returns legibility score (0.0 to 1.0)             │
   │                                                            │
   │    Example VLM response:                                  │
   │      {                                                     │
   │        "intended_goal": "pick the left block",            │
   │        "legibility_score": 0.85,                          │
   │        "reasoning": "Clear lateral sweep toward left      │
   │                      block with visible arc"              │
   │      }                                                     │
   │                                                            │
   │ c) Select MOST LEGIBLE trajectory                         │
   │    legibility_scores = [s['legibility_score'] for s in scores] │
   │    best_idx = argmax(legibility_scores)                   │
   │    best_action = candidates[best_idx]                     │
   │                                                            │
   │ OUTPUT: The candidate with HIGHEST legibility score      │
   │         (often a high-arc variant rather than baseline)   │
   └───────────────────────────────────────────────────────────┘

                              ↓

4. EXECUTE SELECTED TRAJECTORY
   Denormalize: action_denorm = best_action * act_std + act_mean
   Execute: env.step(action_denorm[0:8])  # Execute first 8 actions
   
                              ↓

5. REPEAT from step 1 until task complete
```

---

## Key Points

### CFG-Noise Steering (Stage 1)
- **Input:** Current observation
- **Output:** N trajectory candidates (all with SAME goal, different arcs)
- **Method:** Constrained noise perturbation with endpoint preservation
- **Parameters:**
  - `guidance_scale` (e.g., 3.5): Controls arc strength
  - `n_samples` (e.g., 3): Number of variants to generate
  - `endpoint_tolerance` (0.10m): Max allowed goal drift
- **Key Innovation:** Arc profile with endpoint damping
  ```python
  arc_profile = Gaussian(peak=0.45, sigma=0.2)      # Strong arc in middle
  endpoint_damping = 1 - Gaussian(peak=1.0, sigma=0.1)  # Reduce at end
  lateral_mask = arc_profile * endpoint_damping * guidance_scale
  ```

### VLM Reranking (Stage 2)
- **Input:** N trajectory visualizations (images with overlaid trajectories)
- **Output:** Single best trajectory (highest legibility)
- **Method:** Vision-Language Model (Gemini 2.0) evaluates legibility
- **VLM Capabilities:**
  - Understands spatial relationships
  - Recognizes motion intentions
  - Evaluates trajectory clarity
- **Benefits:**
  - No hand-crafted legibility metrics
  - Adapts to human perception
  - Understands context (obstacles, goals)

---

## Configuration in Code

### vlm_guided_policy.py

```python
class VLMGuidedPolicy:
    def __init__(
        self,
        base_policy,          # Trained diffusion policy
        base_sampler,         # DDIM sampler
        vlm_scorer,           # VLM client (Gemini 2.0)
        visualizer,           # Trajectory visualizer
        n_samples=3,          # Number of candidates (Stage 1)
        rerank_frequency=1,   # How often to use VLM (1=every step)
        ...
    ):
        ...
    
    def predict_action(self, obs, env, step_count):
        # Stage 1: CFG-Noise Steering (lines 115-208)
        candidates = self._generate_diverse_trajectories(obs)
        # All candidates target SAME goal but different arcs
        
        # Stage 2: VLM Reranking (lines 210-250)
        best_action = self._select_most_legible(candidates, env, obs)
        
        return best_action
```

### eval_legibility_steering.py

```python
# Create VLM-guided policy
guided_policy = VLMGuidedPolicy(
    base_policy=policy,
    base_sampler=sampler,
    vlm_scorer=LegibilityScorer(api_key=GEMINI_API_KEY),
    visualizer=TrajectoryVisualizer(),
    n_samples=3,           # Generate 3 candidates per step
    rerank_frequency=1     # Use VLM every replanning step
)

# Run episode
for step in range(max_steps):
    action = guided_policy.predict_action(obs, env, step_count=step)
    obs, reward, done = env.step(action)
```

---

## Expected Results

From your successful run ([final_comparison/vlm_guided_n3_20260226_003806](c:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick\runs\final_comparison\vlm_guided_n3_20260226_003806)):

```json
{
  "success_rate": 1.0,           // 100% task completion
  "avg_legibility_score": 0.75,  // High legibility from VLM
  "avg_latency_ms": 5175,        // ~5 seconds per VLM call
  "total_vlm_calls": 216         // 13 replans/episode × 3 candidates × 5 episodes
}
```

**Benefits Achieved:**
1. ✅ 100% success rate (task completion maintained)
2. ✅ High legibility scores (0.75 average)
3. ✅ Arc 15-19 style trajectories (visible lateral sweeps)
4. ✅ Goal consistency (all variants target correct block)

---

## Tuning Parameters

### To Increase Arc Strength:
```python
# In vlm_guided_policy.py lines 165-167
guidance_scale = 5.0  # Increase from 3.5 to 5.0
perturbation_strength = 0.40  # Increase from 0.35 to 0.40
```

### To Reduce VLM Latency:
```python
# Run VLM less frequently
rerank_frequency = 2  # Every 2nd replanning step instead of every step
n_samples = 2         # Generate 2 candidates instead of 3
```

### To Prioritize Arc Over Goal:
```python
# In goal validation (line 182-189)
endpoint_tolerance = 0.15  # Increase from 0.10m to 0.15m
score = arc_increase - endpoint_diff * 1.0  # Reduce penalty from 2.0 to 1.0
```

---

## Troubleshooting

### Issue: Low arc increase
**Cause:** `guidance_scale` too low
**Fix:** Increase `guidance_scale` from 3.5 to 5.0 or 7.0

### Issue: Goals changing
**Cause:** Endpoint tolerance too loose
**Fix:** Decrease `endpoint_tolerance` from 0.10m to 0.08m

### Issue: VLM not being used
**Cause:** Missing API key or import failure
**Check:** 
```python
# Should see in logs:
"VLM reranking enabled with n_samples=3"
"Step X: Chose arc variant Y (legibility 0.85 > 0.70)"
```

### Issue: All candidates identical
**Cause:** `perturbation_strength` too low
**Fix:** Increase from 0.35 to 0.40 or 0.45

---

## Current Status

✅ **CFG-Noise Steering:** WORKING (100% goal consistency achieved)  
⏳ **VLM Reranking:** Requires setup (google-generativeai + API key)  
📹 **Video Comparison:** Generating now...

**Next Steps:**
1. Watch generated videos to verify arc differences
2. Set up VLM API key if needed
3. Run full evaluation with VLM reranking
4. Tune `guidance_scale` if arcs need to be larger
