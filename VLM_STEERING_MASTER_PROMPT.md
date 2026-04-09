# VLM-Steered Diffusion Policy: 4-Behavior Inference-Time Guidance

## Master Execution Prompt — 2-Day Pipeline

---

## 1. PROBLEM STATEMENT

You have a **trained diffusion policy** for a 7-DOF Panda robot arm performing a **TwoBlockPick** task in PyBullet simulation. The policy is multimodal — given the same observation, it can pick either the left block or the right block depending on sampling noise. The policy uses a U-Net architecture conditioned on timestep and observation, trained on 400 demonstrations (200 left, 200 right) with DDIM sampling (eta=0.3 for stochasticity).

**Your goal:** Use a **Vision-Language Model (Gemini)** at **inference time** to steer the diffusion policy toward **4 distinct behaviors** — with **zero retraining**. The VLM acts as a judge/scorer, not a controller. Everything happens at inference time via **Best-of-K candidate selection**.

### The 4 Behaviors

| # | Behavior | What It Means | VLM's Role |
|---|----------|---------------|------------|
| 1 | **Legibility** | The arm's path makes its intended goal obvious to an observer from the first 30% of the trajectory. Wide arcs toward the target so a human watching can immediately tell which block will be picked. | Score how clearly the arm reveals its intent early. Prefer trajectories with large lateral separation from the non-target block. |
| 2 | **Predictability** (opposite of legibility) | The arm takes the **shortest, most direct** straight-line path to the target. No unnecessary lateral sweeps. A human watching can predict the arm's future position because it moves efficiently in a beeline. | Score how direct and efficient the path is. Penalize any lateral deviation, wide arcs, or unnecessary curvature. |
| 3 | **Safety** (obstacle avoidance) | A red cylindrical obstacle is placed on the table between the robot's home position and the blocks. The arm must reach the target block while maintaining maximum clearance from the obstacle at all times. | Score how safely the arm avoids the obstacle. Penalize trajectories that pass close to or risk collision with the red cylinder. |
| 4 | **Sequential Picking** (waypoint grounding) | The arm must **pass over/near block #3** (a blue waypoint block placed between the two target blocks) **before** picking **block #5** (the target). This tests spatial instruction following — the VLM verifies that the trajectory respects the sequential waypoint constraint. | Score whether the trajectory visits the waypoint before reaching the target. Penalize paths that go directly to the target without passing near the intermediate block. |

---

## 2. WHAT YOU ALREADY HAVE (DO NOT REBUILD)

### Trained Diffusion Policy
- **Architecture:** U-Net with sinusoidal time embedding (128-d), 3 encoder blocks (256→512→1024), bottleneck, 3 decoder blocks with skip connections. GroupNorm(8), Mish activation.
- **Input:** 22-d observation (EE pos/quat/grip + left block pos/quat + right block pos/quat)
- **Output:** 32-step action horizon (5-d: dx, dy, dz, dyaw, grip), noise prediction (no tanh)
- **Sampling:** DDIM with 10 denoising steps, action_scale=0.05m
- **Execute steps:** 8 per replan (critical — do NOT change to 16 or 1)
- **Checkpoint:** `runs/diffusion_*/ckpt_ep100.pt` (contains model weights + obs/act normalization stats + config)
- **Success rate:** ~85-92% baseline, bimodal (picks left ~50%, right ~50%)

### Classifier Guidance (Already Implemented for Legibility Only)
- Injects gradient into DDIM denoising: `ε̃ = ε_θ(x_t) − w·√(1−ᾱ_t)·∇L_early(x̂₀)`
- `L_early_intent`: Bayesian posterior P(goal*|trajectory) using Gaussian proximity model
- **Works:** Δ L_early = +0.039 at w=10 (VLM-synthesized function)
- **Limitation:** Only supports legibility. NOT extensible to safety/grounding without new hand-crafted differentiable scoring functions.

### VLM Integration (Gemini API)
- `tools/vlm_client.py` — Gemini API wrapper
- `gemini_vlm_eval` — sibling project with `GeminiClient` for frame-level evaluation
- Frame capture from PyBullet: 480×480 JPEG at t=0,1,2,3,4,5s (6 frames per candidate)
- Model: `gemini-2.5-flash-preview-05-20`

### Best-of-K Pipeline (Already Implemented in `evaluation/eval_behaviors.py`)
- **This is the core pipeline you will use.** It already handles all 4 behaviors.
- Per episode: reset env → save PyBullet state → generate K candidates → VLM scores each → select best → execute to completion → record video → compute metrics
- Supports environment modification per behavior (obstacle placement, waypoint blocks, color changes)

### Dataset
- 400 demos in `data/demos/demos_combined.npz`: obs (N,T,22), actions (N,T,5), labels ('left'/'right')
- Manifest: `data/manifest_combined_cfg00.jsonl` with trajectory type labels (legible/neutral/deceptive)
- Demo videos: `data/demos/demo_videos_combined/`

---

## 3. THE PIPELINE (Best-of-K with VLM Selection)

This is training-free. You do NOT retrain the policy. The VLM steers behavior purely through **candidate selection** at inference time.

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE-TIME PIPELINE                       │
│                                                                 │
│  For each episode:                                              │
│                                                                 │
│  1. RESET environment with seed                                 │
│     - If safety: spawn red cylinder obstacle at (0.45, 0, 0.42) │
│     - If grounding: spawn blue waypoint block at (0.48, 0.03)   │
│                                                                 │
│  2. SAVE PyBullet state                                         │
│                                                                 │
│  3. GENERATE K=8 candidate trajectories:                        │
│     For k = 1..K:                                               │
│       - Restore saved state                                     │
│       - Set unique random seed per candidate                    │
│       - Run policy for 150 steps (5s), replanning every 8 steps │
│       - Capture 6 frames at t=0,1,2,3,4,5s                     │
│       - Record EE trajectory (Nx3)                              │
│                                                                 │
│  4. VLM SCORING:                                                │
│     For each candidate:                                         │
│       - Send 6 JPEG frames to Gemini                            │
│       - Use BEHAVIOR-SPECIFIC PROMPT (see §4)                   │
│       - Parse JSON response → extract behavior score [0,1]      │
│       - Add feasibility bonus (proximity to target block)        │
│                                                                 │
│  5. SELECT:                                                     │
│     - vlm_selected = argmax(combined_score)                     │
│     - baseline = argmin(combined_score) or random               │
│                                                                 │
│  6. EXECUTE selected candidate to completion (up to 600 steps)  │
│     - Record video                                              │
│     - Compute L_early, success, laterals, clearance metrics     │
│                                                                 │
│  7. COMPARE vlm_selected vs baseline on behavior-specific metric│
└─────────────────────────────────────────────────────────────────┘
```

### Why Best-of-K (Not Gradient Guidance for All 4)

| Method | Legibility | Predictability | Safety | Sequential |
|--------|-----------|----------------|--------|------------|
| Classifier Guidance | ✅ (have differentiable L_early) | ✅ (negate L_early → straight) | ❌ (need differentiable obstacle distance — possible but fragile) | ❌ (sequential waypoint constraint is non-differentiable through VLM) |
| Best-of-K + VLM | ✅ | ✅ | ✅ | ✅ |

Best-of-K is the **only method that works for ALL 4 behaviors** without writing new differentiable scoring functions. K=8 gives enough diversity from DDIM stochasticity (eta=0.3) to find good candidates.

---

## 4. VLM PROMPTS (Behavior-Specific)

### 4a. Legibility Prompt
```
You are evaluating a robot arm trajectory for LEGIBILITY (intent clarity).

Scene: A robot arm starts at the center and must pick one of two blocks.
Goal A: pick the left block. Goal B: pick the right block.

These 6 frames show the arm's motion at t=0,1,2,3,4,5 seconds.

TASK: Rate how LEGIBLE this trajectory is — how clearly does the arm
reveal WHICH block it intends to pick, as early as possible?

A highly legible trajectory curves AWAY from the non-target early on,
making the intended goal obvious to an observer within the first 1-2
seconds. A low-legibility trajectory stays ambiguous — the observer
cannot tell which block until the arm is nearly at the target.

Determine which goal (A or B) the arm appears headed toward, and your
confidence in each. Then rate legibility from 0.0 to 1.0.

Output ONLY valid JSON:
{"pA": X, "pB": X, "legibility": X, "cue": "brief explanation"}
```

### 4b. Predictability (Straight-Line) Prompt
```
You are evaluating a robot arm trajectory for PREDICTABILITY (efficiency).

Scene: A robot arm starts at the center and must pick one of two blocks.
Goal A: pick the left block. Goal B: pick the right block.

These 6 frames show the arm's motion at t=0,1,2,3,4,5 seconds.

TASK: Rate how PREDICTABLE and EFFICIENT this trajectory is.

A highly predictable trajectory takes the SHORTEST, MOST DIRECT path
to its target — a straight beeline with no unnecessary lateral sweeps,
curves, or wide arcs. A human observer can predict where the arm will
be 1 second from now because it moves in a constant, purposeful direction.

A low-predictability trajectory has unnecessary curves, hesitations,
direction changes, or takes a roundabout path.

Determine which goal (A or B) the arm is heading toward.
Rate predictability from 0.0 (erratic/indirect) to 1.0 (perfectly direct).

Output ONLY valid JSON:
{"pA": X, "pB": X, "predictability": X, "cue": "brief explanation"}
```

### 4c. Safety (Obstacle Avoidance) Prompt
```
You are evaluating a robot arm trajectory for SAFETY (obstacle avoidance).

Scene: A robot arm starts at the center and must pick one of two blocks.
Goal A: pick the left block. Goal B: pick the right block.

IMPORTANT: There is a RED CYLINDRICAL OBSTACLE on the table, positioned
between the robot's starting position and the blocks. It is clearly
visible in the frames.

These 6 frames show the arm's motion at t=0,1,2,3,4,5 seconds.

TASK: Rate how SAFELY this trajectory avoids the RED obstacle.

A safe trajectory maintains WIDE clearance from the red cylinder at
all times — it curves around or lifts above the obstacle with generous
margin. An unsafe trajectory passes dangerously close to the obstacle
or appears to risk collision.

Determine which goal (A or B) the arm is heading toward.
Rate safety from 0.0 (very close to obstacle) to 1.0 (wide clearance).

Output ONLY valid JSON:
{"pA": X, "pB": X, "safety": X, "cue": "brief explanation of clearance"}
```

### 4d. Sequential Picking (Waypoint Grounding) Prompt
```
You are evaluating a robot arm trajectory for INSTRUCTION FOLLOWING.

Scene: A robot arm starts at the center. There are blocks on the table:
- A GREEN block (left side) — Goal A
- A RED block (right side) — Goal B
- A BLUE block (center, between the two) — WAYPOINT (do not pick this!)

INSTRUCTION: "Pass near the BLUE block BEFORE picking the GREEN block."

These 6 frames show the arm's motion at t=0,1,2,3,4,5 seconds.

TASK: Rate how well this trajectory follows the spatial instruction.

A high-scoring trajectory clearly passes near or over the BLUE waypoint
block FIRST (visible in early frames), and THEN proceeds to pick the
GREEN block. The arm visits the waypoint as an intermediate stop.

A low-scoring trajectory ignores the blue block entirely and goes
directly to the green block, or goes to the wrong target.

Rate instruction_following from 0.0 (ignores instruction) to 1.0 (clearly follows it).

Output ONLY valid JSON:
{"pA": X, "pB": X, "instruction_following": X, "cue": "description of path relative to waypoint"}
```

---

## 5. DATASET REQUIREMENTS

Your existing dataset of 400 demonstrations is sufficient for the trained base policy. You do NOT need new demonstrations for any of the 4 behaviors — all steering happens at inference time via Best-of-K selection.

However, to **validate** that VLM scoring is meaningful, you need:

### 5a. Validation Set (Create Once, ~2 Hours)

For each behavior, collect **ground-truth labels** on a small set of candidate trajectories:

| Behavior | Ground Truth Metric | How to Compute | Target N |
|----------|-------------------|----------------|----------|
| Legibility | L_early_intent (Bayesian posterior) | Already implemented in `l_early_intent_torch()` | 50 candidates |
| Predictability | Path efficiency ratio = straight-line distance / actual path length | `np.linalg.norm(ee[-1]-ee[0]) / sum(norms of consecutive segments)` | 50 candidates |
| Safety | Min obstacle clearance (meters) | `min(‖ee_t - obstacle_pos‖ for all t)` | 50 candidates |
| Sequential | Waypoint proximity + ordering | `min(‖ee_t - waypoint‖) for t < t_grasp` and `t_nearest_waypoint < t_nearest_target` | 50 candidates |

Then compute **Spearman rank correlation** between VLM scores and ground-truth metrics. If ρ > 0.5, the VLM is usable. If ρ < 0.3, the prompts need revision.

### 5b. Existing Dataset Coverage

Your 400 demos already contain trajectory diversity:
- **arc00-05** (straight, <0.05m lateral) → useful for predictability validation
- **arc15-19** (large sweep, ≥0.15m lateral) → useful for legibility validation
- **legible/neutral/deceptive** labels → already support 2 of 4 behaviors

For **safety** and **sequential picking**, no training data exists (and none is needed — the behaviors are achieved purely at inference time by VLM selection from natural policy diversity + environment modification).

---

## 6. TWO-DAY EXECUTION PLAN

### Day 1: Pipeline Validation (Get Everything Running End-to-End)

**Morning (4h): Verify base pipeline works**

```bash
# 1. Verify checkpoint loads and base policy runs
python evaluation/eval_behaviors.py \
    --checkpoint runs/diffusion_*/ckpt_ep100.pt \
    --behavior legibility --n_episodes 3 --K 4

# 2. Check: does it produce videos? JSON results? No crashes?
# 3. Check: success rate > 60% on 3 episodes?
# 4. Check: VLM returns parseable JSON with scores in [0,1]?
```

**If it crashes:** Common fixes:
- Missing `GEMINI_API_KEY` environment variable
- `gemini_vlm_eval` not on path → check `sys.path.insert` in eval_behaviors.py
- PyBullet state save/restore issues → ensure `cube_jitter=0.0`
- GPU OOM → use CPU for policy (this is fast enough for K=8)

**Afternoon (4h): Run all 4 behaviors, small scale**

```bash
# Run each behavior with K=8 candidates, 5 episodes (quick validation)
for behavior in legibility predictability safety grounding; do
    python evaluation/eval_behaviors.py \
        --checkpoint runs/diffusion_*/ckpt_ep100.pt \
        --behavior $behavior --n_episodes 5 --K 8 \
        --out_dir outputs/behaviors/${behavior}_day1
done
```

**End of Day 1 Checklist:**
- [ ] All 4 behaviors run without crashes
- [ ] Each produces: videos/ directory, results.json, per-episode metrics
- [ ] VLM scores vary across candidates (not all 0.5 — that means prompt failed)
- [ ] VLM-selected candidates score higher than baselines on average
- [ ] At least some episodes show visible behavioral differences in videos

### Day 2: Full Evaluation + Metrics (Publishable Results)

**Morning (4h): Full-scale runs**

```bash
# 20 episodes × 4 behaviors × K=8 candidates = 640 VLM calls
# At ~2s per call + 1.5s sleep = ~37 minutes per behavior
for behavior in legibility predictability safety grounding; do
    python evaluation/eval_behaviors.py \
        --checkpoint runs/diffusion_*/ckpt_ep100.pt \
        --behavior $behavior --n_episodes 20 --K 8 \
        --out_dir outputs/behaviors/${behavior}_full
done
```

**Afternoon (4h): Metrics extraction + validation**

For each behavior, extract from `results.json`:

```python
import json, numpy as np

for behavior in ["legibility", "predictability", "safety", "grounding"]:
    with open(f"outputs/behaviors/{behavior}_full/results.json") as f:
        episodes = json.load(f)

    vlm_scores = [ep["vlm"]["vlm_score"] for ep in episodes]
    base_scores = [ep["baseline"]["vlm_score"] for ep in episodes]
    vlm_success = [ep["vlm"]["success"] for ep in episodes]
    base_success = [ep["baseline"]["success"] for ep in episodes]
    vlm_learly = [ep["vlm"]["L_early"] for ep in episodes]
    base_learly = [ep["baseline"]["L_early"] for ep in episodes]

    print(f"\n{'='*60}")
    print(f"BEHAVIOR: {behavior.upper()}")
    print(f"  VLM  success: {np.mean(vlm_success)*100:.0f}%  "
          f"L_early: {np.mean(vlm_learly):.3f}±{np.std(vlm_learly):.3f}  "
          f"VLM_score: {np.mean(vlm_scores):.3f}±{np.std(vlm_scores):.3f}")
    print(f"  BASE success: {np.mean(base_success)*100:.0f}%  "
          f"L_early: {np.mean(base_learly):.3f}±{np.std(base_learly):.3f}  "
          f"VLM_score: {np.mean(base_scores):.3f}±{np.std(base_scores):.3f}")
    delta = np.mean(vlm_scores) - np.mean(base_scores)
    print(f"  Δ VLM_score: {delta:+.3f}")
```

**Key metrics per behavior:**

| Behavior | Primary Metric | Secondary Metric | Success Threshold |
|----------|---------------|------------------|-------------------|
| Legibility | L_early (VLM > baseline) | VLM legibility score | VLM ≥ 0.90 L_early |
| Predictability | Path efficiency ratio (VLM > baseline) | VLM predictability score | Higher ratio = more direct |
| Safety | Min obstacle clearance in meters (VLM > baseline) | VLM safety score | Clearance > 0.04m |
| Sequential | Waypoint visit before grasp (binary) | VLM instruction_following score | ≥60% correct ordering |

---

## 7. WHAT "FUNCTIONAL PIPELINE" MEANS (Your Acceptance Criteria)

Your pipeline is **functional** if and only if ALL of the following hold:

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 1 | **Runs end-to-end without crashes** | All 4 behaviors × 20 episodes complete. No Python exceptions. No NaN in outputs. |
| 2 | **Produces consistent results** | Run same 5 episodes twice with same seeds → VLM-selected candidate index matches ≥80% of the time (VLM is slightly stochastic, so allow some variance). |
| 3 | **Extractable metrics** | Each run produces a `results.json` with per-episode: success (bool), L_early (float), vlm_score (float), picked (str), steps (int). You can compute mean ± std from these. |
| 4 | **VLM scores correlate with behavior** | For legibility: Spearman ρ(VLM_legibility_score, L_early_groundtruth) > 0.4. For predictability: VLM_predictability_score anti-correlates with lateral displacement. For safety: VLM_safety_score correlates with min obstacle clearance. |
| 5 | **Behavioral difference is visible** | Watch 3 VLM-selected vs 3 baseline videos per behavior. A human can tell the difference without being told which is which. |
| 6 | **VLM selection outperforms baseline** | Paired t-test or Wilcoxon on primary metric: p < 0.05 for at least 3 of 4 behaviors. |

---

## 8. KNOWN PITFALLS (FROM YOUR CODEBASE HISTORY)

These are failure modes your project has already encountered. Avoid them:

| Pitfall | What Happened | How to Avoid |
|---------|---------------|--------------|
| `execute_steps=16` | 0% success — policy never saw 16-step gaps during training | Always use `execute_steps=8` (matches training `n_action_steps`) |
| True DPS gradient injection | 0% success, 38.1% trajectory divergence | Use Best-of-K (no gradients needed). If you must use gradients, use classifier guidance only for legibility. |
| VLM text-only synthesis | Generated scoring function without visual data — not truly VLM-based | In this pipeline, VLM **sees actual frames** (6 JPEGs per candidate). This IS visual grounding. |
| VLM returns unparseable response | Gemini sometimes returns markdown-wrapped JSON or extra text | Parse with `text.find("{")` to `text.rfind("}")` — already implemented in `eval_behaviors.py`. Retry 3x on failure. |
| Modal collapse at low eta | With DDIM eta=0, all K candidates are identical → no selection diversity | Keep eta=0.3. If diversity is still low, increase K from 8 to 12. |
| Obstacle not visible in frames | Camera angle may not show the red cylinder clearly | Verify frame captures before full run. Adjust camera if needed: `env.render(mode="rgb_array")` uses top-down view — ensure obstacle is visible. |
| Rate limiting on Gemini API | Too many rapid calls → 429 errors | Use `--sleep 1.5` (default) between VLM calls. For 20 episodes × K=8, total = 160 calls/behavior, ~10min pace. |

---

## 9. QUICK REFERENCE: FILE MAP

| File | Purpose | Status |
|------|---------|--------|
| `evaluation/eval_behaviors.py` | **MAIN PIPELINE** — Best-of-K for all 4 behaviors | ✅ Ready |
| `evaluation/eval_legibility_guided.py` | Gradient-based legibility (single behavior) | ✅ Works, but use eval_behaviors.py instead |
| `evaluation/legibility_metrics.py` | L_early and other metric implementations | ✅ Reference |
| `scripts/train.py` | Diffusion policy training | ✅ Already trained |
| `envs/twoblockpick_env.py` | PyBullet environment | ✅ Ready |
| `tools/vlm_client.py` | Gemini API wrapper | ✅ Ready |
| `configs/train_combined.yaml` | Training config (for reference) | ✅ Do not modify |
| `data/demos/demos_combined.npz` | Training data (400 demos) | ✅ Do not modify |

---

## 10. SUMMARY: WHAT MAKES THIS WORK

1. **You already have a multimodal policy** — DDIM with eta=0.3 naturally produces diverse trajectories. Some go left, some go right, some arc wide, some go straight. This diversity is your raw material.

2. **The VLM is a selector, not a controller** — You don't need the VLM to generate actions or gradients. You generate K=8 candidates from the policy's natural distribution, and the VLM picks the one that best matches the desired behavior. This is simple, robust, and works for arbitrary behaviors.

3. **Environment modification creates the right test conditions** — For safety, you physically add an obstacle. For sequential picking, you add a waypoint block. The policy doesn't know about these — it just generates diverse trajectories, and the VLM picks the one that happens to avoid the obstacle or pass near the waypoint.

4. **Metrics are grounded** — L_early (Bayesian posterior) for legibility, path efficiency for predictability, clearance distance for safety, waypoint ordering for sequential. Each has a computable ground truth that the VLM score should correlate with.

5. **The pipeline is already implemented** — `eval_behaviors.py` handles all 4 behaviors. You just need to run it, verify outputs, and extract metrics.
