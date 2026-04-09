# Research Prompt: VLM-Guided Steering of a Diffusion Policy

**Context files**: `docs/HONEST_ASSESSMENT.md`, `evaluation/eval_legibility_guided.py`,
`evaluation/eval_behaviors.py`, `outputs/stage1/vlm_score_fn.py`

---

## Background (read before answering)

We have a **trained diffusion policy** (U-Net + DDIM, 10 steps, η=0.3) for a 7-DOF Panda arm
performing a bimodal pick task (TwoBlockPick). The policy is deliberately left in bimodal form
— given the same observation it can pick either block. We steer it at inference time without
retraining.

**Current pipeline results** (20 paired episodes, same seeds):

| Method | Success | L_early |
|--------|---------|---------|
| Baseline (unguided) | 80% | 0.898 |
| Classifier guidance w=15 (hand-crafted `L_early_intent`) | 100% | 0.947 |
| Classifier guidance w=10 (VLM-synthesized score fn) | 100% | 0.937 |
| Best-of-K (K=5, VLM text rerank, N=5 candidates) | 100% | 0.972 |

`L_early` = Bayesian posterior P(correct goal | first 30% of trajectory). Higher is better.
The VLM score function (`outputs/stage1/vlm_score_fn.py`) was generated from a **text-only**
prompt — it never saw any trajectory images. It is a differentiable, 4-criterion function
(proximity 0.35, direction 0.30, lateral 0.25, speed 0.10) injected as a gradient during DDIM.

Our "DPS" is actually **classifier guidance** (noise-prediction modification), not true DPS.
True DPS was tested and failed (0% success, 38% divergence).

---

## Research Questions

### Q1: Is Best-of-K (sampling) the right final method, or should we push toward gradient-based steering?

Specifically:
- Best-of-K gives L_early=0.972 (the best result so far). It requires K=5 full rollouts per
  episode and a VLM call per candidate. Does scaling K further still improve L_early, or does
  it plateau? At what K does the gain saturate?
- Classifier guidance gives L_early=0.937–0.947 with a single rollout. Can it match or
  exceed Best-of-K if the score function that drives the gradient were grounded in **visual
  VLM feedback** rather than a hand-crafted analytic function?
- When should we prefer gradient steering over sampling? Enumerate the axes: compute budget,
  diversity of the policy's distribution, differentiability of the guidance signal,
  out-of-distribution risk from gradient perturbation.

### Q2: The L_early metric and VLM both become informative only after ~0.3t. Does that constrain when steering must act?

The policy's trajectory prefix becomes geometrically informative (separates goal-directed from
ambiguous) after roughly t=0.3 of the task horizon. Both `L_early_intent` and the VLM image
score require some trajectory to have been executed before a meaningful signal is available.

- During DDIM denoising (high noise levels, t≈T→0), the predicted clean action x̂₀ is noisy
  and unreliable. At what DDIM step does x̂₀ become a reliable enough proxy for the executed
  L_early prefix to make classifier-guidance gradients useful? Is this the same 0.3t
  threshold, or a different one?
- If gradient steering must wait until the denoising is mostly complete (low noise levels),
  does that leave enough DDIM steps to steer the trajectory, or does the policy escape to a
  mode before guidance can shift it?
- Contrast: Best-of-K scores the **fully executed** trajectory. For a 10-step DDIM sampler,
  mid-sampling corrections would need x̂₀ to reflect the final executed shape. Propose a
  concrete schedule for when to apply guidance (e.g., steps 6–10 of DDIM only).

### Q3: How should a VLM provide feedback — and which mode is appropriate given what we already have?

Evaluate each of the three roles of a VLM in this pipeline:

**Role A — Reward signal / score synthesis (what we currently do)**
The VLM generates a differentiable Python function offline, then that function drives the
classifier-guidance gradient at every DDIM step. The VLM is never queried during inference.
- Limitation: the function was generated from text only, making it a reparameterization of
  the hand-crafted function rather than a visually grounded signal.
- Improvement path: instead of generating a function, have the VLM score rendered frames of
  x̂₀ mid-denoising to produce a scalar reward, then backpropagate through the renderer.
  Is there a renderer in our pipeline (PyBullet exists) that would make this tractable at
  inference time? What is the compute cost per DDIM step?

**Role B — Trajectory evaluator for mid-sampling corrections (the "DPS-style" role)**
At DDIM step s, compute x̂₀(x_s), render it, send to VLM, receive score, gradient-step x_s.
- Our experiment showed true DPS on an analytic function already diverges (0% success at the
  same scale used for classifier guidance). What changes if the scoring function is replaced
  with VLM image queries? The gradient is now through the renderer, not the analytic function.
  Does this mitigate or worsen divergence? Provide reasoning.
- Given the 1–3 second latency of a Gemini API call, how many mid-sampling VLM corrections
  are feasible without unacceptable episode latency? If DDIM has 10 steps and VLM is called
  at steps 7, 8, 9, 10, is that 4×latency per replan acceptable given we replan every 8 steps?

**Role C — Best-of-K selector (what we currently do with VLM text reranking, already best)**
VLM receives rendered video or annotated frames for each of K candidates and picks the best.
- Already implemented in `evaluation/eval_behaviors.py`.
- This is provably the right approach when the VLM signal is non-differentiable and the
  diversity of the K candidates captured by the policy is sufficient to include a high-quality
  solution. Confirm: given the bimodal policy and K=5, does at least one candidate always
  achieve L_early > 0.96? Report the empirical hit rate.
- What is the theoretical ceiling of Best-of-K as K→∞? Is it bounded by the policy's own
  mode diversity, or by the VLM's scoring accuracy?

### Q4: True CFG for this policy — how to construct the unconditional without retraining?

Standard CFG requires the model to be trained with conditioning dropout so it can run in
two modes. Our policy was NOT trained this way. However, CFG can be approximated at
inference time by exploiting the policy's obs structure.

**The policy obs layout (22-dim):**
```
obs[0:3]   = ee_pos          ← robot end-effector position
obs[3:7]   = ee_quat         ← robot orientation
obs[7]     = gripper         ← gripper state
obs[8:11]  = left_cube_pos   ← LEFT BLOCK position  ← CONDITIONING SIGNAL
obs[11:15] = left_cube_quat
obs[15:18] = right_cube_pos  ← RIGHT BLOCK position ← CONDITIONING SIGNAL
obs[18:22] = right_cube_quat
```

**Constructing the unconditional at inference (no retraining):**

If both block positions are set to their midpoint `m = (left_pos + right_pos) / 2` in the
obs, the policy has a symmetric prior over both goals — it cannot distinguish them. This is
a principled approximation of the null-conditioning `ε_θ(x_t | ∅)`:

```python
obs_uncond = obs.clone()
midpoint = (obs[:, 8:11] + obs[:, 15:18]) / 2.0
obs_uncond[:, 8:11]  = midpoint   # left block → midpoint
obs_uncond[:, 15:18] = midpoint   # right block → midpoint
ε_uncond = model(x_t, t_batch, obs_uncond)   # goal-symmetric noise prediction
ε_cond   = model(x_t, t_batch, obs)           # normal goal-knowing noise prediction
```

Verify this: does `ε_uncond` produce bimodal trajectories (equal left/right mix) while
`ε_cond` produces unimodal goal-directed trajectories? Measure the bimodality gap using
L_early_intent applied to K=20 samples from each. The gap confirms that
`obs_sym → obs` IS the conditioning signal.

**This is real CFG. But it solves goal-commitment, not legibility.**

The CFG update `ε̃ = ε_uncond + w·(ε_cond − ε_uncond)` amplifies goal-directedness, but a
goal-directed trajectory can still be non-legible (straight line → low L_early, high
goal-commitment). We need a SECOND conditioning axis: legibility.

---

## Critical Clarification: Two Separate Time Axes

There are two completely separate "T" values that must never be conflated.

```
Axis 1 — Diffusion denoising "time" (training schedule):
  n_diffusion_steps = 100   ← model trained on 100 noise levels
  DDIM inference uses 10 strides through those 100 steps:
  t ∈ {99, 88, 77, 66, 55, 44, 33, 22, 11, 0}  (evenly spaced, reversed)

Axis 2 — Action horizon "time" (inside each predicted chunk):
  horizon H = 32            ← each DDIM sample produces a 32-step action chunk
  0.3T = 0.3 × 32 = 10     ← L_early uses the FIRST 10 STEPS of the chunk
```

The "0.3T" in L_early_intent has **nothing to do with DDIM steps**. It means: take
x̂₀ (a 32-step action sequence), and evaluate the Bayesian goal posterior on only the
first 10 action steps of it. This is computed on the OUTPUT of DDIM, not on any
intermediate denoising state.

### What α̅ actually looks like at each DDIM stride step

With β linearly spaced from 0.0001→0.1 over 100 steps, approximate ᾱ at each of the
10 DDIM inference timesteps:

| DDIM step (of 10) | Training t | ᾱ_t  | √ᾱ_t | Interpretation |
|:-:|:-:|:-:|:-:|---|
| 1 (start) | 99 | ≈ 0.007 | 0.08 | Pure noise. x̂₀ = x_t/0.08 → meaningless |
| 2 | 88 | ≈ 0.020 | 0.14 | Still nearly pure noise |
| 3 | 77 | ≈ 0.055 | 0.23 | Very noisy |
| 4 | 66 | ≈ 0.116 | 0.34 | Noisy |
| 5 | 55 | ≈ 0.213 | 0.46 | Weak signal emerging |
| 6 | 44 | ≈ 0.346 | 0.59 | Partial arc visible |
| 7 | 33 | ≈ 0.569 | 0.75 | Getting meaningful |
| **8** | **22** | **≈ 0.775** | **0.88** | **x̂₀ mostly reliable** |
| **9** | **11** | **≈ 0.935** | **0.97** | **x̂₀ reliable — render here** |
| 10 (end) | 0  | ≈ 1.000 | 1.00 | Clean action chunk |

x̂₀ = (x_t − √(1−ᾱ_t)·ε_t) / √ᾱ_t. At step 1, √ᾱ_t ≈ 0.08 — we divide by 0.08,
amplifying random noise ×12. The resulting "predicted clean action" is garbage.
Only at steps 8–9 (t=22,11) is x̂₀ a geometrically interpretable 32×5 action sequence
whose first 10 rows form a legible or non-legible arc.

### What ε_cond and ε_uncond look like at each DDIM step

At DDIM step 1 (t=99, pure noise):
- ε_θ(x_t, t=99, obs_true) ≈ x_t  (model recovers the noise; obs barely matters at t=99)
- ε_θ(x_t, t=99, obs_sym) ≈ x_t   (symmetrised obs also barely matters)
- Difference: ε_cond − ε_uncond ≈ **0**
- CFG amplification: w × 0 = **nothing** (regardless of w)

At DDIM step 9 (t=11, nearly clean):
- The model's ε_θ is now a small residual correction; obs content matters
- obs_true (left block at Y=+0.07) pulls ε toward left-reaching arc
- obs_sym (both blocks at midpoint) pulls ε toward centre-ambiguous arc
- Difference: ε_cond − ε_uncond = **meaningful left/right direction vector**
- CFG amplification: w × (left_direction − centre_direction) → sharpens legibility

**Practical consequence:** applying CFG at DDIM steps 1–5 is wasted compute. The
direction vector is near-zero and the denominator (√ᾱ) is small, making any gradient
clipping meaningless. CFG should be applied only at DDIM steps 7–10.

---

## Concrete Algorithm: VLM-Conditional CFG (VC-CFG)

This is the proposed algorithm. It uses:
1. Obs-symmetric dropout → unconditional side (no retraining, no VLM)
2. VLM-weighted averaging over K chains → legibility-conditional side (VLM as the
   estimator of the legibility-conditional distribution)
3. Direct Gemini API + matplotlib rendering — **no `gemini_vlm_eval` folder dependency**

### Why existing VLM eval (`pA`, `pB`) is NOT the CFG conditional

`pA` and `pB` from `tools/vlm_client.py` are unreliable because:
- They require `gemini_vlm_eval.client.evaluate_frame` which applies a fixed prompt
  template designed for post-hoc evaluation, not mid-sampling noise prediction
- The PyBullet 3D camera frame (yaw=135°) has left/right reversal — the VLM confuses
  image-space and world-space directions unless the camera calibration is embedded in prompt
- The VLM is being asked to classify legibility from a 3D camera view when what it
  actually needs is an unambiguous top-down diagram of the trajectory arc shape

`pA`/`pB` IS appropriate for final-step evaluation (which is what it was designed for).
It is NOT appropriate as the conditional signal inside the DDIM loop.

### The 2D projection rendering (replaces PyBullet frames inside DDIM loop)

Since x̂₀ is already a predicted action sequence, the EE trajectory can be computed
analytically from it — **no simulation stepping needed**:

```python
# From x̂₀_k (shape H×5), compute EE trajectory in 2D (XY plane):
delta_pos = x0_pred[k, :, :3] * ACTION_SCALE          # (H, 3), with ACTION_SCALE=0.05m
ee_traj_3d = torch.cumsum(delta_pos, dim=0) + ee_pos_start  # (H, 3)
ee_xy = ee_traj_3d[:, :2].cpu().numpy()               # (H, 2) — top-down X-Y view

# Plot with matplotlib — ~4ms per image, no PyBullet required:
# Mark: start (star), Block A (blue circle), Block B (red circle),
#       early prefix (first 30%) in thick line, rest as dotted
# This produces an unambiguous top-down trajectory diagram.
```

All K trajectory diagrams are combined into ONE grid image (e.g., 2×3 grid for K=5)
and sent in a single Gemini API call. This avoids dependency on `gemini_vlm_eval`.

### The VLM prompt for legibility-conditional scoring (direct Gemini API)

```python
import google.generativeai as genai

prompt = """
You are evaluating robot arm trajectories for legibility.
The attached image shows {K} candidate trajectories (top-down view of a table).

In each panel:
- STAR (★) = robot start position
- BLUE circle = Block A (target the robot should pick)
- RED circle = Block B (the other block, should be avoided)
- THICK colored line = first 30% of the trajectory (the "early prefix")
- DOTTED line = remainder of the trajectory

Your task: Rate each trajectory from 0.0 to 1.0 on how clearly the first 30%
of motion reveals to an observer that the robot intends to pick Block A.
A score of 1.0 means: from the first 30%, a human would be certain the robot
is going to Block A. A score of 0.0 means: ambiguous or pointing toward Block B.

Output ONLY a JSON list of K floats, e.g.: [0.9, 0.4, 0.7, 0.8, 0.3]
"""

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content([prompt, grid_image_part])
scores = json.loads(response.text)  # [s_1, ..., s_K]
```

No camera calibration text needed. The top-down plot IS the calibration.

### Why `ε_cond_leg = Σ_k softmax(s_k) · ε_θ(x_k, obs)` is NOT a valid CFG conditional

CFG requires both sides evaluated at the **same single latent state**:

```
ε̃(x_t) = ε_uncond(x_t) + w · (ε_cond(x_t) − ε_uncond(x_t))
```

The difference `ε_cond(x_t) − ε_uncond(x_t)` is a **direction vector in noise space at x_t**.
This direction only has meaning when both predictions come from the same input point.

After DDIM steps 1–7, each chain k has diverged to a different `x_k^{s=8}` (different noise
seeds → different latent states). Averaging `ε_θ(x_k, obs)` across K different points doesn't
produce a noise prediction at any single state. It produces a weighted centroid of K
predictions from K different locations — the resulting "direction" has no geometric meaning
for steering any particular chain.

### The correct two-role split: VLM selects, CFG amplifies at the selected state

```
STEP 1 — Run K parallel chains for all DDIM steps 1 → 7:
  Each chain k gets its own noise seed.
  All chains use the same obs (true block positions).
  Apply existing analytic classifier-guidance gradient (eval_legibility_guided.py)
  at steps 1–7 to push chains toward legible directions.
  Cost: K × 7 model forward passes. No VLM call.

STEP 2 — VLM selects the best latent state (step 8):
  For k = 1..K:
    x̂₀_k = (x_k^{s=8} − √(1−ᾱ₈)·ε_θ(x_k^{s=8}, obs)) / √ᾱ₈   ← analytic, no sim
    ee_traj_k = cumsum(x̂₀_k[:, :3] * 0.05) + ee_pos_start          ← no PyBullet
    Render top-down matplotlib 2D diagram for chain k

  Pack all K diagrams into one grid image.
  Single Gemini API call → scores [s_1, ..., s_K]
  k* = argmax(s_k)   ← VLM's job is done here: identify the best latent state x_{k*}^{s=8}

STEP 3 — CFG amplification at x_{k*} (both sides evaluated at THE SAME point):
  # Conditional: model with true obs at x_{k*}
  ε_cond   = ε_θ(x_{k*}^{s=8}, obs)

  # Unconditional: same x_{k*}, but obs with goal-symmetry dropout
  obs_uncond = obs.clone()
  m = (obs[8:11] + obs[15:18]) / 2.0   ← both blocks collapsed to midpoint
  obs_uncond[8:11] = obs_uncond[15:18] = m
  ε_uncond = ε_θ(x_{k*}^{s=8}, obs_uncond)   ← SAME x_{k*}, different obs only

  # CFG direction: both evaluated at x_{k*} → this IS a valid direction vector
  ε_guided = ε_uncond + w_cfg · (ε_cond − ε_uncond)
  where w_cfg = 3.0 (tunable)

  # Apply to x_{k*} for DDIM steps 8–10:
  x̂₀_guided = (x_{k*}^{s=8} − √(1−ᾱ₈)·ε_guided) / √ᾱ₈
  x^{s=9} = √ᾱ₉ · x̂₀_guided + √(1−ᾱ₉) · ε_guided
  Continue DDIM steps 9–10 with single chain, no additional guidance.
  Execute resulting chunk for 8 env steps.
```

**What each component does:**
- Analytic guidance (steps 1–7): pushes all K chains into legible regions of latent space
  before any VLM is involved. Cheap and differentiable.
- VLM (step 8): identifies which chain k* landed in the MOST legible region. Selection only.
- CFG (steps 8–10): at the already-good state x_{k*}, amplifies the goal-directedness axis
  (ε_cond − ε_uncond) by w_cfg. This sharpens commitment without the VLM being in the loop.

The VLM constructs the conditional **not** by building a noise prediction, but by identifying
which latent state to evaluate the conditional prediction at. The CFG formula then correctly
uses that state for both sides of the amplification.

### What this algorithm provides that is genuinely new

| Property | Best-of-K (current best) | VC-CFG |
|----------|--------------------------|--------|
| VLM sees rendered images | Yes, post-hoc | Yes, mid-DDIM |
| VLM shapes noise prediction | No (only selects) | **Yes** — ε_guided is a blend |
| Unconditional side | N/A | obs-symmetric dropout |
| CFG amplification of gap | No | **Yes** — w_cfg·(ε_cond − ε_uncond) |
| gemini_vlm_eval dependency | Yes | **No** — direct Gemini API |
| Requires retraining | No | No |
| VLM input quality | PyBullet 3D frame | **2D top-down diagram** |

### Code changes required

| File | Change |
|------|--------|
| `evaluation/eval_legibility_guided.py` | Replace `LPSDDIMSampler.sample()` with `VCCFGSampler.sample()` implementing Steps 1–5 above. |
| `evaluation/vlm_cfg_renderer.py` (new) | `render_topdown_grid(ee_trajs, block_A_pos, block_B_pos, ee_start, K)` → returns matplotlib PNG bytes. |
| `evaluation/vlm_cfg_scorer.py` (new) | Direct Gemini API call using `google.generativeai`. No `gemini_vlm_eval` import. |

### Open empirical questions

1. Does the VLM-weighted ε_cond meaningfully differ in direction from the best single-chain
   ε_k? Measure cos-similarity between ε_cond and each ε_k. If ε_cond ≈ ε_{k*}, then
   VLM-weighting adds nothing over selection.
2. Does the CFG amplification step (ε_uncond + w·(ε_cond − ε_uncond)) with obs-symmetric
   dropout improve L_early beyond ε_cond alone? This tests whether the unconditional
   contrast is actually useful.
3. What w_cfg maximises L_early without trajectory divergence? Test w ∈ {1, 2, 3, 5}.

---

## What NOT to propose

- Do not use `pA`/`pB` from `gemini_vlm_eval` as the CFG conditional — they are
  unreliable for mid-DDIM scoring due to camera perspective confusion in PyBullet frames.
- Do not step the PyBullet environment to render x̂₀ — the 2D matplotlib projection of
  the predicted EE trajectory is faster, more interpretable, and requires no camera calibration.
- Do not propose token/action-level VLM feedback — the VLM operates on rendered 2D diagrams.
- Do not propose final-step-only evaluation as a new contribution — already achieved 0.972.
- Do not suggest retraining as the primary contribution.
