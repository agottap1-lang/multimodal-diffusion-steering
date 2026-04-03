# Hypothesis: VLM-Synthesized Guidance Functions for Diffusion Trajectory Planning

> **⚠️ CORRECTION**: All references to "DPS" and "LPS" in this document should
> read "classifier guidance". See `HONEST_ASSESSMENT.md` for full details.

## Context & Motivation

We have empirically shown (April 1, 2026) that DPS-style gradient guidance during DDIM sampling **works well** for improving trajectory legibility:

| Method | Success | L_early | Notes |
|--------|---------|---------|-------|
| Baseline (DDIM) | 95% | 0.906 | No guidance |
| **LPS w=10 (hand-crafted L_early)** | **100%** | **0.952** | Current best DPS result |
| LegDiff CFG w=3 | 100% | 0.935 | Required separate training |

The hand-crafted guidance function (`l_early_intent_torch`) uses a Gaussian observer model with auto-calibrated σ. This function was designed by a human researcher based on Dragan et al. (HRI 2013). 

**Research Question:** Can a Vision-Language Model (VLM) generate *better* guidance functions than the hand-crafted one, capturing richer legibility criteria that go beyond simple Gaussian proximity?

## Three-Stage Architecture

### Stage 1: VLM Code Synthesis (Eureka-style)

**What:** Gemini generates a differentiable PyTorch scoring function `vlm_legibility_score(ee_traj, goals, obs)` that captures legibility objectives. The function must:
- Accept `ee_traj: Tensor (H, 3)`, `goals: Tensor (K, 3)`, `obs: Tensor (obs_dim,)`
- Return a differentiable scalar (higher = more legible toward committed goal)
- Enable `torch.autograd.grad()` to backprop through it

**Hypothesis (H_S1):** A VLM can generate a valid, differentiable legibility scoring function that captures geometric cues (approach arc, curvature, clearance) beyond simple Gaussian proximity. We expect the VLM-generated function to include:
- Directional commitment (velocity toward goal)
- Lateral deviation (arc away from non-goal)
- Smooth approach patterns

**Prediction:** The VLM will produce syntactically valid code on the first attempt, but may require 1–2 iterations to:
- Fix gradient-breaking operations (argmax, if/else on tensors)
- Correct sign conventions (maximize vs minimize)
- Calibrate constants to action-space scale

### Stage 2: DPS Integration (Plug-and-Play Guidance)

**What:** Swap `l_early_intent_torch()` with the VLM-generated scoring function inside `LPSDDIMSampler`. The DPS formula remains:

$$\hat{\varepsilon} = \varepsilon_\theta(x_t) - w \cdot \sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} f_{\text{VLM}}(\hat{x}_0(x_t))$$

**Hypothesis (H_S2):** The VLM-generated guidance function, when used for DPS at the optimal scale (w ≈ 10), will achieve:
- Task success ≥ 95% (no degradation from baseline)
- L_early ≥ 0.94 (at least matching hand-crafted DPS)
- Potentially L_early > 0.96 if the VLM captures richer geometric cues

**Prediction:** The VLM-generated function may require guidance scale re-tuning (optimal w may differ from 10.0) because the gradient landscape will have different curvature.

### Stage 3: VLM Reranking (Optional Refinement)

**What:** After DPS generates N candidate trajectories (N=3–5), use a text-based VLM call to select the most legible one. No rendered frames — pass trajectory statistics (arc magnitude, L_early, velocity profile) as text.

**Hypothesis (H_S3):** Text-based VLM reranking can provide an additional +1–2% L_early improvement over DPS alone, at minimal cost (text-only API call, no rendering).

**Prediction:** This stage will provide diminishing returns because DPS at w=10 already pushes trajectories close to the legibility optimum. The main value is as a safety net for edge cases where the guidance function gives poor gradients.

## Stage 1 Implementation Plan

### Step 1: Craft the VLM prompt
Provide Gemini with:
- Task description (TwoBlockPick: robot picks one of two blocks, observer must identify which)
- Observation/action space definition (obs_dim=22, act_dim=5, action_scale=0.05)
- The existing L_early_intent function (as reference)
- Requirements: differentiable, PyTorch, specific signature

### Step 2: Generate and validate code
- Parse VLM output for Python code block
- Validate: syntax check, gradient flow test, output shape/range
- Fix any issues (retry with error feedback, up to 3 attempts)

### Step 3: Unit test the generated function
- Run on known trajectories with known L_early values
- Verify gradient exists and is non-zero
- Check that "obviously legible" trajectories score higher than "ambiguous" ones

### Step 4: Plug into LPSDDIMSampler
- Replace `l_early_intent_torch()` with VLM-generated function
- Re-sweep guidance scale (w ∈ {2, 5, 10, 15, 20})
- Evaluate: 20 episodes per condition

### Step 5: Compare against hand-crafted baseline
- Metrics: success rate, L_early, L_early variance, trajectory smoothness
- Statistical test: paired t-test on L_early values (same env seeds)

## Expected Outcomes (Pre-experiment)

| Metric | Hand-crafted DPS (w=10) | VLM-generated DPS (predicted) |
|--------|------------------------|-------------------------------|
| Success Rate | 100% | ≥ 95% |
| L_early | 0.952 | 0.94–0.97 |
| L_early std | 0.025 | 0.01–0.03 |
| Inference time | ~1.5s/episode | ~1.5s/episode (same DPS loop) |
| VLM API calls | 0 | 1 (offline code gen, amortized) |

---

## Critical Flaws in the Three-Stage Design (April 1, 2026)

After running `experiments/verify_true_lps.py` and reviewing the implementation, several **fundamental methodological flaws** were identified:

---

### Flaw 1: Stage 2 formula is classifier guidance, not DPS (CRITICAL)

The formula shown in Stage 2:

$$\hat{\varepsilon} = \varepsilon_\theta(x_t) - w \cdot \sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} f_{\text{VLM}}(\hat{x}_0(x_t))$$

is **classifier guidance** (Dhariwal & Nichol, NeurIPS 2021), NOT Diffusion Posterior Sampling (Chung et al., ICLR 2023). True DPS is:

$$x'_{t-1} = \text{DDIM}(\varepsilon_\theta), \quad x_{t-1} = x'_{t-1} + \zeta_t \nabla_{x_t} L(\hat{x}_0(x_t))$$

Both `eval_legibility_guided.py` (hand-crafted) and `stage1_vlm_guidance.py` (VLM-guided) implement the same classifier-guidance sampler. The document mislabels both as "DPS" throughout.

**Verification:** `experiments/verify_true_lps.py` confirmed **38.1% step-level divergence** between the two formulations. True DPS with the same guidance scale achieves higher legibility (+2.9%) but **destroys task success (0%)**, while the classifier-guidance variant preserves success at 100%.

**Impact:** All Stage 2 results, including the comparison in "Actual Results", describe classifier guidance behavior, not DPS. The claimed contribution "DPS-style gradient injection" is inaccurate.

---

### Flaw 2: VLM improvement not causally attributed (Stage 1 confound)

The Stage 1 comparison is:

| Method | Criteria | L_early |
|--------|----------|---------|
| Hand-crafted | 1 (Gaussian proximity only) | 0.9431 |
| VLM-generated | 4 (Gaussian + directional + lateral + speed) | 0.9486 |

The +0.6% improvement could be **entirely due to adding 3 extra criteria**, not due to the VLM. There is no ablation comparing:
- Hand-crafted 4-criteria function (add directional + lateral + speed by hand)
- VLM 1-criteria function (just Gaussian, as Gemini might write it)

**Without this ablation, there is no evidence the VLM adds anything beyond what a researcher could implement in 20 lines by hand.**

---

### Flaw 3: Stage 1 validation test is too weak

The discrimination test in `stage1_vlm_guidance.py` checks:

```python
legible_score > ambiguous_score
```

Where "legible" = straight line toward left goal, "ambiguous" = straight line to center.

This test is trivially passed by **any** proximity-based function, including the single-criterion Gaussian baseline. It doesn't test:
- Whether directional/lateral criteria provide additional signal
- Whether the function correctly ranks trajectories that the Gaussian model can't distinguish
- Robustness to scale/position variation

**Any function that increases monotonically toward the goal will pass this test.**

---

### Flaw 4: Stage 3 text reranking is informationally circular

Stage 3 proposes passing trajectory statistics (arc magnitude, L_early, velocity profile) to a VLM as text, then asking it to select the most legible trajectory.

The flaw: **L_early is already a legibility score**. Passing L_early to a VLM and asking "which is most legible" is just asking the VLM to return `argmax(L_early)`. The VLM adds no new information beyond the metrics it receives.

For VLM reranking to be non-trivial, it must either:
a) Observe raw video frames and apply visual legibility judgment not captured by L_early, or
b) Receive trajectory coordinates and compute geometric properties the metrics missed.

Text-only statistics passage does neither. This is the same information already available to a deterministic `argmax`.

---

### Flaw 5: Baseline success inconsistency

| Document | Baseline Success | Baseline L_early |
|----------|-----------------|-----------------|
| `DPS_HYPOTHESIS.md` (April 1, 2026) | **95%** | 0.9055 |
| `STAGE123_HYPOTHESIS.md` Stage 1 (April 1, 2026) | **85%** | 0.9063 |

Both claim 20 episodes, same checkpoint, same `cube_jitter=0.0`. The 10% success gap is unexplained. If baselines are not reproducible, neither are the delta improvements claimed.

---

### Flaw 6: Context table uses different numbers than experiment results

The introductory context table claims:

> LPS w=10 (hand-crafted L_early): L_early **0.952**

But Stage 1 actual results show:

> Hand-crafted LPS: L_early **0.9431**

These are different runs of the same method on the same checkpoint. The introductory table (from DPS_HYPOTHESIS.md) is used to motivate the Stage 1 baseline the VLM needs to beat — but that baseline (0.952) was actually never reproduced within Stage 1 (which shows 0.9431). The VLM result (0.9486) does not beat the original claimed 0.952.

---

## Actual Results — Stage 1 (April 1, 2026)

**20 episodes per condition, w=10.0, n_sampling_steps=10, grad_clip=1.0**

| Method | Success | L_early (mean ± std) | Δ vs Baseline |
|--------|---------|---------------------|---------------|
| **Baseline (DDIM)** | **85%** | **0.9063 ± 0.066** | — |
| **Hand-crafted LPS** | **100%** | **0.9431 ± 0.015** | +4.1% |
| **VLM-generated DPS** | **100%** | **0.9486 ± 0.020** | +4.7% |

### VLM-Generated Function Details

Gemini (gemini-2.5-flash) generated a **4-criterion** scoring function on the first attempt:

1. **Gaussian Proximity (P_prox, w=0.35):** Bayesian posterior P(g*|x) — same as hand-crafted baseline
2. **Directional Alignment (P_dir, w=0.30):** Cosine similarity between velocity and goal direction
3. **Lateral Separation (P_lat, w=0.25):** Sigmoid of signed projection onto goal-separation axis
4. **Speed Commitment (P_speed, w=0.10):** Sigmoid of velocity magnitude vs characteristic speed

### Key Findings

1. **H_S1 CONFIRMED:** Gemini generated valid, differentiable code on the first attempt — no retries needed
2. **H_S2 CONFIRMED:** VLM-guided DPS achieves 100% success and L_early ≥ 0.94 (matching prediction)
3. **VLM guidance slightly beats hand-crafted:** L_early 0.9486 vs 0.9431 (+0.6%). The multi-criteria approach captures richer geometric cues
4. **Both DPS variants dramatically outperform baseline:** +15% success rate improvement + +4.7% legibility
5. **Zero retraining, single VLM call:** The function is generated once offline and amortized over all episodes

---

## Flaw Verification Results (experiments/verify_stage123_flaws.py, April 1, 2026)

10 episodes per condition, w=10.0.

### Flaw 3 Confirmed: Discrimination test is trivially easy

| Test case | 1-criteria HC | 4-criteria HC | Result |
|-----------|-------------|-------------|--------|
| Case A (Stage 1 original: straight-to-goal vs center) | +0.087 ✓ | +0.052 ✓ | **Both pass** |
| Case B (hard: equidistant start, direction only) | +0.175 ✓ | +0.222 ✓ | **Both pass** |

**The 1-criteria Gaussian-only function passes both tests, including the hard case.**  
The Stage 1 validation passes any function that monotonically approaches the goal.

### Flaw 5 Confirmed (partially): Baseline is not reproducible at a fixed value

| Run | Success | L_early |
|-----|---------|---------|
| DPS_HYPOTHESIS.md | 95% | 0.906 |
| Stage 1 original | 85% | 0.906 |
| verify_stage123_flaws.py (10 ep) | **90%** | **0.893** |

Baseline success is noisy: **~88–95% across runs**. The 85% used in Stage 1 is a downward outlier; 95% in DPS_HYPOTHESIS.md is an upward outlier. Neither is reliably reproducible.

### Flaw 2 Inconclusive: 4-criteria HC does NOT beat 1-criteria HC

| Method | Success | L_early |
|--------|---------|---------|
| Baseline (DDIM) | 90% | 0.8928 |
| 1-criteria HC (Gaussian, w=10) | 100% | **0.9421 ± 0.022** |
| 4-criteria HC ablation (hand-crafted, w=10) | 100% | 0.9274 ± 0.023 |

**The hand-crafted 4-criteria function scores _lower_ than 1-criteria by −0.015.**  
This means:
- Adding directional/lateral/speed criteria does **not** improve legibility in execution
- The 4-criteria function has a more complex gradient landscape that may interfere with the DDIM update
- The VLM +0.6% result (Stage 1) is therefore **not explainable by multi-criteria** either — it may just be run-to-run noise (within the ~2.2% std of each condition)

**Revised Flaw 2 verdict:** The VLM result (+0.6% over hand-crafted) is within the noise floor (std ≈ 2%). There is no statistical evidence that VLM-generated code outperforms hand-crafted 1-criteria guidance.

---

## Novelty Statement

No prior work combines:
1. A VLM generating differentiable trajectory scoring code (Eureka for trajectories)
2. DPS-style gradient injection into diffusion action denoising (DPS for robot policies)
3. Zero retraining — works on any pre-trained diffusion policy checkpoint

This is distinct from:
- **Eureka** (Ma et al.): VLM writes RL rewards → requires RL training loop
- **DPS** (Chung et al.): Hand-crafted measurement model → no VLM
- **Legibility Diffuser** (Bronars et al.): Requires retraining → not plug-and-play
- **Code as Policies** (Liang et al.): LLM writes control code → no diffusion guidance

**Note:** The gradient injection used here is classifier guidance (Dhariwal & Nichol 2021), which also requires no retraining. The novelty over plain classifier guidance is the use of a VLM to synthesize the guidance function — but as Flaw 2 shows, this synthesis provides no measurable benefit over a hand-crafted 1-criterion function.
