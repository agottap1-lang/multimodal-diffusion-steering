# Week 4: Guidance Methods for Legible Diffusion

## Training-Free Steering of Trajectory Distribution

**Anudeep Gottapu**
Arizona State University

---

## Slide 1: The Guidance Idea

**Key Insight:** Diffusion models generate samples through iterative denoising. At each step, we can add gradient information to steer the trajectory toward desired properties — **without retraining.**

Four methods implemented and compared:
1. **Classifier Guidance** — Modify noise prediction with legibility gradients
2. **Best-of-N Reranking** — Sample multiple, pick best
3. **Classifier-Free Guidance (LegDiff)** — Conditional/unconditional interpolation
4. **VLM Text Reranking** — VLM selects most legible video

---

## Slide 2: Method 1 — Classifier Guidance

**Formulation (Dhariwal & Nichol, 2021)**

$$\hat{\epsilon}_\theta(a_t, t) = \epsilon_\theta(a_t, t) - w \sqrt{1 - \bar{\alpha}_t} \nabla_{a_t} L_{\text{score}}(a_t)$$

- $w$ = guidance scale (controls legibility-task tradeoff)
- $L_{\text{score}}$ = differentiable legibility function
- Applied at every DDIM denoising step

**Scoring Function** (VLM-generated, EUREKA-style):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| P_prox | 0.35 | Gaussian proximity to goal |
| P_dir | 0.30 | Velocity alignment toward goal |
| P_lat | 0.25 | Lateral separation from non-goal |
| P_speed | 0.10 | Speed commitment |

Correlation with hand-crafted baseline: **r = 0.992**

---

## Slide 3: Classifier Guidance — Scale Sweep Results

| Scale w | Success Rate | L_early (mean ± std) |
|---------|:-----------:|:--------------------:|
| 0 (baseline) | 95% | 0.906 ± 0.044 |
| 2 | 95% | 0.911 ± 0.064 |
| 5 | 90% | 0.946 ± 0.019 |
| **10** | **100%** | **0.952 ± 0.025** |
| 20 | 100% | 0.948 ± 0.028 |

**Key finding:** w=10 is optimal — 100% success + highest L_early. Above w=10, diminishing returns with slight L_early decrease.

---

## Slide 4: Terminology Correction — NOT DPS

**Important:** Throughout development, this was called "DPS" or "LPS." This is **wrong**.

| | Classifier Guidance (Ours) | True DPS (Chung 2023) |
|---|---|---|
| Where gradient applied | Noise prediction ε_θ | Denoised sample x_{t-1} |
| Formula | $\hat{\epsilon} = \epsilon_\theta - w\sqrt{1-\bar{\alpha}_t}\nabla L$ | $x_{t-1} = \text{DDIM}(x_t) + \zeta_t \nabla L$ |
| Our result | **100% success** | **0% success** |
| Step divergence | 0% | 38% |

True DPS achieved higher L_early (0.969 vs 0.941) but completely failed the task. **All reported results use classifier guidance.**

---

## Slide 5: Method 2 — Best-of-N Reranking

**Simple but effective:** Sample N trajectories, score each with L_early, execute the best.

| N | L_early (mean ± std) | Δ vs Baseline | Path Efficiency |
|---|:--------------------:|:-------------:|:---------------:|
| 1 | 0.732 ± 0.088 | — | 0.515 |
| 4 | 0.779 ± 0.062 | +6.4% | 0.408 |
| 8 | 0.797 ± 0.051 | +8.8% | 0.374 |
| **16** | **0.806 ± 0.049** | **+10.1%** | 0.354 |

**Trade-off:** More legible ↔ less path-efficient (exaggerated motions). Diminishing returns above N=16.

---

## Slide 6: Method 3 — LegDiff (Classifier-Free Guidance)

**Formulation (Ho & Salimans, 2022)**

$$\hat{\epsilon} = \epsilon_\theta(a_t, \emptyset) + w \cdot (\epsilon_\theta(a_t, G) - \epsilon_\theta(a_t, \emptyset))$$

- Train both conditional (goal-aware) and unconditional models
- Interpolate at inference with scale w
- Uses Conv1d backbone with temporal convolutions (kernel=5)

**Results (20 episodes, w=3):**

| Condition | Success | L_early |
|-----------|:-------:|:-------:|
| Baseline | 100% | 0.922 |
| **LegDiff** | **100%** | **0.935** |

Modest but consistent improvement. All committed goals match true goals.

---

## Slide 7: Method 4 — VLM Text-Based Reranking

**Combines guidance + VLM judgement**

1. Generate K=5 trajectory candidates using classifier guidance
2. Render each as video
3. VLM (Gemini) scores each for goal identifiability
4. Execute the trajectory VLM most confidently identifies

**Results:**

| Condition | Success | L_early |
|-----------|:-------:|:-------:|
| DPS single | 85% | 0.946 |
| Oracle rerank | 95% | 0.968 |
| **VLM rerank** | **100%** | **0.972** |

VLM reranking achieves the best L_early of any method.

---

## Slide 8: Full Pipeline (Staged)

**3-Stage Improvement**

| Stage | Method | Success | L_early |
|-------|--------|:-------:|:-------:|
| 0 | Baseline | 80% | 0.898 |
| 1 | + Classifier guidance (w=10) | 100% | 0.937 |
| 2 | + VLM reranking (K=5) | **100%** | **0.972** |

- VLM vs. Baseline: **Δ = +0.039, p = 0.00042** (statistically significant)
- **Total improvement: +7.4 percentage points** in L_early

---

## Slide 9: Comparison of All Methods

| Method | Requires Training | Success | L_early | Best For |
|--------|:-----------------:|:-------:|:-------:|----------|
| Base policy | Yes | 84% | 0.732 | Simplest baseline |
| Best-of-16 | No | — | 0.806 | Gradient-free setting |
| Classifier guidance | No | **100%** | 0.952 | Best single-method |
| LegDiff (CFG) | Yes | 100% | 0.935 | Principled conditioning |
| VLM reranking | No | **100%** | **0.972** | Maximum legibility |

---

## Slide 10: VLM-Generated vs Hand-Crafted Scoring

**Honest assessment of the EUREKA-style scoring function**

- Generated by Gemini from **text-only** prompt (no visual data)
- Single-shot, no iterative refinement (not full EUREKA protocol)
- Achieves r=0.992 correlation with hand-crafted 4-criteria function
- **Any capable text LLM could produce equivalent output**
- Claimed as: valid use of LLM code generation, NOT a visual contribution

---

## Slide 11: References

- Dhariwal, P. & Nichol, A. (2021). *Diffusion Models Beat GANs.* NeurIPS.
- Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance.* NeurIPS Workshop.
- Chung, H., et al. (2023). *Diffusion Posterior Sampling.* ICLR.
- Janner, M., et al. (2022). *Planning with Diffusion.* ICML.
- Ma, Y. J., et al. (2024). *EUREKA.* ICLR.
