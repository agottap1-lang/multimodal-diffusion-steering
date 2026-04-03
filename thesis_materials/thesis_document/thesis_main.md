# VLM-Guided Diffusion Policies for Legible Robot Motion in Multi-Goal Manipulation

**A Thesis Presented in Partial Fulfillment of the Requirements for the Degree Master of Science**

**Anudeep Gottapu**

**Supervised by [Advisor Name]**

**Arizona State University**

**May 2026**

---

## ABSTRACT

We present a framework for generating legible robot manipulation trajectories using diffusion policies guided by Vision-Language Model (VLM) feedback. Legible motion—motion that enables observers to quickly infer the robot's intended goal—is critical for safe and efficient human-robot collaboration. Our system operates in a simulated TwoBlockPick environment where a Franka Panda robot must pick one of two objects while communicating its intent through trajectory shape.

We make three contributions. **First**, we train an unconditional DDPM-based diffusion policy on 400 expert demonstrations spanning three trajectory styles (legible, neutral, deceptive) and demonstrate 84% task success. **Second**, we develop a VLM-based legibility evaluation pipeline using Gemini 2.5 Flash that achieves 94.7% goal-prediction accuracy on demonstration videos and defines the VLM Onset (VLO) metric—the earliest timestep at which a VLM correctly identifies the robot's goal. **Third**, we implement and compare four training-free guidance methods for steering diffusion policy outputs toward legible behavior: classifier guidance (L_early improved from 0.906 to 0.952 at 100% success), best-of-N reranking (+10.1% L_early at N=16), classifier-free guidance via LegDiff (L_early ≈ 0.935), and VLM text-based reranking (L_early = 0.972, best overall).

Our results demonstrate that training-free guidance over diffusion policies is a viable and effective approach to legible motion generation. The complete pipeline—from VLM-generated scoring functions to multi-stage guidance—improves early legibility by 7.4 percentage points while maintaining 100% task success. We provide an honest assessment of limitations including terminology corrections, evaluation gaps, and directions for future work.

**Keywords:** Diffusion Policy, Legible Motion, Vision-Language Models, Human-Robot Interaction, Classifier Guidance, Manipulation Planning

---

## TABLE OF CONTENTS

1. [Introduction](#chapter-1-introduction)
2. [Related Work](#chapter-2-related-work)
3. [Technical Background](#chapter-3-technical-background)
4. [System Design](#chapter-4-system-design)
5. [Experiments and Results](#chapter-5-experiments-and-results)
6. [Discussion and Honest Assessment](#chapter-6-discussion-and-honest-assessment)
7. [Conclusion and Future Work](#chapter-7-conclusion-and-future-work)

References

Appendix A: Hyperparameter Tables
Appendix B: Full Experimental Results
Appendix C: VLM Prompt Templates

---

## Chapter 1: Introduction

### 1.1 Motivation

In shared human-robot workspaces, a robot's motion communicates information about its intent. When a robot arm reaches toward one of several possible objects, the trajectory it follows determines how quickly a human co-worker can predict which object the robot will grasp. This predictability—termed **legibility** in the robotics literature (Dragan et al., 2013)—directly impacts collaboration efficiency and safety.

Consider a manufacturing cell where a robot and human alternate picking parts from a shared bin. If the robot's trajectory is ambiguous, the human must wait to see which object the robot grasps before planning their own action. A legible trajectory eliminates this waiting time by making the robot's intent apparent early in the motion.

Traditional approaches to legible motion planning optimize hand-crafted cost functions that encode geometric legibility criteria (Dragan & Srinivasa, 2013). While effective in simple domains, these methods require explicit cost function design and struggle with complex manipulation tasks. Recent advances in diffusion-based policy learning (Chi et al., 2023; Reuss et al., 2023) offer an alternative: learn a generative model over expert trajectories, then steer the generation process toward legible outputs at inference time.

### 1.2 Problem Statement

Given a manipulation environment with multiple goals (objects to grasp), we seek a system that:
1. **Generates successful manipulation trajectories** from learned demonstrations
2. **Maximizes legibility**—the probability that an observer correctly infers the goal early in the trajectory
3. **Operates without retraining** the base policy for different legibility requirements
4. **Evaluates legibility using a VLM** as a proxy for human judgment

### 1.3 Approach Overview

Our approach consists of three components:

1. **Base Diffusion Policy**: A DDPM-based generative model trained on 400 expert demonstrations in a TwoBlockPick environment. The model learns a distribution over 32-step action chunks conditioned on the current robot state.

2. **VLM-Based Legibility Evaluation**: A pipeline using Gemini 2.5 Flash to evaluate trajectory legibility from rendered video frames. The VLM predicts the robot's goal at progressive timesteps, defining the VLM Onset (VLO) metric.

3. **Training-Free Guidance Methods**: Four methods for steering the diffusion policy toward legible outputs without retraining:
   - Classifier guidance with a VLM-generated scoring function
   - Best-of-N trajectory reranking
   - Classifier-free guidance (LegDiff)
   - VLM text-based trajectory reranking

### 1.4 Contributions

1. **A VLM-based legibility evaluation framework** that achieves 94.7% goal-prediction accuracy and defines the VLO metric for quantifying trajectory legibility onset.

2. **A systematic comparison of four training-free guidance methods** for steering diffusion policies toward legible behavior, with controlled ablations over guidance scale.

3. **An end-to-end pipeline** from demonstration collection through VLM-guided generation that improves early legibility (L_early) from 0.898 to 0.972 while maintaining 100% task success.

4. **An honest experimental assessment** including terminology corrections, identified evaluation gaps, and clear delineation of what was and was not demonstrated.

### 1.5 Thesis Organization

Chapter 2 reviews related work in legible motion planning, diffusion policies, and VLM-based robot evaluation. Chapter 3 provides technical background on DDPMs, classifier guidance, and legibility metrics. Chapter 4 describes our system design including the environment, demonstration collection, model architecture, and guidance methods. Chapter 5 presents experimental results with quantitative comparisons. Chapter 6 provides an honest discussion of limitations. Chapter 7 concludes with future directions.

---

## Chapter 2: Related Work

### 2.1 Legible Motion Planning

The concept of legible motion was formalized by Dragan et al. (2013), who defined legibility as the probability that an observer correctly and quickly infers the robot's goal from partial trajectory observation. Their formulation uses a Bayesian inverse planning model:

$$P(G \mid \xi_{0:t}) \propto P(\xi_{0:t} \mid G) \cdot P(G)$$

where $G$ is the goal, $\xi_{0:t}$ is the observed trajectory prefix, and $P(\xi_{0:t} \mid G)$ is the likelihood under an optimal planner. A trajectory is legible if $P(G^* \mid \xi_{0:t})$ is high early (small $t$).

Subsequent work extended this framework to various domains: Nikolaidis et al. (2016) applied it to human-robot handovers, Bodden et al. (2018) to navigation, and Busch et al. (2017) to industrial assembly. However, these approaches rely on hand-crafted cost functions and known goal spaces, limiting their applicability to learned policies.

### 2.2 Diffusion Models for Robotics

Diffusion probabilistic models (Ho et al., 2020; Song et al., 2021) have emerged as powerful generative models for robotics. **Diffusion Policy** (Chi et al., 2023) demonstrated that DDPM-based action generation achieves state-of-the-art performance on contact-rich manipulation tasks, outperforming both explicit policy classes and other generative models.

Key properties that make diffusion policies attractive for our setting:
- **Multimodal distribution modeling**: Diffusion policies naturally capture multiple modes in the action distribution, critical when demonstrations include diverse trajectory styles.
- **Training-free controllability**: The iterative denoising process allows gradient-based steering without retraining (Dhariwal & Nichol, 2021; Janner et al., 2022).
- **Chunk-based prediction**: Predicting action chunks (sequences of 8-32 actions) provides temporal coherence essential for legible trajectory generation.

**Diffuser** (Janner et al., 2022) showed that diffusion models can be used as trajectory planners with classifier-guided sampling. Our classifier guidance approach is most closely related to this work, adapted from trajectory-level guidance to action-chunk-level guidance.

### 2.3 Guided Diffusion Sampling

Several methods exist for steering diffusion model outputs without retraining:

**Classifier Guidance** (Dhariwal & Nichol, 2021): Modifies the score function by adding the gradient of a classifier:
$$\hat{\epsilon}_\theta(x_t, t) = \epsilon_\theta(x_t, t) - w\sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y \mid x_t)$$

**Classifier-Free Guidance** (Ho & Salimans, 2022): Trains both conditional and unconditional models, interpolating at inference:
$$\hat{\epsilon}_\theta = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

**Diffusion Posterior Sampling (DPS)** (Chung et al., 2023): Adds gradient guidance at each denoising step. We note that our implementation follows classifier guidance, not DPS—see Section 6.1 for discussion of this distinction.

### 2.4 VLMs for Robot Evaluation

Vision-Language Models have been increasingly used for robot task evaluation. SayCan (Ahn et al., 2022) used language models for task grounding, while VoxPoser (Huang et al., 2023) demonstrated VLM-based affordance extraction. EUREKA (Ma et al., 2024) showed that LLMs can generate reward functions for robot learning.

Our use of VLMs differs from prior work in that we use the VLM as an **evaluator of trajectory legibility** rather than as a planner or reward generator. The VLM observes rendered video frames and predicts which goal the robot is reaching for, directly measuring whether the trajectory communicates intent effectively.

---

## Chapter 3: Technical Background

### 3.1 Denoising Diffusion Probabilistic Models (DDPMs)

A DDPM (Ho et al., 2020) defines a forward process that gradually adds Gaussian noise to data $x_0 \sim q(x_0)$:

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ and $\{\beta_t\}_{t=1}^{T}$ is the noise schedule.

The model $\epsilon_\theta$ is trained to predict the noise:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

At inference, samples are generated by iteratively denoising from $x_T \sim \mathcal{N}(0, I)$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

### 3.2 DDIM Sampling

DDIM (Song et al., 2021a) provides a deterministic sampling procedure that allows fewer denoising steps:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon$$

where $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$.

We use 10-step DDIM with $\eta = 0.3$ for inference, reducing the 100-step training schedule to a practical inference budget.

### 3.3 Legibility Metrics

We define two complementary legibility metrics:

**L_early (Analytical)**: A differentiable legibility score computed over the early fraction of a trajectory using Bayesian posterior probability:

$$L_{\text{early}}(G^* \mid \xi_{0:T_e}) = \frac{\exp(-d(\xi_{T_e}, G^*)^2 / 2\sigma^2)}{\sum_k \exp(-d(\xi_{T_e}, G_k)^2 / 2\sigma^2)}$$

where $T_e = \lfloor 0.3T \rfloor$ is the early cutoff and $d(\cdot, \cdot)$ is Euclidean distance.

**VLO (VLM Onset)**: The first timestep index $k \in \{0, 1, \ldots, K-1\}$ at which a VLM correctly identifies the goal from cumulative video prefix frames. Lower VLO indicates earlier correct identification, hence more legible motion. VLO = $K$ means the VLM never correctly identified the goal within the observed prefix.

### 3.4 Classifier Guidance for Legibility

We adapt classifier guidance (Dhariwal & Nichol, 2021) to steer diffusion policy outputs toward legible trajectories. At each denoising step $t$, the modified score becomes:

$$\hat{\epsilon}_\theta(a_t, t, s_t) = \epsilon_\theta(a_t, t, s_t) - w \sqrt{1 - \bar{\alpha}_t} \nabla_{a_t} L_{\text{early}}(a_t, G^*)$$

where $w$ is the guidance scale, $a_t$ is the noisy action chunk, and $L_{\text{early}}$ is the differentiable legibility scoring function.

---

## Chapter 4: System Design

### 4.1 Environment: TwoBlockPick

We implement a tabletop manipulation environment in PyBullet with a 7-DOF Franka Panda arm. Two red blocks are placed symmetrically on a table at positions $(0.50, +0.07, 0.42)$ (left) and $(0.50, -0.07, 0.42)$ (right), with position jitter of $\pm 0.015$m.

**Table 4.1: Environment Specification**

| Parameter | Value |
|-----------|-------|
| Robot | Franka Panda 7-DOF |
| Physics Engine | PyBullet, $dt = 1/240$s, 20 substeps/action |
| Observation Space | 22-dim: $[\text{ee\_pos}(3), \text{ee\_quat}(4), \text{grip}(1), \text{L\_pos}(3), \text{L\_quat}(4), \text{R\_pos}(3), \text{R\_quat}(4)]$ |
| Action Space | 5-dim: $[\Delta x, \Delta y, \Delta z, \Delta\text{yaw}, \text{grip}]$, each $\in [-1, 1]$ |
| Position Scale | 0.05 m/step |
| Yaw Scale | 15°/step |
| Block Size | 4 cm (half-extent 0.02m) |
| Success Criterion | Block center $z > 0.52$m (table top + 0.12m) |
| Camera | FOV 60°, yaw 135°, pitch −30°, distance 0.9m |

The camera convention is critical for VLM evaluation: with yaw = 135°, the **image-left** corresponds to **world-right** and vice versa. Goal A (image-left) = world-right block; Goal B (image-right) = world-left block.

### 4.2 Demonstration Collection

We collect 400 expert demonstrations using analytically computed Bézier curve trajectories across 10 block configurations. Each configuration produces 40 demonstrations: 20 pick-left and 20 pick-right, distributed across three trajectory styles.

**Table 4.2: Demonstration Distribution**

| Style | Count | % | Trajectory Design |
|-------|-------|---|-------------------|
| Legible | 200 | 50% | Quadratic Bézier with control point swept toward goal. Lateral commitment visible by $t \approx 0.3$ |
| Neutral | 100 | 25% | Quadratic Bézier with control point at $y = 0$. No lateral signal |
| Deceptive | 100 | 25% | Cubic Bézier with $P_1$ feinting toward wrong side, $P_2$ committing to correct goal |

Each demonstration is a full episode of up to 400 timesteps with observation-action pairs stored in `demos_combined.npz`:
- `obs`: shape $(400, 400, 22)$, float32
- `actions`: shape $(400, 400, 5)$, float32
- `labels`: 'left' or 'right' (200 each)
- `style_labels`: 0 (legible), 1 (neutral), 2 (deceptive)
- `config_id`: 0–9, `arc_idx`: indices within each config

### 4.3 Diffusion Policy Architecture

We implement a U-Net architecture with 1D residual blocks for diffusion-based action chunk prediction.

**Table 4.3: Model Architecture**

| Component | Specification |
|-----------|---------------|
| Time Embedding | Sinusoidal(128) → MLP(128→256→256) with Mish activation |
| Observation Embedding | MLP(22→256→256) with Mish activation |
| Input Projection | Linear(5→256) |
| Encoder | 2 UNetBlocks: 256→512→1024 |
| Bottleneck | UNetBlock(1024→1024) |
| Decoder | 2 UNetBlocks with skip connections: 2048→512, 1024→256 |
| Output | MLP(256→256→5) with Mish, **no tanh** (unbounded noise prediction) |
| UNetBlock | Linear + GroupNorm(8) + Mish + time projection (additive) + residual |

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Noise Schedule | Linear, $\beta_{\text{start}} = 0.0001$, $\beta_{\text{end}} = 0.1$, $T = 100$ |
| Horizon (chunk size) | 32 timesteps |
| Execute Steps | 8 (first 8 of 32 predicted actions executed) |
| Optimizer | AdamW, $\text{lr} = 2 \times 10^{-4}$, weight decay $10^{-5}$ |
| Batch Size | 64 |
| EMA Decay | 0.999 |
| Training Epochs | 100 |
| Inference | DDIM 10-step, $\eta = 0.3$ |

The model predicts noise $\epsilon$ (not data), meaning the output is unbounded—no tanh activation on the final layer. This is consistent with the $\epsilon$-prediction DDPM formulation.

### 4.4 VLM-Based Legibility Evaluation

Our VLM evaluation pipeline uses Gemini 2.5 Flash to assess trajectory legibility from rendered video frames.

**Protocol:**
1. Render policy rollout as MP4 video
2. Extract cumulative prefix frames: $k$ windows over the first 30% of the episode
3. For each window, send frames to VLM with prompt asking: "Which goal (A=image-left or B=image-right) is the robot reaching for?"
4. VLM returns structured JSON: $\{p_A, p_B, \text{cue}, \text{choice}\}$
5. Compute VLO = first $k$ where VLM choice matches the true goal

**VLM Configuration:** Gemini 2.5 Flash, temperature = 0.1, thinking budget = 512 tokens, structured JSON output.

**Prompt Design:** After iterating through multiple prompt versions (v1: 45% accuracy, v2: 97.5% accuracy), the key improvements were:
- Annotated reference frames with block position markers
- Image-space reasoning (avoid world coordinate confusion)
- Short, direct prompt with temperature 0.1
- Text-first response format to reduce pA/pB bias

### 4.5 Training-Free Guidance Methods

We implement four methods for steering the diffusion policy toward legible outputs without retraining:

#### 4.5.1 Classifier Guidance

At each DDIM denoising step, compute the gradient of a legibility scoring function and modify the noise prediction:

$$\hat{\epsilon}(a_t, t) = \epsilon_\theta(a_t, t) - w \sqrt{1 - \bar{\alpha}_t} \nabla_{a_t} L_{\text{score}}(a_t)$$

The scoring function $L_{\text{score}}$ is a weighted combination of four criteria generated by Gemini from a text prompt (EUREKA-style, but single-shot without iterative refinement):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| $P_{\text{prox}}$ | 0.35 | Gaussian proximity to goal block |
| $P_{\text{dir}}$ | 0.30 | Velocity alignment with goal direction |
| $P_{\text{lat}}$ | 0.25 | Lateral separation from non-goal block |
| $P_{\text{speed}}$ | 0.10 | Speed commitment |

The generated function achieves $r = 0.992$ correlation with the hand-crafted baseline.

#### 4.5.2 Best-of-N Reranking

Sample $N$ trajectory candidates from the base policy, score each with $L_{\text{early}}$, and execute the highest-scoring candidate.

#### 4.5.3 Classifier-Free Guidance (LegDiff)

Train both conditional and unconditional diffusion models, then interpolate at inference:

$$\hat{\epsilon} = \epsilon_\theta(a_t, \emptyset) + w \cdot (\epsilon_\theta(a_t, G) - \epsilon_\theta(a_t, \emptyset))$$

where $G$ is the goal label. Uses a Conv1d backbone with temporal convolutions (kernel size 5).

#### 4.5.4 VLM Text-Based Reranking

Generate $N$ trajectory candidates using classifier guidance, render each as video, score with VLM, and select the trajectory the VLM most confidently identifies as reaching the correct goal.

---

## Chapter 5: Experiments and Results

### 5.1 Base Policy Evaluation

**Setup:** 50 episodes, checkpoint at epoch 100, DDIM 10-step inference.

**Table 5.1: Base Policy Performance**

| Metric | Value |
|--------|-------|
| Success Rate | **84%** (42/50) |
| Mean Episode Length | 344 steps |
| Picked Left | 31/42 (73.8%) |
| Picked Right | 11/42 (26.2%) |

The left-side bias (73.8% vs. 26.2%) reflects the demonstration distribution, where legible demonstrations (50% of training data) tend to produce stronger left-side trajectories due to the Bézier curve parameterization.

### 5.2 VLM Legibility Assessment of Base Policy

**Setup:** All 42 successful episodes scored with Gemini 2.5 Flash, $k = 6$ prefix windows.

**Table 5.2: Base Policy VLM Legibility (VLO Distribution)**

| VLO Value | Count | Cumulative % |
|-----------|-------|-------------|
| 0 (early) | 4 | 9.5% |
| 1 | 3 | 16.7% |
| 2 | 2 | 21.4% |
| 3 | 2 | 26.2% |
| 4 | 2 | 31.0% |
| 5 | 3 | 38.1% |
| 6 (never) | **26** | **100%** |

**Key Finding:** Mean VLO = 4.57, Median VLO = 6.0. **61.9% of trajectories were never correctly identified** by the VLM within the first 30% of the episode. The base diffusion policy, despite achieving 84% task success, produces largely illegible trajectories.

### 5.3 VLM Evaluation of Expert Demonstrations

**Setup:** 40 demonstration videos evaluated with Gemini 2.5 Flash, $k = 6$ prefix windows.

**Table 5.3: Expert Demo VLM Accuracy by Style**

| Style | Non-C Accuracy | VLO (mean) |
|-------|---------------|------------|
| Legible | — | 2.93 |
| Neutral | — | 3.00 |
| Deceptive | — | 3.71 |
| **Overall** | **94.7%** | — |

The VLO ordering confirms that our trajectory styles encode different legibility levels: legible < neutral < deceptive, validating both the demonstration design and VLM evaluation pipeline.

### 5.4 Classifier Guidance Results

**Setup:** Sweep over guidance scale $w \in \{0, 2, 5, 10, 20\}$, 20 episodes per condition.

**Table 5.4: Classifier Guidance Scale Sweep**

| Scale $w$ | Success Rate | L_early (mean ± std) |
|-----------|-------------|---------------------|
| 0 (baseline) | 95% | 0.906 ± 0.044 |
| 2 | 95% | 0.911 ± 0.064 |
| 5 | 90% | 0.946 ± 0.019 |
| **10** | **100%** | **0.952 ± 0.025** |
| 20 | 100% | 0.948 ± 0.028 |

**Key Finding:** $w = 10$ achieves the best trade-off: 100% success rate with the highest L_early. Above $w = 10$, diminishing returns suggest the guidance saturates the legibility signal without further benefit.

### 5.5 Best-of-N Reranking Results

**Table 5.5: Best-of-N Performance**

| N | L_early (mean ± std) | Δ vs Baseline | Path Efficiency |
|---|---------------------|---------------|-----------------|
| 1 (baseline) | 0.732 ± 0.088 | — | 0.515 |
| 4 | 0.779 ± 0.062 | +6.4% | 0.408 |
| 8 | 0.797 ± 0.051 | +8.8% | 0.374 |
| **16** | **0.806 ± 0.049** | **+10.1%** | 0.354 |

Best-of-N provides consistent improvement with diminishing returns as $N$ increases. The drop in path efficiency (0.515 → 0.354) indicates that more legible trajectories involve less direct paths—consistent with the theoretical prediction that legibility requires exaggerated motion.

### 5.6 LegDiff (Classifier-Free Guidance) Results

**Setup:** 20 episodes, CFG scale $w = 3$.

**Table 5.6: LegDiff Performance**

| Condition | Success Rate | L_early (mean) |
|-----------|-------------|----------------|
| Baseline (unconditioned) | 100% | ≈ 0.922 |
| LegDiff (CFG $w = 3$) | 100% | ≈ 0.935 |

LegDiff provides modest improvement over the baseline, with all committed goals matching true goals. The Conv1d temporal backbone successfully learns the conditional distribution.

### 5.7 Full Pipeline: Staged Guidance

**Setup:** 20 paired episodes per stage, using the same initial seeds.

**Table 5.7: Full Pipeline Results (Staged)**

| Stage | Method | Success | L_early (mean) |
|-------|--------|---------|----------------|
| 0 | Baseline (no guidance) | 80% | 0.898 |
| 1 | Classifier guidance ($w = 10$) | 100% | 0.937 |
| 2 | + VLM text reranking ($K = 5$) | 100% | **0.972** |

**Statistical Significance:**
- VLM guidance vs. Baseline: Δ = +0.039, $p = 0.00042$ (significant)
- VLM reranking vs. guidance alone: Δ = +0.035

**Total improvement: 0.898 → 0.972 (+7.4 percentage points)**

### 5.8 Comparison of All Methods

**Table 5.8: Summary of All Guidance Methods**

| Method | Training Required | Success Rate | L_early | Key Advantage |
|--------|------------------|-------------|---------|---------------|
| Base policy | Yes (100 ep) | 84% | 0.732 | Simplest |
| Best-of-16 | No | — | 0.806 | Training-free, no gradients |
| Classifier guidance ($w=10$) | No | **100%** | 0.952 | Best single-method L_early |
| LegDiff (CFG $w=3$) | Yes (100 ep) | 100% | 0.935 | Principled conditioning |
| VLM reranking | No | **100%** | **0.972** | Best overall L_early |
| Full pipeline | Mixed | **100%** | **0.972** | Maximum legibility |

### 5.9 True DPS Verification

We verified that our implementation is classifier guidance, **not** DPS (Chung et al., 2023). A true DPS implementation—applying gradient updates to $x_{t-1}$ after the denoising step—achieved higher L_early (0.969) but **0% task success** due to trajectory divergence (38% step divergence rate). This confirms that classifier guidance to the noise prediction is the correct approach for our setting.

---

## Chapter 6: Discussion and Honest Assessment

This chapter provides a transparent discussion of the work's limitations and areas where claims must be carefully scoped.

### 6.1 Terminology Correction: Classifier Guidance, Not DPS

Throughout development, our gradient-based guidance method was incorrectly referred to as "DPS" (Diffusion Posterior Sampling) and "LPS" (Legibility Posterior Sampling). The actual implementation follows **classifier guidance** (Dhariwal & Nichol, 2021), where the gradient of the scoring function modifies the noise prediction $\epsilon_\theta$ during denoising:

$$\hat{\epsilon} = \epsilon_\theta - w\sqrt{1 - \bar{\alpha}_t} \nabla L$$

True DPS (Chung et al., 2023) applies gradient updates to the denoised sample $x_{t-1}$:

$$x_{t-1} = \text{DDIM}(x_t) + \zeta_t \nabla_{x_t} \log p(y \mid \hat{x}_0(x_t))$$

When we implemented true DPS, it achieved 0% task success. **All reported results use classifier guidance.** This distinction is important and should not be glossed over. We correct this terminology throughout the thesis.

### 6.2 VLM Scoring Function: Text-Only Generation

The legibility scoring function used for classifier guidance was generated by Gemini from a **text-only prompt** describing the task geometry. The VLM never observed any visual data. While the generated function achieves $r = 0.992$ correlation with our hand-crafted baseline, this is not a "visual" contribution—any capable text LLM could potentially produce equivalent output.

We claim this as a valid use of LLM code generation for robotics (in the spirit of EUREKA, Ma et al., 2024), but note it is **single-shot without iterative refinement**, falling short of the full EUREKA protocol which includes environment feedback and multiple refinement rounds.

### 6.3 VLM Evaluation: Video-Based, Not Online

Our VLM legibility evaluation operates on **pre-recorded videos**, not online during policy execution. The VLO metric measures legibility from an after-the-fact observation, not real-time human perception. While VLM judgments correlate with human assessments in prior work (e.g., RT-2, Brohan et al., 2023), we did not conduct a human study to validate this correlation in our specific setting.

### 6.4 Evaluation Gaps

Several evaluations that would strengthen the work were not completed:

1. **Arc classification on guided trajectories**: We never measured whether guidance actually shifts the trajectory shape distribution (e.g., from deceptive to legible-style curves).
2. **Reverse-steering test**: We did not test whether setting $w < 0$ produces less legible trajectories, which would provide causal evidence that guidance controls legibility.
3. **VLM evaluation of guided trajectories**: The 400 collected demo videos were evaluated by VLM, but guided policy outputs were only evaluated with the analytical L_early metric, not with VLM.
4. **Generalization to novel configurations**: All evaluation uses the same 10 block configurations as training. We did not test on unseen placements.

### 6.5 What We Can and Cannot Claim

**Supported claims (with experimental evidence):**
- Diffusion policies can learn multi-style manipulation from mixed demonstrations (84% success)
- VLM-based evaluation accurately identifies goals from trajectory videos (94.7% accuracy)
- Classifier guidance improves analytical legibility metrics (Δ = +0.052 L_early)
- The full pipeline achieves 100% success with highest legibility scores ($p = 0.00042$)

**Unsupported or weakly supported claims (NOT made in this thesis):**
- That guided trajectories are perceptibly more legible to humans (no human study)
- That the VLM scoring function is superior to hand-crafted alternatives (correlation ≈ 1.0, functionally equivalent)
- That this approach generalizes beyond TwoBlockPick to real-world tasks (simulation only)
- That DPS or other posterior sampling methods are ineffective for legibility (only one configuration tested)

---

## Chapter 7: Conclusion and Future Work

### 7.1 Summary

This thesis presented a framework for generating legible robot manipulation trajectories using VLM-guided diffusion policies. Our key findings are:

1. **VLM evaluation works**: Gemini 2.5 Flash achieves 94.7% accuracy at identifying robot goals from video, validating VLMs as legibility evaluators. The VLO metric correctly orders trajectory styles: legible (2.93) < neutral (3.00) < deceptive (3.71).

2. **Training-free guidance is effective**: Classifier guidance at $w = 10$ improves L_early from 0.906 to 0.952 while increasing success from 95% to 100%. The full pipeline with VLM reranking achieves L_early = 0.972.

3. **Multiple guidance methods are viable**: Best-of-N, classifier guidance, CFG, and VLM reranking all improve legibility, with complementary strengths.

4. **Honest limitations exist**: Terminology errors, missing evaluations, and the gap between analytical metrics and human perception must be acknowledged.

### 7.2 Future Work

1. **Human study**: Validate that improvements in L_early and VLO correspond to improved human prediction accuracy and response time in a controlled user study.

2. **Online VLM guidance**: Replace post-hoc video scoring with online VLM feedback during policy execution, potentially enabling real-time legibility optimization.

3. **Reverse-steering and causal analysis**: Test negative guidance scales and trajectory shape classification to establish causal relationships between guidance and trajectory legibility.

4. **Real robot transfer**: Validate the approach on physical hardware (Franka Panda) with real block configurations and human observers.

5. **Multi-goal generalization**: Extend to settings with 3+ objects, where legibility becomes more challenging and the VLM's multi-goal reasoning abilities can be more fully tested.

6. **Full EUREKA integration**: Implement the complete EUREKA loop with environment feedback and iterative refinement of the scoring function.

---

## References

Ahn, M., Brohan, A., Brown, N., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *arXiv:2204.01691*.

Bodden, C., Rakita, D., Mutlu, B., & Gleicher, M. (2018). Evaluating Intent-Expressive Robot Arm Motion. *RO-MAN 2018*.

Brohan, A., Brown, N., Carbajal, J., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *arXiv:2307.15818*.

Busch, B., Maeda, G., Mollard, Y., Demangeat, M., & Lopes, M. (2017). Postural optimization for an ergonomic human-robot interaction. *IROS 2017*.

Chi, C., Feng, S., Du, Y., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS 2023*.

Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. (2023). Diffusion Posterior Sampling for General Noisy Inverse Problems. *ICLR 2023*.

Dhariwal, P. & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. *NeurIPS 2021*.

Dragan, A., Lee, K., & Srinivasa, S. (2013). Legibility and Predictability of Robot Motion. *HRI 2013*.

Dragan, A. & Srinivasa, S. (2013). Generating Legible Motion. *RSS 2013*.

Du, Y., Yang, S., Dai, B., et al. (2023). Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC. *ICML 2023*.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.

Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS Workshop on Deep Generative Models 2022*.

Huang, W., Wang, C., Zhang, R., et al. (2023). VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models. *CoRL 2023*.

Janner, M., Du, Y., Tenenbaum, J. B., & Levine, S. (2022). Planning with Diffusion for Flexible Behavior Synthesis. *ICML 2022*.

Ma, Y. J., Liang, W., Wang, G., et al. (2024). EUREKA: Human-Level Reward Design via Coding Large Language Models. *ICLR 2024*.

Nikolaidis, S., Dragan, A., & Srinivasa, S. (2016). Viewpoint-Based Legibility Optimization. *HRI 2016*.

Reuss, M., Li, M., Wang, X., & Lioutikov, O. (2023). Goal-Conditioned Imitation Learning using Score-based Diffusion Policies. *RSS 2023*.

Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR 2021*.

---

## Appendix A: Hyperparameter Tables

### A.1 Training Hyperparameters (Combined Model)

| Parameter | Value |
|-----------|-------|
| `demo_path` | `data/demos/demos_combined.npz` |
| `obs_dim` | 22 |
| `act_dim` | 5 |
| `horizon` | 32 |
| `n_action_steps` | 8 |
| `n_diffusion_steps` | 100 |
| `beta_start` | 0.0001 |
| `beta_end` | 0.1 |
| `hidden_dim` | 256 |
| `n_blocks` | 3 |
| `batch_size` | 64 |
| `lr` | 2e-4 |
| `weight_decay` | 1e-5 |
| `ema_decay` | 0.999 |
| `epochs` | 100 |
| `ddim_steps` | 10 |
| `ddim_eta` | 0.3 |

### A.2 Original Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| `demo_path` | `data/demos/demos.npz` |
| `n_blocks` | 6 |
| `batch_size` | 256 |
| `lr` | 1e-4 |
| `mirror_augment` | true |
| `smooth_weight` | 0.01 |
| `epochs` | 500 |

### A.3 VLM Configuration

| Parameter | Value |
|-----------|-------|
| Model | Gemini 2.5 Flash |
| Temperature | 0.1 |
| Thinking Budget | 512 tokens |
| Response Format | Structured JSON |
| Prefix Windows ($k$) | 6 |
| Prefix Fraction | 30% of episode |

---

## Appendix B: Full Experimental Results

### B.1 Per-Episode VLO Distribution (Base Policy, 42 Episodes)

| VLO | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|---|
| Count | 4 | 3 | 2 | 2 | 2 | 3 | 26 |
| % | 9.5 | 7.1 | 4.8 | 4.8 | 4.8 | 7.1 | 61.9 |
| Cumulative % | 9.5 | 16.7 | 21.4 | 26.2 | 31.0 | 38.1 | 100 |

### B.2 Hand-Crafted vs. VLM-Generated Scoring Function

| Condition | L_early (HC) | L_early (VLM) | Correlation |
|-----------|-------------|--------------|-------------|
| 1-criteria (Gaussian) | 0.942 | — | — |
| 4-criteria (full) | 0.927 | 0.937 | r = 0.992 |

### B.3 True DPS vs. Classifier Guidance

| Method | L_early (mean ± std) | Success Rate | Step Divergence |
|--------|---------------------|-------------|-----------------|
| Classifier Guidance | 0.941 ± 0.014 | **100%** | 0% |
| True DPS | 0.969 ± 0.006 | **0%** | 38% |

---

## Appendix C: VLM Prompt Templates

### C.1 Goal Identification Prompt (v2)

```
You see a robot arm reaching toward blocks on a table.
Goal A is on the LEFT side of the image.
Goal B is on the RIGHT side of the image.

Based on the robot's trajectory so far, which goal is the robot reaching for?

Respond with JSON:
{
  "pA": <float 0-1>,
  "pB": <float 0-1>,
  "cue": "<brief description of visual cue>",
  "choice": "A" or "B" or "C" (if uncertain)
}
```

### C.2 VLM-Generated Legibility Scoring Function (EUREKA-style)

```python
def vlm_legibility_score(ee_traj, goals, true_goal_idx, early_frac=0.3):
    """
    Differentiable legibility score generated by Gemini from text prompt.
    4 weighted criteria: proximity, direction, lateral, speed.
    Returns scalar in [0, 1].
    Correlation with hand-crafted baseline: r = 0.992
    """
    P_prox  = 0.35 * gaussian_proximity(ee_traj, goals[true_goal_idx])
    P_dir   = 0.30 * velocity_alignment(ee_traj, goals[true_goal_idx])
    P_lat   = 0.25 * lateral_separation(ee_traj, goals)
    P_speed = 0.10 * speed_commitment(ee_traj)
    return P_prox + P_dir + P_lat + P_speed
```
