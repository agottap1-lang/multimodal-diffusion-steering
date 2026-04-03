# Week 3: VLM-Based Legibility Evaluation

## Using Gemini as a Legibility Judge

**Anudeep Gottapu**
Arizona State University

---

## Slide 1: The Evaluation Problem

**How do we measure legibility at scale?**

- Human studies are gold standard but expensive and slow
- Analytical metrics (L_early) assume a specific Bayesian observer model
- **Proposal:** Use a Vision-Language Model (VLM) as a proxy human observer
- VLMs can watch trajectory videos and predict intent — just like a human would

**VLM Onset (VLO):** The first timestep at which the VLM correctly predicts the robot's goal from cumulative video prefix. Lower VLO = more legible.

---

## Slide 2: VLM Evaluation Pipeline

**Steps:**

1. Render policy rollout → MP4 video
2. Extract **cumulative prefix frames**: k=6 windows over first 30% of episode
3. For each window, send frames to Gemini 2.5 Flash with goal identification prompt
4. VLM returns: {pA, pB, cue, choice} as structured JSON
5. VLO = first k where choice matches true goal

**Key design choices:**
- Cumulative windows (not independent) — VLM sees progressively more context
- 30% prefix cutoff — evaluates early intent communication only
- Temperature 0.1 — reduces VLM randomness for reproducible scoring

---

## Slide 3: Prompt Engineering Journey

**V1 (45% accuracy) → V2 (97.5% accuracy)**

| Version | Accuracy | Key Issue |
|---------|----------|-----------|
| V1 | 45% | World-coordinate confusion, long prompt, high temp |
| V2 | **97.5%** | Annotated reference frames, image-space reasoning, temp=0.1 |

**Critical improvements:**
- Annotated reference frame with block position markers (visual grounding)
- Image-space reasoning ("LEFT/RIGHT in image") instead of world coordinates
- Short, direct prompt — removed verbose instructions
- Text-first response format to reduce pA/pB bias
- Temperature lowered from default to 0.1

---

## Slide 4: Expert Demo Evaluation (Ground Truth)

**40 cfg00 demos evaluated, k=6 prefix windows**

| Metric | Value |
|--------|-------|
| Non-C accuracy | **94.7%** (36/38 decisive) |
| C-rate (uncertain) | 84.2% of windows |
| Left arc accuracy | 95.0% |
| Right arc accuracy | 100.0% |

**VLO by Style (confirms demo design):**

| Style | Mean VLO |
|-------|----------|
| Legible | **2.93** |
| Neutral | 3.00 |
| Deceptive | **3.71** |

**Result:** legible (2.93) < neutral (3.00) < deceptive (3.71) — VLM correctly distinguishes trajectory styles

---

## Slide 5: Base Policy VLM Assessment

**42 successful episodes scored with k=6**

| VLO | Count | Cumulative % |
|-----|-------|:------------:|
| 0 | 4 | 9.5% |
| 1 | 3 | 16.7% |
| 2 | 2 | 21.4% |
| 3 | 2 | 26.2% |
| 4 | 2 | 31.0% |
| 5 | 3 | 38.1% |
| **6 (never)** | **26** | **100%** |

- **Mean VLO = 4.57** (out of 6)
- **Median VLO = 6.0** (worst possible)
- **61.9% never correctly identified**
- Only 9.5% legible at first window

---

## Slide 6: Key Insight

**The base policy achieves 84% task success but produces largely illegible trajectories.**

This is the central motivation for guidance:
- Training on mixed (legible + neutral + deceptive) demonstrations averages out trajectory styles
- The diffusion policy doesn't prefer legible modes without explicit steering
- We need training-free guidance to push the policy toward the legible region of the learned distribution

---

## Slide 7: VLM Configuration Details

| Parameter | Value |
|-----------|-------|
| Model | Gemini 2.5 Flash |
| Temperature | 0.1 |
| Thinking Budget | 512 tokens |
| Response Format | `application/json` |
| Frame Sampling | Evenly spaced within each prefix window |
| Prefix Fraction | First 30% of episode |
| k (windows) | 6 |

**Camera convention fix:** picked_side='left' → VLM correct answer = 'B', picked_side='right' → 'A'

---

## Slide 8: What VLM Evaluation Does and Does NOT Show

**Supported:**
- VLMs can accurately predict robot goals from video (94.7% on demos)
- VLO correctly orders trajectory styles (legible < neutral < deceptive)
- Base policy trajectories are mostly illegible per VLM judgment

**Limitations:**
- VLM ≠ human observer — no human study validates the correlation
- Evaluation is post-hoc on recorded video, not real-time
- Only tested on 10 block configurations (same as training)
- VLO is a proxy metric — may not perfectly reflect human prediction speed

---

## Slide 9: References

- Ma, Y. J., et al. (2024). *EUREKA: Human-Level Reward Design via Coding LLMs.* ICLR.
- Brohan, A., et al. (2023). *RT-2: Vision-Language-Action Models.* arXiv:2307.15818.
- Dragan, A. & Srinivasa, S. (2013). *Generating Legible Motion.* RSS.
