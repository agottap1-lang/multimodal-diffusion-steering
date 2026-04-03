# Honest Assessment: VLM-Guided Trajectory Legibility

**Date**: June 2025  
**Status**: Critical self-audit addressing 12 fundamental concerns

---

## Executive Summary

After thorough investigation, several claims in our prior documentation are **wrong or misleading**. This document corrects the record and identifies what genuinely works, what doesn't, and what was never actually tested.

**Bottom line**: We have a working training-free guidance mechanism that improves trajectory legibility. But (a) it's classifier guidance, not DPS; (b) the "VLM-synthesized" function was generated from a text-only prompt with the solution essentially handed to it; and (c) the VLM never saw any visual data from our task.

---

## Issue 1: DPS vs Classifier Guidance — TERMINOLOGY IS WRONG

### What we claimed
All docs and code comments call our method "Diffusion Posterior Sampling (DPS)" or "LPS" (Legibility Posterior Sampling).

### What we actually implemented
**Classifier guidance** (Dhariwal & Nichol, NeurIPS 2021):

```
ε̃ = ε_θ(x_t) − w · √(1−ᾱ_t) · ∇_{x_t} L(x̂₀(x_t))
```

This modifies the noise prediction, then applies standard DDIM update.

### What true DPS is (Chung et al., ICLR 2023)
```
x_{t-1} = DDIM_step(x_t) + ζ_t · ∇_{x_t} L(x̂₀(x_t))
```

DPS adds the gradient to the *denoised sample* x_{t-1}, not to the noise prediction.

### Evidence
- Code at `evaluation/eval_legibility_guided.py` line 358:
  ```python
  guided_eps = eps_pred.detach() - self.guidance_scale * sqrt_1m_ab * g
  ```
  This is noise-prediction modification = **classifier guidance**.
  
- True DPS was tested (`experiments/verify_true_lps.py`) and has **0% success rate** at the same guidance scale. The two methods diverge by 38.1% at a single step.

### Correction needed
**Every document, comment, and variable name that says "DPS" or "LPS" should say "classifier guidance" or "training-free classifier guidance."** The method works, but calling it DPS is factually incorrect.

---

## Issue 2: VLM Function Was Generated From Text-Only Prompt

### What happened
The EUREKA_PROMPT in `evaluation/stage1_vlm_guidance.py` (lines 74-180) is a **text-only** prompt sent to Gemini. It contains:

1. A description of the TwoBlockPick task geometry (block positions, EE start)
2. The **complete hand-crafted `l_early_intent_torch` function** as a "reference"
3. Instructions: "Write a BETTER function that captures richer geometric cues"
4. Specific requirement list: proximity, directional commitment, lateral separation, curvature

### What was NOT in the prompt
- ❌ No trajectory images or video frames
- ❌ No demonstration data from our 400 collected demos
- ❌ No env observations or action sequences
- ❌ No arc examples (arc00-05, arc10-14, arc15-19)
- ❌ No Gemini "prefix mode" visual evaluation
- ❌ No reference to what makes arcs legible from a visual observer's perspective

### What the VLM actually produced
A function (`outputs/stage1/vlm_score_fn.py`) with 4 weighted criteria:
- P_prox (0.35): Gaussian proximity — **identical concept** to the reference function
- P_dir (0.30): Velocity alignment with goal direction
- P_lat (0.25): Lateral separation from non-goal
- P_speed (0.10): Speed commitment

### Honest evaluation
The VLM function is a **richer reparameterization of the reference function it was given**. It adds directional and lateral cues, which are genuinely useful. But:
- It never "saw" what makes a trajectory look legible to a human
- It was given the solution structure and told to elaborate
- Any competent ML engineer could write this function in 30 minutes by reading the Dragan 2013 paper
- The "VLM" part adds no visual grounding — it's code completion from a text prompt

---

## Issue 3: Eureka Standard — Is Text-Only Prompt Valid?

### Yes, for Eureka's original use case
Eureka (Ma et al., ICLR 2024) uses **environment source code** as text context for zero-shot reward function synthesis. This IS the published standard.

### No, for our claim of "VLM-synthesized guidance"
Key differences from Eureka:
1. **Eureka iterates**: It generates a function → runs RL training → feeds back training statistics → generates improved function. We did **one-shot generation with no feedback loop**.
2. **Eureka doesn't hand over the solution**: Eureka gives environment code, not a working reward function. We gave Gemini our hand-crafted function as explicit reference.
3. **Our claim is about VLM advantage**: If we claim VLMs bring something novel, the function should leverage vision capabilities. Our function was generated from text alone — a text-only LLM (GPT-4, Claude) would produce equivalent output.
4. **No visual grounding**: The whole point of a VLM in our context should be that it can *see* trajectories and understand what makes them legible to a human observer. We never used this capability.

### What we should claim instead
"We used an LLM to generate a multi-criteria scoring function from a text description of the task, building on a reference implementation." This is honest but less novel.

---

## Issue 4: Arc Classification — Never Computed for Guided Trajectories

### What exists
Arc classification code in `analysis/analyze_arc_structure.py` categorizes **demo trajectories** into:
- arc00-05: max_arc < 0.05m (straight approach)
- arc10-14: 0.05m ≤ max_arc < 0.15m (moderate curve)
- arc15-19: max_arc ≥ 0.15m (large lateral sweep)

### What's missing
We never computed arc categories for:
- Baseline (unguided) trajectories
- Hand-crafted guided trajectories
- VLM-guided trajectories
- The 3-way comparison across guidance conditions

**This is a critical metric gap.** If guidance doesn't shift the arc distribution toward more legible arcs (arc15-19), the L_early improvement is just numerical noise in the Gaussian posterior, not actual trajectory shape change.

### What we need to do
Run episodes, save full EE trajectories (not just scalar L_early), classify arcs, report distribution shift.

---

## Issue 5: Best-of-N Reranking — Is It "Cheating"?

### Evidence from literature
Universal Guidance (Bansal et al., 2023) generates 20 images and selects top-k via the guidance function. This is **standard practice** even in gradient-guided diffusion papers. The approach is:
1. Generate N candidates with gradient guidance
2. Rank by the guidance function
3. Report best result

### But...
If we use **only** best-of-N with no gradient guidance, that's just sampling + selection — it doesn't demonstrate that the scoring function can steer generation. The value of gradient guidance is that it shifts the *entire distribution*, not just picks lucky samples.

### Our Stage 3 situation
Stage 3 uses VLM text-based reranking (Gemini evaluates trajectory descriptions). This is selection without gradient guidance during generation. The question is whether this demonstrates the VLM's contribution or just finds the best sample from an already-good distribution.

### Honest answer
Best-of-N with gradient guidance = standard practice. Pure best-of-N without guidance = weak contribution. We should report both and clearly distinguish them.

---

## Issue 6: Collected Demos Never Used

### What we have
400 demonstrations across 10 configs:
- 50% legible (Bézier curves toward goal)
- 25% neutral (straight-line)
- 25% deceptive (curve away then correct)
- MP4 videos for each demo
- Stored in `data/demos/`

### What we should have done
1. **Show demo videos to Gemini** (prefix mode) with labels: "This trajectory is legible because..." / "This trajectory is NOT legible because..."
2. **Use visual examples** to teach the VLM what legible arcs look like from an observer perspective
3. **Test VLM evaluation on demo trajectories** — can Gemini correctly classify legible vs. neutral vs. deceptive?
4. **Generate scoring function from visual understanding** — the function should emerge from seeing examples, not from reading code

### Why this matters
The gap between "LLM generates code from text description" and "VLM learns from visual demonstrations" is the entire novel contribution of the paper. Without it, we have standard code synthesis.

---

## Issue 7: Visual VLM Infrastructure Exists But Was Never Used

### eval_combined_prefix.py
Sends trajectory frames at 0.5s intervals to Gemini for visual goal inference:
- Progressive windows n∈{3,4,5,6}
- Annotated reference frame showing block locations
- Returns P(goal_A) and P(goal_B) + reasoning

### gemini_vlm_eval project
Full pipeline with:
- `single_frame` mode: One frame → goal prediction
- `prefix_frames` mode: Frame sequence → goal prediction
- Video generation and annotation
- Client code for Gemini API

### What should happen
1. Use prefix_frames mode to evaluate VLM's ability to distinguish legible vs. non-legible trajectories
2. Feed visual VLM evaluation back into the scoring function design
3. Test whether VLM visual evaluation correlates with our numerical L_early metric
4. Iterate: if the VLM disagrees with L_early, the scoring function should be updated

---

## Issue 8: Reverse-Steering Never Tested

### What this means
If our guidance function works, we should be able to:
- **Positive guidance** (w > 0): Push toward legibility → more arc15-19
- **Negative guidance** (w < 0): Push away from legibility → more arc00-05 (predictable/straight)

### Why this is important
Reverse steering is the strongest proof that the guidance function actually controls trajectory shape. If negative guidance produces straight trajectories and positive produces arced trajectories, that's causal evidence.

### Status
Never attempted.

---

## Issue 9: What We Actually Have That Works

Despite the issues above, the core mechanism **does work**:

### Confirmed results (paired evaluation, same seeds)

| Condition | Success Rate | L_early (mean) | Δ vs Baseline |
|-----------|-------------|----------------|---------------|
| Baseline (no guidance) | 80% | 0.898 | — |
| HC w=5 | 90% | 0.928 | +0.030 |
| HC w=10 | 95% | 0.930 | +0.032 |
| HC w=15 | 100% | 0.947 | +0.049 |
| VLM w=5 | 100% | 0.929 | +0.031 |
| VLM w=10 | 100% | 0.937 | +0.039 |
| VLM w=15 | 85% | 0.946 | +0.048 |

### What this shows
1. Training-free classifier guidance genuinely improves L_early over baseline
2. Both HC and VLM functions produce similar improvements
3. The VLM function achieves 100% success at w=5 and w=10 (better than HC at w=5: 90%)
4. Higher guidance scales push L_early up, but VLM w=15 drops success to 85%

### What this does NOT show
1. That the VLM adds anything over a hand-crafted function (improvements are comparable)
2. That visual understanding is involved (it's not — pure text synthesis)
3. Arc distribution shift (never measured)
4. That an observer would actually perceive the trajectories as more legible (no human study, no VLM visual evaluation)

---

## Issue 10: What Needs To Be Done

### Must-fix (blocking)
1. **Correct DPS→Classifier Guidance terminology everywhere**
2. **Compute arc classification** on all conditions (baseline, HC, VLM at each scale)
3. **Run reverse-steering test** (w < 0) to prove causal control

### Should-do (for paper validity)
4. **Send demo videos to Gemini** — teach VLM what legible arcs look like visually
5. **Generate scoring function from visual understanding** — VLM sees example trajectories, then writes function
6. **Use eval_combined_prefix.py on guided trajectories** — does VLM visual evaluation agree with L_early numbers?
7. **Test VLM failure modes** — when does the VLM scoring function fail? Which episodes?
8. **Eureka-style iteration** — generate function, run guidance, feed back results, regenerate

### Nice-to-have (strengthens claims)
9. **Action-space grounding** — include action dimensions in VLM prompt
10. **Multi-modal scoring** — combine text-synthesized function with visual VLM evaluation
11. **Ablation** — VLM function components (remove P_dir, P_lat, P_speed one at a time)

---

## Summary Table of All 12 Concerns

| # | Concern | Status | Severity |
|---|---------|--------|----------|
| 1 | DPS terminology wrong | CONFIRMED — it's classifier guidance | High |
| 2 | VLM used text-only prompt | CONFIRMED — no visual data | High |
| 3 | No arc classification metrics | CONFIRMED — never computed on guided trajectories | High |
| 4 | Best-of-N legitimacy | PARTIALLY OK — standard WITH gradient guidance, weak WITHOUT | Medium |
| 5 | VLM function possibly hardcoded/trivial | PARTIALLY VALID — richer than reference but same concept | Medium |
| 6 | Collected demos unused | CONFIRMED — 400 demos with videos never shown to VLM | High |
| 7 | No prefix-mode visual VLM evaluation | CONFIRMED — infrastructure exists, unused | High |
| 8 | No reverse-steering test | CONFIRMED — never attempted | High |
| 9 | Need more paper research | DONE — Eureka, Universal Guidance, DPS reviewed | Low |
| 10 | VLM I/O not shown | DOCUMENTED — see Issue 2 above | Low |
| 11 | Action dimensions not in prompt | CONFIRMED — prompt has positions only, not action space | Medium |
| 12 | VLM failure modes not tested | CONFIRMED — no per-episode failure analysis | Medium |

**High severity issues: 6 of 12. These must be addressed before any paper submission.**
