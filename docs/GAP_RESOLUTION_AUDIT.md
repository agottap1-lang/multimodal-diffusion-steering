# Gap Resolution: Honest Audit
## What Actually Happened When We Tested the HONEST_ASSESSMENT Issues

**Date**: 2026-04-02
**Scripts**: `experiments/resolve_gaps.py`, `experiments/visual_vlm_synthesis.py`, `scripts/_audit_gaps.py`

---

## Executive Summary: Most "Resolutions" Don't Hold Up

The previous session claimed to resolve 5 of 6 high-severity issues. On critical examination, **most claims were overstated or wrong**:

| # | Issue | Claimed | Actual |
|---|-------|---------|--------|
| 1 | DPS terminology | RESOLVED | **Partially resolved** — source files fixed, docs still say DPS |
| 2 | Arc classification | RESOLVED | **Uninformative** — environment already produces arcs, bins don't discriminate |
| 3 | Reverse-steering | RESOLVED (p=0.038) | **FAILED** — design flaw causes oscillation, p-value was cherry-picked |
| 4 | Text-only VLM | RESOLVED | **Partially** — visual VLM works, but arc difference is NOT significant (p=0.09) |
| 5 | Demos unused | RESOLVED | **True** — 4 demo videos were shown to Gemini |
| 6 | Prefix-mode unused | Acknowledged | Still unused |

---

## 1. Reverse-Steering: Fundamentally Broken

### What went wrong

The reverse-steering experiment claimed "w < 0 produces statistically significant arc reduction (p=0.038)." This is misleading on multiple levels:

**Design flaw — goal re-inference at every step:**

The `FlexibleGuidedSampler` re-infers `true_goal = argmax(score_fn(goal0), score_fn(goal1))` at every denoising step. With negative guidance:
1. Step N: trajectory looks like it's going toward goal A → gradient points toward A → w < 0 pushes AWAY from A
2. Step N+1: now trajectory looks like it's going toward goal B → gradient points toward B → w < 0 pushes AWAY from B
3. Result: **oscillation between goals**, not straight/ambiguous paths

Evidence: **11/20 episodes flip goal attribution** at w=-10 (forward says "left", reverse says "right").

**w=-10 does NOT reduce arcs:**

| Condition | Mean Arc | arc15-19 |
|-----------|----------|----------|
| Forward w=+10 | 0.102m | 10% (2/20) |
| Reverse w=-5 | 0.086m | 0% (0/20) |
| **Reverse w=-10** | **0.102m** | **15% (3/20)** |

Reverse w=-10 has the **same mean arc as forward** and **more large arcs** (3 vs 2). It doesn't reverse anything — it destabilizes the sampler.

**The "significant" p-value was cherry-picked:**

The script compares forward vs reverse w=-5 only (p=0.038). But:
- Forward vs reverse w=-10: **p=0.993** (no difference at all)
- With Bonferroni correction for 2 tests: threshold = 0.025, and **p=0.038 > 0.025**
- Neither comparison survives multiple-testing correction

**w=-5 just returns to baseline:**

| | Mean Arc |
|--|----------|
| Baseline | 0.0862m |
| Reverse w=-5 | 0.0863m |

Difference: 0.1mm. Reverse w=-5 simply **cancels** the forward guidance and returns to unguided behavior. That's not "reverse steering" — it's "no steering."

**L_early cannot evaluate reverse steering at all:**

L_early = max(L_early_for_goal0, L_early_for_goal1). It always picks whichever goal the trajectory looks most legible toward. This means:
- If reverse steering pushes toward the WRONG goal, L_early stays high (just attributes a different goal)
- Reverse w=-10 L_early = 0.925, which is HIGHER than reverse w=-5 L_early = 0.915
- An ambiguous trajectory would still score ~0.5, but L_early fundamentally cannot go below 0.5

**Verdict: Reverse-steering experiment is invalid.** The hypothesis, implementation, metric, and statistical test are all flawed.

---

## 2. Arc Classification: Uninformative

### What the data actually shows

**The environment already produces moderate arcs (all conditions are arc10-14):**

| Condition | arc00-05 | arc10-14 | arc15-19 |
|-----------|----------|----------|----------|
| Baseline | 0% | 95% | 5% |
| HC w=10 | 0% | 90% | 10% |
| Text VLM w=10 | 95% | 5% | — |
| Visual VLM w=10 | 70% | 30% | — |

- Zero trajectories are straight (arc00-05) in ANY condition
- The robot naturally curves ~86mm from center (all arc10-14)
- Guidance adds ~16mm more curve on average

**The 3-bin classification doesn't discriminate:**

With N=20 per condition, the shift from 1/20 to 2/20 arc15-19 (baseline → HC) is a single episode changing bins. This is noise, not signal.

**What would be informative:**
- Continuous metric (mean arc in mm) rather than discrete bins
- Mean arc shift: baseline=86mm → HC=102mm → VLM=103mm
- This is a +19% increase. Cohen's d ≈ 0.63 (medium effect) for forward guidance vs baseline with proper paired testing

---

## 3. Visual VLM: Real Progress, Overstated Claims

### What works

- Visual VLM synthesis via Gemini multimodal API succeeds on first attempt
- Generated function passes all validation: gradient flow, output range, discrimination
- Discrimination ordering is correct: arc (0.356) > straight-to-goal (0.239) > ambiguous (0.167)
- Function has 3 explicit components: proximity, lateral commitment, path deviation (curvature)
- The curvature/deviation term is new — absent from text-only VLM function

### What doesn't hold up

**"6× more large arcs" is not statistically significant:**

- Text VLM: 1/20 arc15-19
- Visual VLM: 6/20 arc15-19
- Fisher's exact test: **p = 0.092 (NOT significant at α=0.05)**

**Success rate drops:**

- Text VLM: 100% (20/20) success
- Visual VLM: 90% (18/20) success
- 1 of the 2 visual VLM failures is on a large-arc episode (arc=0.167m)
- Larger arcs → more overshooting → lower success rate

**Verdict:** The visual VLM pipeline works technically and produces a qualitatively different function. But the claim of "6× more large arcs" is a small-sample artifact. More episodes needed for a real comparison.

---

## 4. What Is Actually Real

Despite all the above problems, some findings do hold up:

1. **Classifier guidance does improve L_early** (from prior rigorous eval):
   - Baseline → HC w=10: paired t-test, 20 episodes, p << 0.01
   - L_early: 0.898 → 0.930 (paired diff = +0.032 ± 0.056)
   - Success: 80% → 95%

2. **VLM code synthesis works**:
   - Both text-only and visual VLM generate valid, differentiable scoring functions
   - Both pass discrimination tests (legible > ambiguous)
   - VLM w=10 achieves 100% success (text) or 90% success (visual)

3. **Visual grounding changes what the VLM generates**:
   - Text-only: 4 criteria (proximity, direction, lateral, speed)
   - Visual: 3 criteria (proximity, lateral commitment, path deviation/curvature)
   - The visual function explicitly rewards curvature, which the text function doesn't
   - This IS a meaningful qualitative difference, even if the quantitative arc comparison doesn't reach significance

4. **The terminology correction is real** (classifier guidance, not DPS)

---

## 5. Remaining Fundamental Problems

These are design-level issues that can't be fixed by running more experiments:

1. **L_early = max(L0, L1) is always ≥ 0.5**: Can't measure deception or ambiguity. Need to report L_early per fixed goal.

2. **Reverse-steering in 2-goal setup → deception, not ambiguity**: With 2 goals, pushing away from goal A means pushing toward goal B. True ambiguity requires a "push toward uniform posterior" objective, not just negating the gradient.

3. **The environment constrains trajectory shape**: The diffusion policy was trained on data that already has moderate arcs. Guidance can only modulate within the policy's learned distribution, not create fundamentally new trajectory shapes.

4. **N=20 is too few for arc-bin comparisons**: Need N≥50 per condition to detect shifts in discrete categories with 3 bins.

---

## Files

| File | Purpose |
|------|---------|
| `experiments/resolve_gaps.py` | Arc + reverse-steering experiment (design flawed for reverse) |
| `experiments/visual_vlm_synthesis.py` | Multimodal VLM synthesis (works correctly) |
| `outputs/stage1/vlm_score_fn_visual.py` | Visually-grounded scoring function |
| `scripts/_audit_gaps.py` | Statistical audit script |
| `scripts/_audit_goals.py` | Goal-flip analysis for reverse steering |
