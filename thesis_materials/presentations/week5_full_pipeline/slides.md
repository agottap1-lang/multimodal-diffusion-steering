# Week 5: Full Pipeline Integration & Lessons Learned

## From Training to VLM-Guided Legible Diffusion

**Anudeep Gottapu**
Arizona State University

---

## Slide 1: The Complete Pipeline

```
Demo Collection (400 Bézier trajectories)
         ↓
Diffusion Policy Training (DDPM, 100 epochs)
         ↓
Base Policy (84% success, VLO=4.57 — illegible)
         ↓
Classifier Guidance (w=10 → L_early=0.952, 100% success)
         ↓
VLM Reranking (K=5 → L_early=0.972, 100% success)
         ↓
VLM Evaluation (VLO metric, Gemini 2.5 Flash)
```

End-to-end: from demonstration collection to VLM-verified legible execution.

---

## Slide 2: What Worked

| Component | Result | Why It Worked |
|-----------|--------|---------------|
| Mixed-style demos | 84% success | Bézier gives precise control; 400 demos sufficient |
| DDPM U-Net | Learns distribution | Suitable for multimodal action data |
| Classifier guidance | +5.2% L_early | Gradient signal meaningful; w=10 balances task/legibility |
| VLM evaluation | 94.7% accuracy | Prompt engineering (v1→v2) was critical |
| VLM reranking | Best L_early | VLM as judge catches what metrics miss |

---

## Slide 3: What Didn't Work

| Attempt | Outcome | Root Cause |
|---------|---------|------------|
| True DPS implementation | **0% success** | Gradient to x_{t-1} causes trajectory divergence |
| Initial eval (13% success) | Nearly total failure | Checkpoint bug: normalized stats wrong |
| VLM v1 prompt | 45% accuracy (chance) | World-coordinate confusion in prompt |
| 4-criteria vs 1-criteria HC | 4-criteria slightly worse | Lateral + speed dilute core proximity signal |
| 87% failure in seed eval | Modal collapse | Some seeds produce degenerate behavior |

---

## Slide 4: Engineering Lessons

**Bug Fixes That Made the Difference**

1. **Video saving bug** — `reset()` cleared `_video_frames` unconditionally, wiping recorded data. Fix: guard with `if not self._video_path`
2. **Video recording order** — `record_video()` before `reset()` → initial frame erased. Fix: call after `reset()`
3. **Camera convention** — Image-left ≠ world-left. Caused systematic VLM evaluation errors until corrected.
4. **Normalization stats** — Checkpoint saved mean=0, std=1 instead of actual data stats → policy predictions wildly wrong at inference

---

## Slide 5: Numbers at a Glance

**Demonstration Data:**
- 400 demos: 200 legible, 100 neutral, 100 deceptive
- 10 block configs, 400 steps/episode max, 22-dim obs, 5-dim action

**Model:**
- ~5.5M parameters, U-Net with 3 encoder blocks
- Training: 100 epochs, loss 0.154 → 0.045

**Evaluation:**
- Base policy: 84% success, VLO=4.57 (mostly illegible)
- Classifier guidance (w=10): 100% success, L_early=0.952
- Full pipeline + VLM rerank: L_early=0.972 (p=0.00042 vs baseline)
- Expert demo VLO: legible=2.93, neutral=3.00, deceptive=3.71

---

## Slide 6: Honest Assessment (6 Issues)

1. **DPS terminology wrong** — Method is classifier guidance, not DPS. True DPS fails.
2. **VLM scoring from text only** — Gemini generated scoring function from text description, never saw visual data
3. **Not full EUREKA** — Single-shot generation, no iterative refinement with environment feedback
4. **No human study** — VLM ≠ human; correlation assumed but not validated
5. **No reverse-steering test** — w < 0 not tested; no causal proof of control
6. **Same configs for train and eval** — Generalization not demonstrated

---

## Slide 7: What We CAN Claim (With Evidence)

✓ Diffusion policies learn multi-style trajectories from mixed demos (84% success, Table 5.1)

✓ VLMs accurately predict robot goals from video (94.7%, Table 5.3)

✓ VLO correctly orders trajectory styles (2.93 < 3.00 < 3.71, Table 5.3)

✓ Classifier guidance significantly improves legibility (p=0.00042, Table 5.7)

✓ Full pipeline: 0.898 → 0.972 L_early at 100% success (Table 5.7)

✓ w=10 is empirically optimal for classifier guidance (Table 5.4)

---

## Slide 8: What We CANNOT Claim

✗ Guided trajectories are more legible to humans (no human study)

✗ VLM scoring function is better than hand-crafted (r≈1.0, functionally equivalent)

✗ Approach generalizes to real robots or novel configs

✗ DPS is inherently unsuitable for this task (only one configuration tested)

✗ This is a visual contribution (scoring function generated from text)

---

## Slide 9: Future Directions

1. **Human validation study** — Does improved VLO correspond to faster human prediction?
2. **Online VLM guidance** — Real-time VLM feedback during execution
3. **Reverse-steering** — Test w < 0 for causal evidence
4. **Real robot transfer** — Franka Panda physical experiments
5. **Multi-goal settings** — 3+ objects for richer legibility challenges
6. **Full EUREKA loop** — Iterative refinement with environment feedback

---

## Slide 10: Thank You

**Code:** Full codebase with all scripts, configs, and evaluation tools

**Data:** 400 demos, 42 evaluation videos, legibility scores

**Key Result:** Training-free guidance over diffusion policies is a viable approach to legible motion generation, improving early legibility by 7.4 percentage points while maintaining 100% task success.
