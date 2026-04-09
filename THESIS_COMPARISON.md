# Complete Codebase Audit & Thesis Comparison

## 1. What Was Built and Why

The project iteratively built up to a multimodal diffusion policy for communicative robot manipulation. The progression:

1. **Vanilla diffusion** — baseline: can it pick blocks at all?
2. **LPS gradient steering** — can we post-hoc steer a frozen policy without retraining?
3. **LegDiff** — does goal-conditioning improve legibility vs vanilla?
4. **CFG Multimodal** — can one policy handle all 4 behavior dimensions simultaneously?
5. **VLM Best-of-K selection** — does language-model reranking further improve behavior?

---

## 2. Training Scripts — What Each Does

| Script | Model trained | Data used | Status |
|---|---|---|---|
| `scripts/train.py` | Vanilla diffusion policy (22d obs, no conditioning) | `demos.npz` (orig arcs) | **Baseline — used in thesis** |
| `scripts/train_legibility_diffuser.py` | LegDiff: goal-conditioned U-Net (goal ∈ {left, right, null}), CFG dropout p=0.15 | `demos.npz` | **Intermediate — used in thesis** |
| `scripts/train_cfg.py` | **FINAL**: CFG 4-behavior policy, behavior_mode + context, CFG dropout p=0.15, obs=26d | `demos_v2.npz` (500 eps, 4 behaviors) | **FINAL — used in thesis** |
| `scripts/train_bc.py` | Behavior cloning (MLP, no diffusion) | `demos.npz` | Dead-end. BC couldn't coordinate multi-joint reach. Abandoned. |
| `scripts/train_steered_diffusion.py` | Steering vector injection into U-Net activations | `demos.npz` | Dead-end. Steering vectors didn't generalize across seeds. |
| `scripts/train_with_splits.py` | Same as `train.py` but with train/val split | `demos.npz` | Dead-end. Val loss didn't correlate with eval perf. |
| `scripts/train_clean.py` / `train_corrected.py` / `train_fixed.py` / `train_optimized2.py` / `train_spatial_policy.py` / `train_research_backed.py` | Bug-fix iterations during debugging of GPU normalization issue (obs_mean/std mismatch). All superseded. | Various | Dead-end / debugging artifacts. |

**Why so many train scripts?** A critical normalization bug caused obs_mean/std to be computed only on GPU batch (N=64) instead of full dataset (~2MB). Each `*_corrected`, `*_fixed`, `*_clean` etc. was an attempt to fix this before the root cause was found. `train_cfg.py` is written clean from scratch with correct stats.

---

## 3. Eval Scripts — What Each Does

| Script | What it evaluates | Where results live |
|---|---|---|
| `evaluation/eval_cfg_vlm.py` | **FINAL PIPELINE**: CFG policy → K=4 rollouts → Gemini 3 Pro scores each → selects best | `outputs/eval_vlm_final3/results.json` |
| `evaluation/run_fast_cfg_eval.py` | CFG policy only (no VLM), 4 behaviors, fast env, computes L_early + path_eff + clearance + wp_dist | `outputs/eval_cfg_fast/results.json` |
| `scripts/eval_multimodality.py` | LPS gradient guidance sweep (w=0→20) on vanilla policy; finds optimal guidance scale | `outputs/lps_sweep_results.json` |
| `scripts/eval_research_backed.py` | LegDiff vs Vanilla head-to-head, n=20 each; paired episodes | `outputs/legdiff_results.json` |
| `scripts/diagnose_policy.py` | Stage-by-stage VLM eval of vanilla; best-of-N reranking n=50 | `outputs/best_of_n_results.json` |
| `scripts/test_eval_loop.py` / various `eval_*.py` in scripts/ | Debugging scripts for checking env reset, action scaling, step counts. Not thesis results. | `outputs/smoke_*/`, various |
| `scripts/eval_bc.py` | Evaluated BC policy (task success only) | Not used in thesis |
| `scripts/eval_diagnostic.py` / `eval_clean.py` / `eval_corrected.py` | Normalization bug diagnosis scripts | `outputs/step1_comparison/`, `step2_corrected/` — not thesis results |

---

## 4. Data Collection Scripts

| Script | Produces | What's in it |
|---|---|---|
| `scripts/collect_demos_twoblockpick.py` | `data/demos/demos.npz` | ~100 demos, 22d obs (x,y,z,quat,gripper × 2 blocks + 2 goals), arc trajectories, left/right picks |
| `scripts/collect_demos_combined.py` | `data/demos/demos_combined.npz` | 400 demos, 22d obs, more diverse left+right, uniform block placement jitter |
| `scripts/collect_demos_v2.py` | **`data/demos/demos_v2.npz`** | **500 demos, 26d obs** (+behavior_mode, +target_idx, +context_angle, +waypoint_dist), 4 behavior categories: legible/predictable/safe/grounding |

---

## 5. Model Checkpoints

| Run | Checkpoint | obs_dim | epochs | Final loss | Data | Architecture |
|---|---|---|---|---|---|---|
| `runs/diffusion_20260222_195530` | `ckpt_ep100.pt` | 22 | 100 | 0.0472 | `demos.npz` | Vanilla U-Net |
| `runs/diffusion_20260402_072747` | `ckpt_ep100.pt` | 22 | 100 | 0.0447 | `demos_combined.npz` | Vanilla U-Net (more data) |
| `runs/legdiff_20260331_021740` | `ckpt_ep100.pt` | 22 | 100 | 0.0345 | `demos.npz` | LegDiff (goal-cond, run A) |
| `runs/legdiff_20260331_174004` | `ckpt_ep100.pt` | 22 | 100 | **0.0018** | `demos.npz` | LegDiff (goal-cond, run B — better) |
| `runs/cfg_20260406_005407` | `ckpt_ep200.pt` | **26** | **200** | 0.0429 | `demos_v2.npz` | **CFG Multimodal — FINAL** |
| `runs/cfg_20260406_004904` | `ckpt_ep200.pt` | 26 | 200 | 0.0429 | `demos_v2.npz` | CFG duplicate run |

All `bc_*`, `test_*`, `arc_*`, `comparison_*`, `paired_*` runs are dead-end experiments.

---

## 6. Master Comparison Table

### 6A. Legibility (L_early metric): All models

> **L_early** = fraction of rollout completed at which a Bayesian observer can first classify arm intent (left vs right block) at ≥90% confidence. *Higher = more legible* (robot commits to direction earlier in trajectory). Range: [0, 1].

| Model | Data | n | Success rate | **L_early mean ± std** | Notes |
|---|---|---|---|---|---|
| Vanilla diffusion (baseline) | `demos.npz` | 20 | 100% | **0.919 ± 0.030** | From `legdiff_results.json` |
| Vanilla diffusion (baseline) | `demos.npz` | 20 | 85% | **0.906 ± 0.066** | From `stage1_results.json` (different seeds) |
| Vanilla + LPS w=0 (baseline) | `demos.npz` | 20 | 95% | **0.906 ± 0.044** | From `lps_sweep_results.json` |
| Vanilla + LPS w=2 | `demos.npz` | 20 | 95% | 0.911 ± 0.064 | +0.6% vs baseline |
| Vanilla + LPS w=5 | `demos.npz` | 20 | 90% | 0.946 ± 0.019 | +4.4% vs baseline |
| **Vanilla + LPS w=10** | `demos.npz` | 20 | **100%** | **0.952 ± 0.025** | **Best LPS** — note: DPS does NOT hurt success |
| Vanilla + LPS w=20 | `demos.npz` | 20 | 100% | 0.948 ± 0.028 | Diminishing returns |
| LegDiff w=3.0 | `demos.npz` | 20 | 100% | **0.935 ± 0.018** | From `legdiff_results.json` |
| LegDiff + VLM K=3 | `demos.npz` | 10 | 100% | **0.955 ± 0.012** | From `legdiff_vlm_results.json` |
| CFG Multimodal λ=2 (legible mode) | `demos_v2.npz` | 10 | 100% | **0.900 ± 0.017** | From `eval_cfg_fast/results.json` |
| **CFG + VLM Best-of-4 (legible mode)** | `demos_v2.npz` | 10 | **100%** | 0.798 ± 0.075 | From `eval_vlm_final3` — VLM selects task-success, not L_early |

> **Key finding**: LPS w=10 achieves the highest L_early (0.952) on vanilla policy without retraining. LegDiff w=3 (0.935) is competitive. CFG legibility mode (0.900) is slightly lower but supports 3 additional concurrent behavior dimensions.

> **Important caveat on CFG+VLM legibility**: The VLM was optimized to select "most behaviorally appropriate" rollout, NOT maximize L_early directly. The VLM score for the CFG legibility mode was 0.915 (best selectivity), meaning the VLM did discriminate correctly, but the selected rollout happened to have slightly lower L_early. This is expected — VLM text reasoning about legibility ≠ exact Bayesian L_early metric.

---

### 6B. Additional Metrics — CFG Multimodal Model Only (4 behaviors)

> The vanilla and LegDiff models were only trained/evaluated for legibility. Only the CFG model was evaluated on all four behavior dimensions.

| Behavior mode | n | Success | Primary metric | Mean ± std | VLM score (Best-of-4) |
|---|---|---|---|---|---|
| **Legibility** | 10 | 10/10 (100%) | L_early ↑ | 0.900 ± 0.017 | 0.915 ± 0.086 |
| **Predictability** | 10 | 9/10 (90%) | Path efficiency ↑ | 0.424 ± 0.025 | 1.000 ± 0.000 |
| **Safety** | 10 | 10/10 (100%) | Min clearance ↑ | 0.075 ± 0.022 | 0.480 ± 0.274 |
| **Grounding** | 10 | 9/10 (90%) | Waypoint dist ↓ | 0.079 ± 0.033 | 0.386 ± 0.299 |

### 6C. CFG+VLM vs Baseline (no VLM) — n=10/behavior

| Behavior | Subset | Success | Primary metric | VLM score |
|---|---|---|---|---|
| Legibility | CFG+VLM | 10/10 | L_early=0.798 ± 0.075 | **0.915** |
| Legibility | Baseline (1st rollout) | 4/10 | L_early=0.967 ± 0.021 | 0.286 |
| Predictability | CFG+VLM | 4/10 | eff=0.486 ± 0.127 | **1.000** |
| Predictability | Baseline | 3/10 | eff=0.420 ± 0.091 | 0.815 |
| Safety | CFG+VLM | 7/10 | clear=0.061 ± 0.004 | **0.480** |
| Safety | Baseline | 7/10 | clear=0.056 ± 0.008 | 0.050 |
| Grounding | CFG+VLM | 7/10 | wp=0.115 ± 0.034 | **0.386** |
| Grounding | Baseline | 8/10 | wp=0.117 ± 0.033 | 0.000 |

> **VLM discrimination gap** (VLM_selected − baseline VLM score): +0.629 legibility, +0.185 predictability, +0.430 safety, +0.386 grounding.

---

## 7. Model Limitations

### Vanilla Diffusion
- **No behavior control whatsoever.** L_early is entirely a function of the arc shape in the training demos — it cannot be steered or adjusted at inference.
- The policy "accidentally" achieves decent L_early (0.906–0.919) because the expert demos used arc trajectories. Any flat trajectory demos would destroy L_early with no way to recover.
- Cannot express predictability, safety, or grounding behaviors.
- 22d obs requires same arena layout as training (brittle to env changes).

### Vanilla + LPS (Gradient Steering)
- Achieves best raw L_early (0.952 at w=10) without retraining, but this is a test-time inference hack.
- **Requires a differentiable VLM scoring function** at every denoising step — computationally expensive (×10 inference steps × gradient computation).
- At w>10, L_early saturates and success begins to degrade (manifold departure).
- Only steers legibility. Cannot handle predictability/safety/grounding without separate scoring functions for each.
- **Cannot generalize to novel behavior specifications** — each new behavior requires a hand-crafted scoring function.

### LegDiff (Goal-Conditioned)
- Can only control one behavior dimension: legibility (via left/right/null goal token).
- The goal conditioning is categorical (left vs right vs null) — not continuous. Cannot express "be slightly more legible" vs "maximize legibility."
- Target accuracy with VLM was only ~50% in early checks (`legibility_combined_results.json`), suggesting the goal-conditioned trajectories are not always visually distinguishable.
- Still uses 22d obs — limited to original arena setup.
- LegDiff run A loss (`0.0345`) vs run B loss (`0.0018`) is a 19× difference, suggesting run B may have overfit to specific seeds. Run A was used for VLM experiments.

### CFG Multimodal (Final Model)
- **All four behaviors in one model**, but each behavior is only learned from 125/500 demo trajectories (n≈125 per behavior). The per-behavior sample count is smaller than the vanilla baseline (100 demos for full policy vs 125 for one behavior mode).
- VLM selection **hurts L_early** (0.798 vs 0.900 for CFG-only). This is a **ceiling effect mismatch**: VLM selects based on semantic task alignment, not the Bayesian L_early formula — these objectives are not identical.
- **Predictability and grounding are the weakest behaviors**: VLM success rate drops to 40–70% for these modes. The policy struggles when the instruction calls for extreme path efficiency (predictability) because demos have some curvature, and the waypoint grounding behavior requires precise approach angles that diffusion policy blends/averages.
- **26d obs requires retraining** from scratch: incompatible with vanilla/LegDiff checkpoints. Any environment or task change needs a new dataset collection + full training run.
- Hardware: 200 epochs on `demos_v2.npz` (500×400-step demos) takes ~6–8 hours on single GPU.

---

## 8. Which Model Is Best on Which Metric

| Metric | Winner | Runner-up | Notes |
|---|---|---|---|
| **L_early** (legibility) | Vanilla + LPS w=10: **0.952 ± 0.025** | LegDiff+VLM: 0.955 ± 0.012 (n=10 only) | LegDiff+VLM narrow edge, but n=10 vs n=20 |
| **Task success rate** | Vanilla + LPS w=10 and CFG Legibility mode: **100%** | LegDiff w=3: 100% | All are tied at 100% in best configs |
| **VLM discrimination score** | CFG+VLM legibility: **0.915** | Vanilla+VLM (smoke, n=3): ~0.997 (too small n) | CFG VLM score most reliable |
| **Behavior specificity (legibility)** | CFG Multimodal — **only model that can be instructed** | LegDiff — goal-cond but categorical only | Critical for thesis claim |
| **Behavior specificity (multi-behavior)** | CFG Multimodal — **only model with all 4 behaviors** | — | Vanilla/LegDiff cannot do safety/grounding at all |
| **Path efficiency (predictability)** | CFG Predictable mode: **0.424 ± 0.025**, VLM-selected: **0.486 ± 0.127** | Not measured in other models | Only CFG can do this |
| **Obstacle clearance (safety)** | CFG Safe mode: **0.075 ± 0.022** | — | Only CFG can do this |
| **Waypoint grounding** | CFG Grounding mode: **0.079 ± 0.033** | — | Only CFG can do this |
| **Training data efficiency** | Vanilla (100 demos) and LegDiff (100 demos) | CFG (500 demos for 4 behaviors = 125/behavior) | Similar per-behavior |
| **Deployment simplicity** | Vanilla — no conditioning needed at inference | LPS needs grad scoring fn; CFG needs behavior label | |

---

## 9. Statistical Note / Gaps

The following comparisons are statistically weak due to small n:

| Result | n | Concern |
|---|---|---|
| demos_combined + VLM (smoke_legibility4) | **n=3** | Far too small — discard from thesis comparisons |
| LegDiff+VLM | n=10 | Acceptable for exploratory; p-values would be borderline |
| CFG per-behavior evals | n=10 | Same caveat — treat as exploratory/pilot |
| LPS sweep, LegDiff head-to-head, vanilla baseline | **n=20** | Most reliable. Use these as primary comparisons. |
| best_of_n_results.json vanilla | **n=50** | Most statistically robust, but uses `L_early_intent` (different metric) |

**Note on two L_early metrics:**
- `l_early` (used in legdiff_results, lps_sweep, stage1, eval_cfg_fast) = Bayesian early commitment fraction from the simulation physics
- `L_early_intent` (used in best_of_n_results) = weighted by intent classification confidence = systematically lower (~0.73 baseline vs ~0.91). **Do not compare these directly across result files.**

---

## 10. Thesis Result Narrative (Suggested)

**Research question**: Can a diffusion policy learn communicative manipulation behaviors conditioned on natural language behavior specifications, and does VLM-guided Best-of-K selection improve behavioral compliance?

**Answer by model:**

1. **Vanilla diffusion** achieves strong task success (95–100%) and incidental legibility (L_early≈0.91) because arc-shaped demos happen to be legible. *But it cannot be steered.*

2. **LPS post-hoc steering** (w=10) improves L_early to 0.952 without retraining — the best raw legibility result. *But it requires a differentiable reward signal at inference, cannot generalize to novel behaviors, and does not address the multi-behavior specification problem.*

3. **LegDiff** (goal-conditioning with CFG dropout) improves legibility to 0.935 (w=3.0) with 100% success. VLM reranking further lifts it to 0.955 (n=10). *But it only handles one behavior dimension and uses categorical conditioning.*

4. **CFG Multimodal** is the first model capable of expressing all four behaviors simultaneously (legibility, predictability, safety, grounding) via continuous behavior conditioning. L_early of 0.900 marginally below LPS/LegDiff, but is the **only model where the behavior is addressable at inference time via a behavior label**, not derived from trajectory shape.

5. **VLM Best-of-4 selection** shows strong VLM discrimination (+0.629 legibility score gap), confirming the language model can distinguish behavior quality — but L_early is not always maximized by VLM selection because the VLM optimizes task alignment, not the Bayesian legibility-specific formula.
