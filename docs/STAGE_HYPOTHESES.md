# Per-Stage Hypotheses: VLM-Synthesized Guidance for Diffusion Trajectory Planning

**Principal Investigator Protocol**: Each stage is an independent experiment with its own hypothesis, method, metrics, and acceptance criteria. No stage's result is assumed true until tested.

---

## Stage 1: VLM Code Synthesis (Eureka-Style)

### Question
Can a VLM (Gemini 2.5 Flash) generate a **differentiable PyTorch scoring function** that correctly captures trajectory legibility, without any environment rollouts?

### Hypothesis H_S1
The VLM-generated function `vlm_legibility_score(ee_traj, goals, true_goal_idx, early_frac)` will:

| Sub-hypothesis | Criterion | Metric | Accept if |
|---|---|---|---|
| H_S1(a) | Gradient existence | `‖∇_traj f‖` | > 1e-10 |
| H_S1(b) | Discrimination | f(legible) vs f(ambiguous) | `f(legible) > f(center) > f(opposite)` |
| H_S1(c) | Correlation with L_early | Pearson r over 21 synthetic trajectories | r > 0.80 |
| H_S1(d) | Gradient direction | Early y-gradient sign | Negative when true_goal = left (y=-0.07) |
| H_S1(e) | Score range | All outputs | ∈ [0, 1] |
| H_S1(f) | Symmetry | Scores for goal 0 and goal 1 | Correct ordering for both goals |

### Method
- Load the VLM-generated function from `outputs/stage1/vlm_score_fn.py`
- Run 6 synthetic tests on a fixed grid of trajectories (no environment, no rollout, no policy)
- Compare against hand-crafted `l_early_intent_torch` on same synthetic trajectories
- Ablation: evaluate each VLM criterion component in isolation (if function is modular)

### What this does NOT test
- Whether the function improves actual rollout performance (that's Stage 2)
- Whether the DPS gradient guidance preserves task success (that's Stage 2)

---

## Stage 2: Diffusion Posterior Sampling (DPS) Integration

### Question
When used as a gradient guidance signal during DDIM sampling, does the VLM-generated function improve trajectory legibility without degrading task success?

### Hypothesis H_S2
At the optimal guidance scale w*:

| Sub-hypothesis | Criterion | Metric | Accept if |
|---|---|---|---|
| H_S2(a) | Task preservation | Success rate | ≥ 95% |
| H_S2(b) | Legibility improvement | Paired L_early, VLM vs baseline | Δ > 0 with p < 0.05 |
| H_S2(c) | Competitive with hand-crafted | L_early difference | |VLM − HC| < 0.02 |
| H_S2(d) | Scale sensitivity | Optimal w* | Reported; may differ from HC |

### Method
- **Paired design**: Generate N=20 seed pairs (env_seed, sample_seed). ALL conditions use SAME pairs.
- **Conditions**: Baseline (w=0), Hand-crafted LPS (w ∈ {5, 10, 15}), VLM-guided (w ∈ {5, 10, 15})
- **Primary metric**: L_early_intent (Dragan 2013) computed on executed EE trajectoryone 
- **Statistical test**: Paired t-test on L_early values (same seed pairs → paired observations)
- **Report**: Success rate, L_early mean ± std, guidance scale sweep table

### Controls
- Same checkpoint, same DDIM steps (10), same action_scale (0.05)
- Same gradient clipping (clip=1.0) for both HC and VLM
- Same number of diffusion steps (100), same beta schedule

---

## Stage 3: VLM Text Reranking (Best-of-N Selection)

### Question
Does generating multiple candidate trajectories and using VLM text-based analysis to select the best one further improve legibility beyond single-sample DPS?

### Hypothesis H_S3
Best-of-N reranking with N=5 candidates:

| Sub-hypothesis | Criterion | Metric | Accept if |
|---|---|---|---|
| H_S3(a) | Task preservation | Success rate | ≥ 95% |
| H_S3(b) | L_early gain | Oracle reranking vs single DPS | Δ > +0.01 |
| H_S3(c) | VLM ≈ Oracle | VLM text reranking vs oracle | Agreement > 60% on best candidate |

### Method
- **Oracle reranking**: Generate N candidates, pick the one with highest predicted L_early
- **VLM text reranking**: Generate N candidates, send trajectory statistics to Gemini, Gemini picks
- **Baseline**: Single-sample DPS (N=1) at best w* from Stage 2
- **Same paired seeds** as Stage 2

### Rationale
This tests whether N>1 + selection provides a Pareto improvement. The oracle gives the ceiling; VLM reranking tests whether LLM text understanding matches numerical quality.

---

## Experiment Execution Order

```
Stage 1 → Validate VLM function in isolation (NO rollouts)
   ↓ (must pass before proceeding)
Stage 2 → DPS integration with paired rollouts
   ↓ (identifies best w*)
Stage 3 → Reranking on top of Stage 2's best config
```

Each stage gates the next. If Stage 1 fails, we regenerate the VLM function before touching Stage 2.

---

## Infrastructure

- **Script**: `experiments/eval_stages_rigorous.py`
- **Checkpoint**: `runs/diffusion_20260222_195530/ckpt_ep100.pt`
- **VLM function**: `outputs/stage1/vlm_score_fn.py`
- **Output**: `outputs/rigorous_eval/{stage1,stage2,stage3}_results.json`
