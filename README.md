# Steering Diffusion Policies for Legible Robot Manipulation — CFG + VLM Reranking

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Sim](https://img.shields.io/badge/sim-PyBullet-orange.svg)](https://pybullet.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **behavior-conditioned diffusion policy** for robot pick-and-place that can be **steered at inference time — without retraining** — to produce trajectories that are more *legible, predictable, safe, or grounded*. Steering is done two ways: **Classifier-Free Guidance (CFG)** and **VLM best-of-N reranking**, where a vision-language model (Gemini / GPT / Claude) scores candidate trajectories and the best is executed. Franka Panda + PyBullet; full data → train → steer → evaluate pipeline with statistically validated results.

> 🎓 This is the diffusion-policy half of my MS thesis, *Vision-Language Models as Proxies for Human Judgment of Robot Motion Legibility* (ASU, 2026). The VLM goal-inference benchmark lives in the companion repo **[gemini-vlm-goal-inference](https://github.com/agottap1-lang/gemini-vlm-goal-inference)**.

<p align="center">
  <img src="figures/fig7_system_overview.png" alt="System overview: diffusion policy + CFG + VLM reranking" width="88%">
</p>

---

## Why this matters
Robots that share space with people must not only *succeed* at a task but also **communicate their intent** through motion. A diffusion policy naturally generates *many* valid ways to do the same task — so instead of retraining for each desired behavior, this project **selects and steers** among those candidates at inference time using a VLM as an observer-style critic.

## Highlights
- **Behavior-conditioned diffusion policy** (DDPM U-Net, 8.8M params) trained on 500 demos across 4 behavior styles.
- **Two inference-time steering knobs, no retraining:** CFG (guidance scale λ) and **VLM best-of-N** trajectory reranking.
- **VLM-as-critic:** renders candidate rollouts to frames, queries a VLM with a behavior-specific prompt, executes the highest-scoring candidate.
- **Rigorous evaluation:** compositional train/test splits, paired baselines, statistical tests, and behavior-specific metrics (legibility, path efficiency, clearance, grounding).
- **Reproducible:** unified CLI, YAML configs, seeded runs, checkpoints bundling normalization stats.

---

## Results

**Inference-time steering improves behavior without hurting task success.** Across 4 behaviors (10 episodes each):

| Behavior | CFG-only success | Key metric (CFG) | + VLM best-of-N |
|---|:--:|---|:--:|
| **Legibility** | 10/10 | L_early = 0.900 ± 0.017 | **100% vs 40% random — 6× gain** |
| Safety | 10/10 | min clearance, **0 collisions** | success 7/10, clearance +0.43 vs random |
| Grounding | 9/10 | waypoint hover 0.079 ± 0.033 | +0.39 vs random |
| Predictability | 9/10 | path efficiency 0.424 ± 0.025 | weak (straight paths look alike) |

**Full 3-stage pipeline (20 paired episodes):**

| Stage | Success | Legibility (L_early) |
|---|:--:|:--:|
| Baseline (no steering) | 80% | 0.898 |
| + CFG guidance | 100% | 0.948 |
| + VLM reranking | 100% | **0.972** |

➡️ **Headline finding:** the VLM critic is **most valuable for legibility** — it reliably picks the trajectory that curves decisively toward the intended block, a **6× improvement over random candidate selection** and **+7.4 pp** L_early end-to-end (p < 0.0001). For behaviors where candidates look visually similar (predictability), reranking adds little — a finding I report honestly rather than hide.

<p align="center">
  <img src="figures/fig5_final_results.png" alt="Final results across behaviors" width="80%">
  <img src="figures/vlm_table1_main_results.png" alt="Main results table" width="80%">
</p>

---

## How steering works

<p align="center">
  <img src="figures/ppt4_inference_scoring.png" alt="VLM best-of-N reranking pipeline" width="82%">
</p>

**1. Classifier-Free Guidance (CFG).** The policy is trained with 15% condition dropout so it learns both conditional and unconditional scores. At inference the denoising direction is pushed toward the target behavior:

$$\tilde{\epsilon} = \epsilon_\theta(\mathbf{x}_t, t, \varnothing) + \lambda\,\big(\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \varnothing)\big)$$

Sampled with DDIM (η = 0.3) to stay stable while preserving the policy's multimodality.

**2. VLM best-of-N reranking.** Generate N candidate trajectories with different seeds → render key frames → ask a VLM to score each against a behavior-specific rubric → execute the top-scored candidate. The VLM scoring rubric was validated against a hand-crafted metric (r = 0.992).

The two mechanisms are **complementary**: CFG shapes the *distribution* of candidates; reranking *selects* the best realization within it.

---

## Quickstart

```bash
python -m venv .venv && .\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

python cli.py list                                     # show CLI commands
python cli.py quick-eval --episodes 3                  # smoke test
python cli.py evaluate-paired --episodes 10            # CFG vs baseline
python cli.py generate-videos --n-videos 5             # rollout videos
```

<details>
<summary><b>Full pipeline (collect → train → evaluate)</b></summary>

```bash
# 1. Collect demonstrations (4 behavior styles)
python scripts/collect_demos_cfg.py --n 500
# 2. Train behavior-conditioned diffusion policy (CFG, 200 epochs)
python scripts/train_cfg.py --config configs/train_combined.yaml
# 3. Evaluate CFG steering across behaviors
python evaluation/eval_cfg.py --ckpt runs/cfg_20260406_005407/ckpt_ep200.pt
# 4. Evaluate CFG + VLM best-of-N reranking
python evaluation/eval_cfg_vlm.py --ckpt runs/cfg_20260406_005407/ckpt_ep200.pt --n-candidates 4
```
The VLM key is read from the environment / `.env` (`GEMINI_API_KEY`) — never hardcoded.
</details>

---

## Method & data details

<details>
<summary><b>Model & training configuration</b></summary>

| Parameter | Value |
|---|---|
| Architecture | DDPM 1-D U-Net (8.8M params), 6 ResBlocks, 256 hidden, FiLM time embedding |
| Obs / action dim | 26 (22 obs + 4 behavior one-hot) / 5 (Δx, Δy, Δz, Δyaw, gripper) |
| Horizon | 32 (predict 32, execute 8), closed-loop replanning |
| Diffusion | 100 steps, linear β 1e-4→0.1; DDIM η = 0.3 at inference |
| Training | 200–500 epochs, batch 256, AdamW 1e-4, EMA 0.999, CFG dropout 0.15 |
</details>

<details>
<summary><b>Task, behaviors & evaluation protocol</b></summary>

- **Task — TwoBlockPick:** a Franka Panda must pick one of two blocks while expressing a target behavior.
- **Behaviors:** *legibility* (curve toward target early), *predictability* (direct path), *safety* (clearance from the non-target / obstacle), *grounding* (pass through a task waypoint).
- **Legibility metric (L_early):** Bayesian observer posterior over goals from a partial trajectory (higher = intent revealed earlier).
- **Compositional splits:** held-out scene configs *and* held-out trajectory arcs to test generalization, not memorization.
</details>

<details>
<summary><b>Known limitations (reported, not hidden)</b></summary>

- VLM predictability scoring is weak — near-straight paths look identical to the model.
- DDPM at eval amplifies actions → use DDIM (η = 0.3); `execute_steps=8` avoids OOD observations.
- Re-execution variance: the VLM selects from simulated candidates, then re-executes; PyBullet stochasticity can cause small drift.
- Simulation only; sim-to-real transfer of the legibility signal is future work.
</details>

---

## Repository map
```
configs/        training configs (YAML)        envs/         PyBullet TwoBlockPick env
scripts/        data collection + training     evaluation/   CFG / VLM / BC evaluators
experiments/    rigorous staged evaluation     analysis/     legibility / arc / VLM analysis
figures/        paper-ready figures            thesis_materials/  thesis figures & LaTeX
cli.py          unified entry point            FINAL_RESULTS.md / THESIS_COMPARISON.md
```

## Status
The core method and evaluation were defended as my MS thesis (peer-reviewed). This repository is the **living codebase** and also contains **additional, in-development experiments** beyond the thesis scope — those are exploratory and still being validated.

## Citation
```bibtex
@mastersthesis{gottapu2026vlmlegibility,
  title  = {Vision-Language Models as Proxies for Human Judgment of Robot Motion Legibility},
  author = {Gottapu, Anudeep Sai},
  school = {Arizona State University},
  year   = {2026}
}
```

## License
MIT — see [LICENSE](LICENSE). Companion repo: [gemini-vlm-goal-inference](https://github.com/agottap1-lang/gemini-vlm-goal-inference). Advised by Prof. Nakul Gopalan (LOGOS Robotics Lab, ASU).
