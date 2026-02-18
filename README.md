
# Multimodal Diffusion Policy: TwoBlockPick

## Overview

This project implements a state-of-the-art unconditional diffusion policy for robotic pick-and-place, trained on a perfectly balanced dataset of left-pick and right-pick demonstrations. Multimodality emerges from stochastic diffusion sampling: the same initial state, with different noise seeds, leads the robot to pick either the left or right block.

**Key Features:**
- PyBullet-based simulation with a 7-DOF Panda arm and two cubes
- 400 high-quality demonstrations (10 configs × 20 arcs × 2 modes)
- Rigorous compositional train/test splits for generalization
- Full pipeline: data collection → training → evaluation → analysis
- Multimodality proof protocol and robust diagnostics

---

## Motivation

Robotic policies must generalize and exhibit multimodal behavior in ambiguous scenarios. This project demonstrates that diffusion models, when properly trained and evaluated, can learn to stochastically select between distinct strategies (left/right pick) without explicit conditioning.

---

## Pipeline Quickstart

```powershell
# 1. Environment setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 2. Full pipeline (collect → train → eval)
.\run_all.ps1
```

### Manual Steps

```powershell
# Collect demonstrations (100 left + 100 right)
python scripts/collect_demos_twoblockpick.py --episodes_left 100 --episodes_right 100

# Inspect demo quality and balance
python scripts/inspect_demos.py --path data/demos/demos.npz

# Train diffusion policy (default: 500 epochs)
python scripts/train_diffusion_policy.py --config configs/train.yaml

# Evaluate multimodality (10 env seeds × 20 sample seeds)
python scripts/eval_multimodality.py --ckpt runs/latest/ckpt.pt --K 10 --M 20

# (Optional) Train + evaluate BC baseline
python scripts/train_bc.py --config configs/train.yaml
python scripts/eval_bc.py  --ckpt runs/bc_latest/bc_ckpt.pt --K 10 --M 5
```

---

## Project Structure

```
├── configs/                  # Training configs (YAML)
├── envs/                     # PyBullet environment
├── scripts/                  # Data collection, training, evaluation
├── data/demos/               # Saved demonstrations
├── runs/                     # Training checkpoints
├── outputs/                  # Evaluation results, videos, metrics
├── run_all.ps1               # Full pipeline script
├── requirements.txt
└── README.md
```

---

## Data Format

Demonstrations are stored in `data/demos/demos.npz`:

| Key               | Shape         | Description                       |
|-------------------|--------------|-----------------------------------|
| `obs`             | (N, T, 22)   | Observations per timestep         |
| `actions`         | (N, T, 5)    | Actions per timestep              |
| `episode_lengths` | (N,)         | Valid length of each episode      |
| `labels`          | (N,) str     | "left" / "right" (for eval)      |

**Observation (22-d):**
`ee_pos(3), ee_quat(4), gripper(1), left_cube_pos(3), left_cube_quat(4), right_cube_pos(3), right_cube_quat(4)`

**Action (5-d):**
`dx, dy, dz, delta_yaw, gripper` (all ∈ [-1, 1])

---

## Training & Evaluation Configuration

See [`configs/train.yaml`](configs/train.yaml) for all hyperparameters.

| Parameter          | Value (default)                       |
|--------------------|---------------------------------------|
| Horizon            | 32 (predict 32, execute 8)            |
| Diffusion steps    | 100 (linear β 0.0001→0.1)             |
| Network            | 6 ResBlock MLP, 256 hidden, FiLM time |
| Epochs             | 500                                   |
| Batch size         | 256                                   |
| Learning rate      | 1e-4, AdamW                           |
| Action norm        | identity (actions ∈ [-1,1])           |
| Obs norm           | per-dim mean/std, std floored at 0.01 |

---

## Evaluation & Multimodality Protocol

1. Fix **K** environment seeds (identical cube placement per seed)
2. For each env seed, run **M** rollouts with different `sample_seed` (diffusion noise)
3. Record outcome: `left_success` / `right_success` / `failure`
4. Compute per-seed entropy over {left, right} among successes
5. Flag `BIMODAL` if both left and right picks occur, `COLLAPSE` if >90% go to one side

**Compositional Split:**
- Train: Configs 0-6, Arcs 0-15 (224 demos)
- Val: Config 7, Arcs 0-15 (32 demos)
- Test-trajectory: Configs 0-7, Arcs 16-19 (64 demos)
- Test-scene: Configs 8-9, Arcs 0-15 (64 demos)
- Test-full: Configs 8-9, Arcs 16-19 (16 demos)

---

## Troubleshooting & Lessons Learned

**Common Pitfalls:**
- Changing horizon or n_action_steps without matching demo structure breaks policy
- DDPM sampling for evaluation causes action amplification; use DDIM with eta=0.3 for stability and multimodality
- execute_steps=16 creates out-of-distribution observations; use execute_steps=8
- Deterministic DDIM (eta=0) disables multimodality testing
- Gripper temporal ensemble can cause grasp delays; ensemble only position/orientation

**Best Practices:**
- Always match training and evaluation distributions (e.g., cube_jitter)
- Use compositional splits for rigorous generalization testing
- Monitor action std and obs z-score for diagnostics

---

## Outputs

| File                        | Description                               |
|-----------------------------|-------------------------------------------|
| outputs/metrics.json        | Overall counts, rates, entropy, collapse  |
| outputs/results.csv         | Per-rollout outcome table                 |
| outputs/entropy_by_seed.csv | Per-seed entropy + flags                  |
| outputs/multimodality_bar.png | Bar chart: left/right/fail per env seed |
| outputs/videos/*.mp4        | Rollout videos (named by outcome)         |
| outputs/bc/bc_metrics.json  | BC baseline metrics                       |

---

## References

- [Compositional Split Strategy](COMPOSITIONAL_SPLIT_STRATEGY.md)
- [Final Root Cause Analysis](FINAL_ROOT_CAUSE_ANALYSIS.md)
- [Pipeline Diagnosis](PIPELINE_DIAGNOSIS.md)
- [Demo Analysis & Recommendations](DEMO_ANALYSIS_AND_RECOMMENDATIONS.md)

---

## Reproducibility

- All random seeds are configurable (`seed:` in YAML, `--env_seed_start`)
- Checkpoints include normalization stats for exact replay
- Tested on Windows 10/11, Python 3.12, PyBullet DIRECT mode, CPU
