# Multimodal Diffusion Policy — TwoBlockPick

Unconditional DDPM imitation policy trained on a balanced 50/50 dataset of
left-pick and right-pick demonstrations.  Multimodality emerges from
**stochastic diffusion sampling**: same initial state, different noise seed →
the robot sometimes picks the LEFT block, sometimes the RIGHT block.

## Quickstart

```powershell
# 1. Create venv and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 2. Full pipeline (collect → train → eval)
.\run_all.ps1
```

### Step by step

```powershell
# Collect 100 left + 100 right demos
python scripts/collect_demos_twoblockpick.py --episodes_left 100 --episodes_right 100

# Inspect demos (prints stats + guards)
python scripts/inspect_demos.py --path data/demos/demos.npz

# Train diffusion policy (500 epochs)
python scripts/train_diffusion_policy.py --config configs/train.yaml

# Evaluate multimodality (10 env seeds × 20 sample seeds)
python scripts/eval_multimodality.py --ckpt runs/latest/ckpt.pt --K 10 --M 20

# (Optional) Train + evaluate BC baseline
python scripts/train_bc.py --config configs/train.yaml
python scripts/eval_bc.py  --ckpt runs/bc_latest/bc_ckpt.pt --K 10 --M 5
```

## Repo structure

```
├── configs/
│   └── train.yaml                # training hyper-parameters
├── envs/
│   └── twoblockpick_env.py       # PyBullet env (Panda + table + 2 cubes)
├── scripts/
│   ├── collect_demos_twoblockpick.py
│   ├── inspect_demos.py          # dataset inspection + balance guards
│   ├── train_diffusion_policy.py
│   ├── eval_multimodality.py     # K×M evaluation + multimodality proof
│   ├── train_bc.py               # MLP BC baseline (MSE)
│   └── eval_bc.py                # BC evaluation (same protocol)
├── data/demos/                   # saved trajectories (demos.npz)
├── runs/                         # training checkpoints
│   ├── latest/ckpt.pt            # diffusion policy
│   └── bc_latest/bc_ckpt.pt      # BC baseline
├── outputs/                      # evaluation artifacts
│   ├── metrics.json              # overall + per-seed metrics
│   ├── results.csv               # one row per rollout
│   ├── entropy_by_seed.csv       # per-seed entropy + collapse flags
│   ├── multimodality_bar.png     # bar chart
│   ├── videos/                   # rollout videos (named by outcome)
│   └── bc/                       # BC baseline outputs
├── run_all.ps1
├── requirements.txt
└── README.md
```

## Data format (`data/demos/demos.npz`)

| Key               | Shape         | Description                       |
|--------------------|---------------|-----------------------------------|
| `obs`             | `(N, T, 22)` | observations per timestep         |
| `actions`         | `(N, T, 5)`  | actions per timestep              |
| `episode_lengths` | `(N,)`       | valid length of each episode      |
| `labels`          | `(N,)` str   | `"left"` / `"right"` (eval only) |

**Observation (22-d):** `ee_pos(3), ee_quat(4), gripper(1),
left_cube_pos(3), left_cube_quat(4), right_cube_pos(3), right_cube_quat(4)`.

**Action (5-d):** `dx, dy, dz, delta_yaw, gripper` — each ∈ [-1, 1].

## Training config

See [`configs/train.yaml`](configs/train.yaml).

| Parameter          | Value                                  |
|--------------------|----------------------------------------|
| Horizon            | 16 (predict 16, execute 8)             |
| Diffusion steps    | 100 (DDPM, linear β 0.0001→0.1)       |
| Network            | 6 ResBlock MLP, 256 hidden, FiLM time  |
| Epochs             | 500                                    |
| Batch size         | 256                                    |
| Learning rate      | 1e-4, AdamW                            |
| Action normalisation | identity (actions already ∈ [-1,1])  |
| Obs normalisation  | per-dim mean/std, std floored at 0.01  |

## Multimodality proof protocol

1. Fix **K** environment seeds (identical cube placement per seed).
2. For each env seed, run **M** rollouts with different `sample_seed`
   (controls `torch.manual_seed`, set **once** per rollout).
3. Record outcome: `left_success` / `right_success` / `failure`.
4. Compute per-seed binary entropy over {left, right} among successes
   (only when ≥5 successes).
5. Flag `BIMODAL` if both left and right picks occur,
   `COLLAPSE` if >90% of successes go to one side.

### Seed separation

| Seed           | Controls                             | Set when            |
|----------------|--------------------------------------|---------------------|
| `env_seed`     | Cube placement jitter (via `_rng`)   | `env.reset(seed=…)` |
| `sample_seed`  | Diffusion noise (torch global RNG)   | Once at rollout start |

## Results (placeholder — fill after running eval)

> Run `python scripts/eval_multimodality.py --ckpt runs/latest/ckpt.pt --K 10 --M 20`
> to populate `outputs/metrics.json` and the table below.

| Metric                      | Value         |
|-----------------------------|---------------|
| Total rollouts              | `<fill>`      |
| Success rate                | `<fill>`      |
| Left / Right successes      | `<fill>`      |
| Bimodal seeds               | `<fill>`      |
| Collapsed seeds             | `<fill>`      |
| Mean entropy (≥5 successes) | `<fill>`      |

### BC baseline comparison

| Metric       | Diffusion | BC (MLP) |
|--------------|-----------|----------|
| Success rate | `<fill>`  | `<fill>` |
| Multimodal?  | `<fill>`  | `<fill>` |

> See `outputs/metrics.json` and `outputs/bc/bc_metrics.json` for full details.

## Outputs (auto-generated after eval)

| File                              | Description                               |
|-----------------------------------|-------------------------------------------|
| `outputs/metrics.json`           | Overall counts, rates, entropy, collapse  |
| `outputs/results.csv`           | Per-rollout outcome table                 |
| `outputs/entropy_by_seed.csv`   | Per-seed entropy + flags                  |
| `outputs/multimodality_bar.png` | Bar chart: left/right/fail per env seed   |
| `outputs/videos/*.mp4`          | Rollout videos (named `…_{outcome}.mp4`)  |
| `outputs/bc/bc_metrics.json`    | BC baseline metrics                       |

## Key design choices

- **Unconditional policy** — no goal_id, no left/right conditioning.
  Modes must emerge from diffusion noise alone.
- **Identity action normalisation** — actions are already ∈ [-1,1];
  mean/std normalisation would crush dx/dy variance.
- **Obs std floor = 0.01** — prevents near-constant quaternion dims
  from exploding normalised values.
- **execute_steps=8** — execute 8 actions from a 16-step horizon
  before replanning. Smaller values cause mode-switching between plans.
- **Commitment mechanism** — early-step dy nudge (seeded by sample_seed)
  helps the policy commit to one side when initial actions are ambiguous.

## Reproducibility

- All random seeds are configurable (`seed:` in YAML, `--env_seed_start`).
- Checkpoints include normalisation stats for exact replay.
- Tested on Windows 10/11, Python 3.12, PyBullet DIRECT mode, CPU.
