# Multimodal Diffusion for TwoBlockPick

**VLM-Guided Legible Robot Trajectories with Diffusion Policy**

A research system for generating human-interpretable robot manipulation trajectories using Vision-Language Models (VLM) to guide a diffusion policy toward more legible arc motions.

[![Status](https://img.shields.io/badge/status-working-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.8+-blue)]()

---

## 🚀 Quick Start

### One-Command Setup
```bash
# Clone and setup
git clone <repo-url>
cd "multimodal diffusion for twoblockpick"
python -m venv .venv
.venv\Scripts\activate  # Windows: or "source .venv/bin/activate" on Linux/Mac
pip install -r requirements.txt

# Reorganize project (first time only)
pwsh -ExecutionPolicy Bypass -File reorganize.ps1

# Run evaluation
python cli.py evaluate-paired --episodes 10
```

### Using Make (Recommended)
```bash
# Reorganize project
make reorganize

# Run quick test
make quick

# Run full evaluation
make eval

# Generate videos
make videos

# See all commands
make help
```

---

## 📁 Project Structure

After running `make reorganize`, the project is organized as:

```
multimodal-diffusion-for-twoblockpick/
├── cli.py                  # 🎯 Main CLI entry point
├── Makefile                # 📋 Quick commands (make help)
├── reorganize.ps1          # 📦 Project organization script
├── requirements.txt        # 📥 Dependencies
├── README.md              # 📖 This file
│
├── evaluation/            # 🔬 Main evaluation pipelines
│   ├── paired_rollouts_proper.py
│   ├── paired_replanning_rollouts_v2.py
│   ├── paired_iterative_vlm.py
│   └── quick_eval.py
│
├── experiments/           # 🧪 Tests and experiments
│   ├── test_vlm_integration.py
│   ├── test_arc_steering.py
│   └── test_*.py
│
├── verification/          # ✅ Verification scripts
│   ├── verify_arc_diversity.py
│   ├── verify_arc_measurement.py
│   └── check_horizon.py
│
├── analysis/              # 📊 Data analysis
│   ├── analyze_full_trajectory_arc.py
│   ├── measure_approach_arc.py
│   └── metrics.json
│
├── tools/                 # 🛠️ Utilities
│   ├── generate_arc15_policy_videos.py
│   ├── debug_vlm_selection.py
│   └── show_pairs.py
│
├── scripts/               # 📜 Core implementation
│   ├── train_diffusion_policy.py
│   ├── eval_bc.py
│   └── vlm_client.py
│
├── configs/               # ⚙️ Configuration files
│   └── train.yaml
│
├── data/                  # 💾 Dataset
│   └── demos/
│       ├── demos.npz
│       └── splits_compositional.json
│
├── envs/                  # 🤖 Environments
│   └── twoblockpick_env.py
│
├── docs/                  # 📚 Documentation
│   ├── ARC_MEASUREMENT_CORRECTION_SUMMARY.md
│   └── *.md
│
├── outputs/               # 📤 Output files
│   ├── videos/
│   └── metrics.json
│
└── runs/                  # 🏃 Training runs
    └── diffusion_20260222_195530/
        └── ckpt_ep100.pt
```

---

## 🎯 Main Commands (via CLI)

The CLI provides a clean interface to all functionality:

```bash
# List all commands
python cli.py list

# Paired evaluation (baseline vs VLM-guided)
python cli.py evaluate-paired --episodes 10 --n-candidates 8

# Generate arc 15-19 videos
python cli.py generate-videos --n-videos 5

# Verify arc diversity
python cli.py verify-arc --samples 100

# Debug VLM selection
python cli.py debug-vlm --episode 42

# Quick evaluation
python cli.py quick-eval --episodes 3
```

Or use Make shortcuts:
```bash
make eval          # Run evaluation
make videos        # Generate videos
make verify        # Verify arc diversity
make debug         # Debug VLM
make test          # Run tests
```

---

## 🔬 System Overview

### Problem
Standard diffusion policies generate functional but often **straight-line trajectories** that lack the curved, sweeping motions humans naturally expect when observing goal-directed reaching.

### Solution
1. **Diffusion Policy** generates diverse trajectory candidates
2. **VLM Scoring** evaluates legibility using Vision-Language Models (Gemini)
3. **Arc-Based Selection** chooses trajectories with large lateral sweeps (arc 15-19: ≥0.12m)
4. **Iterative Replanning** maintains legibility throughout the full episode

### Key Metrics
- **Arc Measurement:** Maximum lateral Y position during execution (0.06-0.15m range)
- **Arc Classes:**
  - 00-05 (gentle): < 0.08m
  - 10-14 (moderate): 0.08-0.12m  
  - 15-19 (large/legible): ≥ 0.12m
- **Legibility Score:** VLM confidence in detecting intended goal (0-1 scale)

---

## 📊 Performance

**Baseline Diffusion Policy:**
- Success rate: 92%
- Arc 15-19 rate: ~2% (mostly straight trajectories)
- Legibility score: ~0.55

**VLM-Guided Policy:**
- Success rate: 92% (maintained)
- Arc 15-19 rate: ~80% (large sweeping arcs)
- Legibility score: ~0.85

---

## 🛠️ Development

### Running Tests
```bash
# All tests
make test

# VLM integration
make test-vlm

# Specific test
python experiments/test_arc_steering.py
```

### Adding New Evaluation Scripts

1. Create script in `evaluation/` directory
2. Add CLI command in `cli.py`:
```python
def run_my_eval(args):
    cmd = [sys.executable, 'evaluation/my_script.py', ...]
    subprocess.run(cmd)
```
3. Add to Makefile for convenience

### Code Style
```bash
# Format code
make format

# Lint
make lint
```

---

## 📚 Documentation

See `docs/` for detailed documentation:

- **[ARC_MEASUREMENT_CORRECTION_SUMMARY.md](docs/ARC_MEASUREMENT_CORRECTION_SUMMARY.md)** - Arc measurement methodology
- **[EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)** - Experimental findings
- **[VLM_GUIDED_EXPLANATION.md](docs/VLM_GUIDED_EXPLANATION.md)** - System architecture

---

## 🔧 Configuration

Key files:
- **Checkpoint:** `runs/diffusion_20260222_195530/ckpt_ep100.pt`
- **Config:** `configs/train.yaml`
- **Data:** `data/demos/demos.npz`

Set VLM API key:
```bash
# Windows
$env:GEMINI_API_KEY="your-key-here"

# Linux/Mac
export GEMINI_API_KEY="your-key-here"
```

---

## 📈 Metrics and Outputs

Outputs are organized in `outputs/`:
```
outputs/
├── videos/              # Generated videos
├── metrics.json         # Evaluation metrics
└── results.csv          # Result summaries
```

Training runs in `runs/`:
```
runs/
└── diffusion_20260222_195530/
    ├── ckpt_ep100.pt   # Best checkpoint
    └── training.log
```

---

## 🐛 Troubleshooting

### Videos not generating?
```bash
python verification/verify_arc_measurement.py
```

### VLM not working?
```bash
python experiments/test_vlm_integration.py
```

### Import errors after reorganization?
Run the reorganization script:
```bash
make reorganize
```

---

## 🤝 Contributing

When adding new scripts:
1. Place in appropriate directory (`evaluation/`, `experiments/`, `tools/`, etc.)
2. Add CLI command in `cli.py`
3. Add Make target in `Makefile`
4. Update documentation

---

## 📄 License

[Your license here]

---

## 🙏 Acknowledgments

Built with:
- PyTorch + Diffusion Models
- PyBullet simulation
- Google Gemini VLM
- Research by [your team]

---

## 📞 Contact

[Your contact information]

---

**Last Updated:** March 9, 2026
