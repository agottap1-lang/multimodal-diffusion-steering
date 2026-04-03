# 🚀 Quick Command Cheatsheet

## Most Common Commands

```bash
# See all available commands
make help
# or
python cli.py list

# Run quick test (3 episodes)
make quick

# Run full evaluation (10 episodes)
make eval

# Generate arc 15-19 videos
make videos

# Verify arc diversity
make verify

# Debug VLM selection
make debug

# Run all tests
make test

# Clean up outputs
make clean
```

## Via CLI (More Options)

```bash
# Paired evaluation with custom settings
python cli.py evaluate-paired --episodes 20 --n-candidates 12 --seed 200

# Generate specific number of videos
python cli.py generate-videos --n-videos 15 --output-dir outputs/my_videos

# Verify with videos
python cli.py verify-arc --with-videos

# Debug specific episode
python cli.py debug-vlm --episode 123 --n-candidates 25

# Quick eval with custom checkpoint
python cli.py quick-eval --checkpoint runs/my_model/ckpt.pt --episodes 5
```

## VS Code Tasks

Press `Ctrl+Shift+P` → Type "Tasks: Run Task" → Select:
- 🚀 Quick Eval
- 🔬 Paired Evaluation
- 🎥 Generate Videos
- 🔍 Verify Arc Diversity
- 🐛 Debug VLM
- 📋 List Commands

## PowerShell Scripts

```powershell
# Reorganize project (first time)
pwsh -ExecutionPolicy Bypass -File reorganize.ps1
```

## File Locations

| What | Where |
|------|-------|
| Evaluation scripts | `evaluation/` |
| Tests | `experiments/` |
| Verification | `verification/` |
| Analysis | `analysis/` |
| Tools/Utilities | `tools/` |
| Core scripts | `scripts/` |
| Documentation | `docs/` |
| Outputs | `outputs/` |
| Checkpoints | `runs/` |

## Quick Navigation

```bash
# Go to evaluation scripts
cd evaluation

# Go to experiments
cd experiments

# Go to outputs
cd outputs

# Back to root
cd ..
```

## Environment Setup

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set API key
$env:GEMINI_API_KEY="your-key"  # Windows
export GEMINI_API_KEY="your-key"  # Linux/Mac
```

## Troubleshooting

```bash
# Import errors after reorganization?
make reorganize

# VLM not working?
python experiments/test_vlm_integration.py

# Videos not generating?
python verification/verify_arc_measurement.py

# Clean everything
make clean
```

---

**Save this file for quick reference!** 📌
