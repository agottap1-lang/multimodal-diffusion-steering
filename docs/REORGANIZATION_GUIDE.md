# 🎯 Project Reorganization Guide

## What Changed?

Your project has been reorganized following senior developer best practices for better maintainability and clarity.

## ✅ Before vs After

### Before (Cluttered Root)
```
multimodal-diffusion/
├── 50+ Python scripts at root level 😵
├── Mix of tests, tools, evaluation scripts
├── Documentation scattered everywhere
└── Hard to find what you need
```

### After (Clean Structure)
```
multimodal-diffusion/
├── cli.py                 # 🎯 ONE entry point for everything
├── Makefile               # 📋 Quick commands
├── evaluation/            # 🔬 Core evaluation pipelines
├── experiments/           # 🧪 Tests and experiments
├── verification/          # ✅ Verification scripts
├── analysis/              # 📊 Data analysis
├── tools/                 # 🛠️ Utilities
├── scripts/               # 📜 Core implementation
├── docs/                  # 📚 All documentation
└── [data, configs, etc.]  # Supporting files
```

---

## 🚀 How to Use the New Structure

### 1️⃣ Reorganize (First Time Only)
```bash
# Option A: Using PowerShell script
pwsh -ExecutionPolicy Bypass -File reorganize.ps1

# Option B: Using Make
make reorganize
```

### 2️⃣ Use the CLI
Instead of remembering 50 different script names, use ONE command:

```bash
# See all available commands
python cli.py list

# Run evaluation
python cli.py evaluate-paired --episodes 10

# Generate videos
python cli.py generate-videos --n-videos 5

# Verify arc
python cli.py verify-arc --samples 100

# Debug VLM
python cli.py debug-vlm --episode 42
```

### 3️⃣ Or Use Make Shortcuts
Even simpler:
```bash
make help          # See all commands
make eval          # Run evaluation
make videos        # Generate videos
make verify        # Verify arc
make debug         # Debug VLM
make test          # Run tests
make clean         # Clean up
```

### 4️⃣ Or Use VS Code Tasks
Press `Ctrl+Shift+P` → "Tasks: Run Task" → Choose from menu:
- 🚀 Quick Eval
- 🔬 Paired Evaluation
- 🎥 Generate Videos
- 🔍 Verify Arc Diversity
- 🐛 Debug VLM
- 📋 List Commands

---

## 📁 Where Did My Files Go?

### Evaluation Scripts → `evaluation/`
- `paired_rollouts_proper.py`
- `paired_replanning_rollouts_v2.py`
- `paired_iterative_vlm.py`
- `rollout_policy_vs_vlm_guided.py`
- `compare_policy_vs_vlm_videos.py`
- `eval_goal_locked_vlm.py`
- `quick_eval.py`

### Test Scripts → `experiments/`
- `test_*.py` (all test files)
- `quick_variant_test.py`
- `run_baseline_steering_comparison.py`

### Verification Scripts → `verification/`
- `verify_*.py` (all verification files)
- `check_horizon.py`
- `inspect_frames.py`

### Analysis Scripts → `analysis/`
- `analyze_*.py` (all analysis files)
- `measure_*.py`
- `find_correct_arc_metric.py`
- `detailed_arc_analysis.py`

### Utility Scripts → `tools/`
- `generate_arc15_policy_videos.py`
- `debug_vlm_selection.py`
- `save_partial_summary.py`
- `show_pairs.py`

### Documentation → `docs/`
- All `.md` files (except README.md)
- Summary files
- Analysis documents
- Fix documentation

### Output Files → `outputs/`
- `.txt` files
- `.json` files (except from configs/)
- `.log` files

---

## 🎓 Benefits of New Structure

### 1. **Clear Intent**
- Know immediately where to find scripts
- `evaluation/` = production code
- `experiments/` = testing/development
- `tools/` = utilities

### 2. **Better Workflows**
```bash
# Old way 😵
python paired_rollouts_proper.py --checkpoint runs/... --episodes 10 --n-candidates 8 --seed 100

# New way 😎
make eval

# Or
python cli.py evaluate-paired --episodes 10
```

### 3. **Professional Standards**
- Follows Python packaging best practices
- Similar to projects like: scikit-learn, pytorch, transformers
- Easy for new team members to understand

### 4. **Easier Collaboration**
- PRs are cleaner (files in logical directories)
- No more "where should this go?" confusion
- Clear separation of concerns

---

## 🔧 What If Something Breaks?

### Import Errors
If you get import errors, the script might have hardcoded paths. Update them:

```python
# Before
from scripts.vlm_client import LegibilityScorer

# After (works from anywhere)
from scripts.vlm_client import LegibilityScorer  # Still works!
```

The CLI handles paths automatically, so prefer using it.

### Script Not Found
Check the new location:
```bash
# Find where a script moved
ls evaluation/ | grep "script_name"
ls experiments/ | grep "script_name"
ls tools/ | grep "script_name"
```

Or use the CLI which knows the locations:
```bash
python cli.py list  # Shows all available commands
```

---

## 🎯 Common Tasks - Quick Reference

| Task | Old Way | New Way |
|------|---------|---------|
| **Run eval** | `python paired_rollouts_proper.py ...` | `make eval` or `python cli.py evaluate-paired` |
| **Generate videos** | `python generate_arc15_policy_videos.py ...` | `make videos` or `python cli.py generate-videos` |
| **Verify arc** | `python verify_arc_diversity.py ...` | `make verify` or `python cli.py verify-arc` |
| **Debug VLM** | `python debug_vlm_selection.py ...` | `make debug` or `python cli.py debug-vlm` |
| **Run tests** | `python test_*.py` | `make test` |
| **Clean up** | Manually delete | `make clean` |

---

## 📋 Checklist After Reorganization

- [ ] Run `make reorganize` to organize files
- [ ] Test with `make quick` (3-episode test)
- [ ] Verify imports work: `python -c "from scripts.vlm_client import LegibilityScorer; print('OK')"`
- [ ] Update any custom scripts that import from old locations
- [ ] Update VS Code launch configurations if needed
- [ ] Update team documentation/wiki

---

## 🤝 Team Guidelines

### Adding New Scripts
```bash
# 1. Place in appropriate directory
touch evaluation/my_new_eval.py    # For evaluation pipelines
touch experiments/test_feature.py  # For tests
touch tools/my_utility.py          # For utilities

# 2. Add CLI command (optional but recommended)
# Edit cli.py to add new command

# 3. Add Make target (optional)
# Edit Makefile to add shortcut
```

### Running Others' Code
```bash
# Always prefer CLI commands over direct script execution
python cli.py <command>    # ✅ Good
python scripts/foo.py      # ❌ Avoid (unless necessary)
```

---

## 🆘 Rolling Back (If Needed)

If something goes wrong and you need to undo:

```bash
# 1. Stash uncommitted changes
git stash

# 2. Reset to before reorganization
git reset --hard HEAD~1

# 3. Restore your work
git stash pop
```

---

## 📞 Questions?

- **Can't find a script?** → Check the directory mapping above
- **Import errors?** → Use the CLI instead of direct script execution
- **Want to add new command?** → Edit `cli.py` and `Makefile`
- **Need help?** → Run `make help` or `python cli.py list`

---

## ✨ Next Steps

1. **Try it out:** Run `make quick` to test everything works
2. **Update your workflow:** Start using `make` commands
3. **Explore the CLI:** Run `python cli.py list`
4. **Read new README:** Check `README_NEW.md` for updated documentation
5. **Give feedback:** Let the team know what works/doesn't work

---

**Happy coding! 🚀**
