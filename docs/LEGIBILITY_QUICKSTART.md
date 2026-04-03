# 🎯 VLM-Guided Legibility Steering - Quick Start Guide

**Implementation Complete!** This implements **Method 1: VLM-Guided Trajectory Reranking** from `LEGIBILITY_STEERING_PLAN.md`.

## What We've Built

✅ **VLM Client** (`scripts/vlm_client.py`) - Gemini 2.0 Flash integration for legibility scoring  
✅ **Trajectory Visualizer** (`scripts/trajectory_visualizer.py`) - Renders predicted actions on environment frames  
✅ **Reranking Policy** (`scripts/vlm_guided_policy.py`) - Samples N trajectories and picks the most legible  
✅ **Evaluation Script** (`scripts/eval_legibility_steering.py`) - Run rollouts with legibility steering  

---

## 🚀 How It Works

1. **At each replanning step:**
   - Sample N=5 candidate trajectories from your trained diffusion policy
   - Visualize each trajectory (overlay arrows showing predicted motion)
   - Send visualizations to Gemini VLM with prompt: "Which is more legible?"
   - Execute the most legible trajectory

2. **VLM evaluates based on:**
   - How clearly the motion indicates the goal (left block vs right block)
   - Visual cues like direction, approach angle, speed
   - Returns legibility scores: 0.0 (ambiguous) to 1.0 (very clear)

---

## 📋 Prerequisites

1. **Set Gemini API Key:**
   ```powershell
   $env:GEMINI_API_KEY="your-key-here"
   ```

2. **Install dependencies:**
   ```powershell
   pip install google-generativeai pillow opencv-python
   ```

---

## 🎬 Running Experiments

### 1. Baseline (No Steering)
```powershell
py scripts/eval_legibility_steering.py `
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt `
    --n_episodes 20 `
    --baseline `
    --save_videos `
    --output runs/baseline_eval
```

**What this does:** Runs your policy without any VLM guidance (for comparison).

---

### 2. VLM-Guided Steering (Recommended)
```powershell
py scripts/eval_legibility_steering.py `
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt `
    --n_episodes 20 `
    --n_samples 5 `
    --rerank_frequency 1 `
    --save_videos `
    --output runs/legibility_eval
```

**What this does:** 
- Samples 5 candidate trajectories at each replanning step
- Queries Gemini VLM to score legibility
- Executes the most legible trajectory

**Parameters:**
- `--n_samples`: Number of candidates (higher = better legibility, slower)
  - Try: 3 (fast), 5 (recommended), 10 (best quality)
- `--rerank_frequency`: Rerank every N replans
  - 1 = every time (most legible, slowest)
  - 2 = every other time (good balance)
  - 4 = less frequent (faster, less steering)

---

### 3. Random Reranking Baseline
```powershell
py scripts/eval_legibility_steering.py `
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt `
    --n_episodes 20 `
    --n_samples 5 `
    --random_baseline `
    --save_videos `
    --output runs/random_baseline
```

**What this does:** Tests if VLM actually helps (vs. just sampling more trajectories).

---

## 📊 Expected Results

| Method | Success Rate | Legibility Score | VLM Calls | Time per Episode |
|--------|-------------|------------------|-----------|------------------|
| Baseline (N=1) | ~50% | - | 0 | ~2s |
| Random (N=5) | ~50% | - | 0 | ~4s |
| VLM-Guided (N=5) | ~45-55% | **0.70-0.85** | ~250 | ~30s |

**Key Insights:**
- Task success rate should remain similar (legibility ≠ task success)
- Legibility scores should increase by **20-40%**
- VLM adds latency (~1-2s per call) but makes motion more interpretable

---

## 📹 Analyzing Results

### View Videos
```powershell
# Videos are saved to:
runs/legibility_eval/vlm_guided_n5_<timestamp>/videos/

# Compare baseline vs steered:
# - Baseline: Direct, efficient but ambiguous early motion
# - Steered: Exaggerated, overshoot toward target, clearer intent
```

### Check Metrics
```powershell
py -c "import json; print(json.dumps(json.load(open('runs/legibility_eval/vlm_guided_n5_*/results.json')), indent=2))"
```

**Key metrics:**
- `success_rate`: Task completion
- `legibility_stats.avg_legibility_score`: How legible (higher = better)
- `legibility_stats.total_vlm_calls`: Number of VLM queries
- `legibility_stats.avg_latency_ms`: VLM response time

---

## 🎯 Comparative Study

Run all three methods:

```powershell
# 1. Baseline
py scripts/eval_legibility_steering.py --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt --n_episodes 20 --baseline --save_videos --output runs/comparison/baseline

# 2. Random reranking
py scripts/eval_legibility_steering.py --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt --n_episodes 20 --n_samples 5 --random_baseline --save_videos --output runs/comparison/random

# 3. VLM-guided
py scripts/eval_legibility_steering.py --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt --n_episodes 20 --n_samples 5 --save_videos --output runs/comparison/vlm_guided
```

Then compare results:

```powershell
py scripts/compare_legibility_methods.py --results_dirs runs/comparison/*
```

---

## 🔬 Research Questions to Explore

1. **Does VLM steering improve legibility without hurting task success?**
   - Compare success rates across methods
   - Check if legibility scores correlate with human judgments

2. **What's the optimal N (number of candidates)?**
   - Try N = 3, 5, 7, 10
   - Plot legibility vs. computation cost

3. **When should we rerank?**
   - Compare rerank_frequency = 1, 2, 4, 8
   - Early motion may be more important than late

4. **What visual cues does VLM use?**
   - Analyze `cue` field in VLM responses
   - Do humans agree with VLM judgments?

---

## 🐛 Troubleshooting

### Error: "GEMINI_API_KEY must be set"
```powershell
# Set in current session:
$env:GEMINI_API_KEY="your-key-here"

# Or add to .env file:
echo "GEMINI_API_KEY=your-key-here" > .env
```

### VLM calls are slow
- Try `--rerank_frequency 2` or `4` (rerank less often)
- Reduce `--n_samples` to 3
- Use fewer episodes for quick tests

### "Module not found" errors
```powershell
pip install google-generativeai opencv-python pillow
```

### Low legibility scores
- This is expected! Baseline policy wasn't trained for legibility
- The goal is to see VLM-guided > random > baseline
- Absolute scores matter less than relative improvements

---

## 📚 Next Steps

1. **Run quick test (5 episodes):**
   ```powershell
   py scripts/eval_legibility_steering.py --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt --n_episodes 5 --n_samples 3 --save_videos
   ```

2. **Watch generated videos** to see steering in action

3. **Run full comparison** (baseline vs VLM-guided)

4. **Analyze results** and iterate on prompts/parameters

5. **Consider Method 2** (Classifier-Free Guidance) if results are promising

---

## 🎓 Related Research

- **Diffusion Policy**: [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)
- **Legible Motion**: Dragan et al. 2013 - "Legibility and Predictability of Robot Motion"
- **VLM for Robotics**: Du et al. 2023 - "Vision-Language Models as Success Detectors"
- **Test-Time Guidance**: Dhariwal & Nichol 2021 - "Diffusion Models Beat GANs"

---

## 💡 Tips for Best Results

1. **Start small**: Test with 5 episodes before running 50
2. **Save videos**: Visual inspection is crucial for understanding legibility
3. **Compare methods**: Always run baseline to measure improvement
4. **Tune prompts**: Modify VLM prompts in `vlm_client.py` for your task
5. **Monitor costs**: Each VLM call costs API credits (track with `total_vlm_calls`)

---

**Ready to test?** Run the quick test above and let me know what you see! 🚀
