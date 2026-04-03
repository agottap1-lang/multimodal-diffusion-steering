# VLM Pipeline Integration Complete

## ✅ What Was Done

Successfully integrated your existing **gemini_vlm_eval** production pipeline into the legibility steering system.

### Updated Files

1. **scripts/vlm_client.py** (182 lines)
   - Now wraps your existing `GeminiClient` from `C:\Users\anude\OneDrive\Documents\gemini_vlm_eval`
   - Uses proper `ManifestEntry` and `EvaluationResult` schemas
   - Full metadata tracking (latency, confidence, API details)
   - 3-retry logic from your production client

2. **Integration Architecture**
   ```
   VLMGuidedPolicy → LegibilityScorer → GeminiClient → Gemini API
                         (adapter)        (your pipeline)
   ```

## 🏗️ How It Works

### LegibilityScorer API (Maintained Compatibility)
```python
# Your vlm_guided_policy.py code works without changes!
scorer = LegibilityScorer()

# Score single trajectory
result = scorer.score_trajectory(
    image_bytes=img,
    goal_A="pick left block",
    goal_B="pick right block",
    mode="single_frame"  # or "prefix_frames"
)

# Returns: 
# {
#   'pA': 0.75, 'pB': 0.25, 
#   'legibility_score': 0.75,
#   'confidence': 75,
#   'choice': 'A',
#   'cue': 'Robot moving toward left block',
#   'legible': 'legible',
#   'latency_ms': 1200,
#   'model': 'gemini-2.0-flash-exp'
# }
```

### Under the Hood
For each trajectory evaluation:
1. Creates `ManifestEntry` with:
   - `video_id`, `goal_A`, `goal_B`
   - `scene_id="twoblockpick"`, `task_family="block_pick"`
   - `traj_type="predicted"`

2. Calls `GeminiClient.evaluate_frame()`:
   - Passes image bytes + manifest entry
   - Gets full `EvaluationResult` with metadata
   - Has 3-retry logic, latency tracking

3. Converts to simplified dict format for compatibility with existing code

## 🚀 Running Experiments

### Prerequisites
```powershell
# Set your Gemini API key
$env:GEMINI_API_KEY = "your-api-key-here"
```

### Quick Test (3 episodes, 3 samples)
```powershell
py scripts/eval_legibility_steering.py `
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt `
    --n_episodes 3 `
    --n_samples 3 `
    --save_videos `
    --output runs/legibility_quick_test
```

### Full Evaluation (20 episodes, 5 samples)
```powershell
py scripts/eval_legibility_steering.py `
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt `
    --n_episodes 20 `
    --n_samples 5 `
    --rerank_frequency 1 `
    --save_videos `
    --output runs/legibility_eval_20ep
```

### Baseline (No Steering)
```powershell
py scripts/eval_legibility_steering.py `
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt `
    --n_episodes 20 `
    --baseline `
    --save_videos `
    --output runs/baseline_no_steering
```

## 📊 Expected Output

### Console Output
```
Episode 1/3: Success=True, Reward=1.000, Steps=95, VLM Calls=12, Avg Legibility=0.823
======================================================================
Method: VLM-Guided Legibility Steering
Success Rate: 100.0% (3/3)
Average Reward: 1.000
Average Steps: 92.3

Legibility Steering Stats:
  Avg Legibility Score: 0.823
  Min/Max Legibility: 0.651 / 0.891
  Total VLM Calls: 36
  Avg Latency per Call: 1234ms
======================================================================
```

### Saved Files
```
runs/legibility_quick_test/
├── results.json              # Full metrics + per-episode data
└── videos/
    ├── episode_000.mp4       # Rollout with trajectory overlays
    ├── episode_001.mp4
    └── episode_002.mp4
```

## 🔍 What to Look For

### In the Videos
- **Green trajectories**: Selected (most legible) path
- **Yellow trajectories**: Candidate paths that were rejected
- **Robot motion**: Should show clear intent toward one block early

### In results.json
```json
{
  "method": "VLM-Guided Legibility Steering",
  "success_rate": 0.85,
  "avg_reward": 0.85,
  "legibility_stats": {
    "avg_legibility_score": 0.78,
    "total_vlm_calls": 180,
    "avg_latency_ms": 1200
  },
  "episodes": [
    {
      "episode": 0,
      "success": true,
      "reward": 1.0,
      "n_vlm_calls": 12,
      "avg_legibility": 0.823
    }
  ]
}
```

## 🎯 Key Metrics

### Success Metrics
- **Success Rate**: % of episodes where robot picks correct block
- **Avg Reward**: Average per-episode reward (1.0 = success)

### Legibility Metrics
- **Avg Legibility Score**: Mean max(pA, pB) across all rерланing steps
- **Legibility Distribution**: How many trajectories scored >0.7 (legible)?
- **VLM Calls**: How many times we queried Gemini per episode

### Efficiency Metrics
- **Avg Latency per Call**: Gemini API response time
- **Steps to Success**: How quickly robot completes task

## 🔧 Tuning Parameters

### n_samples
Controls how many candidate trajectories to compare:
- **n_samples=3**: Fast, may miss best trajectory
- **n_samples=5**: Balanced (recommended)
- **n_samples=10**: Thorough but 2x slower

### rerank_frequency
How often to use VLM guidance:
- **rerank_frequency=1**: Every replanning step (most legible, slowest)
- **rerank_frequency=2**: Every other step (faster)
- **rerank_frequency=4**: Every 4 steps (minimal VLM overhead)

### Example: Fast mode
```powershell
py scripts/eval_legibility_steering.py `
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt `
    --n_episodes 10 `
    --n_samples 3 `
    --rerank_frequency 2 `
    --output runs/fast_legibility
```

## 📋 Next Steps

1. **Run quick test** (3 episodes) to verify everything works
2. **Compare to baseline** (no steering) to see improvement
3. **Tune parameters** (n_samples, rerank_frequency) for best tradeoff
4. **Analyze videos** - do trajectories look more legible?
5. **Collect 20+ episodes** for statistically significant results

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found"
```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

### "Failed to import gemini_vlm_eval"
The path is hardcoded to:
```python
GEMINI_VLM_PATH = Path(r"C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\src")
```
Verify this path exists and contains the package.

### "Checkpoint not found"
Use absolute path or ensure you're in the project root:
```powershell
cd "C:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick"
```

### VLM calls too slow
- Use `--rerank_frequency 2` or higher
- Reduce `--n_samples` to 3
- Gemini API latency is typically 1-2 seconds per call

## 📚 Related Files

- [LEGIBILITY_QUICKSTART.md](LEGIBILITY_QUICKSTART.md) - User guide
- [LEGIBILITY_IMPLEMENTATION_SUMMARY.md](LEGIBILITY_IMPLEMENTATION_SUMMARY.md) - Technical details
- [LEGIBILITY_STEERING_PLAN.md](LEGIBILITY_STEERING_PLAN.md) - Original plan (5 methods)

---

**Status**: ✅ Integration complete, ready to run experiments!
