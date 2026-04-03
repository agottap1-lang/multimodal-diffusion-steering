# VLM-Guided Trajectory Selection - FULL VISUALIZATION

## Overview

The new **`evaluate-visualized`** command provides complete visibility into the VLM-guided trajectory selection process, addressing all your requirements:

## What You'll See

### 1. **Attempt Tracking for Arc 15-19**
```
🎲 Attempt 1/15: Sampling trajectory (temp=1.34)... • Arc 0.0821m (05-09)
🎲 Attempt 2/15: Sampling trajectory (temp=1.67)... ✓ Arc 0.1623m (15-19) ⭐ ARC 15-19 FOUND!
🎲 Attempt 3/15: Sampling trajectory (temp=0.92)... • Arc 0.1034m (10-14)
...
📊 Generation Summary:
   Total attempts: 12
   Valid candidates: 8
   Arc 15-19 found: 3 (25.0%)
```

### 2. **Baseline Waiting State**
```
PHASE 1: GENERATING 5-SECOND TRAJECTORY CANDIDATES
──────────────────────────────────────────────────
⏸️  BASELINE PAUSED - Waiting at initial state...
🔄 Policy will generate candidates until we have arc 15-19 options
```

### 3. **VLM Frame Sending Process**
```
PHASE 2: VLM EVALUATION
──────────────────────────────
📤 Sending 8 trajectories to VLM for legibility scoring...
🖼️  Each trajectory: 6 frames (t=0,1,2,3,4,5 seconds)

🔍 Scoring candidate 0... Legibility: 0.723
🔍 Scoring candidate 1... Legibility: 0.891  ⭐ High legibility!
🔍 Scoring candidate 2... Legibility: 0.654
...
```

### 4. **VLM Selection Decision**
```
PHASE 3: TRAJECTORY SELECTION
──────────────────────────────
🔍 Selection criteria:
   1. Legibility ≥ 0.70
   2. Arc 15-19 (≥0.15m)
   3. Arc ≤ 0.85m (not extreme)

✅ STRATEGY: Arc 15-19 + Legible
   Found 3 candidates meeting criteria
   Selected candidate with highest arc: 0.1623m

🎯 SELECTED CANDIDATE #1:
   Arc: 0.1623m (15-19)
   Legibility: 0.891
   Temperature: 1.67
   Found in attempt: 2
   Selection method: arc15_legible
```

### 5. **Execution Resume**
```
PHASE 4: EXECUTION
──────────────────
▶️  RESUMING BASELINE - Executing VLM-selected trajectory...
🎬 First 5 seconds: Using VLM-selected arc 0.1623m trajectory
🔄 After 5 seconds: Standard replanning if needed

✅ Execution complete:
   Success: True | Picked: LEFT (target: LEFT)
   Arc: 0.1623m (15-19)
   Legibility: 0.891
   Total steps: 312
   Replans needed: 2
```

### 6. **Visual Outputs Generated**

The script automatically saves visualization images:

#### **All Candidates View** (`ep{seed}_candidates_all.png`)
- Shows all 8 candidates side-by-side
- 6 frames per candidate (t=0 to 5 seconds)
- Arc measurement and class labeled
- Selected candidate highlighted in green

#### **VLM Decision View** (`ep{seed}_vlm_decision.png`)
- Top: Selected trajectory with all 6 frames
- Bottom Left: Arc vs Legibility scatter plot
  - Shows all candidates
  - Selected candidate in green
  - Arc 15-19 threshold line
  - Legibility threshold line
- Bottom Right: Arc classification summary
  - Distribution counts (00-04, 05-09, 10-14, 15-19)
  - Total candidates
  - Arc 15-19 success rate

## Usage

```bash
# Quick demo (1 episode, 5 candidates)
python cli.py evaluate-visualized --episodes 1 --n-candidates 5

# Standard evaluation (5 episodes, 8 candidates, up to 20 attempts)
python cli.py evaluate-visualized --episodes 5 --n-candidates 8 --max-attempts 20

# Full evaluation (10 episodes, 10 candidates, aggressive sampling)
python cli.py evaluate-visualized --episodes 10 --n-candidates 10 --max-attempts 30
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 5 | Number of episodes to evaluate |
| `--n-candidates` | 8 | Candidates to generate per episode |
| `--max-attempts` | 20 | Max attempts to find arc 15-19 trajectories |
| `--seed` | 100 | Random seed for reproducibility |
| `--output-dir` | `outputs/vlm_visualized` | Where to save visualizations |

## What Makes This Different?

### Standard `evaluate-paired`:
- Minimal logging
- No attempt tracking
- No visualizations
- Basic terminal output

### New `evaluate-visualized`:
- ✅ **Detailed logging** of every step
- ✅ **Attempt tracking** for arc 15-19 generation
- ✅ **Frame-by-frame visualization** of what VLM sees
- ✅ **Statistical summaries** (arc distribution, success rates)
- ✅ **Visual decision explanation** (scatter plots, distributions)
- ✅ **Clear "waiting" states** showing baseline pause
- ✅ **Saved PNG visualizations** for presentation/debugging

## Output Files

```
outputs/vlm_visualized/
├── ep100_candidates_all.png        # All candidates with frames
├── ep100_vlm_decision.png          # VLM's selection visualization
├── ep101_candidates_all.png
├── ep101_vlm_decision.png
├── ...
└── results_20260309_143022.json    # JSON summary of all results
```

## Understanding the Arc 15-19 Generation Rate

The script tracks **how efficient the policy is** at generating legible trajectories:

```json
{
  "total_attempts": 47,
  "arc15_count": 12,
  "success_rate": "25.5%"
}
```

This tells you:
- **Total attempts**: 47 trajectory samples across all episodes
- **Arc 15-19 found**: 12 trajectories met the legibility threshold
- **Success rate**: 25.5% of samples are naturally legible

**Higher success rate** = Policy learned legible motions well
**Lower success rate** = Policy needs more diverse sampling/steering

## Statistics Provided

```
FINAL STATISTICS
════════════════════════════════════════

Success Rate:
  Baseline:    4/5 (80.0%)
  VLM-Guided:  5/5 (100.0%)

Arc Statistics:
  Baseline:    0.0847m ± 0.0234m
  VLM-Guided:  0.1612m ± 0.0187m

Arc 15-19 Generation:
  Total attempts: 47
  Arc 15-19 found: 12
  Success rate: 25.5%
  Average attempts per episode: 9.4
```

## Debugging

If you see issues:

1. **No arc 15-19 found**: Increase `--max-attempts` to 30 or 40
2. **VLM API errors**: Check your Gemini API key in environment variables
3. **Out of memory**: Reduce `--n-candidates` to 5 or lower
4. **Slow execution**: Expected - VLM scoring takes ~2-3 sec per candidate

## Technical Details

### Arc Classification (4-tier system):
- **00-04**: < 0.07m (nearly straight)
- **05-09**: 0.07-0.11m (slight curve)
- **10-14**: 0.11-0.15m (moderate curve)
- **15-19**: ≥ 0.15m (sharp curve, HIGH legibility) ⭐ TARGET

### VLM Selection Strategy:
1. **Primary**: Arc 15-19 + Legibility ≥ 0.70 + Arc ≤ 0.85m
2. **Fallback 1**: Legible (≥0.70) + Highest arc available
3. **Fallback 2**: Highest legibility among all candidates

### 5-Second Video Generation:
- **150 steps** @ 30Hz = 5 seconds
- **6 frames** captured at t = 0, 1, 2, 3, 4, 5 seconds
- Frames sent as PNG format to VLM
- VLM sees full reaching motion for legibility assessment

## Next Steps

After running the visualization:

1. **Check the PNG files** in `outputs/vlm_visualized/` to see what VLM evaluated
2. **Review arc 15-19 success rate** - if < 20%, consider temperature tuning
3. **Compare baseline vs VLM-guided arcs** - VLM should consistently pick higher arcs
4. **Use insights** to tune steering parameters or training data

## Example Terminal Output

See the sections above for examples of what you'll see in the terminal. Every phase is clearly marked with separators and emoji indicators for easy reading.
