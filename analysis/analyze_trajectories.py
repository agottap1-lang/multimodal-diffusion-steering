"""
Analyze trajectory curvature from saved videos
Helps validate that steering produces different arc styles (like demos)
"""

import json
import numpy as np
from pathlib import Path
import sys

def analyze_trajectory_curvature(results_dir: Path):
    """Analyze trajectory style from evaluation results"""
    
    results_file = results_dir / "results.json"
    if not results_file.exists():
        print(f"No results.json found in {results_dir}")
        return None
    
    with open(results_file) as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Trajectory Analysis: {results_dir.name}")
    print(f"{'='*70}")
    
    episodes = data.get('episodes', [])
    if not episodes:
        print("No episode data found")
        return None
    
    for ep in episodes:
        ep_idx = ep.get('episode', 0)
        success = ep.get('success', False)
        steps = ep.get('steps', 0)
        legibility = ep.get('legibility_score', 0)
        
        status = "✓" if success else "✗"
        print(f"\nEpisode {ep_idx:03d}: {status} | Steps: {steps:3d}", end="")
        
        if legibility > 0:
            print(f" | Legibility: {legibility:.3f}", end="")
            
            # Classify arc style by legibility
            if legibility < 0.65:
                arc_style = "straight/direct"
            elif legibility < 0.80:
                arc_style = "medium arc"
            else:
                arc_style = "large sweep"
            print(f" ({arc_style})")
        else:
            print()
    
    # Summary stats
    avg_legibility = np.mean([ep.get('legibility_score', 0) for ep in episodes if ep.get('legibility_score', 0) > 0])
    success_rate = np.mean([ep.get('success', False) for ep in episodes])
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Success Rate: {success_rate*100:.1f}%")
    if avg_legibility > 0:
        print(f"  Avg Legibility: {avg_legibility:.3f}")
    print(f"{'='*70}\n")
    
    return {
        'success_rate': success_rate,
        'avg_legibility': avg_legibility,
        'episodes': episodes
    }


def compare_baseline_vs_steering(baseline_dir: Path, steering_dir: Path):
    """Compare baseline vs steering trajectories"""
    
    print("\n" + "="*70)
    print("BASELINE vs STEERING COMPARISON")
    print("="*70)
    
    print("\n[BASELINE]")
    baseline_stats = analyze_trajectory_curvature(baseline_dir)
    
    print("\n[STEERING]")
    steering_stats = analyze_trajectory_curvature(steering_dir)
    
    if baseline_stats and steering_stats:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        
        legibility_improvement = steering_stats['avg_legibility'] - baseline_stats.get('avg_legibility', 0)
        
        print(f"Legibility: {baseline_stats.get('avg_legibility', 0):.3f} → {steering_stats['avg_legibility']:.3f}")
        print(f"Improvement: +{legibility_improvement:.3f} ({legibility_improvement/max(baseline_stats.get('avg_legibility', 1), 0.01)*100:.1f}%)")
        print(f"\nExpected: Steering uses LARGER ARCS (like arc17) vs baseline")
        print(f"         This makes intent clearer earlier in the motion")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        baseline_dir = Path(sys.argv[1])
        steering_dir = Path(sys.argv[2])
        compare_baseline_vs_steering(baseline_dir, steering_dir)
    elif len(sys.argv) == 2:
        results_dir = Path(sys.argv[1])
        analyze_trajectory_curvature(results_dir)
    else:
        print("Usage:")
        print("  Single analysis:  py analyze_trajectories.py <results_dir>")
        print("  Comparison:       py analyze_trajectories.py <baseline_dir> <steering_dir>")
        print("\nExample:")
        print("  py analyze_trajectories.py runs/final_baseline/baseline_* runs/final_steering/vlm_guided_*")
