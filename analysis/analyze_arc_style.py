"""
Analyze Arc Style: Compare trajectory lateral displacement to verify arc 15-19
Arc style indicator: Large Y-displacement (lateral sweep)
  - Arc 00-05: Y_displacement < 0.05m (straight)
  - Arc 10-14: Y_displacement ~0.10-0.15m (moderate curve)
  - Arc 15-19: Y_displacement > 0.18m (large sweep)
"""

import json
import sys
from pathlib import Path

def analyze_arc_style(steering_dir):
    """Analyze trajectory arc styles from results"""
    results_path = Path(steering_dir) / "results.json"
    
    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("ARC STYLE ANALYSIS - Lateral Displacement (Y-axis)")
    print("=" * 80)
    print("\nArc Style Guide:")
    print("  Arc 00-05:  Y < 0.05m   (straight, low legibility)")
    print("  Arc 10-14:  Y ~0.10-0.15m (moderate curve)")
    print("  Arc 15-19:  Y > 0.18m   (large sweep, HIGH legibility) ← TARGET")
    print()
    
    # Extract trajectory stats if available
    if 'episodes' in data:
        for i, ep in enumerate(data['episodes']):
            print(f"\nEpisode {i}:")
            print(f"  Success: {'✓' if ep.get('success', False) else '✗'}")
            print(f"  Steps: {ep.get('steps', 'N/A')}")
            leg_mean = ep.get('legibility', {}).get('mean', None)
            leg_str = f"{leg_mean:.3f}" if leg_mean is not None else "N/A"
            print(f"  Legibility: {leg_str}")
            
            # Check if trajectory data available
            if 'trajectory_stats' in ep:
                stats = ep['trajectory_stats']
                y_disp = stats.get('max_y_displacement', 0)
                print(f"  Max Y Displacement: {y_disp:.3f}m", end="")
                
                if y_disp > 0.18:
                    print(" ← Arc 15-19 style ✓✓")
                elif y_disp > 0.10:
                    print(" ← Arc 10-14 style")
                else:
                    print(" ← Arc 00-05 style (too straight)")
    
    # Summary stats
    if 'stats' in data:
        stats = data['stats']
        print("\n" + "-" * 80)
        print("SUMMARY:")
        print(f"  Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
        print(f"  Avg Legibility: {stats.get('legibility', {}).get('mean', 0):.3f}")
        
        if stats.get('legibility', {}).get('mean', 0) > 0.78:
            print("  ✓✓ HIGH LEGIBILITY achieved (target: >0.75)")
        else:
            print("  ⚠️  Legibility below target")
    
    print("\n" + "=" * 80)
    print("\nNOTE: Without trajectory_stats in results, watch videos to verify:")
    print("  → Large lateral sweeps visible = Arc 15-19 ✓")
    print("  → Nearly straight paths = Arc 00-05 ✗")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent steering result
        test_dir = Path("runs/test_arc_steering")
        if test_dir.exists():
            subdirs = sorted([d for d in test_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
            if subdirs:
                analyze_arc_style(subdirs[0])
            else:
                print("No results found in runs/test_arc_steering")
        else:
            print("Usage: py analyze_arc_style.py <steering_results_dir>")
    else:
        analyze_arc_style(sys.argv[1])
