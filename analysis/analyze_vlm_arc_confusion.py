"""
Analyze VLM predictions per-arc to identify confusion patterns.
Generates exact accuracy statistics and presentable figures.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def load_results(jsonl_path: Path) -> pd.DataFrame:
    """Load results.jsonl into a DataFrame."""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def compute_per_arc_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each arc (0-19) at each time point (0-10s).
    Returns DataFrame with: arc_idx, t_sec, n_samples, accuracy, avg_confidence
    """
    # Add correctness column
    df['correct'] = df.apply(
        lambda row: (row['choice'] == row['goal_gt']) if row['choice'] != 'C' else False,
        axis=1
    )
    
    # Group by arc_idx and t_sec
    accuracy_df = df.groupby(['arc_idx', 't_sec']).agg({
        'correct': ['sum', 'count', 'mean'],
        'confidence': 'mean',
        'pA': 'mean',
        'pB': 'mean',
        'demo_side': 'first'
    }).reset_index()
    
    # Flatten column names
    accuracy_df.columns = ['arc_idx', 't_sec', 'n_correct', 'n_samples', 'accuracy', 
                           'avg_confidence', 'avg_pA', 'avg_pB', 'demo_side']
    
    return accuracy_df

def create_accuracy_heatmap(accuracy_df: pd.DataFrame, output_path: Path):
    """Create heatmap showing accuracy across arcs and time."""
    # Pivot to create matrix
    heatmap_data = accuracy_df.pivot(index='arc_idx', columns='t_sec', values='accuracy')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'},
                ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Arc Index', fontsize=12, fontweight='bold')
    ax.set_title('VLM Prediction Accuracy by Arc and Time\n(Green=Correct, Red=Wrong)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add arc class boundaries
    for y in [4.5, 9.5, 14.5]:
        ax.axhline(y=y, color='white', linewidth=3, linestyle='--')
    
    # Add annotations for arc classes
    ax.text(-1.5, 2, 'Arc 00-04\n(Straight)', ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(-1.5, 7, 'Arc 05-09\n(Slight)', ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(-1.5, 12, 'Arc 10-14\n(Moderate)', ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(-1.5, 17, 'Arc 15-19\n(High)', ha='right', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy heatmap to {output_path}")

def create_accuracy_curves(accuracy_df: pd.DataFrame, output_path: Path):
    """Create line plots showing accuracy over time for each arc."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    arc_classes = {
        'Arc 00-04 (Straight)': range(0, 5),
        'Arc 05-09 (Slight)': range(5, 10),
        'Arc 10-14 (Moderate)': range(10, 15),
        'Arc 15-19 (High)': range(15, 20)
    }
    
    for (title, arc_range), ax in zip(arc_classes.items(), axes.flatten()):
        for arc_idx in arc_range:
            arc_data = accuracy_df[accuracy_df['arc_idx'] == arc_idx]
            demo_side = arc_data['demo_side'].iloc[0]
            label = f"Arc {arc_idx:02d} ({'L' if demo_side == 'left' else 'R'})"
            ax.plot(arc_data['t_sec'], arc_data['accuracy'], marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    
    plt.suptitle('VLM Accuracy Over Time by Arc Class', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy curves to {output_path}")

def create_confidence_heatmap(accuracy_df: pd.DataFrame, output_path: Path):
    """Create heatmap showing VLM confidence across arcs and time."""
    heatmap_data = accuracy_df.pivot(index='arc_idx', columns='t_sec', values='avg_confidence')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='viridis', 
                vmin=50, vmax=100, cbar_kws={'label': 'Confidence (%)'},
                ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Arc Index', fontsize=12, fontweight='bold')
    ax.set_title('VLM Confidence by Arc and Time', fontsize=14, fontweight='bold', pad=20)
    
    # Add arc class boundaries
    for y in [4.5, 9.5, 14.5]:
        ax.axhline(y=y, color='white', linewidth=3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence heatmap to {output_path}")

def analyze_prediction_flips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze when VLM predictions flip over time for each video.
    Returns DataFrame with flip statistics per arc.
    """
    flip_stats = []
    
    for video_id in df['video_id'].unique():
        video_df = df[df['video_id'] == video_id].sort_values('t_sec')
        
        arc_idx = video_df['arc_idx'].iloc[0]
        demo_side = video_df['demo_side'].iloc[0]
        goal_gt = video_df['goal_gt'].iloc[0]
        
        # Track prediction sequence
        predictions = []
        flips = 0
        prev_choice = None
        
        for _, row in video_df.iterrows():
            choice = row['choice']
            predictions.append(choice)
            
            if prev_choice is not None and prev_choice != choice and choice != 'C' and prev_choice != 'C':
                flips += 1
            
            prev_choice = choice
        
        # Final prediction
        final_choice = video_df[video_df['t_sec'] == video_df['t_sec'].max()]['choice'].iloc[0]
        final_correct = (final_choice == goal_gt)
        
        # First time correct
        first_correct_time = None
        for _, row in video_df.iterrows():
            if row['choice'] == goal_gt:
                first_correct_time = row['t_sec']
                break
        
        # First time wrong
        first_wrong_time = None
        for _, row in video_df.iterrows():
            if row['choice'] != 'C' and row['choice'] != goal_gt:
                first_wrong_time = row['t_sec']
                break
        
        flip_stats.append({
            'video_id': video_id,
            'arc_idx': arc_idx,
            'demo_side': demo_side,
            'n_flips': flips,
            'final_correct': final_correct,
            'first_correct_time': first_correct_time,
            'first_wrong_time': first_wrong_time,
            'prediction_sequence': ''.join(predictions)
        })
    
    return pd.DataFrame(flip_stats)

def create_flip_analysis_plot(flip_df: pd.DataFrame, output_path: Path):
    """Create visualizations of prediction flips."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Number of flips by arc
    arc_flips = flip_df.groupby('arc_idx')['n_flips'].mean()
    axes[0, 0].bar(arc_flips.index, arc_flips.values, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Arc Index', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Average Number of Flips', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Prediction Instability by Arc', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add arc class boundaries
    for x in [4.5, 9.5, 14.5]:
        axes[0, 0].axvline(x=x, color='red', linestyle='--', alpha=0.5)
    
    # 2. Final accuracy by arc
    arc_accuracy = flip_df.groupby('arc_idx')['final_correct'].mean()
    colors = ['green' if acc >= 0.5 else 'red' for acc in arc_accuracy.values]
    axes[0, 1].bar(arc_accuracy.index, arc_accuracy.values, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Chance')
    axes[0, 1].set_xlabel('Arc Index', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Final Accuracy', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Final Prediction Accuracy by Arc', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].legend()
    
    # Add arc class boundaries
    for x in [4.5, 9.5, 14.5]:
        axes[0, 1].axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Time to first correct vs first wrong
    arc_correct_time = flip_df.groupby('arc_idx')['first_correct_time'].mean()
    arc_wrong_time = flip_df.groupby('arc_idx')['first_wrong_time'].mean()
    
    x = np.arange(20)
    width = 0.35
    
    axes[1, 0].bar(x - width/2, arc_correct_time.values, width, label='First Correct', 
                    color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].bar(x + width/2, arc_wrong_time.values, width, label='First Wrong', 
                    color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Arc Index', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Time to First Correct vs Wrong Prediction', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter: flips vs final accuracy
    for arc_class, color, label in [
        (range(0, 5), 'blue', 'Arc 00-04'),
        (range(5, 10), 'green', 'Arc 05-09'),
        (range(10, 15), 'orange', 'Arc 10-14'),
        (range(15, 20), 'red', 'Arc 15-19')
    ]:
        class_df = flip_df[flip_df['arc_idx'].isin(arc_class)]
        arc_stats = class_df.groupby('arc_idx').agg({'n_flips': 'mean', 'final_correct': 'mean'})
        axes[1, 1].scatter(arc_stats['n_flips'], arc_stats['final_correct'], 
                           s=150, alpha=0.7, color=color, edgecolor='black', linewidth=2, label=label)
    
    axes[1, 1].set_xlabel('Average Number of Flips', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Final Accuracy', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Prediction Instability vs Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle('VLM Prediction Flip Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved flip analysis to {output_path}")

def create_detailed_csv(df: pd.DataFrame, output_path: Path):
    """
    Create detailed CSV similar to user_study_timepoints.csv format.
    Columns: video_id, arc_idx, demo_side, t_sec, pA, pB, choice, confidence, 
             goal_gt, correct, legible
    """
    # Add correctness column
    df['correct'] = df.apply(
        lambda row: (row['choice'] == row['goal_gt']) if row['choice'] != 'C' else False,
        axis=1
    )
    
    # Select relevant columns
    output_df = df[[
        'video_id', 'arc_idx', 'demo_side', 't_sec', 
        'pA', 'pB', 'choice', 'confidence', 'goal_gt', 'correct', 'legible', 'cue'
    ]].copy()
    
    # Sort by arc_idx and t_sec
    output_df = output_df.sort_values(['arc_idx', 'demo_side', 't_sec'])
    
    # Save to CSV
    output_df.to_csv(output_path, index=False, float_format='%.3f')
    print(f"Saved detailed CSV to {output_path}")
    print(f"  Total rows: {len(output_df)}")

def create_summary_statistics(df: pd.DataFrame, accuracy_df: pd.DataFrame, 
                               flip_df: pd.DataFrame, output_path: Path):
    """Create comprehensive summary statistics."""
    summary = {
        "overall_stats": {
            "total_evaluations": len(df),
            "total_videos": df['video_id'].nunique(),
            "total_arcs": df['arc_idx'].nunique(),
            "time_points": sorted(df['t_sec'].unique().tolist()),
            "overall_accuracy": float(df.apply(lambda r: r['choice'] == r['goal_gt'] if r['choice'] != 'C' else False, axis=1).mean())
        },
        "per_arc_stats": [],
        "arc_class_stats": []
    }
    
    # Per-arc statistics
    for arc_idx in sorted(df['arc_idx'].unique()):
        arc_data = df[df['arc_idx'] == arc_idx]
        arc_correct = arc_data.apply(lambda r: r['choice'] == r['goal_gt'] if r['choice'] != 'C' else False, axis=1)
        
        # Final time point accuracy
        final_data = arc_data[arc_data['t_sec'] == arc_data['t_sec'].max()]
        final_accuracy = float(final_data.apply(lambda r: r['choice'] == r['goal_gt'] if r['choice'] != 'C' else False, axis=1).mean())
        
        # Time to legible
        legible_times = arc_data[arc_data['legible'] == 'legible_now']['t_sec']
        time_to_legible = float(legible_times.min()) if len(legible_times) > 0 else None
        
        # Flip stats
        arc_flip_stats = flip_df[flip_df['arc_idx'] == arc_idx]
        
        summary["per_arc_stats"].append({
            "arc_idx": int(arc_idx),
            "demo_side": arc_data['demo_side'].iloc[0],
            "n_evaluations": int(len(arc_data)),
            "overall_accuracy": float(arc_correct.mean()),
            "final_accuracy": final_accuracy,
            "time_to_legible": time_to_legible,
            "avg_confidence": float(arc_data['confidence'].mean()),
            "avg_flips": float(arc_flip_stats['n_flips'].mean()),
            "final_correct_rate": float(arc_flip_stats['final_correct'].mean())
        })
    
    # Arc class statistics
    arc_classes = {
        'arc_00-04': range(0, 5),
        'arc_05-09': range(5, 10),
        'arc_10-14': range(10, 15),
        'arc_15-19': range(15, 20)
    }
    
    for class_name, arc_range in arc_classes.items():
        class_data = df[df['arc_idx'].isin(arc_range)]
        class_correct = class_data.apply(lambda r: r['choice'] == r['goal_gt'] if r['choice'] != 'C' else False, axis=1)
        
        # Final accuracy
        final_data = class_data[class_data['t_sec'] == class_data['t_sec'].max()]
        final_accuracy = float(final_data.apply(lambda r: r['choice'] == r['goal_gt'] if r['choice'] != 'C' else False, axis=1).mean())
        
        # Flip stats
        class_flip_stats = flip_df[flip_df['arc_idx'].isin(arc_range)]
        
        summary["arc_class_stats"].append({
            "arc_class": class_name,
            "n_arcs": len([a for a in arc_range if a in df['arc_idx'].unique()]),
            "n_videos": int(class_data['video_id'].nunique()),
            "overall_accuracy": float(class_correct.mean()),
            "final_accuracy": final_accuracy,
            "avg_confidence": float(class_data['confidence'].mean()),
            "avg_time_to_legible": float(class_data[class_data['legible'] == 'legible_now'].groupby('video_id')['t_sec'].min().mean()),
            "avg_flips": float(class_flip_stats['n_flips'].mean())
        })
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary statistics to {output_path}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS: VLM Arc Confusion Analysis")
    print("="*80)
    
    print("\n📊 OVERALL STATISTICS:")
    print(f"  Total Evaluations: {summary['overall_stats']['total_evaluations']}")
    print(f"  Total Videos: {summary['overall_stats']['total_videos']}")
    print(f"  Overall Accuracy: {summary['overall_stats']['overall_accuracy']:.1%}")
    
    print("\n🎯 ARC CLASS PERFORMANCE:")
    for stats in summary["arc_class_stats"]:
        print(f"\n  {stats['arc_class'].upper()}:")
        print(f"    Final Accuracy: {stats['final_accuracy']:.1%}")
        print(f"    Time to Legible: {stats['avg_time_to_legible']:.1f}s")
        print(f"    Avg Confidence: {stats['avg_confidence']:.1f}%")
        print(f"    Avg Flips: {stats['avg_flips']:.2f}")
    
    print("\n🔍 CRITICAL INSIGHT:")
    high_arc_final = [s for s in summary["arc_class_stats"] if s['arc_class'] == 'arc_15-19'][0]['final_accuracy']
    high_arc_time = [s for s in summary["arc_class_stats"] if s['arc_class'] == 'arc_15-19'][0]['avg_time_to_legible']
    moderate_arc_final = [s for s in summary["arc_class_stats"] if s['arc_class'] == 'arc_10-14'][0]['final_accuracy']
    moderate_arc_time = [s for s in summary["arc_class_stats"] if s['arc_class'] == 'arc_10-14'][0]['avg_time_to_legible']
    
    print(f"  High arcs (15-19): {high_arc_final:.1%} accuracy at {high_arc_time:.1f}s")
    print(f"  Moderate arcs (10-14): {moderate_arc_final:.1%} accuracy at {moderate_arc_time:.1f}s")
    
    if high_arc_final < moderate_arc_final and high_arc_time <= moderate_arc_time:
        print("\n  ⚠️  PARADOX CONFIRMED:")
        print("     High arcs become legible FASTER but have LOWER final accuracy!")
        print("     This suggests VLM is CONFUSING high lateral arcs (flipping left/right).")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Analyze VLM arc confusion patterns")
    parser.add_argument('--input', type=str, 
                        default='outputs/demo_legibility_prefix_cfg00/results.jsonl',
                        help='Path to results.jsonl')
    parser.add_argument('--output', type=str,
                        default='analysis/vlm_arc_confusion',
                        help='Output directory for analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VLM ARC CONFUSION ANALYSIS")
    print("="*80)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}\n")
    
    # Load data
    print("Loading results...")
    df = load_results(Path(args.input))
    print(f"  Loaded {len(df)} evaluations from {df['video_id'].nunique()} videos")
    
    # Compute per-arc accuracy
    print("\nComputing per-arc accuracy...")
    accuracy_df = compute_per_arc_accuracy(df)
    
    # Analyze flips
    print("Analyzing prediction flips...")
    flip_df = analyze_prediction_flips(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_accuracy_heatmap(accuracy_df, output_dir / 'accuracy_heatmap.png')
    create_accuracy_curves(accuracy_df, output_dir / 'accuracy_curves.png')
    create_confidence_heatmap(accuracy_df, output_dir / 'confidence_heatmap.png')
    create_flip_analysis_plot(flip_df, output_dir / 'flip_analysis.png')
    
    # Create outputs
    print("\nGenerating output files...")
    create_detailed_csv(df, output_dir / 'vlm_timepoints_detailed.csv')
    create_summary_statistics(df, accuracy_df, flip_df, output_dir / 'summary_statistics.json')
    
    # Save additional data
    accuracy_df.to_csv(output_dir / 'per_arc_accuracy.csv', index=False, float_format='%.3f')
    print(f"Saved per-arc accuracy to {output_dir / 'per_arc_accuracy.csv'}")
    
    flip_df.to_csv(output_dir / 'flip_statistics.csv', index=False)
    print(f"Saved flip statistics to {output_dir / 'flip_statistics.csv'}")
    
    print("\n✅ Analysis complete!")
    print(f"\nOutputs saved to: {output_dir}")
    print("  - accuracy_heatmap.png")
    print("  - accuracy_curves.png")
    print("  - confidence_heatmap.png")
    print("  - flip_analysis.png")
    print("  - vlm_timepoints_detailed.csv")
    print("  - per_arc_accuracy.csv")
    print("  - flip_statistics.csv")
    print("  - summary_statistics.json")

if __name__ == '__main__':
    main()
