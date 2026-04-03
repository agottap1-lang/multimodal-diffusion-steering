#!/usr/bin/env python3
"""
Multimodal Diffusion CLI - Main entry point for running evaluations and experiments
Usage: python cli.py <command> [options]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Diffusion Policy CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run paired evaluation (baseline vs VLM-guided)
  python cli.py evaluate-paired --episodes 10
  
  # Generate arc 15-19 videos
  python cli.py generate-videos --n-videos 5
  
  # Verify arc diversity
  python cli.py verify-arc --samples 100
  
  # Debug VLM selection
  python cli.py debug-vlm --episode 42
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate paired rollouts
    eval_parser = subparsers.add_parser('evaluate-paired', help='Run paired evaluation (baseline vs VLM)')
    eval_parser.add_argument('--checkpoint', default='runs/diffusion_20260222_195530/ckpt_ep100.pt')
    eval_parser.add_argument('--episodes', type=int, default=10)
    eval_parser.add_argument('--n-candidates', type=int, default=8)
    eval_parser.add_argument('--seed', type=int, default=100)
    eval_parser.add_argument('--version', choices=['v1', 'v2', 'iterative', 'proper'], default='proper')
    
    # Generate videos
    video_parser = subparsers.add_parser('generate-videos', help='Generate arc 15-19 policy videos')
    video_parser.add_argument('--checkpoint', default='runs/diffusion_20260222_195530/ckpt_ep100.pt')
    video_parser.add_argument('--n-videos', type=int, default=10)
    video_parser.add_argument('--output-dir', default='outputs/videos')
    
    # Verify arc diversity
    verify_parser = subparsers.add_parser('verify-arc', help='Verify arc diversity in policy samples')
    verify_parser.add_argument('--checkpoint', default='runs/diffusion_20260222_195530/ckpt_ep100.pt')
    verify_parser.add_argument('--samples', type=int, default=100)
    verify_parser.add_argument('--with-videos', action='store_true')
    
    # Debug VLM
    debug_parser = subparsers.add_parser('debug-vlm', help='Debug VLM selection process')
    debug_parser.add_argument('--checkpoint', default='runs/diffusion_20260222_195530/ckpt_ep100.pt')
    debug_parser.add_argument('--episode', type=int, default=42)
    debug_parser.add_argument('--n-candidates', type=int, default=20)
    
    # Quick eval
    quick_parser = subparsers.add_parser('quick-eval', help='Quick evaluation on test set')
    quick_parser.add_argument('--checkpoint', default='runs/diffusion_20260222_195530/ckpt_ep100.pt')
    quick_parser.add_argument('--episodes', type=int, default=10)
    
    # Visualized evaluation (NEW - Full VLM process visualization)
    viz_parser = subparsers.add_parser('evaluate-visualized', help='Paired evaluation with FULL visualization and tracking')
    viz_parser.add_argument('--checkpoint', default='runs/diffusion_20260222_195530/ckpt_ep100.pt')
    viz_parser.add_argument('--episodes', type=int, default=5)
    viz_parser.add_argument('--n-candidates', type=int, default=8)
    viz_parser.add_argument('--max-attempts', type=int, default=20)
    viz_parser.add_argument('--seed', type=int, default=100)
    viz_parser.add_argument('--output-dir', default='outputs/vlm_visualized')
    
    # List commands
    subparsers.add_parser('list', help='List all available commands')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        print("\n📋 Available Commands:\n")
        print("  evaluate-paired      - Run paired evaluation (baseline vs VLM-guided)")
        print("  evaluate-visualized  - Paired evaluation with FULL visualization & attempt tracking ⭐ NEW")
        print("  generate-videos      - Generate arc 15-19 policy videos")
        print("  verify-arc           - Verify arc diversity in policy samples")
        print("  debug-vlm            - Debug VLM selection process")
        print("  quick-eval           - Quick evaluation on test set")
        print("\nRun 'python cli.py <command> --help' for command-specific options\n")
        return
    
    # Dispatch to appropriate script
    if args.command == 'evaluate-paired':
        run_paired_evaluation(args)
    elif args.command == 'evaluate-visualized':
        run_visualized_evaluation(args)
    elif args.command == 'generate-videos':
        run_generate_videos(args)
    elif args.command == 'verify-arc':
        run_verify_arc(args)
    elif args.command == 'debug-vlm':
        run_debug_vlm(args)
    elif args.command == 'quick-eval':
        run_quick_eval(args)


def run_paired_evaluation(args):
    """Run paired evaluation"""
    print(f"\n🚀 Running paired evaluation (version: {args.version})...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Candidates: {args.n_candidates}")
    
    script_map = {
        'proper': 'evaluation/paired_rollouts_proper.py',
        'v2': 'evaluation/paired_replanning_rollouts_v2.py',
        'v1': 'evaluation/paired_replanning_rollouts.py',
        'iterative': 'evaluation/paired_iterative_vlm.py'
    }
    
    script = script_map[args.version]
    cmd = [
        sys.executable, script,
        '--checkpoint', args.checkpoint,
        '--episodes', str(args.episodes),
        '--n-candidates', str(args.n_candidates),
        '--seed', str(args.seed)
    ]
    
    import subprocess
    subprocess.run(cmd)


def run_generate_videos(args):
    """Generate arc 15-19 videos"""
    print(f"\n🎥 Generating {args.n_videos} arc 15-19 videos...")
    
    cmd = [
        sys.executable, 'tools/generate_arc15_policy_videos.py',
        '--checkpoint', args.checkpoint,
        '--n-videos', str(args.n_videos),
        '--output-dir', args.output_dir
    ]
    
    import subprocess
    subprocess.run(cmd)


def run_verify_arc(args):
    """Verify arc diversity"""
    print(f"\n🔍 Verifying arc diversity ({args.samples} samples)...")
    
    script = 'verification/verify_arc_measurement.py' if args.with_videos else 'verification/verify_arc_diversity.py'
    
    cmd = [sys.executable, script, '--checkpoint', args.checkpoint]
    if not args.with_videos:
        cmd.extend(['--samples', str(args.samples)])
    
    import subprocess
    subprocess.run(cmd)


def run_debug_vlm(args):
    """Debug VLM selection"""
    print(f"\n🐛 Debugging VLM selection (episode {args.episode})...")
    
    cmd = [
        sys.executable, 'tools/debug_vlm_selection.py',
        '--checkpoint', args.checkpoint,
        '--episode-seed', str(args.episode),
        '--n-candidates', str(args.n_candidates)
    ]
    
    import subprocess
    subprocess.run(cmd)


def run_visualized_evaluation(args):
    """Run paired evaluation with FULL visualization"""
    print(f"\n🎬 Running VISUALIZED paired evaluation...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Candidates: {args.n_candidates}")
    print(f"   Max attempts for arc 15-19: {args.max_attempts}")
    print(f"   Output directory: {args.output_dir}")
    print(f"\n   This will show:")
    print(f"   ✓ How many attempts to generate arc 15-19")
    print(f"   ✓ What frames are sent to VLM")
    print(f"   ✓ VLM's selection decision")
    print(f"   ✓ Baseline waiting for VLM input")
    print(f"   ✓ Visualizations saved to {args.output_dir}/\n")
    
    cmd = [
        sys.executable, 'evaluation/paired_rollouts_visualized.py',
        '--checkpoint', args.checkpoint,
        '--episodes', str(args.episodes),
        '--n-candidates', str(args.n_candidates),
        '--max-attempts', str(args.max_attempts),
        '--seed', str(args.seed),
        '--output-dir', args.output_dir
    ]
    
    import subprocess
    subprocess.run(cmd)


def run_quick_eval(args):
    """Quick evaluation"""
    print(f"\n⚡ Running quick evaluation ({args.episodes} episodes)...")
    
    cmd = [
        sys.executable, 'evaluation/quick_eval.py',
        '--checkpoint', args.checkpoint,
        '--episodes', str(args.episodes)
    ]
    
    import subprocess
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
