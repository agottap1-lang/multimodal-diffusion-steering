"""
Analyze the trajectory directly from the generated video file.
Extract frames and track the trajectory visually.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def extract_trajectory_from_video(video_path: str):
    """Extract frames from video and analyze trajectory"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*80}")
    print(f"VIDEO ANALYSIS: {Path(video_path).name}")
    print(f"{'='*80}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.2f}s")
    
    # Extract sample frames
    sample_frames = []
    sample_indices = np.linspace(0, frame_count-1, min(20, frame_count), dtype=int)
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sample_frames.append((idx, frame))
    
    cap.release()
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    rows = 4
    cols = 5
    for i, (idx, frame) in enumerate(sample_frames):
        if i >= rows * cols:
            break
        ax = plt.subplot(rows, cols, i + 1)
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.set_title(f'Frame {idx}\n({idx/fps:.2f}s)', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Video Frame Samples: {Path(video_path).name}', fontsize=14)
    plt.tight_layout()
    
    output_path = Path("runs/arc_measurement_analysis") / f"video_frames_{Path(video_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📸 Frame samples saved: {output_path}")
    plt.close()
    
    print(f"\n💬 Please describe what you observe:")
    print(f"   - Does the robot arm sweep in a large arc?")
    print(f"   - Does it take a direct/straight path?")
    print(f"   - How pronounced is the lateral deviation?")


def compare_both_videos():
    """Compare both generated videos"""
    video1 = "runs/policy_arc15_verification/policy_arc0.9835m_left_seed1001_temp0.81.mp4"
    video2 = "runs/policy_arc15_verification/policy_arc0.7945m_left_seed1002_temp1.12.mp4"
    
    for video in [video1, video2]:
        if Path(video).exists():
            extract_trajectory_from_video(video)
            print("\n")
        else:
            print(f"Video not found: {video}")
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("\nBased on perpendicular distance measurements:")
    print("  Seed 1001: 0.024m (gentle curve)")
    print("  Seed 1002: 0.089m (moderate curve)")
    print("\nBased on cumsum measurements:")
    print("  Seed 1001: 0.98m → 1.35m (high oscillation)")
    print("  Seed 1002: 0.79m → 2.30m (very high oscillation)")
    print("\nDemo arc 15-19: 0.23-0.28m perpendicular distance")
    print("\n❓ Question: What do you see in the videos?")
    print("   If the videos show LARGE sweeping arcs visually,")
    print("   then the perpendicular measurement might be wrong.")
    print("   High cumsum with low perpendicular suggests wobbling/oscillating trajectory.")


if __name__ == "__main__":
    compare_both_videos()
