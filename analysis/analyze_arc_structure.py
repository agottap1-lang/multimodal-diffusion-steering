"""
Analyze demo structure to understand arcun 15-19 characteristics
"""
import numpy as np
import json

# Load demos
data = np.load('data/demos/demos.npz', allow_pickle=True)
print(f"Available keys: {list(data.keys())}\n")

actions = data['actions']  # (N, T, 5): dx, dy, dz, dyaw, grip

# Try to load metadata if exists
metadata = data.get('metadata', None)
if metadata is not None:
    if isinstance(metadata, np.ndarray) and metadata.dtype == object:
        metadata = metadata.item()
    print(f"Metadata: {metadata}\n")

print("="*80)
print("DEMO ARC STRUCTURE ANALYSIS")
print("="*80)

# Analyze first 30 demos (covers arc 00, 05, 10, 14, 15, 19)
for i in range(min(30, len(actions))):
    traj = actions[i]
    
    # Compute cumulative Y-displacement
    dy_cumsum = np.cumsum(traj[:, 1])  # Cumulative lateral displacement
    max_arc = np.max(np.abs(dy_cumsum))
    
    # Classify arc level
    if max_arc < 0.05:
        arc_class = "00-05 (straight)"
    elif max_arc < 0.15:
        arc_class = "10-14 (moderate)"
    else:
        arc_class = "15-19 (large sweep)"
    
    print(f"Demo {i:03d}: Max arc={max_arc:.4f}m, Class={arc_class}")

print("\n" + "="*80)
print("ARC CLASSIFICATION THRESHOLDS")
print("="*80)
print("Arc 00-05: max_arc < 0.05m  (straight approach)")
print("Arc 10-14: 0.05m ≤ max_arc < 0.15m  (moderate curve)")
print("Arc 15-19: max_arc ≥ 0.15m  (large lateral sweep)")
print("\nTarget for steering: INCREASE max_arc to ≥ 0.15m")
