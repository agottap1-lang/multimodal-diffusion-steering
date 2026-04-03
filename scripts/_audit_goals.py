#!/usr/bin/env python3
"""Check goal flips and anomalous reverse episodes."""
import json
import numpy as np

with open("outputs/gap_resolution/gap_resolution_results.json") as f:
    data = json.load(f)

print("=== TRUE GOAL ATTRIBUTION ===")
for cond in ["forward_hc_w10", "reverse_hc_w-5.0", "reverse_hc_w-10.0"]:
    eps = data[cond]["episodes"]
    goals = [e["true_goal"] for e in eps]
    left_count = goals.count("left")
    right_count = goals.count("right")
    print(f"  {cond:25s}: left={left_count} right={right_count}")

print()
print("=== Does reverse flip which goal looks intended? ===")
fwd = data["forward_hc_w10"]["episodes"]
rev5 = data["reverse_hc_w-5.0"]["episodes"]
rev10 = data["reverse_hc_w-10.0"]["episodes"]

flips_5 = sum(1 for f, r in zip(fwd, rev5) if f["true_goal"] != r["true_goal"])
flips_10 = sum(1 for f, r in zip(fwd, rev10) if f["true_goal"] != r["true_goal"])
print(f"  Forward vs Reverse w=-5:  {flips_5}/20 goal flips")
print(f"  Forward vs Reverse w=-10: {flips_10}/20 goal flips")

print()
print("=== w=-10 episodes with arc >= 0.15m (anomalies) ===")
for i, e in enumerate(rev10):
    if e["arc"]["max_arc"] >= 0.15:
        fwd_arc = fwd[i]["arc"]["max_arc"]
        print(f"  ep{i}: seed=({e['env_seed']},{e['sample_seed']}) "
              f"fwd_arc={fwd_arc:.4f}m rev10_arc={e['arc']['max_arc']:.4f}m "
              f"fwd_goal={fwd[i]['true_goal']} rev_goal={e['true_goal']} "
              f"rev_success={e['success']}")

print()
print("=== Are failed reverse episodes the large-arc ones? ===")
for i, e in enumerate(rev10):
    if not e["success"]:
        print(f"  FAILED ep{i}: arc={e['arc']['max_arc']:.4f}m "
              f"goal={e['true_goal']} L={e['l_early_actual']:.3f}")
