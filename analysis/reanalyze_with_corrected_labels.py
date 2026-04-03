#!/usr/bin/env python3
"""
Re-analyze the existing 240 results but with corrected goal_gt labels.

Previously all gt labels were inverted in the manifest, so "correct"
meant wrong and vice versa. This script verifies the correction.
"""
import json
from collections import defaultdict

# Load the CORRECTED manifest
manifest = {}
with open("data/manifest_combined_cfg00.jsonl") as f:
    for line in f:
        e = json.loads(line)
        manifest[e["video_id"]] = e["goal_gt"]

# Load the existing 240 results (pA/pB/choice from old prompt)
results = []
with open("outputs/combined_prefix_gemini_eval_k6/results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        # Replace goal_gt with the corrected label
        vid_id = r["video_id"]
        corrected_gt = manifest.get(vid_id, r["goal_gt"])
        results.append({**r, "goal_gt": corrected_gt})

total = len(results)
print(f"Total: {total}")
c_count = sum(1 for r in results if r["choice"] == "C")
print(f"C rate: {c_count/total*100:.1f}% (same as before - no change here)")
print()

# Now recompute accuracy with corrected labels
decisive = [r for r in results if r["choice"] != "C"]
correct = [r for r in decisive if r["choice"] == r["goal_gt"]]
print(f"Decisive: {len(decisive)}/{total}")
print(f"Correct/decisive: {len(correct)}/{len(decisive)} = {len(correct)/len(decisive)*100:.1f}%")
print()

# By traj_type
by_type = defaultdict(list)
for r in results:
    by_type[r["traj_type"]].append(r)

print("By trajectory type (corrected gt):")
for tt in ["legible", "deceptive", "neutral"]:
    rows = by_type.get(tt, [])
    if not rows:
        continue
    c_rate = sum(1 for r in rows if r["choice"] == "C") / len(rows) * 100
    dec_rows = [r for r in rows if r["choice"] != "C"]
    corr = sum(1 for r in dec_rows if r["choice"] == r["goal_gt"])
    print(f"  {tt:12s}: n={len(rows)}, C={c_rate:.0f}%, "
          f"decisive_acc={corr}/{len(dec_rows)} = {corr/len(dec_rows)*100:.0f}% "
          f"(was listed as WRONG before)")

print()
# Argmax analysis with corrected gt
all_asym = [r for r in results if r["pA"] != 0.50]
argmax_correct = sum(1 for r in all_asym if
                     ("A" if r["pA"] > r["pB"] else "B") == r["goal_gt"])
if all_asym:
    print(f"Argmax accuracy on all asymmetric records ({len(all_asym)}): "
          f"{argmax_correct}/{len(all_asym)} = {argmax_correct/len(all_asym)*100:.0f}%")
    print("(should now be ~100% since we just inverted the labels)")
