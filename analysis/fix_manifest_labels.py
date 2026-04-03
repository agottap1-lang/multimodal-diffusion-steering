#!/usr/bin/env python3
"""Fix inverted goal_gt labels in manifest_combined_cfg00.jsonl.

Root cause: The naming convention in collect_demos_combined.py is:
  leg_left / dec_left / neu_left  →  target="left"  →  picks LEFT block = Goal A
  leg_right / dec_right / neu_right → target="right" → picks RIGHT block = Goal B

But the manifest was written with goal_gt A and B swapped.  This script
inverts goal_gt for every entry (A→B, B→A) to fix the bug.
"""
import json, shutil, pathlib

src = pathlib.Path("data/manifest_combined_cfg00.jsonl")
bak = pathlib.Path("data/manifest_combined_cfg00.jsonl.bak")
shutil.copy(src, bak)

entries = []
with open(src) as f:
    for line in f:
        e = json.loads(line)
        # Swap the goal label
        e["goal_gt"] = "A" if e["goal_gt"] == "B" else "B"
        entries.append(e)

with open(src, "w") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"Fixed {len(entries)} entries. Backup saved to {bak}")

# Verify
with open(src) as f:
    rows = [json.loads(l) for l in f]
print("\nSample after fix:")
for r in rows[:6]:
    vid = r["video_id"]
    gt  = r["goal_gt"]
    gA  = r["goal_A"]
    print(f"  {vid:32s}  gt={gt}  goalA={gA}")
