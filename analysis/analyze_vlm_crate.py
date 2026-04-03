#!/usr/bin/env python3
"""Analyze why the VLM individual evaluation has 84% C rate."""
import json
from collections import defaultdict

results = []
with open('outputs/combined_prefix_gemini_eval_k6/results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

print(f'Total records: {len(results)}')
c_count = sum(1 for r in results if r['choice'] == 'C')
print(f'C rate: {c_count}/{len(results)} = {c_count/len(results)*100:.1f}%')
print()

# Breakdown by traj_type
by_type = defaultdict(list)
for r in results:
    by_type[r['traj_type']].append(r)

print('By trajectory type:')
for tt, rows in sorted(by_type.items()):
    c_rate = sum(1 for r in rows if r['choice'] == 'C') / len(rows) * 100
    correct = sum(1 for r in rows if r['choice'] == r['goal_gt']) / len(rows) * 100
    avg_pA = sum(r['pA'] for r in rows) / len(rows)
    print(f'  {tt}: n={len(rows)}, C-rate={c_rate:.0f}%, correct={correct:.0f}%, avg_pA={avg_pA:.3f}')

print()

# pA distribution by traj_type at each timepoint
print('pA distribution at t=5s by traj_type:')
for tt, rows in sorted(by_type.items()):
    late = [r for r in rows if r['t_sec'] == 5]
    if late:
        vals = sorted([r['pA'] for r in late])
        choices = [r['choice'] for r in late]
        n_correct = sum(1 for r in late if r['choice'] == r['goal_gt'])
        print(f'  {tt} (n={len(late)}): pA vals={[round(v,2) for v in vals]}')
        print(f'    choices={choices}, n_correct={n_correct}')

print()

# The pA histogram
print('Histogram of pA across ALL records:')
buckets = defaultdict(int)
for r in results:
    bucket = round(r['pA'] * 20) / 20  # bin to nearest 0.05
    buckets[bucket] += 1
for k in sorted(buckets):
    bar = '#' * (buckets[k] // 2)
    print(f'  pA={k:.2f}: {buckets[k]:3d} {bar}')

print()

# Sample cues when C
c_rows = [r for r in results if r['choice'] == 'C']
print('Sample cues on C decisions (first 10):')
for r in c_rows[:10]:
    print(f'  {r["video_id"]} t={r["t_sec"]}s pA={r["pA"]:.2f} pB={r["pB"]:.2f} cue: {r["cue"][:70]}')

print()

# Non-C decisions
non_c = [r for r in results if r['choice'] != 'C']
print(f'Non-C decisions: {len(non_c)}')
for r in non_c[:10]:
    correct_str = 'CORRECT' if r['choice'] == r['goal_gt'] else 'WRONG'
    print(f'  {r["video_id"]} t={r["t_sec"]}s pA={r["pA"]:.2f} pB={r["pB"]:.2f} choice={r["choice"]} gt={r["goal_gt"]} [{correct_str}]')
    print(f'    cue: {r["cue"][:70]}')
