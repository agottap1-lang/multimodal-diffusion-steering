#!/usr/bin/env python3
"""
Offline test: what would happen if we applied two fixes to the existing 240 results?

Fix 1: Lower threshold from 0.60 -> 0.52
Fix 2: Hard argmax (threshold=0.50+epsilon) - assumes any asymmetry is signal
Fix 3 (aspirational): New arc-aware prompt (requires re-running API; estimated here
       by checking whether existing directional responses were correct).

This does NOT re-call the API. It re-processes the existing pA/pB values.
"""
import json
from collections import defaultdict

results = []
with open('outputs/combined_prefix_gemini_eval_k6/results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

def apply_threshold(results, threshold):
    """Re-apply threshold to existing pA/pB values."""
    recoded = []
    for r in results:
        pA, pB = r['pA'], r['pB']
        max_p = max(pA, pB)
        if max_p >= threshold:
            choice = 'A' if pA >= pB else 'B'
        else:
            choice = 'C'
        recoded.append({**r, 'choice': choice})
    return recoded

def summarize(results, label):
    total = len(results)
    c_count = sum(1 for r in results if r['choice'] == 'C')
    correct = sum(1 for r in results if r['choice'] != 'C' and r['choice'] == r['goal_gt'])
    wrong = sum(1 for r in results if r['choice'] != 'C' and r['choice'] != r['goal_gt'])
    decisive = total - c_count
    acc_on_decisive = correct / decisive if decisive > 0 else 0
    print(f"\n{label}")
    print(f"  C-rate:          {c_count}/{total} = {c_count/total*100:.1f}%")
    print(f"  Decisive:        {decisive}/{total} = {decisive/total*100:.1f}%")
    print(f"  Correct/decisive:{correct}/{decisive} = {acc_on_decisive*100:.1f}%  (correct = right prediction)")
    print(f"  Wrong/decisive:  {wrong}/{decisive} = {wrong/decisive*100:.1f}% (wrong prediction)")
    # By traj_type
    by_type = defaultdict(list)
    for r in results:
        by_type[r['traj_type']].append(r)
    for tt in ['deceptive', 'neutral', 'legible']:
        rows = by_type.get(tt, [])
        if not rows:
            continue
        c_r = sum(1 for r in rows if r['choice']=='C') / len(rows) * 100
        dec = [r for r in rows if r['choice'] != 'C']
        corr = sum(1 for r in dec if r['choice'] == r['goal_gt'])
        print(f"    {tt:12s}: C={c_r:.0f}%  decisive_acc={corr}/{len(dec)}")

original = results
thresh60 = apply_threshold(results, 0.60)
thresh52 = apply_threshold(results, 0.52)
argmax   = apply_threshold(results, 0.501)  # pure argmax

summarize(original, "ORIGINAL (threshold=0.60, old prompt)")
summarize(thresh60, "THRESHOLD=0.60 (same as original, crosscheck)")
summarize(thresh52, "THRESHOLD=0.52 (new default, old pA/pB values)")
summarize(argmax,   "ARGMAX (threshold=0.501, old pA/pB values)")

print("\n\n=== KEY INSIGHT ===")
# At t=5 how many have pA exactly 0.50 vs asymmetric?
t5 = [r for r in results if r['t_sec'] == 5]
exact_half = sum(1 for r in t5 if r['pA'] == 0.50)
asym = sum(1 for r in t5 if r['pA'] != 0.50)
asym_correct = sum(1 for r in t5 if r['pA'] != 0.50 and
                   ('A' if r['pA'] > r['pB'] else 'B') == r['goal_gt'])
print(f"At t=5s: {len(t5)} records, {exact_half} have pA=0.50 exactly, {asym} are asymmetric")
if asym > 0:
    print(f"  Of the {asym} asymmetric ones, argmax would give {asym_correct} correct ({asym_correct/asym*100:.0f}%)")
    print(f"  -> The DIRECTION of the existing pA/pB signal is {'useful' if asym_correct/asym > 0.5 else 'noisy/anti-correlated'}")

print()
# Check whether pA>0.5 correlates with gt=A across all records
all_asym = [r for r in results if r['pA'] != 0.50]
argmax_correct_all = sum(1 for r in all_asym if
                         ('A' if r['pA'] > r['pB'] else 'B') == r['goal_gt'])
if all_asym:
    print(f"All {len(all_asym)} asymmetric records: argmax gives {argmax_correct_all}/{len(all_asym)} = {argmax_correct_all/len(all_asym)*100:.0f}% correct")
    print(f"  (50% = chance, >50% = some signal, <50% = anti-correlated)")
