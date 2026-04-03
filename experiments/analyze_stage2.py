#!/usr/bin/env python3
"""Re-analyze Stage 2 results with correct best-scale selection:
   first require ≥95% success, THEN pick highest L_early."""

import json
import numpy as np
from math import erfc
from pathlib import Path

def paired_ttest(a, b):
    a, b = np.array(a), np.array(b)
    diff = b - a
    n = len(diff)
    m = float(diff.mean())
    s = float(diff.std(ddof=1))
    se = s / np.sqrt(n) if n > 1 else 0.0
    t = m / se if se > 1e-12 else 0.0
    p = erfc(abs(t) / np.sqrt(2)) if n >= 2 else 1.0
    ci = 1.96 * se
    return dict(mean_diff=round(m,5), std_diff=round(s,5), t_stat=round(t,3),
                p_value=round(float(p),5), ci_95=[round(m-ci,5), round(m+ci,5)],
                n=n, significant=float(p) < 0.05)

data = json.loads(Path('outputs/rigorous_eval/stage2_results.json').read_text())

bl = data['baseline']
bl_l = [r['l_early_actual'] for r in bl['episodes']]
bl_s = [r['success'] for r in bl['episodes']]

print("="*72)
print("  STAGE 2 CORRECTED ANALYSIS")
print("  Selection criterion: max L_early WHERE success ≥ 95%")
print("="*72)

# Full sweep table
print(f"\n  {'Condition':<25} {'Success':>8} {'L_early':>9} {'± std':>8}")
print(f"  {'-'*52}")
print(f"  {'Baseline (w=0)':<25} {bl['success_rate']:>7.0%} {bl['l_early_mean']:>9.4f} {bl['l_early_std']:>8.4f}")

for label, key in [("Hand-crafted", "handcrafted"), ("VLM-guided", "vlm_guided")]:
    for wk in sorted(data[key].keys()):
        r = data[key][wk]
        star = ""
        if r['success_rate'] >= 0.95:
            star = " *"
        print(f"  {label+' w='+str(r['guidance_scale']):<25} "
              f"{r['success_rate']:>7.0%} {r['l_early_mean']:>9.4f} {r['l_early_std']:>8.4f}{star}")

# Correct best-scale selection
def pick_best(results_dict, min_success=0.95):
    viable = {k: v for k, v in results_dict.items() if v['success_rate'] >= min_success}
    if not viable:
        print(f"  WARNING: No scale meets ≥{min_success:.0%} success! Using highest success.")
        return max(results_dict.values(), key=lambda v: v['success_rate'])
    return max(viable.values(), key=lambda v: v['l_early_mean'])

best_hc = pick_best(data['handcrafted'])
best_vlm = pick_best(data['vlm_guided'])

print(f"\n  Best Hand-crafted: w={best_hc['guidance_scale']} "
      f"(success={best_hc['success_rate']:.0%}, L_early={best_hc['l_early_mean']:.4f})")
print(f"  Best VLM-guided:  w={best_vlm['guidance_scale']} "
      f"(success={best_vlm['success_rate']:.0%}, L_early={best_vlm['l_early_mean']:.4f})")

hc_l = [r['l_early_actual'] for r in best_hc['episodes']]
vlm_l = [r['l_early_actual'] for r in best_vlm['episodes']]

# Paired t-tests
t_vlm_bl = paired_ttest(bl_l, vlm_l)
t_hc_bl = paired_ttest(bl_l, hc_l)
t_vlm_hc = paired_ttest(hc_l, vlm_l)

print(f"\n  Paired t-tests (N={len(bl_l)} paired seeds):")
print(f"    VLM vs Baseline:     Δ={t_vlm_bl['mean_diff']:+.4f}  p={t_vlm_bl['p_value']:.5f}  "
      f"CI95={t_vlm_bl['ci_95']}  {'SIG' if t_vlm_bl['significant'] else 'n.s.'}")
print(f"    HC vs Baseline:      Δ={t_hc_bl['mean_diff']:+.4f}  p={t_hc_bl['p_value']:.5f}  "
      f"CI95={t_hc_bl['ci_95']}  {'SIG' if t_hc_bl['significant'] else 'n.s.'}")
print(f"    VLM vs HC:           Δ={t_vlm_hc['mean_diff']:+.4f}  p={t_vlm_hc['p_value']:.5f}  "
      f"CI95={t_vlm_hc['ci_95']}  {'SIG' if t_vlm_hc['significant'] else 'n.s.'}")

# Per-episode paired comparison
print(f"\n  Per-episode paired comparison (L_early):")
print(f"  {'EP':>4} {'Seed':>12} {'Baseline':>9} {'HC best':>9} {'VLM best':>9} {'VLM-BL':>8} {'VLM-HC':>8}")
print(f"  {'-'*68}")
for i in range(len(bl_l)):
    ep_bl = bl['episodes'][i]
    ep_hc = best_hc['episodes'][i]
    ep_vlm = best_vlm['episodes'][i]
    print(f"  {i+1:>4} ({ep_bl['env_seed']:>4},{ep_bl['sample_seed']:>5}) "
          f"{bl_l[i]:>9.4f} {hc_l[i]:>9.4f} {vlm_l[i]:>9.4f} "
          f"{vlm_l[i]-bl_l[i]:>+8.4f} {vlm_l[i]-hc_l[i]:>+8.4f}")

vlm_wins_bl = sum(1 for a,b in zip(bl_l, vlm_l) if b > a)
vlm_wins_hc = sum(1 for a,b in zip(hc_l, vlm_l) if b > a)
print(f"\n  VLM wins vs Baseline: {vlm_wins_bl}/{len(bl_l)}")
print(f"  VLM wins vs HC:      {vlm_wins_hc}/{len(hc_l)}")

# Verdict
print(f"\n{'='*72}")
print("  CORRECTED STAGE 2 VERDICT")
print(f"{'='*72}")
print(f"  H_S2(a) VLM success ≥ 95%     : {'CONFIRMED' if best_vlm['success_rate']>=0.95 else 'REJECTED'} ({best_vlm['success_rate']:.0%})")
print(f"  H_S2(b) VLM > baseline (p<.05): {'CONFIRMED' if t_vlm_bl['significant'] else 'REJECTED'} (Δ={t_vlm_bl['mean_diff']:+.4f}, p={t_vlm_bl['p_value']:.5f})")
print(f"  H_S2(c) VLM ≈ HC (|Δ|<0.02)  : {'CONFIRMED' if abs(t_vlm_hc['mean_diff'])<0.02 else 'REJECTED'} (Δ={t_vlm_hc['mean_diff']:+.4f})")
w_diff = best_vlm['guidance_scale'] != best_hc['guidance_scale']
print(f"  H_S2(d) Optimal w* differs     : {'CONFIRMED' if w_diff else 'REJECTED'} (VLM:{best_vlm['guidance_scale']}, HC:{best_hc['guidance_scale']})")
