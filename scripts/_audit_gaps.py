#!/usr/bin/env python3
"""Critical audit of all gap resolution claims."""
import json
import numpy as np
from scipy import stats

with open("outputs/gap_resolution/gap_resolution_results.json") as f:
    data = json.load(f)

print("=" * 70)
print("CRITICAL AUDIT OF ALL CLAIMS")
print("=" * 70)

fwd = data["forward_hc_w10"]["episodes"]
rev5 = data["reverse_hc_w-5.0"]["episodes"]
rev10 = data["reverse_hc_w-10.0"]["episodes"]

# =============================================
# PROBLEM 1: L_early uses max(L0, L1)
# =============================================
print("\n--- PROBLEM 1: L_early cannot detect de-legibilization ---")
print("L_early = max(L_early_goal0, L_early_goal1)")
print("Always >= 0.5 because one goal always looks intended.")
print("Reverse steering CANNOT make trajectory ambiguous by this metric.\n")

fwd_l = [e["l_early_actual"] for e in fwd]
r5_l = [e["l_early_actual"] for e in rev5]
r10_l = [e["l_early_actual"] for e in rev10]
print(f"Forward w=+10 L_early: {np.mean(fwd_l):.4f}")
print(f"Reverse w=-5  L_early: {np.mean(r5_l):.4f}")
print(f"Reverse w=-10 L_early: {np.mean(r10_l):.4f}")
print("Reverse w=-10 L_early (0.925) > reverse w=-5 (0.915)")
print("VERDICT: L_early metric is INVALID for evaluating reverse steering.")

# =============================================
# PROBLEM 2: Reverse w=-10 MORE large arcs
# =============================================
print("\n--- PROBLEM 2: Reverse w=-10 has MORE large arcs than forward ---")
fwd_big = sum(1 for e in fwd if e["arc"]["max_arc"] >= 0.15)
r5_big = sum(1 for e in rev5 if e["arc"]["max_arc"] >= 0.15)
r10_big = sum(1 for e in rev10 if e["arc"]["max_arc"] >= 0.15)
print(f"Forward w=+10: {fwd_big}/20 large arcs")
print(f"Reverse w=-5:  {r5_big}/20 large arcs")
print(f"Reverse w=-10: {r10_big}/20 large arcs")
print("Reverse w=-10 = 3 large arcs > Forward w=+10 = 2 large arcs!")
print("VERDICT: CONTRADICTS the reverse-steering hypothesis at w=-10.")

# =============================================
# PROBLEM 3: Cherry-picked significance
# =============================================
print("\n--- PROBLEM 3: p=0.038 cherry-picks forward vs w=-5 only ---")
fwd_arcs = [e["arc"]["max_arc"] for e in fwd]
r5_arcs = [e["arc"]["max_arc"] for e in rev5]
r10_arcs = [e["arc"]["max_arc"] for e in rev10]

t1, p1 = stats.ttest_ind(fwd_arcs, r5_arcs)
t2, p2 = stats.ttest_ind(fwd_arcs, r10_arcs)
print(f"Forward vs Reverse w=-5:  t={t1:.3f}, p={p1:.4f}")
print(f"Forward vs Reverse w=-10: t={t2:.3f}, p={p2:.4f}")
print(f"Bonferroni threshold (2 tests): 0.025")
print(f"Forward vs w=-5 survives Bonferroni? {p1 < 0.025}")
print("VERDICT: One of two comparisons is not significant. Other barely passes.")

# =============================================
# PROBLEM 4: w=-5 = baseline, not below it
# =============================================
print("\n--- PROBLEM 4: Reverse w=-5 just returns to baseline ---")
baseline_arc = 0.0862  # from previous run
print(f"Baseline mean arc:     {baseline_arc:.4f}m")
print(f"Forward w=+10 mean:    {np.mean(fwd_arcs):.4f}m")
print(f"Reverse w=-5 mean:     {np.mean(r5_arcs):.4f}m")
print(f"Reverse w=-10 mean:    {np.mean(r10_arcs):.4f}m")
diff = np.mean(r5_arcs) - baseline_arc
print(f"Reverse w=-5 vs baseline diff: {diff*1000:+.1f}mm")
print("VERDICT: w=-5 cancels guidance. Does NOT reverse it.")
print("w=-10 mean arc (0.102) = forward w=+10 mean arc (0.102). No reversal.")

# =============================================
# PROBLEM 5: Effect size is tiny
# =============================================
print("\n--- PROBLEM 5: Effect sizes ---")
pooled_std = np.sqrt((np.std(fwd_arcs)**2 + np.std(r5_arcs)**2) / 2)
d = (np.mean(fwd_arcs) - np.mean(r5_arcs)) / pooled_std
diff_mm = (np.mean(fwd_arcs) - np.mean(r5_arcs)) * 1000
print(f"Arc difference (fwd - rev5): {diff_mm:.1f}mm")
print(f"Cohen's d: {d:.3f} (small=0.2, medium=0.5, large=0.8)")
print(f"VERDICT: d={d:.2f} is a {'small' if d < 0.5 else 'medium' if d < 0.8 else 'large'} effect.")
print(f"Absolute difference is {diff_mm:.0f}mm on a ~140mm inter-goal distance.")

# =============================================
# PROBLEM 6: Arc classification bins
# =============================================
print("\n--- PROBLEM 6: Arc classification is nearly uniform ---")
print("All conditions: 0% arc00-05 (straight). Nobody goes straight.")
print("The environment naturally produces moderate arcs (arc10-14).")
print("Baseline already has 95% arc10-14.")
print("VERDICT: Guidance does not CREATE arcs. Environment already has arcs.")
print("The arc classification distinguishes nothing meaningful.")

# =============================================
# PROBLEM 7: Visual VLM arcs might be noise
# =============================================
print("\n--- PROBLEM 7: Visual VLM 30% large arcs may be noise ---")
# Fisher's exact test: text VLM 1/20 arc15+, visual VLM 6/20 arc15+
table = np.array([[1, 19], [6, 14]])
oddsratio, fisher_p = stats.fisher_exact(table)
print(f"Text VLM:   1/20 large arcs (5%)")
print(f"Visual VLM: 6/20 large arcs (30%)")
print(f"Fisher's exact test: p={fisher_p:.4f}")
print(f"VERDICT: {'Significant' if fisher_p < 0.05 else 'NOT significant'} (p {'<' if fisher_p < 0.05 else '>'} 0.05)")

# Also: visual VLM failures
print(f"\nVisual VLM success: 90% vs Text VLM 100%")
print("2 failures in visual VLM — are the large arcs from failing episodes?")

# =============================================
# SUMMARY
# =============================================
print("\n" + "=" * 70)
print("SUMMARY OF AUDIT FINDINGS")
print("=" * 70)
print("""
1. REVERSE STEERING DOES NOT WORK AS CLAIMED:
   - w=-10 produces MORE large arcs than w=+10 (contradicts hypothesis)
   - w=-5 just returns to baseline (cancels guidance, doesn't reverse it)
   - L_early metric cannot detect de-legibilization (always max over goals)
   - The p=0.038 cherry-picks the one favorable comparison

2. ARC CLASSIFICATION IS TRIVIAL:
   - Environment naturally produces ~0.08m arcs (all arc10-14)
   - 0% straight trajectories in ANY condition
   - Guidance shifts arc magnitude by ~16mm — practically negligible

3. VISUAL VLM CLAIMS ARE WEAK:
   - "6x more large arcs" = 1 vs 6 absolute counts
   - Fisher's exact test needed to check if significant
   - Success rate drops (90% vs 100%) — larger arcs may be overshooting
   
4. THE FUNDAMENTAL PROBLEM REMAINS:
   - All trajectories already curve (baseline mean arc = 86mm)
   - Guidance adds ~16mm more curve on avg
   - This is a SMALL effect masked by environmental stochasticity
""")
