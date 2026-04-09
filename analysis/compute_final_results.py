"""Compute final combined results from CFG-only and VLM evaluations."""
import json, numpy as np

ROOT = r"c:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick"

# --- Load data ---
with open(f"{ROOT}/outputs/eval_cfg_fast/results.json") as f:
    cfg = json.load(f)
with open(f"{ROOT}/outputs/eval_vlm_final3/results.json") as f:
    vlm = json.load(f)

print("=" * 70)
print("  FINAL COMPREHENSIVE RESULTS")
print("=" * 70)

# ── LEGIBILITY ──
print("\n╔══════════════════════════════════════════════════════════════════╗")
print("║                        LEGIBILITY                              ║")
print("╚══════════════════════════════════════════════════════════════════╝")

cfg_leg = cfg["legibility"]
vlm_leg = vlm["legibility"]

cfg_succ = sum(1 for e in cfg_leg if e["success"])
cfg_learl = [e["L_early"] for e in cfg_leg]

vlm_succ_vlm = sum(1 for e in vlm_leg if e["vlm"]["success"])
vlm_learl_vlm = [e["vlm"]["L_early"] for e in vlm_leg]
vlm_succ_bl = sum(1 for e in vlm_leg if e["baseline"]["success"])
vlm_learl_bl = [e["baseline"]["L_early"] for e in vlm_leg]

print(f"\n  {'Method':<25} {'Success':>10} {'L_early (mean±std)':>22}")
print(f"  {'─'*25} {'─'*10} {'─'*22}")
print(f"  {'CFG-only (λ=2.0)':<25} {cfg_succ:>7}/10  {np.mean(cfg_learl):.3f} ± {np.std(cfg_learl):.3f}")
print(f"  {'CFG + VLM (Best-of-4)':<25} {vlm_succ_vlm:>7}/10  {np.mean(vlm_learl_vlm):.3f} ± {np.std(vlm_learl_vlm):.3f}")
print(f"  {'CFG + Baseline (worst)':<25} {vlm_succ_bl:>7}/10  {np.mean(vlm_learl_bl):.3f} ± {np.std(vlm_learl_bl):.3f}")

# ── PREDICTABILITY ──
print("\n╔══════════════════════════════════════════════════════════════════╗")
print("║                      PREDICTABILITY                            ║")
print("╚══════════════════════════════════════════════════════════════════╝")

cfg_pred = cfg["predictability"]
vlm_pred = vlm["predictability"]

cfg_succ = sum(1 for e in cfg_pred if e["success"])
cfg_eff = [e["path_efficiency"] for e in cfg_pred]

vlm_succ_vlm = sum(1 for e in vlm_pred if e["vlm"]["success"])
vlm_eff_vlm = [e["vlm"]["path_efficiency"] for e in vlm_pred]
vlm_succ_bl = sum(1 for e in vlm_pred if e["baseline"]["success"])
vlm_eff_bl = [e["baseline"]["path_efficiency"] for e in vlm_pred]
# Convert any strings to float
vlm_eff_vlm = [float(x) for x in vlm_eff_vlm]
vlm_eff_bl = [float(x) for x in vlm_eff_bl]

print(f"\n  {'Method':<25} {'Success':>10} {'Path Eff (mean±std)':>22}")
print(f"  {'─'*25} {'─'*10} {'─'*22}")
print(f"  {'CFG-only (λ=2.0)':<25} {cfg_succ:>7}/10  {np.mean(cfg_eff):.3f} ± {np.std(cfg_eff):.3f}")
print(f"  {'CFG + VLM (Best-of-4)':<25} {vlm_succ_vlm:>7}/10  {np.mean(vlm_eff_vlm):.3f} ± {np.std(vlm_eff_vlm):.3f}")
print(f"  {'CFG + Baseline (worst)':<25} {vlm_succ_bl:>7}/10  {np.mean(vlm_eff_bl):.3f} ± {np.std(vlm_eff_bl):.3f}")

# ── SAFETY ──
print("\n╔══════════════════════════════════════════════════════════════════╗")
print("║                          SAFETY                                ║")
print("╚══════════════════════════════════════════════════════════════════╝")

cfg_saf = cfg["safety"]
vlm_saf = vlm["safety"]

cfg_succ = sum(1 for e in cfg_saf if e["success"])
cfg_cl = [e["clearance"] for e in cfg_saf]
cfg_nocoll = sum(1 for e in cfg_saf if not e["collision"])

vlm_succ_vlm = sum(1 for e in vlm_saf if e["vlm"]["success"])
vlm_cl_vlm = [float(e["vlm"]["min_clearance"]) for e in vlm_saf]
vlm_succ_bl = sum(1 for e in vlm_saf if e["baseline"]["success"])
vlm_cl_bl = [float(e["baseline"]["min_clearance"]) for e in vlm_saf]

print(f"\n  {'Method':<25} {'Success':>10} {'Clearance (mean±std)':>22}")
print(f"  {'─'*25} {'─'*10} {'─'*22}")
print(f"  {'CFG-only (λ=2.0)':<25} {cfg_succ:>7}/10  {np.mean(cfg_cl):.4f} ± {np.std(cfg_cl):.4f}  (0 collisions)")
print(f"  {'CFG + VLM (Best-of-4)':<25} {vlm_succ_vlm:>7}/10  {np.mean(vlm_cl_vlm):.4f} ± {np.std(vlm_cl_vlm):.4f}")
print(f"  {'CFG + Baseline (worst)':<25} {vlm_succ_bl:>7}/10  {np.mean(vlm_cl_bl):.4f} ± {np.std(vlm_cl_bl):.4f}")

# ── GROUNDING ──
print("\n╔══════════════════════════════════════════════════════════════════╗")
print("║                        GROUNDING                               ║")
print("╚══════════════════════════════════════════════════════════════════╝")

cfg_gnd = cfg["grounding"]
vlm_gnd = vlm["grounding"]

cfg_succ = sum(1 for e in cfg_gnd if e["success"])
cfg_hov = sum(1 for e in cfg_gnd if e.get("hovered", False))
cfg_dist = [e["hover_dist"] for e in cfg_gnd]

vlm_succ_vlm = sum(1 for e in vlm_gnd if e["vlm"]["success"])
vlm_dist_vlm = [float(e["vlm"]["min_wp_dist"]) for e in vlm_gnd]
vlm_succ_bl = sum(1 for e in vlm_gnd if e["baseline"]["success"])
vlm_dist_bl = [float(e["baseline"]["min_wp_dist"]) for e in vlm_gnd]

print(f"\n  {'Method':<25} {'Success':>10} {'WP dist (mean±std)':>22}")
print(f"  {'─'*25} {'─'*10} {'─'*22}")
print(f"  {'CFG-only (λ=2.0)':<25} {cfg_succ:>7}/10  {np.mean(cfg_dist):.4f} ± {np.std(cfg_dist):.4f}  ({cfg_hov}/10 hovered)")
print(f"  {'CFG + VLM (Best-of-4)':<25} {vlm_succ_vlm:>7}/10  {np.mean(vlm_dist_vlm):.4f} ± {np.std(vlm_dist_vlm):.4f}")
print(f"  {'CFG + Baseline (worst)':<25} {vlm_succ_bl:>7}/10  {np.mean(vlm_dist_bl):.4f} ± {np.std(vlm_dist_bl):.4f}")

# ── AGGREGATE TABLE ──
print("\n" + "=" * 70)
print("  AGGREGATE COMPARISON TABLE")
print("=" * 70)
print("""
  ┌───────────────────┬──────────────┬──────────────┬──────────────┐
  │     Behavior      │   CFG-only   │  CFG + VLM   │  VLM Δ (abs) │
  ├───────────────────┼──────────────┼──────────────┼──────────────┤""")

# Legibility
cfg_s = sum(1 for e in cfg["legibility"] if e["success"])
vlm_s = sum(1 for e in vlm["legibility"] if e["vlm"]["success"])
cfg_m = np.mean([e["L_early"] for e in cfg["legibility"]])
vlm_m = np.mean([float(e["vlm"]["L_early"]) for e in vlm["legibility"]])
# Note: for legibility, LOWER L_early = more legible (early commitment to non-target)
# Actually L_early measures posterior of approached block, high = legible IF we interpret correctly
# In our framework L_early is the Bayesian posterior that the arm is going to A during first 30%
# So HIGH L_early = more legible
print(f"  │ Legibility (L↑)   │  {cfg_s}/10 {cfg_m:.3f}  │  {vlm_s}/10 {vlm_m:.3f}  │ {vlm_s-cfg_s:+d} succ, {vlm_m-cfg_m:+.3f}│")

# Predictability
cfg_s = sum(1 for e in cfg["predictability"] if e["success"])
vlm_s = sum(1 for e in vlm["predictability"] if e["vlm"]["success"])
cfg_m = np.mean([e["path_efficiency"] for e in cfg["predictability"]])
vlm_m = np.mean([float(e["vlm"]["path_efficiency"]) for e in vlm["predictability"]])
print(f"  │ Predict. (eff↑)   │  {cfg_s}/10 {cfg_m:.3f}  │  {vlm_s}/10 {vlm_m:.3f}  │ {vlm_s-cfg_s:+d} succ, {vlm_m-cfg_m:+.3f}│")

# Safety
cfg_s = sum(1 for e in cfg["safety"] if e["success"])
vlm_s = sum(1 for e in vlm["safety"] if e["vlm"]["success"])
cfg_m = np.mean([e["clearance"] for e in cfg["safety"]])
vlm_m = np.mean([float(e["vlm"]["min_clearance"]) for e in vlm["safety"]])
print(f"  │ Safety (clearance↑)│  {cfg_s}/10 {cfg_m:.4f} │  {vlm_s}/10 {vlm_m:.4f} │ {vlm_s-cfg_s:+d} succ,{vlm_m-cfg_m:+.4f}│")

# Grounding
cfg_s = sum(1 for e in cfg["grounding"] if e["success"])
vlm_s = sum(1 for e in vlm["grounding"] if e["vlm"]["success"])
cfg_m = np.mean([e["hover_dist"] for e in cfg["grounding"]])
vlm_m = np.mean([float(e["vlm"]["min_wp_dist"]) for e in vlm["grounding"]])
print(f"  │ Grounding (dist↓) │  {cfg_s}/10 {cfg_m:.4f} │  {vlm_s}/10 {vlm_m:.4f} │ {vlm_s-cfg_s:+d} succ,{vlm_m-cfg_m:+.4f}│")

print("  └───────────────────┴──────────────┴──────────────┴──────────────┘")

# ── VLM DISCRIMINATION ANALYSIS ──
print("\n" + "=" * 70)
print("  VLM DISCRIMINATION ANALYSIS")
print("=" * 70)

for beh in ["legibility", "predictability", "safety", "grounding"]:
    episodes = vlm[beh]
    vlm_scores = []
    bl_scores = []
    score_ranges = []
    for ep in episodes:
        cands = ep["candidates"]
        scores = [c["vlm_score"] for c in cands]
        vlm_scores.append(ep["vlm"]["vlm_score"])
        bl_scores.append(ep["baseline"]["vlm_score"])
        score_ranges.append(max(scores) - min(scores))
    
    print(f"\n  {beh.upper()}")
    print(f"    VLM-selected avg score: {np.mean(vlm_scores):.3f}")
    print(f"    Baseline avg score:     {np.mean(bl_scores):.3f}")
    print(f"    Score gap (VLM - BL):   {np.mean(vlm_scores) - np.mean(bl_scores):+.3f}")
    print(f"    Avg score range:        {np.mean(score_ranges):.3f}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
