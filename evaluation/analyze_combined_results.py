"""
analyze_combined_results.py – Deep analysis of eval_combined_prefix.py outputs.
Run: python evaluation/analyze_combined_results.py
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PREFIX_DIR  = Path("outputs/combined_prefix_test/prefix")
VIDEO30_DIR = Path("outputs/combined_prefix_test/video30")

# ── load ─────────────────────────────────────────────────────────────────────
prefix_data  = [json.loads(f.read_text()) for f in sorted(PREFIX_DIR.glob("*.json"))]
video30_data = [json.loads(f.read_text()) for f in sorted(VIDEO30_DIR.glob("*.json"))]

STYLES = ["dec", "neu", "leg"]
SIDES  = ["left", "right"]

# ── 1. Per-(style,side) accuracy at each window ───────────────────────────────
by_ss_n = defaultdict(lambda: defaultdict(list))
for r in prefix_data:
    style = r["meta"]["style"]
    side  = r["meta"]["side"]
    for vr in r["vlm_results"]:
        by_ss_n[(style, side)][vr["n_frames"]].append(vr["correct"])

print("=" * 68)
print("PREFIX ACCURACY  per (style / side) x window size")
print("=" * 68)
print(f"{'style/side':<14}  {'n=3':>6}  {'n=4':>6}  {'n=5':>6}  {'n=6':>6}  {'all':>6}")
print("-" * 68)
for style in STYLES:
    for side in SIDES:
        key = (style, side)
        if key not in by_ss_n:
            continue
        vals = by_ss_n[key]
        rows = []
        for n in [3, 4, 5, 6]:
            v   = vals.get(n, [])
            acc = np.mean(v) if v else float("nan")
            rows.append(f"{acc:>6.0%}")
        all_v = [c for v in vals.values() for c in v]
        all_acc = np.mean(all_v) if all_v else float("nan")
        print(f"  {style}/{side:<10}  {'  '.join(rows)}  {all_acc:>6.0%}")
    print()

# ── 2. Mean pA per (style, side) ──────────────────────────────────────────────
by_ss_pA = defaultdict(list)
for r in prefix_data:
    style = r["meta"]["style"]
    side  = r["meta"]["side"]
    for vr in r["vlm_results"]:
        by_ss_pA[(style, side)].append(vr.get("pA", 0.5))

print("=" * 68)
print("MEAN pA  per (style / side)  —  A=image-left=world-right block")
print("=" * 68)
print(f"{'style/side':<14}  {'mean_pA':>8}  {'mean_pB':>8}  {'n_calls':>8}")
print("-" * 68)
for style in STYLES:
    for side in SIDES:
        key = (style, side)
        if key not in by_ss_pA:
            continue
        vals = by_ss_pA[key]
        print(f"  {style}/{side:<10}  {np.mean(vals):>8.3f}  {1-np.mean(vals):>8.3f}  {len(vals):>8}")
    print()

# ── 3. Right-side-only comparison (removes pA bias direction effect) ──────────
print("=" * 68)
print("RIGHT-SIDE ONLY  (correct=A for all)  — bias-neutral comparison")
print("=" * 68)
print(f"{'style':<8}  {'acc':>6}  {'mean_pA':>8}  {'frac_legible':>14}")
print("-" * 68)
for style in STYLES:
    items = [vr
             for r in prefix_data
             for vr in r["vlm_results"]
             if r["meta"]["style"] == style and r["meta"]["side"] == "right"]
    if not items:
        continue
    acc   = np.mean([v["correct"] for v in items])
    mpA   = np.mean([v.get("pA", 0.5) for v in items])
    fleg  = np.mean([1 if v.get("legible") == "legible_now" else 0 for v in items])
    print(f"  {style:<8}  {acc:>6.0%}  {mpA:>8.3f}  {fleg:>14.0%}")

# ── 4. Deceptive feint: pA trajectory across window sizes ────────────────────
print()
print("=" * 68)
print("DEC FEINT SIGNAL: mean pA as function of n (per side)")
print("  Expected: dec/right pA drops from hi (feint toward B) to lo")
print("            dec/left  pA rises from hi (feint toward A) to lo")
print("=" * 68)
for side in SIDES:
    correct = "A" if side == "right" else "B"
    rows = []
    for n in [3, 4, 5, 6]:
        vals = [
            vr.get("pA", 0.5)
            for r in prefix_data
            for vr in r["vlm_results"]
            if r["meta"]["style"] == "dec"
            and r["meta"]["side"] == side
            and vr["n_frames"] == n
        ]
        rows.append(f"n={n}: pA={np.mean(vals):.3f}" if vals else f"n={n}: -")
    print(f"  dec/{side} (correct={correct}):  " + "  ".join(rows))

# ── 5. Legible variant arc-strength analysis ──────────────────────────────────
print()
print("=" * 68)
print("LEG VARIANT ACCURACY  (variant 0=small arc, 9=large arc)")
print("Only right-side shown to avoid pA-bias direction issue")
print("=" * 68)
for r in sorted(prefix_data, key=lambda x: (x["meta"]["side"], x["meta"]["variant"])):
    if r["meta"]["style"] != "leg" or r["meta"]["side"] != "right":
        continue
    vrs  = r["vlm_results"]
    acc  = np.mean([v["correct"] for v in vrs]) if vrs else 0
    mpA  = np.mean([v.get("pA", 0.5) for v in vrs]) if vrs else 0.5
    accs = {v["n_frames"]: v["correct"] for v in vrs}
    var  = r["meta"]["variant"]
    detail = "  ".join(f"{'OK' if accs.get(n) else 'XX'}"
                       for n in [3, 4, 5, 6])
    print(f"  v{var:02d} (arc_strength={var}):  acc={acc:.0%}  pA={mpA:.2f}  [{detail}]")

# ── 6. VIDEO30 per-(style,side) ───────────────────────────────────────────────
by_vss = defaultdict(list)
for r in video30_data:
    style = r["meta"]["style"]
    side  = r["meta"]["side"]
    by_vss[(style, side)].append(r.get("vlm_result", {}))

print()
print("=" * 68)
print("VIDEO-30% MODE  per (style / side)")
print("=" * 68)
print(f"{'style/side':<14}  {'acc':>6}  {'mean_pA':>8}  {'frac_leg':>10}  {'n':>4}")
print("-" * 68)
for style in STYLES:
    for side in SIDES:
        key = (style, side)
        if key not in by_vss:
            continue
        items = by_vss[key]
        ok    = [v for v in items if "error" not in v]
        if not ok:
            continue
        acc  = np.mean([v.get("correct", False) for v in ok])
        mpA  = np.mean([v.get("pA", 0.5) for v in ok])
        fleg = np.mean([1 if v.get("legible") == "legible_now" else 0 for v in ok])
        print(f"  {style}/{side:<10}  {acc:>6.0%}  {mpA:>8.3f}  {fleg:>10.0%}  {len(ok):>4}")
    print()

# ── 7. Quick verdict ──────────────────────────────────────────────────────────
print("=" * 68)
print("SUMMARY VERDICT")
print("=" * 68)

# Right-side only (bias-neutral)
stats = {}
for style in STYLES:
    items = [vr
             for r in prefix_data
             for vr in r["vlm_results"]
             if r["meta"]["style"] == style and r["meta"]["side"] == "right"]
    if items:
        stats[style] = {
            "acc":  np.mean([v["correct"] for v in items]),
            "pA":   np.mean([v.get("pA", 0.5) for v in items]),
            "fleg": np.mean([1 if v.get("legible")=="legible_now" else 0 for v in items]),
        }

if {"dec","neu","leg"}.issubset(stats):
    print(f"  [Right-side bias-neutral]  dec={stats['dec']['acc']:.0%}  "
          f"neu={stats['neu']['acc']:.0%}  leg={stats['leg']['acc']:.0%}")
    # Check expected ordering: dec < neu < leg
    if stats["dec"]["acc"] < stats["neu"]["acc"] < stats["leg"]["acc"]:
        print("  -> CORRECT ordering: dec < neu < leg  *** VALIDATES HYPOTHESIS ***")
    elif stats["dec"]["acc"] <= stats["neu"]["acc"]:
        print("  -> Partial: dec <= neu  (deception effect present)")
    else:
        print("  -> INVERTED or flat — hypothesis not supported by this metric")

    delta_dec_leg = stats["leg"]["acc"] - stats["dec"]["acc"]
    print(f"  Legible - Deceptive accuracy gap: {delta_dec_leg:+.0%}")
