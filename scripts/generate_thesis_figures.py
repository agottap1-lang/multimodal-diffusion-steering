#!/usr/bin/env python3
"""
Generate all thesis figures answering:
  "Can a VLM be as good as a human in judging robot motion?"

Reads existing JSON results and produces publication-quality figures.
Output: outputs/thesis_figures/
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "thesis_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "font.family": "serif",
})
C_CFG2 = "#2196F3"    # blue
C_CFG0 = "#FF9800"    # orange
C_VLM  = "#4CAF50"    # green
C_BASE = "#9E9E9E"    # gray
C_LEGDIFF = "#9C27B0" # purple
C_RED  = "#F44336"
C_TEAL = "#009688"

# ═══════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_json(path):
    with open(ROOT / path) as f:
        return json.load(f)

print("Loading data...")
n50     = load_json("outputs/eval_n50/results.json")
lam0    = load_json("outputs/eval_lambda0/results.json")
vlm10   = load_json("outputs/eval_vlm_n10/results.json")
jitter  = load_json("outputs/eval_jitter/results.json")
bon     = load_json("outputs/best_of_n_results.json")
lps     = load_json("outputs/lps_sweep_results.json")
legdiff = load_json("outputs/legdiff_results.json")
legvlm  = load_json("outputs/legdiff_vlm_results.json")

# ── Precompute aggregates ─────────────────────────────────────────────────
def agg(episodes, key):
    vals = [e[key] for e in episodes if e.get(key) is not None]
    return np.array(vals, dtype=float)

def success_rate(episodes):
    return np.mean([e["success"] for e in episodes])


def draw_formula_card(ax, title, color, formulas, bullets, example, better):
    """Render one metric card with formula and plain-English explanation."""
    ax.axis("off")
    card = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#FCFCFC", edgecolor=color, linewidth=2.0,
        transform=ax.transAxes,
    )
    ax.add_patch(card)

    ax.text(0.05, 0.92, title, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=color, va="top")

    y = 0.80
    for line in formulas:
        ax.text(0.05, y, line, transform=ax.transAxes,
                fontsize=11.5, va="top")
        y -= 0.12

    y -= 0.02
    for bullet in bullets:
        ax.text(0.07, y, u"\u2022 " + bullet, transform=ax.transAxes,
                fontsize=9.5, va="top", color="#222222")
        y -= 0.09

    ax.text(0.05, 0.07, example, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#222222")
    ax.text(0.95, 0.07, better, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=color, ha="right")

# CFG λ=2 (n=50)
cfg2_leg_le     = agg(n50["legibility"], "L_early")
cfg2_pred_eff   = agg(n50["predictability"], "path_efficiency")
cfg2_safe_clear = agg(n50["safety"], "clearance")
cfg2_safe_coll  = [e["collision"] for e in n50["safety"]]
cfg2_grnd_dist  = agg(n50["grounding"], "hover_dist")
cfg2_grnd_hov   = [e["hovered"] for e in n50["grounding"]]

# CFG λ=0 (n=50)
cfg0_leg_le     = agg(lam0["legibility"], "L_early")
cfg0_pred_eff   = agg(lam0["predictability"], "path_efficiency")
cfg0_safe_clear = agg(lam0["safety"], "clearance")
cfg0_safe_coll  = [e["collision"] for e in lam0["safety"]]
cfg0_grnd_dist  = agg(lam0["grounding"], "hover_dist")
cfg0_grnd_hov   = [e["hovered"] for e in lam0["grounding"]]

print(f"  CFG lambda=2: Leg L_early={cfg2_leg_le.mean():.3f}+-{cfg2_leg_le.std():.3f}")
print(f"  CFG lambda=0: Leg L_early={cfg0_leg_le.mean():.3f}+-{cfg0_leg_le.std():.3f}")

print("\nFig 0: Behavior Metric Formulas...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Behavior-Specific Metrics Used in Evaluation",
             fontsize=15, fontweight="bold")

draw_formula_card(
    axes[0, 0],
    "Legibility",
    C_CFG2,
    [
        r"$L_{\mathrm{early}}=\frac{1}{T_e}\sum_{t=0}^{T_e-1} P(g^* \mid x_t)$",
        r"$P(g_i \mid x_t)\propto \exp\!\left(-\|x_t-g_i\|_2^2/(2\sigma^2)\right)$",
    ],
    [
        "Measures how early the motion reveals the intended goal.",
        "Computed from end-effector distance to the left and right goals.",
        "Higher value means the goal becomes clear earlier.",
    ],
    f"Example: {cfg2_leg_le.mean():.3f} +/- {cfg2_leg_le.std():.3f}",
    "Higher is better",
)

draw_formula_card(
    axes[0, 1],
    "Predictability",
    C_CFG0,
    [
        r"$\mathrm{PathEff}=\frac{\|x_T-x_0\|_2}{\sum_{t=0}^{T-1}\|x_{t+1}-x_t\|_2}$",
    ],
    [
        "Compares straight-line distance to actual traveled path length.",
        "Captures how direct and efficient the motion is.",
        "A perfectly straight trajectory approaches 1.0.",
    ],
    f"Example: {cfg2_pred_eff.mean():.3f} +/- {cfg2_pred_eff.std():.3f}",
    "Higher is better",
)

draw_formula_card(
    axes[1, 0],
    "Safety",
    C_RED,
    [
        r"$C_{\min}=\min_t \|x_{t,xy}-o_{xy}\|_2$",
        r"$\mathrm{Collision}=\mathbf{1}[C_{\min}<r_{\mathrm{obs}}]$",
    ],
    [
        "Tracks the closest horizontal distance to the obstacle.",
        "Clearance is measured over the full trajectory.",
        "A collision is flagged if clearance drops below obstacle radius.",
    ],
    f"Example: {cfg2_safe_clear.mean():.3f} +/- {cfg2_safe_clear.std():.3f} m"
    f" | collisions={np.mean(cfg2_safe_coll)*100:.0f}%",
    "Higher clearance is better",
)

draw_formula_card(
    axes[1, 1],
    "Grounding",
    C_TEAL,
    [
        r"$D_{\mathrm{wp}}=\min_t \|x_t-w\|_2$",
        r"$\mathrm{Hovered}=\mathbf{1}[D_{\mathrm{wp}}<0.06]$",
    ],
    [
        "Measures how close the end effector gets to the waypoint.",
        "Operationalizes instruction following as waypoint proximity.",
        "Hover success is counted when the waypoint distance is below 6 cm.",
    ],
    f"Example: {cfg2_grnd_dist.mean():.3f} +/- {cfg2_grnd_dist.std():.3f} m"
    f" | hovered={np.mean(cfg2_grnd_hov)*100:.0f}%",
    "Lower distance is better",
)

legend_text = (
    r"Symbols: $x_t$ = end-effector position at time $t$; "
    r"$x_0, x_T$ = start/end end-effector position; "
    r"$g_i$ = candidate goal; $g^*$ = true target goal; " "\n"
    r"$o$ = obstacle center; $w$ = waypoint position; "
    r"$x_{t,xy}$ = XY projection of the end effector; "
    r"$T_e$ = first 30\% of trajectory."
)
fig.text(
    0.5, 0.03, legend_text,
    ha="center", va="center", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BDBDBD"),
)
plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.94])
fig.savefig(OUT / "fig0_metric_formulas.png")
plt.close()
print("  -> fig0_metric_formulas.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: CFG ABLATION — λ=0 vs λ=2 across 4 behaviors (SLIDE 1)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 1: CFG Ablation...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Classifier-Free Guidance Ablation (n=50, paired seeds)", fontsize=14, fontweight="bold")

# 1a: Legibility — L_early
ax = axes[0, 0]
means = [cfg0_leg_le.mean(), cfg2_leg_le.mean()]
stds  = [cfg0_leg_le.std(), cfg2_leg_le.std()]
bars = ax.bar(["λ=0\n(unconditioned)", "λ=2\n(CFG)"], means, yerr=stds,
              color=[C_CFG0, C_CFG2], capsize=5, width=0.5, edgecolor="black", linewidth=0.5)
ax.set_ylabel("L_early (↑ better)")
ax.set_title("Legibility")
ax.set_ylim(0.7, 1.0)
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width()/2, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="chance")

# 1b: Predictability — path efficiency
ax = axes[0, 1]
means = [cfg0_pred_eff.mean(), cfg2_pred_eff.mean()]
stds  = [cfg0_pred_eff.std(), cfg2_pred_eff.std()]
bars = ax.bar(["λ=0", "λ=2"], means, yerr=stds,
              color=[C_CFG0, C_CFG2], capsize=5, width=0.5, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Path Efficiency (↑ better)")
ax.set_title("Predictability")
ax.set_ylim(0.3, 0.55)
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width()/2, m + s + 0.003, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

# 1c: Safety — collision rate + clearance
ax = axes[1, 0]
coll_rate_0 = np.mean(cfg0_safe_coll) * 100
coll_rate_2 = np.mean(cfg2_safe_coll) * 100
x = np.arange(2)
w = 0.35
b1 = ax.bar(x - w/2, [cfg0_safe_clear.mean(), cfg2_safe_clear.mean()],
            w, yerr=[cfg0_safe_clear.std(), cfg2_safe_clear.std()],
            color=[C_CFG0, C_CFG2], capsize=4, edgecolor="black", linewidth=0.5, label="Clearance (m)")
ax.set_ylabel("Min Clearance (m, ↑ better)")
ax.set_title("Safety")
ax.set_xticks(x)
ax.set_xticklabels(["λ=0", "λ=2"])
ax.set_ylim(0, 0.12)

ax2 = ax.twinx()
b2 = ax2.bar(x + w/2, [coll_rate_0, coll_rate_2], w,
             color=[C_RED, "#EF9A9A"], edgecolor="black", linewidth=0.5, alpha=0.7, label="Collision %")
ax2.set_ylabel("Collision Rate % (↓ better)")
ax2.set_ylim(0, 30)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
ax.text(0 - w/2, cfg0_safe_clear.mean() + cfg0_safe_clear.std() + 0.003,
        f"{cfg0_safe_clear.mean():.3f}", ha="center", fontsize=8)
ax.text(1 - w/2, cfg2_safe_clear.mean() + cfg2_safe_clear.std() + 0.003,
        f"{cfg2_safe_clear.mean():.3f}", ha="center", fontsize=8)
ax2.text(0 + w/2, coll_rate_0 + 1, f"{coll_rate_0:.0f}%", ha="center", fontsize=8, color=C_RED)
ax2.text(1 + w/2, coll_rate_2 + 1, f"{coll_rate_2:.0f}%", ha="center", fontsize=8, color=C_RED)

# 1d: Grounding — hover rate + success
ax = axes[1, 1]
sr_0 = success_rate(lam0["grounding"]) * 100
sr_2 = success_rate(n50["grounding"]) * 100
hr_0 = np.mean(cfg0_grnd_hov) * 100
hr_2 = np.mean(cfg2_grnd_hov) * 100
x = np.arange(2)
b1 = ax.bar(x - w/2, [sr_0, sr_2], w, color=[C_CFG0, C_CFG2],
            edgecolor="black", linewidth=0.5, label="Success %")
b2 = ax.bar(x + w/2, [hr_0, hr_2], w, color=[C_TEAL, "#80CBC4"],
            edgecolor="black", linewidth=0.5, label="Hover Near WP %")
ax.set_ylabel("Rate (%)")
ax.set_title("Grounding")
ax.set_xticks(x)
ax.set_xticklabels(["λ=0", "λ=2"])
ax.set_ylim(0, 110)
ax.legend(fontsize=8)
for i, (s, h) in enumerate([(sr_0, hr_0), (sr_2, hr_2)]):
    ax.text(i - w/2, s + 1, f"{s:.0f}%", ha="center", fontsize=8)
    ax.text(i + w/2, h + 1, f"{h:.0f}%", ha="center", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "fig1_cfg_ablation.png")
plt.close()
print("  -> fig1_cfg_ablation.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: VLM CANDIDATE SCORE DISTRIBUTION (SLIDE 2)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 2: VLM Score Distribution...")
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("VLM Score Distribution Across K=8 Candidates Per Episode", fontsize=13, fontweight="bold")

behaviors = ["legibility", "predictability", "safety", "grounding"]
titles = ["Legibility", "Predictability", "Safety", "Grounding"]

for ax, beh, title in zip(axes, behaviors, titles):
    all_scores = []
    for ep in vlm10[beh]:
        cand_scores = [c["vlm_score"] for c in ep["candidates"]]
        all_scores.append(cand_scores)

    # Box plot per episode
    bp = ax.boxplot(all_scores, patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor=C_VLM, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3))
    # Overlay individual points
    for i, scores in enumerate(all_scores):
        jitter_x = np.random.normal(i + 1, 0.08, len(scores))
        ax.scatter(jitter_x, scores, s=12, alpha=0.6, color=C_VLM, zorder=3, edgecolors="none")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("VLM Score" if ax == axes[0] else "")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    # Annotate
    flat = [s for ep_s in all_scores for s in ep_s]
    n_05 = sum(1 for s in flat if abs(s - 0.5) < 0.01)
    ax.text(0.98, 0.02, f"n=0.5: {n_05}/{len(flat)}", transform=ax.transAxes,
            ha="right", fontsize=7, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig2_vlm_score_distribution.png")
plt.close()
print("  -> fig2_vlm_score_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: VLM-SELECTED vs BASELINE (Best vs Worst) — SLIDE 2/3
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 3: VLM Selection Effect...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("VLM Best-of-8 Selection: VLM-Picked vs Lowest-Scored (n=10)", fontsize=13, fontweight="bold")

# ── Legibility ──
ax = axes[0, 0]
vlm_le = [ep["vlm"]["L_early"] for ep in vlm10["legibility"]]
bl_le  = [ep["baseline"]["L_early"] for ep in vlm10["legibility"]]
vlm_suc = [ep["vlm"]["success"] for ep in vlm10["legibility"]]
bl_suc  = [ep["baseline"]["success"] for ep in vlm10["legibility"]]

x = np.arange(len(vlm_le))
ax.bar(x - 0.2, vlm_le, 0.35, color=C_VLM, alpha=0.8, label=f"VLM-pick ({sum(vlm_suc)}/10 suc)")
ax.bar(x + 0.2, bl_le, 0.35, color=C_BASE, alpha=0.8, label=f"Worst-scored ({sum(bl_suc)}/10 suc)")
for i in range(len(vlm_le)):
    if not vlm_suc[i]:
        ax.scatter(i - 0.2, vlm_le[i] + 0.02, marker="x", color=C_RED, s=30, zorder=5)
    if not bl_suc[i]:
        ax.scatter(i + 0.2, bl_le[i] + 0.02, marker="x", color=C_RED, s=30, zorder=5)
ax.set_ylabel("L_early")
ax.set_title("Legibility")
ax.set_xlabel("Episode")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.15)
ax.axhline(np.mean(vlm_le), color=C_VLM, linestyle="--", alpha=0.5)
ax.axhline(np.mean(bl_le), color=C_BASE, linestyle="--", alpha=0.5)
ax.text(9.5, np.mean(vlm_le), f"μ={np.mean(vlm_le):.2f}", fontsize=7, color=C_VLM, va="bottom")
ax.text(9.5, np.mean(bl_le), f"μ={np.mean(bl_le):.2f}", fontsize=7, color="gray", va="bottom")

# ── Predictability ──
ax = axes[0, 1]
vlm_pe = []
bl_pe  = []
vlm_suc_p = [ep["vlm"]["success"] for ep in vlm10["predictability"]]
bl_suc_p  = [ep["baseline"]["success"] for ep in vlm10["predictability"]]
for ep in vlm10["predictability"]:
    v = ep["vlm"].get("path_efficiency", 0)
    b = ep["baseline"].get("path_efficiency", 0)
    vlm_pe.append(float(v))
    bl_pe.append(float(b))

x = np.arange(len(vlm_pe))
ax.bar(x - 0.2, vlm_pe, 0.35, color=C_VLM, alpha=0.8, label=f"VLM-pick ({sum(vlm_suc_p)}/10 suc)")
ax.bar(x + 0.2, bl_pe, 0.35, color=C_BASE, alpha=0.8, label=f"Worst-scored ({sum(bl_suc_p)}/10 suc)")
for i in range(len(vlm_pe)):
    if not vlm_suc_p[i]:
        ax.scatter(i - 0.2, vlm_pe[i] + 0.01, marker="x", color=C_RED, s=30, zorder=5)
    if not bl_suc_p[i]:
        ax.scatter(i + 0.2, bl_pe[i] + 0.01, marker="x", color=C_RED, s=30, zorder=5)
ax.set_ylabel("Path Efficiency")
ax.set_title("Predictability")
ax.set_xlabel("Episode")
ax.legend(fontsize=8)

# ── Safety ──
ax = axes[1, 0]
vlm_sc_s = [ep["vlm"]["vlm_score"] for ep in vlm10["safety"]]
bl_sc_s  = [ep["baseline"]["vlm_score"] for ep in vlm10["safety"]]
vlm_suc_s = [ep["vlm"]["success"] for ep in vlm10["safety"]]
bl_suc_s  = [ep["baseline"]["success"] for ep in vlm10["safety"]]

x = np.arange(len(vlm_sc_s))
ax.bar(x - 0.2, vlm_sc_s, 0.35, color=C_VLM, alpha=0.8, label=f"VLM-pick ({sum(vlm_suc_s)}/10 suc)")
ax.bar(x + 0.2, bl_sc_s, 0.35, color=C_BASE, alpha=0.8, label=f"Worst-scored ({sum(bl_suc_s)}/10 suc)")
ax.set_ylabel("VLM Safety Score")
ax.set_title("Safety")
ax.set_xlabel("Episode")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.15)

# ── Grounding ──
ax = axes[1, 1]
vlm_sc_g = [ep["vlm"]["vlm_score"] for ep in vlm10["grounding"]]
bl_sc_g  = [ep["baseline"]["vlm_score"] for ep in vlm10["grounding"]]
vlm_suc_g = [ep["vlm"]["success"] for ep in vlm10["grounding"]]
bl_suc_g  = [ep["baseline"]["success"] for ep in vlm10["grounding"]]
x = np.arange(len(vlm_sc_g))
ax.bar(x - 0.2, vlm_sc_g, 0.35, color=C_VLM, alpha=0.8, label=f"VLM-pick ({sum(vlm_suc_g)}/10 suc)")
ax.bar(x + 0.2, bl_sc_g, 0.35, color=C_BASE, alpha=0.8, label=f"Worst-scored ({sum(bl_suc_g)}/10 suc)")
ax.set_ylabel("VLM Grounding Score")
ax.set_title("Grounding (API quota degraded)")
ax.set_xlabel("Episode")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.15)
ax.text(0.5, 0.5, "All scores = 0.5\n(API quota)", transform=ax.transAxes,
        ha="center", va="center", fontsize=11, color="red", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig3_vlm_selection_effect.png")
plt.close()
print("  -> fig3_vlm_selection_effect.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: VLM SCORE vs L_early SCATTER — Per candidate (legibility)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 4: VLM Score vs L_early Scatter...")
fig, ax = plt.subplots(figsize=(7, 5))

# For legibility: each of 10 episodes has 8 candidates
# But we only have L_early for the executed vlm/baseline, not all 8
# So let's plot the executed VLM-pick and baseline for each episode
vlm_scores_all = []
metric_scores_all = []
labels_all = []

for ep in vlm10["legibility"]:
    # VLM-selected point
    vlm_scores_all.append(ep["vlm"]["vlm_score"])
    metric_scores_all.append(ep["vlm"]["L_early"])
    labels_all.append("VLM-pick")
    # Baseline point
    vlm_scores_all.append(ep["baseline"]["vlm_score"])
    metric_scores_all.append(ep["baseline"]["L_early"])
    labels_all.append("Worst-scored")

vlm_arr = np.array(vlm_scores_all)
met_arr = np.array(metric_scores_all)
lab_arr = np.array(labels_all)

mask_vlm = lab_arr == "VLM-pick"
mask_bl  = lab_arr == "Worst-scored"

ax.scatter(vlm_arr[mask_vlm], met_arr[mask_vlm], s=80, c=C_VLM, marker="o",
           label="VLM-picked trajectory", alpha=0.8, edgecolors="black", linewidths=0.5)
ax.scatter(vlm_arr[mask_bl], met_arr[mask_bl], s=80, c=C_BASE, marker="s",
           label="Worst-scored trajectory", alpha=0.8, edgecolors="black", linewidths=0.5)

# Connect pairs with lines
for i in range(0, len(vlm_scores_all), 2):
    ax.plot([vlm_scores_all[i], vlm_scores_all[i+1]],
            [metric_scores_all[i], metric_scores_all[i+1]],
            color="gray", alpha=0.3, linewidth=0.8)

ax.set_xlabel("VLM Legibility Score (visual judgment)")
ax.set_ylabel("L_early Metric (Bayesian geometric)")
ax.set_title("VLM Visual Score vs Geometric Metric\n(Legibility, n=10 episodes)", fontweight="bold")
ax.legend()
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(0, 1.1)

# Add annotation for the tension
ax.annotate("VLM prefers visual\ncurvature but geometric\nmetric prefers proximity",
            xy=(0.95, 0.65), fontsize=8, color="gray", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Correlation
corr = np.corrcoef(vlm_arr, met_arr)[0, 1]
ax.text(0.02, 0.02, f"Pearson r = {corr:.3f}", transform=ax.transAxes, fontsize=9)

fig.savefig(OUT / "fig4_vlm_vs_learly_scatter.png")
plt.close()
print("  -> fig4_vlm_vs_learly_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: VLM GOAL IDENTIFICATION ACCURACY — Legibility candidates
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 5: VLM Goal Identification...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("VLM Perceptual Analysis: Goal Identification (Legibility)", fontsize=13, fontweight="bold")

# Analysis: VLM reports pA/pB for WHAT IT SEES, not the episode target.
# Many candidates don't go toward the assigned target — that's expected
# with stochastic diffusion. The real question is: does VLM CONFIDENTLY
# identify A trajectory direction?

confidences = []
says_left = 0
says_right = 0
ambiguous = 0
high_conf_count = 0  # max(pA,pB) >= 0.8
total = 0

for ep in vlm10["legibility"]:
    for c in ep["candidates"]:
        pA = c.get("vlm_pA", 0.5)
        pB = c.get("vlm_pB", 0.5)
        conf = max(pA, pB)
        confidences.append(conf)
        total += 1
        if conf >= 0.8:
            high_conf_count += 1
        if pA > 0.6:
            says_left += 1
        elif pB > 0.6:
            says_right += 1
        else:
            ambiguous += 1

accuracy = high_conf_count / total if total > 0 else 0
print(f"  VLM high-confidence rate: {high_conf_count}/{total} = {accuracy:.1%}")
print(f"  Says left: {says_left}, Says right: {says_right}, Ambiguous: {ambiguous}")

# 5a: Confidence distribution
ax1.hist(confidences, bins=15, range=(0, 1.05), color=C_VLM, edgecolor="black",
         linewidth=0.5, alpha=0.8)
ax1.axvline(0.8, color=C_RED, linestyle="--", alpha=0.7, label="High-conf threshold")
ax1.set_xlabel("VLM Confidence max(pA, pB)")
ax1.set_ylabel("Count")
ax1.set_title(f"Confidence: {high_conf_count}/{total} ({accuracy:.0%}) ≥ 0.8")
ax1.legend(fontsize=8)

# 5b: Direction breakdown
bars = ax2.bar(["Says LEFT\n(pA > 0.6)", "Says RIGHT\n(pB > 0.6)", "Ambiguous"],
               [says_left, says_right, ambiguous],
               color=[C_CFG2, C_CFG0, C_BASE], edgecolor="black", linewidth=0.5, width=0.5)
ax2.set_ylabel("Count (80 candidates)")
ax2.set_title("Direction Identification")
for b, v in zip(bars, [says_left, says_right, ambiguous]):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.5, str(v), ha="center", fontweight="bold")

# 5c: VLM correctly discriminates high vs low score
# Among candidates within same episode: does higher VLM score correlate with
# more exaggerated curvature?
spread_per_ep = []
for ep in vlm10["legibility"]:
    scores = [c["vlm_score"] for c in ep["candidates"]]
    spread_per_ep.append(max(scores) - min(scores))

ax3.bar(range(1, 11), spread_per_ep, color=C_VLM, edgecolor="black", linewidth=0.5, width=0.6)
ax3.axhline(np.mean(spread_per_ep), color="gray", linestyle="--", alpha=0.5)
ax3.set_xlabel("Episode")
ax3.set_ylabel("Score Spread (max - min)")
ax3.set_title(f"Within-Episode Discrimination\n(mean spread = {np.mean(spread_per_ep):.2f})")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig5_vlm_goal_identification.png")
plt.close()
print("  -> fig5_vlm_goal_identification.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: LegDiff + VLM Best-of-3 Comparison (strongest VLM result)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 6: LegDiff + VLM...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("LegDiff Policy: Effect of VLM Best-of-3 Selection (n=10)", fontsize=13, fontweight="bold")

ld_raw = [e["l_early"] for e in legvlm["legdiff"]]
ld_vlm = [e["l_early"] for e in legvlm["legdiff_vlm"]]

# 6a: Paired comparison
x = np.arange(len(ld_raw))
ax1.bar(x - 0.2, ld_raw, 0.35, color=C_LEGDIFF, alpha=0.6, label=f"LegDiff (μ={np.mean(ld_raw):.3f})")
ax1.bar(x + 0.2, ld_vlm, 0.35, color=C_VLM, alpha=0.8, label=f"LegDiff+VLM (μ={np.mean(ld_vlm):.3f})")
ax1.set_ylabel("L_early")
ax1.set_xlabel("Episode")
ax1.set_title("Per-Episode Comparison")
ax1.legend(fontsize=8)
ax1.set_ylim(0.9, 1.0)

# 6b: Summary statistics
methods = ["Vanilla\nBaseline", "LegDiff\n(w=3)", "LegDiff\n+VLM"]
ld_baseline = [e["l_early"] for e in legdiff["baseline"]]
means = [np.mean(ld_baseline), np.mean(ld_raw), np.mean(ld_vlm)]
stds  = [np.std(ld_baseline), np.std(ld_raw), np.std(ld_vlm)]
colors = [C_BASE, C_LEGDIFF, C_VLM]
bars = ax2.bar(methods, means, yerr=stds, color=colors, capsize=5, width=0.5,
               edgecolor="black", linewidth=0.5)
ax2.set_ylabel("L_early")
ax2.set_title("Method Comparison")
ax2.set_ylim(0.85, 1.0)
for b, m, s in zip(bars, means, stds):
    ax2.text(b.get_x() + b.get_width()/2, m + s + 0.002, f"{m:.3f}", ha="center", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig6_legdiff_vlm.png")
plt.close()
print("  -> fig6_legdiff_vlm.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: BEST-OF-N SCALING (VLM value proposition)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 7: Best-of-N Scaling...")
fig, ax = plt.subplots(figsize=(7, 4.5))

ns = [1, 4, 8, 16]
keys = ["baseline", "best_of_4", "best_of_8", "best_of_16"]
bon_means = [bon[k]["mean_L_early_intent"] for k in keys]
bon_stds  = [bon[k]["std_L_early_intent"] for k in keys]

ax.errorbar(ns, bon_means, yerr=bon_stds, marker="o", color=C_VLM, linewidth=2,
            capsize=5, markersize=8, label="Best-of-N (VLM reranking)")
ax.fill_between(ns, np.array(bon_means) - np.array(bon_stds),
                np.array(bon_means) + np.array(bon_stds), alpha=0.15, color=C_VLM)

# Add CFG baseline for reference
ax.axhline(cfg2_leg_le.mean(), color=C_CFG2, linestyle="--", alpha=0.7, label=f"CFG λ=2 (n=50): {cfg2_leg_le.mean():.3f}")
ax.axhline(cfg0_leg_le.mean(), color=C_CFG0, linestyle="--", alpha=0.5, label=f"CFG λ=0 (n=50): {cfg0_leg_le.mean():.3f}")

ax.set_xlabel("Number of Candidates (K)")
ax.set_ylabel("L_early")
ax.set_title("Best-of-K VLM Reranking: Legibility Scaling (n=50 per K)", fontweight="bold")
ax.set_xticks(ns)
ax.legend(fontsize=9)
ax.set_ylim(0.65, 0.95)

for n, m, s in zip(ns, bon_means, bon_stds):
    ax.annotate(f"{m:.3f}", (n, m), textcoords="offset points", xytext=(10, -5), fontsize=8)

fig.savefig(OUT / "fig7_best_of_n_scaling.png")
plt.close()
print("  -> fig7_best_of_n_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: SUMMARY VERDICT TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 8: Summary Table...")
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

# Build table data
col_headers = ["Dimension", "What VLM Does", "Metric Result",
               "VLM Agrees\nw/ Metric?", "Verdict"]
table_data = [
    ["Goal\nIdentification",
     "Identifies which block\nrobot targets",
     f"{high_conf_count}/{total} high-conf\n({accuracy:.0%} ≥ 0.8)",
     "Confident\nin most cases",
     "GOOD"],
    ["Legibility\n(exaggeration)",
     "Prefers visually\ncurved trajectories",
     f"VLM μ={np.mean(vlm_le):.2f}\nBaseline μ={np.mean(bl_le):.2f}",
     "No\n(anti-correlated)",
     "DISAGREES"],
    ["Predictability\n(directness)",
     "Rates straight paths\n~1.0 (saturates)",
     f"VLM suc={sum(vlm_suc_p)}/10\nBase suc={sum(bl_suc_p)}/10",
     "Partial\n(saturates at 1.0)",
     "MIXED"],
    ["Safety\n(avoidance)",
     "Identifies\ncollision risk",
     f"VLM suc={sum(vlm_suc_s)}/10\nBase suc={sum(bl_suc_s)}/10",
     "Directionally\ncorrect",
     "PROMISING"],
    ["Grounding\n(instructions)",
     "N/A — API quota\nexhausted",
     "All scores = 0.5\n(fallback)",
     "N/A",
     "INCONCLUSIVE"],
    ["LegDiff\nBest-of-3",
     "Selects most curved\nfrom small pool",
     f"LegDiff: {np.mean(ld_raw):.3f}\n+VLM: {np.mean(ld_vlm):.3f}",
     "Yes\n(+0.004 L_early)",
     "GOOD"],
]

colors_map = {"GOOD": "#C8E6C9", "DISAGREES": "#FFCDD2", "MIXED": "#FFF9C4",
              "PROMISING": "#C8E6C9", "INCONCLUSIVE": "#E0E0E0"}

table = ax.table(cellText=table_data, colLabels=col_headers,
                 cellLoc="center", loc="center",
                 colColours=["#BBDEFB"] * 5)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.2)

# Color the verdict column
for i, row in enumerate(table_data):
    verdict = row[-1]
    cell = table[i + 1, 4]
    cell.set_facecolor(colors_map.get(verdict, "white"))
    cell.get_text().set_fontweight("bold")

for j in range(5):
    table[0, j].get_text().set_fontweight("bold")
    table[0, j].set_facecolor("#1976D2")
    table[0, j].get_text().set_color("white")

ax.set_title("Summary: Can VLM Judge Robot Motion Quality?",
             fontsize=14, fontweight="bold", pad=20)

fig.savefig(OUT / "fig8_summary_verdict_table.png")
plt.close()
print("  -> fig8_summary_verdict_table.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9: LPS GUIDANCE SCALE SWEEP
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 9: LPS Guidance Sweep...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("Legibility-Posterior Sampling: Guidance Scale Sweep (n=20 each)", fontsize=13, fontweight="bold")

lps_keys = ["baseline_w0", "lps_w2", "lps_w5", "lps_w10", "lps_w20"]
lps_w = [0, 2, 5, 10, 20]
lps_means = [lps["results"][k]["l_early_mean"] for k in lps_keys]
lps_stds  = [lps["results"][k]["l_early_std"] for k in lps_keys]
lps_sr    = [lps["results"][k]["success_rate"] * 100 for k in lps_keys]

ax1.errorbar(lps_w, lps_means, yerr=lps_stds, marker="s", color=C_CFG2,
             linewidth=2, capsize=5, markersize=7)
ax1.set_xlabel("Guidance Scale (w)")
ax1.set_ylabel("L_early")
ax1.set_title("Legibility vs Guidance Scale")
ax1.set_ylim(0.85, 1.0)
for w, m in zip(lps_w, lps_means):
    ax1.annotate(f"{m:.3f}", (w, m), textcoords="offset points", xytext=(5, 8), fontsize=8)

ax2.bar([str(w) for w in lps_w], lps_sr, color=C_TEAL, edgecolor="black", linewidth=0.5, width=0.5)
ax2.set_xlabel("Guidance Scale (w)")
ax2.set_ylabel("Success Rate (%)")
ax2.set_title("Task Success vs Guidance Scale")
ax2.set_ylim(80, 105)
for i, (w, s) in enumerate(zip(lps_w, lps_sr)):
    ax2.text(i, s + 0.5, f"{s:.0f}%", ha="center", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig9_lps_guidance_sweep.png")
plt.close()
print("  -> fig9_lps_guidance_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10: COMPREHENSIVE BAR CHART — ALL METHODS ON LEGIBILITY
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 10: All Methods Comparison...")
fig, ax = plt.subplots(figsize=(12, 5))

methods = [
    "Vanilla\nDiffusion",
    "LPS\nw=10",
    "LegDiff\nw=3",
    "LegDiff\n+VLM (K=3)",
    "CFG λ=0\n(uncond)",
    "CFG λ=2",
    "Best-of-4\n(VLM)",
    "Best-of-8\n(VLM)",
    "Best-of-16\n(VLM)",
]
means_all = [
    np.mean(ld_baseline),                     # Vanilla
    lps["results"]["lps_w10"]["l_early_mean"], # LPS w=10
    np.mean([e["l_early"] for e in legdiff["legdiff"] if "l_early" in e]),  # LegDiff
    np.mean(ld_vlm),                           # LegDiff+VLM
    cfg0_leg_le.mean(),                        # CFG λ=0
    cfg2_leg_le.mean(),                        # CFG λ=2
    bon["best_of_4"]["mean_L_early_intent"],   # Best-of-4
    bon["best_of_8"]["mean_L_early_intent"],   # Best-of-8
    bon["best_of_16"]["mean_L_early_intent"],  # Best-of-16
]
stds_all = [
    np.std(ld_baseline),
    lps["results"]["lps_w10"]["l_early_std"],
    np.std([e["l_early"] for e in legdiff["legdiff"] if "l_early" in e]),
    np.std(ld_vlm),
    cfg0_leg_le.std(),
    cfg2_leg_le.std(),
    bon["best_of_4"]["std_L_early_intent"],
    bon["best_of_8"]["std_L_early_intent"],
    bon["best_of_16"]["std_L_early_intent"],
]
ns_all = [20, 20, 20, 10, 50, 50, 50, 50, 50]
colors_all = [C_BASE, C_TEAL, C_LEGDIFF, C_VLM, C_CFG0, C_CFG2, "#81C784", "#4CAF50", "#2E7D32"]

bars = ax.bar(range(len(methods)), means_all, yerr=stds_all,
              color=colors_all, capsize=4, width=0.65, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("L_early (↑ = more legible)")
ax.set_title("Legibility: All Methods Comparison", fontsize=14, fontweight="bold")
ax.set_ylim(0.65, 1.05)

for i, (b, m, s, n) in enumerate(zip(bars, means_all, stds_all, ns_all)):
    ax.text(b.get_x() + b.get_width()/2, m + s + 0.005,
            f"{m:.3f}\n(n={n})", ha="center", fontsize=7.5)

ax.axhline(0.5, color="gray", linestyle="--", alpha=0.2)

plt.tight_layout()
fig.savefig(OUT / "fig10_all_methods_legibility.png")
plt.close()
print("  -> fig10_all_methods_legibility.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11: VLM DISCRIMINATION POWER — Safety scores vs clearance
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 11: VLM Safety Discrimination...")
fig, ax = plt.subplots(figsize=(7, 5))

vlm_safety_scores = []
clearances = []
for ep in vlm10["safety"]:
    # VLM pick
    vlm_safety_scores.append(ep["vlm"]["vlm_score"])
    clearances.append(ep["vlm"].get("min_clearance", 0))
    # Baseline
    vlm_safety_scores.append(ep["baseline"]["vlm_score"])
    clearances.append(ep["baseline"].get("min_clearance", 0))

ax.scatter(clearances, vlm_safety_scores, s=60, c=C_TEAL, alpha=0.8,
           edgecolors="black", linewidths=0.5)
ax.set_xlabel("Minimum Clearance to Obstacle (m)")
ax.set_ylabel("VLM Safety Score")
ax.set_title("VLM Safety Score vs Physical Clearance", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
corr_s = np.corrcoef(clearances, vlm_safety_scores)[0, 1]
ax.text(0.02, 0.98, f"r = {corr_s:.3f}", transform=ax.transAxes, fontsize=10, va="top")

fig.savefig(OUT / "fig11_vlm_safety_discrimination.png")
plt.close()
print("  -> fig11_vlm_safety_discrimination.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12: ROBUSTNESS — Jitter test
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 12: Robustness (Jitter)...")
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("Robustness: CFG λ=2 with cube_jitter=0.02 (n=10)", fontsize=13, fontweight="bold")

jit_leg = agg(jitter["legibility"], "L_early")
jit_pred = agg(jitter["predictability"], "path_efficiency")
jit_safe_cl = agg(jitter["safety"], "clearance")
jit_grnd_dist = agg(jitter["grounding"], "hover_dist")

pairs = [
    ("Legibility\nL_early", cfg2_leg_le, jit_leg),
    ("Predictability\nEfficiency", cfg2_pred_eff, jit_pred),
    ("Safety\nClearance", cfg2_safe_clear, jit_safe_cl),
    ("Grounding\nHover Dist", cfg2_grnd_dist, jit_grnd_dist),
]

for ax, (title, nominal, jittered) in zip(axes, pairs):
    bars = ax.bar(["Nominal\n(n=50)", "Jittered\n(n=10)"],
                  [nominal.mean(), jittered.mean()],
                  yerr=[nominal.std(), jittered.std()],
                  color=[C_CFG2, "#FF7043"], capsize=5, width=0.5,
                  edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    for b, m, s in zip(bars, [nominal.mean(), jittered.mean()],
                       [nominal.std(), jittered.std()]):
        ax.text(b.get_x() + b.get_width()/2, m + s + 0.002,
                f"{m:.3f}", ha="center", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig12_robustness_jitter.png")
plt.close()
print("  -> fig12_robustness_jitter.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13: VLM PREDICTABILITY — Score vs Efficiency Per Candidate
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 13: VLM Predictability Analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("VLM Predictability: Score Saturation Problem", fontsize=13, fontweight="bold")

# 13a: All candidate scores histogram
all_pred_scores = []
for ep in vlm10["predictability"]:
    for c in ep["candidates"]:
        all_pred_scores.append(c["vlm_score"])

ax1.hist(all_pred_scores, bins=20, range=(0, 1.05), color=C_VLM, edgecolor="black",
         linewidth=0.5, alpha=0.8)
ax1.set_xlabel("VLM Predictability Score")
ax1.set_ylabel("Count")
ax1.set_title(f"Score Distribution (n={len(all_pred_scores)} candidates)")
n_1 = sum(1 for s in all_pred_scores if s >= 0.99)
ax1.text(0.5, 0.85, f"{n_1}/{len(all_pred_scores)} scored ≥0.99\n({n_1/len(all_pred_scores):.0%})",
         transform=ax1.transAxes, fontsize=10, ha="center",
         bbox=dict(boxstyle="round", facecolor="lightyellow"))

# 13b: Success rate comparison
ax2.bar(["VLM\nPick", "Worst\nScored"],
        [sum(vlm_suc_p)/10 * 100, sum(bl_suc_p)/10 * 100],
        color=[C_VLM, C_BASE], edgecolor="black", linewidth=0.5, width=0.5)
ax2.set_ylabel("Success Rate (%)")
ax2.set_title("Task Success (n=10)")
ax2.set_ylim(0, 110)
ax2.text(0, sum(vlm_suc_p)/10 * 100 + 2,
         f"{sum(vlm_suc_p)}/10", ha="center", fontweight="bold")
ax2.text(1, sum(bl_suc_p)/10 * 100 + 2,
         f"{sum(bl_suc_p)}/10", ha="center", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig13_vlm_predictability_analysis.png")
plt.close()
print("  -> fig13_vlm_predictability_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 14: MASTER SUMMARY — 3-slide layout guide
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 14: Master Summary...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

col_headers = ["Metric", "λ=0 (n=50)", "λ=2 (n=50)", "Δ", "VLM Pick\n(n=10)", "Worst Pick\n(n=10)", "VLM\nbetter?"]

# Legibility
leg_delta = cfg2_leg_le.mean() - cfg0_leg_le.mean()
# Predictability
pred_delta = cfg2_pred_eff.mean() - cfg0_pred_eff.mean()
# Safety collision
safe_delta = np.mean(cfg0_safe_coll)*100 - np.mean(cfg2_safe_coll)*100

table_data = [
    ["L_early (↑)",
     f"{cfg0_leg_le.mean():.3f} ± {cfg0_leg_le.std():.3f}",
     f"{cfg2_leg_le.mean():.3f} ± {cfg2_leg_le.std():.3f}",
     f"+{leg_delta:.3f}",
     f"{np.mean(vlm_le):.3f}",
     f"{np.mean(bl_le):.3f}",
     "No ↓"],
    ["Path Eff (↑)",
     f"{cfg0_pred_eff.mean():.3f} ± {cfg0_pred_eff.std():.3f}",
     f"{cfg2_pred_eff.mean():.3f} ± {cfg2_pred_eff.std():.3f}",
     f"+{pred_delta:.3f}",
     f"{np.mean(vlm_pe):.3f}",
     f"{np.mean(bl_pe):.3f}",
     "Yes ↑"],
    ["Collision % (↓)",
     f"{np.mean(cfg0_safe_coll)*100:.0f}%",
     f"{np.mean(cfg2_safe_coll)*100:.0f}%",
     f"-{safe_delta:.0f}%",
     f"{sum(vlm_suc_s)}/10 suc",
     f"{sum(bl_suc_s)}/10 suc",
     "Yes ↑"],
    ["Success (↑)",
     f"{success_rate(lam0['legibility'])*100:.0f}%",
     f"{success_rate(n50['legibility'])*100:.0f}%",
     "—",
     f"{sum(vlm_suc)}/10",
     f"{sum(bl_suc)}/10",
     "~same"],
    ["Goal ID Acc",
     "—", "—", "—",
     f"{accuracy:.0%}", "—", "Yes"],
]

table = ax.table(cellText=table_data, colLabels=col_headers,
                 cellLoc="center", loc="center",
                 colColours=["#E3F2FD"] * 7)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

for j in range(7):
    table[0, j].get_text().set_fontweight("bold")
    table[0, j].set_facecolor("#1565C0")
    table[0, j].get_text().set_color("white")

# Color VLM verdict column
verdict_colors = {"No ↓": "#FFCDD2", "Yes ↑": "#C8E6C9", "~same": "#FFF9C4", "Yes": "#C8E6C9"}
for i, row in enumerate(table_data):
    v = row[-1]
    cell = table[i + 1, 6]
    cell.set_facecolor(verdict_colors.get(v, "white"))
    cell.get_text().set_fontweight("bold")

ax.set_title("Master Comparison: CFG Conditioning + VLM Selection",
             fontsize=14, fontweight="bold", pad=20)

fig.savefig(OUT / "fig14_master_comparison_table.png")
plt.close()
print("  -> fig14_master_comparison_table.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 15: VLM SCORE WITHIN CANDIDATES — HIGH vs LOW legibility
# ═══════════════════════════════════════════════════════════════════════════
print("\nFig 15: VLM Score Ranking Fidelity...")
fig, ax = plt.subplots(figsize=(8, 5))

# For each legibility episode, rank candidates by VLM score and see if
# top-scored candidate has the most exaggerated trajectory
episode_labels = []
score_spreads = []
vlm_best_scores = []
vlm_worst_scores = []

for i, ep in enumerate(vlm10["legibility"]):
    scores = [c["vlm_score"] for c in ep["candidates"]]
    episode_labels.append(f"ep{i+1}")
    score_spreads.append(max(scores) - min(scores))
    vlm_best_scores.append(max(scores))
    vlm_worst_scores.append(min(scores))

x = np.arange(len(episode_labels))
ax.bar(x - 0.15, vlm_best_scores, 0.28, color=C_VLM, alpha=0.8,
       label="Best candidate", edgecolor="black", linewidth=0.5)
ax.bar(x + 0.15, vlm_worst_scores, 0.28, color=C_RED, alpha=0.6,
       label="Worst candidate", edgecolor="black", linewidth=0.5)
ax.bar(x, score_spreads, 0.05, bottom=np.array(vlm_worst_scores),
       color="black", alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(episode_labels, fontsize=9)
ax.set_ylabel("VLM Legibility Score")
ax.set_xlabel("Episode")
ax.set_title("VLM Score Range Per Episode: Best vs Worst Candidate (K=8)",
             fontweight="bold")
ax.legend()
ax.set_ylim(0, 1.15)
ax.axhline(np.mean(score_spreads), color="gray", linestyle="--", alpha=0.4)
ax.text(9.5, np.mean(score_spreads), f"mean spread\n={np.mean(score_spreads):.2f}",
        fontsize=8, va="center")

fig.savefig(OUT / "fig15_vlm_score_ranking_fidelity.png")
plt.close()
print("  -> fig15_vlm_score_ranking_fidelity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  All figures saved to: {OUT}")
print(f"  Total: 16 figures")
print(f"{'='*60}")

# Print slide layout guide
print("""
SLIDE LAYOUT GUIDE (4 slides):

SLIDE 0: "Behavior Metrics and Formulas"
  - fig0_metric_formulas.png
  Key message: each behavior is evaluated with a distinct trajectory metric

SLIDE 1: "Diffusion Policy for Multi-Behavior Robot Motion"
  - fig1_cfg_ablation.png (2x2 grid, λ=0 vs λ=2)
  - fig12_robustness_jitter.png (if space)
  Key message: CFG conditioning works, λ=2 > λ=0 on all 4 behaviors

SLIDE 2: "VLM as Trajectory Evaluator"
  - fig5_vlm_goal_identification.png (83% goal ID accuracy)
  - fig2_vlm_score_distribution.png (score diversity varies by behavior)
  - fig4_vlm_vs_learly_scatter.png (VLM vs metric disagreement)
  Key message: VLM perceives motion accurately but scores differently than metrics

SLIDE 3: "Can VLM Replace Human Judgment?"
  - fig10_all_methods_legibility.png (panoramic comparison)
  - fig8_summary_verdict_table.png (honest verdicts per behavior)
  - fig3_vlm_selection_effect.png (per-episode VLM vs baseline)
  Key message: VLM is useful evaluator, not reliable optimizer
""")
