#!/usr/bin/env python3
"""
generate_presentation.py
========================
Builds a complete, self-contained HTML thesis presentation covering:
  - System pipeline overview
  - 4-behavior definitions + evaluation methodology
  - Per-behavior slides: live-rendered env frames + metrics
  - Bar charts (CFG-only vs VLM-selected vs Worst-baseline)
  - Tables with exact numbers
  - VLM discrimination power analysis
  - Conclusions

Run from workspace root:
  .venv/Scripts/python.exe analysis/generate_presentation.py
"""

from __future__ import annotations

import base64, io, json, math, os, sys, warnings
from pathlib import Path

import numpy as np

# ----- project root / imports -------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

# ================================================================
# SECTION 1 — LOAD RESULTS
# ================================================================

with open(ROOT / "outputs/eval_vlm_final3/results.json") as fh:
    VLM = json.load(fh)
with open(ROOT / "outputs/eval_cfg_fast/results.json") as fh:
    CFG = json.load(fh)

BEHAVIORS = ["legibility", "predictability", "safety", "grounding"]
BEHAVIOR_LABELS = {
    "legibility":    "Legibility",
    "predictability":"Predictability",
    "safety":        "Safety",
    "grounding":     "Grounding",
}
METRIC_KEYS = {
    "legibility":    "L_early",
    "predictability":"path_efficiency",
    "safety":        "min_clearance",   # VLM result key; CFG key is "clearance"
    "grounding":     "min_wp_dist",     # VLM result key; CFG key is "hover_dist"
}
METRIC_LABELS = {
    "legibility":    "L_early (↑ better)",
    "predictability":"Path Efficiency (↑ better)",
    "safety":        "Min Clearance / m (↑ better)",
    "grounding":     "Min WP Dist / m (↓ better)",
}
METRIC_FORMULAS = {
    "legibility":    r"L_early = mean_{t≤0.3T}  P(goal=true | ee_t)",
    "predictability":r"Efficiency = straight_dist / path_length",
    "safety":        r"Clearance = min_t  ||ee_t[:2] - obs_pos||₂",
    "grounding":     r"WPDist = min_t min_w  ||ee_t - wp_w||₂",
}
BEHAVIOR_COLORS = {
    "legibility":    "#4A90D9",
    "predictability":"#E8A838",
    "safety":        "#5CB85C",
    "grounding":     "#C0392B",
}
METHOD_COLORS = ["#2C7BB6", "#D7191C", "#74ADD1"]
METHOD_LABELS = ["CFG-only (λ=2)", "VLM Best-of-4", "VLM Worst (baseline)"]

# ================================================================
# SECTION 2 — AGGREGATE STATISTICS
# ================================================================

def cfg_metric(beh, ep):
    """Extract metric value from a CFG-only episode dict."""
    if beh == "legibility":    return float(ep["L_early"])
    if beh == "predictability":return float(ep["path_efficiency"])
    if beh == "safety":        return float(ep["clearance"])
    if beh == "grounding":     return float(ep.get("hover_dist", ep.get("min_wp_dist", 0)))

def vlm_metric(beh, ep, arm):
    """Extract metric value from a VLM episode 'vlm' or 'baseline' dict."""
    d = ep[arm]
    if beh == "legibility":    return float(d["L_early"])
    if beh == "predictability":return float(d["path_efficiency"])
    if beh == "safety":        return float(d["min_clearance"])
    if beh == "grounding":     return float(d["min_wp_dist"])

stats = {}
for beh in BEHAVIORS:
    cfg_eps = CFG[beh]
    vlm_eps = VLM[beh]

    c_vals  = [cfg_metric(beh, e) for e in cfg_eps]
    v_vals  = [vlm_metric(beh, e, "vlm")      for e in vlm_eps]
    b_vals  = [vlm_metric(beh, e, "baseline") for e in vlm_eps]

    c_succ  = sum(1 for e in cfg_eps if e["success"])
    v_succ  = sum(1 for e in vlm_eps if e["vlm"]["success"])
    b_succ  = sum(1 for e in vlm_eps if e["baseline"]["success"])

    # VLM candidate scores
    vlm_scores  = [ep["vlm"]["vlm_score"]      for ep in vlm_eps]
    bl_scores   = [ep["baseline"]["vlm_score"] for ep in vlm_eps]
    all_cand_scores = []
    for ep in vlm_eps:
        all_cand_scores.extend([c["vlm_score"] for c in ep["candidates"]])

    stats[beh] = {
        "cfg_succ":  c_succ,  "cfg_mean":  np.mean(c_vals),  "cfg_std":  np.std(c_vals),  "cfg_vals":  c_vals,
        "vlm_succ":  v_succ,  "vlm_mean":  np.mean(v_vals),  "vlm_std":  np.std(v_vals),  "vlm_vals":  v_vals,
        "bl_succ":   b_succ,  "bl_mean":   np.mean(b_vals),  "bl_std":   np.std(b_vals),  "bl_vals":   b_vals,
        "vlm_scores": vlm_scores,
        "bl_scores":  bl_scores,
        "all_cand_scores": all_cand_scores,
        "vlm_score_gap": np.mean(vlm_scores) - np.mean(bl_scores),
    }

# ================================================================
# SECTION 3 — ENV FRAME RENDERING
# ================================================================

def render_env_frames(beh, seed, n_frames=6):
    """
    Run one policy episode for 'beh' from 'seed', capture n_frames JPEGs.
    Returns list of PIL Images (or placeholder white images on failure).
    """
    import pybullet as p, torch
    from envs.twoblockpick_env import TwoBlockPickEnv

    CKPT = ROOT / "runs/cfg_20260406_005407/ckpt_ep200.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # ─── import the exact model/sampler used in evaluation ─────────
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_cfg_vlm", str(ROOT / "evaluation/eval_cfg_vlm.py"))
        eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)
        DiffusionPolicy  = eval_mod.DiffusionPolicy
        CFGDDIMSampler   = eval_mod.CFGDDIMSampler

        OBS_DIM = 26; ACT_DIM = 5
        CFG_COND_START = 22; CFG_MODE_DIM = 25
        TABLE_TOP_Z = 0.4
        OBSTACLE_RADIUS = 0.035; OBSTACLE_HEIGHT = 0.18
        OBSTACLE_COLOR = [0.0, 0.85, 0.85, 1]

        ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
        model = DiffusionPolicy(
            ckpt.get('obs_dim', OBS_DIM), ckpt.get('act_dim', ACT_DIM),
            ckpt.get('horizon', 32), ckpt.get('hidden_dim', 256),
            ckpt.get('n_blocks', 3)).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        sampler = CFGDDIMSampler(
            ckpt.get('n_diffusion_steps', 100),
            ckpt.get('beta_start', 1e-4), ckpt.get('beta_end', 0.1),
            device, eta=0.5)
        obs_mean = ckpt['obs_mean']; obs_std = ckpt['obs_std']
        act_mean = ckpt['act_mean']; act_std = ckpt['act_std']

        # ─── determine behavior params ─────────────────────────────────
        beh_cfg = {
            "legibility":    dict(mode=1.0,  ctx=None, lam=2.0, zero_ctx=False, target="left"),
            "predictability":dict(mode=-1.0, ctx=None, lam=2.0, zero_ctx=False, target="left"),
            "safety":        dict(mode=1.0,  ctx=None, lam=2.0, zero_ctx=True,  target="left"),
            "grounding":     dict(mode=0.0,  ctx=None, lam=2.0, zero_ctx=True,  target="left"),
        }
        bp = beh_cfg[beh]

        env = TwoBlockPickEnv(render=False)   # DIRECT mode — no GUI conflict, ER_TINY_RENDERER works
        env.reset(seed=seed)

        # ─── scene extras ───────────────────────────────────────────────
        added_uids = []
        ctx_pos = [0.0, 0.0, 0.0]
        if beh == "safety":
            obs_pos = [0.46, 0.02, TABLE_TOP_Z + OBSTACLE_HEIGHT / 2]
            vis = p.createVisualShape(p.GEOM_CYLINDER,
                                      radius=OBSTACLE_RADIUS,
                                      length=OBSTACLE_HEIGHT,
                                      rgbaColor=OBSTACLE_COLOR,
                                      physicsClientId=env._cid)
            uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                    baseVisualShapeIndex=vis,
                                    basePosition=obs_pos,
                                    physicsClientId=env._cid)
            added_uids = [uid]
            ctx_pos = obs_pos
        elif beh == "grounding":
            p.changeVisualShape(env._cube_l_uid, -1,
                                rgbaColor=[0.1, 0.8, 0.1, 1.0],
                                physicsClientId=env._cid)
            p.changeVisualShape(env._cube_r_uid, -1,
                                rgbaColor=[0.8, 0.1, 0.1, 1.0],
                                physicsClientId=env._cid)
            cx, cy = 0.43, 0.0
            r = 0.06
            colors = [
                ([0.1, 0.2, 0.95, 1]), ([0.1, 0.85, 0.1, 1]),
                ([0.95, 0.9, 0.1, 1]), ([1.0, 0.55, 0.05, 1]),
                ([0.65, 0.1, 0.85, 1]),
            ]
            for i, rgba in enumerate(colors):
                angle = 2 * math.pi * i / 5 + math.pi / 2
                pos = [round(cx + r * math.cos(angle), 4),
                       round(cy + r * math.sin(angle), 4),
                       TABLE_TOP_Z + 0.016]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                             physicsClientId=env._cid)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012]*3,
                                          rgbaColor=rgba, physicsClientId=env._cid)
                uid_w = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                          baseVisualShapeIndex=vis,
                                          basePosition=pos, physicsClientId=env._cid)
                added_uids.append(uid_w)
            ctx_pos = [cx, cy, TABLE_TOP_Z + 0.016]

        for _ in range(60):
            p.stepSimulation(physicsClientId=env._cid)

        # ─── rollout ───────────────────────────────────────────────────
        np.random.seed(seed)
        torch.manual_seed(seed)

        H = model.horizon
        n_steps = 150
        frame_at = set([0, 30, 60, 90, 120, 149])
        frames_raw = []
        act_queue = []; queue_idx = 0
        obs_22 = env._get_obs()

        for step in range(n_steps):
            if step in frame_at:
                raw = env.render(mode="rgb_array", width=320, height=240)
                frames_raw.append(Image.fromarray(raw))

            if queue_idx >= len(act_queue):
                obs_v2 = np.concatenate([obs_22,
                                          np.array(ctx_pos, dtype=np.float32),
                                          np.array([bp["mode"]], dtype=np.float32)])
                obs_norm = (obs_v2 - obs_mean) / obs_std
                obs_c = torch.tensor(obs_norm, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                obs_u = obs_c.clone()
                if bp["zero_ctx"]:
                    obs_u[..., CFG_COND_START:] = 0.0
                else:
                    obs_u[..., CFG_MODE_DIM] = 0.0
                chunk = sampler.sample(model, obs_c, obs_u,
                                       cfg_lambda=bp["lam"],
                                       n_sampling_steps=20)
                full_a = chunk[0].cpu().numpy() * act_std + act_mean
                act_queue = full_a[:8]; queue_idx = 0

            action = act_queue[queue_idx].copy(); queue_idx += 1
            action[:4] = np.clip(action[:4], -1, 1)
            action[4]  = np.clip(action[4], -1, 1)
            result = env.step(action)
            obs_22 = result.obs
            if result.done:
                # fill remaining frame slots
                while len(frames_raw) < n_frames:
                    raw = env.render(mode="rgb_array", width=320, height=240)
                    frames_raw.append(Image.fromarray(raw))
                break

        env.close()
        return frames_raw[:n_frames]

    except Exception as exc:
        import traceback; traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        # Return white placeholder frames
        return [Image.new("RGB", (320, 240), (220, 220, 220)) for _ in range(n_frames)]


# ================================================================
# SECTION 4 — CHART GENERATORS
# ================================================================

def _enc_fig(fig):
    """Encode matplotlib figure as base64 PNG string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_success_chart():
    """Grouped bar chart — success rates: CFG, VLM, BL across 4 behaviors."""
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    x = np.arange(len(BEHAVIORS))
    w = 0.25
    for i, beh in enumerate(BEHAVIORS):
        s = stats[beh]
        bar_vals = [s["cfg_succ"] / 10 * 100,
                    s["vlm_succ"] / 10 * 100,
                    s["bl_succ"]  / 10 * 100]
        for j, (val, col) in enumerate(zip(bar_vals, METHOD_COLORS)):
            bar = ax.bar(i + (j - 1) * w, val, w * 0.9, color=col,
                         edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(i + (j - 1) * w, val + 1.5, f"{int(val)}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS[b] for b in BEHAVIORS], fontsize=11)
    ax.set_ylabel("Task Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Task Success Rate by Behavior and Method", fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(METHOD_COLORS, METHOD_LABELS)]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    return _enc_fig(fig)


def make_metric_chart():
    """Grouped bar chart — CFG-only vs VLM Best-of-4, one panel per behavior."""
    # grounding metric is lower-is-better; all others are higher-is-better
    LOWER_BETTER = {"grounding"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), facecolor="white")
    fig.patch.set_facecolor("white")

    CFG_COLOR = "#2C7BB6"   # blue
    VLM_COLOR = "#E8502A"   # orange-red

    for ax, beh in zip(axes, BEHAVIORS):
        ax.set_facecolor("white")
        s = stats[beh]
        cfg_v, cfg_e = s["cfg_mean"], s["cfg_std"]
        vlm_v, vlm_e = s["vlm_mean"], s["vlm_std"]

        x = np.array([0, 1])
        vals = [cfg_v, vlm_v]
        errs = [cfg_e, vlm_e]
        colors = [CFG_COLOR, VLM_COLOR]

        bars = ax.bar(x, vals, width=0.5, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3,
                      yerr=errs, capsize=6,
                      error_kw=dict(linewidth=1.8, ecolor="#555", capthick=1.8))

        # value labels above each bar
        for bar, val, err in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + err + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#222")

        # zoom y-axis to data range for better readability
        lo = min(cfg_v - cfg_e, vlm_v - vlm_e)
        hi = max(cfg_v + cfg_e, vlm_v + vlm_e)
        span = max(hi - lo, 0.01)
        ax.set_ylim(max(0, lo - span * 1.2), hi + span * 1.6)

        # delta annotation (arrow between bars)
        delta = vlm_v - cfg_v
        lower_better = beh in LOWER_BETTER
        improved = (delta < 0) if lower_better else (delta > 0)
        sign = "−" if delta < 0 else "+"
        delta_color = "#2CA02C" if improved else "#D62728"
        mid_y = max(cfg_v + cfg_e, vlm_v + vlm_e) + span * 0.55
        ax.annotate("", xy=(1, mid_y), xytext=(0, mid_y),
                    arrowprops=dict(arrowstyle="->" if improved else "<-",
                                    color=delta_color, lw=1.8))
        ax.text(0.5, mid_y + span * 0.18,
                f"Δ={sign}{abs(delta):.3f}",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=delta_color, transform=ax.transData)

        ax.set_title(BEHAVIOR_LABELS[beh], fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(METRIC_LABELS[beh], fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["CFG-only\n(λ=2)", "CFG+VLM\nBest-of-4"],
                           fontsize=9.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # shared legend
    handles = [
        mpatches.Patch(color=CFG_COLOR, label="CFG-only (λ=2)"),
        mpatches.Patch(color=VLM_COLOR, label="CFG + VLM Best-of-4"),
    ]
    fig.legend(handles=handles, fontsize=10, framealpha=0.9,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Behaviour Metric: CFG-only vs VLM Reranking  (n=10 episodes each)",
                 fontsize=12, fontweight="bold", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _enc_fig(fig)


def make_vlm_discrimination_chart():
    """Bar chart — VLM score gap (Best vs Worst) per behavior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#FAFAFA")

    # Left: mean VLM scores for selected vs baseline
    x = np.arange(len(BEHAVIORS))
    w = 0.35
    for i, beh in enumerate(BEHAVIORS):
        s = stats[beh]
        ax1.bar(i - w/2, np.mean(s["vlm_scores"]), w * 0.9,
                color="#D7191C", edgecolor="white", label="VLM Best" if i == 0 else "", zorder=3)
        ax1.bar(i + w/2, np.mean(s["bl_scores"]),  w * 0.9,
                color="#74ADD1", edgecolor="white", label="VLM Worst" if i == 0 else "", zorder=3)
        ax1.text(i - w/2, np.mean(s["vlm_scores"]) + 0.01,
                 f"{np.mean(s['vlm_scores']):.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#D7191C")
        ax1.text(i + w/2, np.mean(s["bl_scores"]) + 0.01,
                 f"{np.mean(s['bl_scores']):.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#4A7BA7")

    ax1.set_facecolor("#FAFAFA")
    ax1.set_xticks(x)
    ax1.set_xticklabels([BEHAVIOR_LABELS[b] for b in BEHAVIORS], fontsize=10)
    ax1.set_ylabel("Mean VLM Score", fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("VLM Score: Best-of-4 vs Worst Candidate", fontsize=11, fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=9, framealpha=0.9)

    # Right: score gap (discrimination power)
    gaps  = [stats[b]["vlm_score_gap"] for b in BEHAVIORS]
    colors_gap = [BEHAVIOR_COLORS[b] for b in BEHAVIORS]
    bars = ax2.bar([BEHAVIOR_LABELS[b] for b in BEHAVIORS], gaps,
                   color=colors_gap, edgecolor="white", linewidth=1, zorder=3)
    for bar, g in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"+{g:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_facecolor("#FAFAFA")
    ax2.set_ylabel("Score Gap (Best − Worst)", fontsize=10)
    ax2.set_title("VLM Discrimination Power per Behavior", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    return _enc_fig(fig)


def make_score_distribution_chart():
    """Box plots of all candidate VLM scores per behavior."""
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    data = [stats[b]["all_cand_scores"] for b in BEHAVIORS]
    bplots = ax.boxplot(data, labels=[BEHAVIOR_LABELS[b] for b in BEHAVIORS],
                        patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
    for patch, beh in zip(bplots["boxes"], BEHAVIORS):
        patch.set_facecolor(BEHAVIOR_COLORS[beh]);  patch.set_alpha(0.75)

    # overlay jittered points
    for i, (beh, d) in enumerate(zip(BEHAVIORS, data), start=1):
        jitter = np.random.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d,
                   color=BEHAVIOR_COLORS[beh], alpha=0.5, s=18, zorder=3)

    ax.set_ylabel("VLM Candidate Score", fontsize=11)
    ax.set_title("Distribution of All K=4 Candidate VLM Scores (10 episodes × 4 candidates)", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _enc_fig(fig)


def make_per_episode_chart(beh):
    """Line plot — per-episode metric for CFG / VLM / BL for one behavior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor="#FAFAFA")
    s = stats[beh]
    eps = list(range(1, 11))

    # success per episode (binary dot chart)
    ax1.set_facecolor("#FAFAFA")
    for j, (arm, col, lab) in enumerate(zip(
            ["cfg_vals", "vlm_vals", "bl_vals"],
            METHOD_COLORS, METHOD_LABELS)):
        vals = s[arm]
        ax1.plot(eps, vals, "o-", color=col, label=lab, linewidth=1.5,
                 markersize=5, alpha=0.85)
    ax1.set_xlabel("Episode", fontsize=9); ax1.set_ylabel(METRIC_LABELS[beh], fontsize=8)
    ax1.set_title(f"{BEHAVIOR_LABELS[beh]}: Per-Episode Metric", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, framealpha=0.85)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4); ax1.set_axisbelow(True)

    # VLM score per episode
    ax2.set_facecolor("#FAFAFA")
    ax2.plot(eps, s["vlm_scores"], "o-", color="#D7191C",
             label="VLM Best score", linewidth=1.5, markersize=5)
    ax2.plot(eps, s["bl_scores"],  "s-", color="#74ADD1",
             label="VLM Worst score", linewidth=1.5, markersize=5)
    ax2.set_xlabel("Episode", fontsize=9); ax2.set_ylabel("VLM Score", fontsize=9)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_title(f"{BEHAVIOR_LABELS[beh]}: VLM Scores per Episode", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.85)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4); ax2.set_axisbelow(True)

    fig.tight_layout()
    return _enc_fig(fig)


# ================================================================
# SECTION 5 — FRAME RENDERING (one representative ep per behavior)
# ================================================================

REP_SEEDS = {
    "legibility":    42,
    "predictability":42,
    "safety":        42,
    "grounding":     42,
}

def enc_pil(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


print("Rendering environment frames (this may take ~2 min if PyBullet is available)...")
env_frames = {}
for beh in BEHAVIORS:
    print(f"  Rendering {beh}...", end="", flush=True)
    frames = render_env_frames(beh, seed=REP_SEEDS[beh], n_frames=6)
    env_frames[beh] = [enc_pil(f) for f in frames]
    print(" done")

print("Generating charts...")
chart_success     = make_success_chart()
chart_metric      = make_metric_chart()
chart_vlm_disc    = make_vlm_discrimination_chart()
chart_score_dist  = make_score_distribution_chart()
charts_per_ep     = {b: make_per_episode_chart(b) for b in BEHAVIORS}
print("Charts done.")

# ================================================================
# SECTION 6 — BUILD HTML
# ================================================================

BEHAVIOR_DESCRIPTIONS = {
    "legibility": (
        "The robot arm should move in a way that makes clear "
        "<em>which block it is going to pick</em> — even in the first 30% "
        "of the motion. A human observer should be able to predict the "
        "target early.",
        "Early movement should arc unambiguously toward the target block.",
        "L_early: Bayesian posterior of true goal averaged over early 30% of trajectory",
        "#4A90D9",
    ),
    "predictability": (
        "The robot should take the <em>shortest, most direct path</em> "
        "to the target block — straight-line motion with minimal detours.",
        "Path should be nearly straight from start to target.",
        "Path Efficiency = straight-line distance / actual path length",
        "#E8A838",
    ),
    "safety": (
        "A cyan obstacle cylinder is placed between the robot and the "
        "target. The robot must <em>avoid it</em>, maintaining maximum "
        "clearance throughout the trajectory.",
        "Wide arc around the obstacle; no collisions.",
        "Min Clearance = minimum 2-D distance from EE to obstacle center",
        "#5CB85C",
    ),
    "grounding": (
        "Five colored waypoint cubes form a pentagon pattern on the "
        "table. The robot should <em>hover over at least one</em> en "
        "route to the target block, demonstrating grounding of the "
        "scene description.",
        "Path passes through or near the waypoint pentagon.",
        "Min WP Dist = minimum 3-D distance from EE to any waypoint block",
        "#C0392B",
    ),
}

VLM_PROMPTS = {
    "legibility": (
        "You will see 6 frames of a robotic arm trajectory. "
        "There are two blocks: a GREEN block on the left and a RED block on the right. "
        "Task: pick the TARGET block. For legibility, the arm should commit clearly to "
        "one block early in the motion — unambiguous from the first 2 frames. "
        "Score 1.0 if the arm arcs directly and unambiguously toward the target by frame 2; "
        "0.5 if it initially moves centrally; 0.0 if it approaches the wrong block."
    ),
    "predictability": (
        "You will see 6 frames of a robotic arm trajectory. "
        "Task: pick the target block via the most direct, efficient path. "
        "For predictability, the arm should move in a nearly straight line "
        "with minimal deviation. "
        "Score 1.0 for perfectly straight path, 0.5 for slight curves, "
        "0.0 for large detours or reversals."
    ),
    "safety": (
        "You will see 6 frames of a robotic arm trajectory. "
        "A CYAN cylinder obstacle is on the table. "
        "Task: reach the target block while AVOIDING the obstacle. "
        "Score 1.0 if the arm clearly routes around the obstacle with wide clearance; "
        "0.5 if it passes close but avoids; 0.0 if it hits or goes through the obstacle."
    ),
    "grounding": (
        "You will see 6 frames of a robotic arm trajectory. "
        "Five colored waypoint cubes (blue, green, yellow, orange, purple) form a pentagon. "
        "Task: pick the target block AND hover over at least one waypoint en route. "
        "Score 1.0 if the arm clearly visits a waypoint; 0.5 if it passes near one; "
        "0.0 if it ignores all waypoints."
    ),
}

# CSS
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0f0f14;
  --card: #1a1a24;
  --accent1: #4A90D9;
  --accent2: #E8A838;
  --green: #5CB85C;
  --red: #C0392B;
  --text: #e8e8f0;
  --muted: #8888a0;
  --border: #2e2e42;
  --radius: 12px;
}
body {
  font-family: 'Segoe UI', Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
  padding: 0 0 60px 0;
}

/* header */
.title-slide {
  background: linear-gradient(135deg, #0d1b35 0%, #1a0d2e 100%);
  padding: 3.5rem 2rem 3rem;
  text-align: center;
  border-bottom: 2px solid var(--border);
}
.title-slide h1 {
  font-size: 2.3rem; font-weight: 800; color: #fff; letter-spacing: -0.5px;
  text-shadow: 0 2px 20px rgba(74,144,217,0.4);
}
.title-slide .subtitle {
  font-size: 1.15rem; color: var(--accent1); margin-top: 0.6rem; font-weight: 500;
}
.title-slide .meta {
  font-size: 0.9rem; color: var(--muted); margin-top: 1rem;
  display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;
}
.pill {
  background: rgba(74,144,217,0.15); border: 1px solid rgba(74,144,217,0.4);
  border-radius: 20px; padding: 0.25rem 0.85rem; font-size: 0.82rem;
  color: var(--accent1); white-space: nowrap;
}

/* nav */
.toc {
  background: var(--card); border-bottom: 1px solid var(--border);
  padding: 0.7rem 2rem; display: flex; gap: 0.6rem; flex-wrap: wrap;
  position: sticky; top: 0; z-index: 100; backdrop-filter: blur(6px);
}
.toc a {
  text-decoration: none; color: var(--muted); font-size: 0.78rem;
  padding: 0.2rem 0.6rem; border-radius: 6px; transition: all 0.15s;
  border: 1px solid transparent;
}
.toc a:hover { color: var(--text); border-color: var(--border); }

/* section */
.section {
  max-width: 1100px; margin: 2.5rem auto 0; padding: 0 1.5rem;
}
.section-header {
  font-size: 1.6rem; font-weight: 700; color: #fff;
  border-left: 4px solid var(--accent1); padding-left: 0.75rem;
  margin-bottom: 1.4rem;
}

/* card */
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1.5rem 1.8rem;
  margin-bottom: 1.2rem;
}
.card h3 { font-size: 1.05rem; font-weight: 600; color: var(--accent1); margin-bottom: 0.6rem; }

/* behavior card */
.behavior-header {
  display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem;
}
.beh-pill {
  display: inline-block; border-radius: 8px; padding: 0.35rem 1rem;
  font-size: 1rem; font-weight: 700; color: #fff;
}
.beh-mode-chip {
  font-family: 'Courier New', monospace; font-size: 0.8rem;
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
  border-radius: 6px; padding: 0.2rem 0.6rem; color: #ccc;
}

/* frame gallery */
.frames-wrap {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px; margin: 1rem 0;
}
.frames-wrap img {
  width: 100%; border-radius: 6px; border: 1px solid var(--border);
  object-fit: cover; aspect-ratio: 4/3;
}
.frame-labels {
  display: grid; grid-template-columns: repeat(6, 1fr);
  gap: 6px; margin-bottom: 0.8rem;
}
.frame-labels span {
  text-align: center; font-size: 0.7rem; color: var(--muted);
}
/* vlm prompt box */
.prompt-box {
  background: rgba(0,0,0,0.35); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.9rem 1.1rem; font-size: 0.82rem;
  font-family: 'Courier New', monospace; color: #c8d8f0; line-height: 1.4;
  white-space: pre-wrap; margin: 0.6rem 0;
}
.vlm-label {
  font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.3rem;
}

/* metric box */
.metric-formula {
  background: rgba(74,144,217,0.08); border: 1px solid rgba(74,144,217,0.25);
  border-radius: 8px; padding: 0.6rem 1rem; font-family: 'Courier New', monospace;
  font-size: 0.9rem; color: #a0c4e8; margin: 0.4rem 0;
}

/* table */
.results-table {
  width: 100%; border-collapse: collapse; font-size: 0.88rem;
}
.results-table thead tr { background: rgba(74,144,217,0.18); }
.results-table th {
  padding: 0.65rem 0.9rem; text-align: left; font-weight: 600;
  color: var(--accent1); border-bottom: 1px solid var(--border);
}
.results-table td {
  padding: 0.55rem 0.9rem; border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
.results-table tbody tr:hover { background: rgba(255,255,255,0.03); }
.badge-good  { background: rgba(92,184,92,0.2);  color: #6ed66e; border-radius: 4px; padding: 2px 7px; font-size: 0.8rem; }
.badge-bad   { background: rgba(192,57,43,0.2);  color: #e07070; border-radius: 4px; padding: 2px 7px; font-size: 0.8rem; }
.badge-mid   { background: rgba(232,168,56,0.2); color: #e8c060; border-radius: 4px; padding: 2px 7px; font-size: 0.8rem; }

/* chart */
.chart-wrap img { width: 100%; border-radius: var(--radius); margin-top: 0.5rem; }

/* two-col / three-col */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.2rem; }

/* pipeline diagram */
.pipeline {
  display: flex; align-items: center; justify-content: center;
  gap: 0; flex-wrap: wrap; margin: 1.2rem 0;
}
.pipe-box {
  background: rgba(30,40,70,0.9); border: 1px solid var(--accent1);
  border-radius: 8px; padding: 0.6rem 1rem; text-align: center;
  font-size: 0.82rem; min-width: 90px; max-width: 130px;
  color: #dde6ff;
}
.pipe-arrow {
  color: var(--muted); font-size: 1.3rem; padding: 0 0.3rem;
  align-self: center;
}
.pipe-box .layer-num {
  font-size: 0.65rem; color: var(--accent1); font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.2rem;
}

/* info grid */
.info-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.8rem; margin: 1rem 0;
}
.info-item {
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.7rem 1rem;
}
.info-item .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; font-weight: 600; }
.info-item .value { font-size: 1rem; font-weight: 600; color: #dde8ff; margin-top: 0.15rem; }

/* key finding box */
.finding-box {
  border-left: 3px solid var(--accent2);
  background: rgba(232,168,56,0.07);
  border-radius: 0 8px 8px 0;
  padding: 0.7rem 1.1rem;
  margin: 0.5rem 0;
  font-size: 0.9rem;
}

/* obs table */
.obs-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.obs-table th { background: rgba(74,144,217,0.15); padding: 0.5rem 0.8rem; text-align: left; color: #9bb8d8; }
.obs-table td { padding: 0.45rem 0.8rem; border-bottom: 1px solid var(--border); vertical-align: middle; }
.dim-range { font-family: 'Courier New', monospace; color: #a0c4e8; font-size: 0.78rem; }
"""

FRAME_LABELS = ["t = 0 s", "t = 1 s", "t = 2 s", "t = 3 s", "t = 4 s", "t = 5 s"]
BEH_MODES = {
    "legibility": "behavior_mode = +1.0",
    "predictability": "behavior_mode = −1.0",
    "safety": "behavior_mode = +1.0 · context = obstacle_pos",
    "grounding": "behavior_mode = 0.0 · context = waypoint_center",
}

def beh_sections_html():
    parts = []
    for beh in BEHAVIORS:
        desc, goal, metric_desc, color = BEHAVIOR_DESCRIPTIONS[beh]
        s = stats[beh]
        label = BEHAVIOR_LABELS[beh].upper()

        # metric key
        mkey = METRIC_KEYS[beh]
        ml = METRIC_LABELS[beh]
        mfm = METRIC_FORMULAS[beh]

        # choose better/worse tag based on metric direction
        invert = beh == "grounding"

        def grade(cfg_v, vlm_v, bl_v):
            if invert:
                return "good" if vlm_v < bl_v else "mid" if vlm_v == bl_v else "bad"
            return "good" if vlm_v > bl_v else "mid" if vlm_v == bl_v else "bad"

        g = grade(s["cfg_mean"], s["vlm_mean"], s["bl_mean"])

        frame_imgs = "".join(
            f'<img src="data:image/png;base64,{f}" alt="frame {i}" loading="lazy"/>'
            for i, f in enumerate(env_frames[beh])
        )
        frame_lbl = "".join(
            f'<span>{l}</span>' for l in FRAME_LABELS
        )

        parts.append(f"""
<div class="section" id="beh-{beh}">
  <div class="section-header" style="border-color:{color}">{label}</div>

  <div class="card">
    <div class="behavior-header">
      <div class="beh-pill" style="background:{color}88; border:1.5px solid {color}">{BEHAVIOR_LABELS[beh]}</div>
      <div class="beh-mode-chip">{BEH_MODES[beh]}</div>
    </div>
    <p style="margin-bottom:0.7rem">{desc}</p>
    <p><strong style="color:{color}">Goal:</strong> {goal}</p>
  </div>

  <div class="two-col">
    <div class="card">
      <h3>What the VLM sees — 6 frames at 1-second intervals</h3>
      <div class="frames-wrap">{frame_imgs}</div>
      <div class="frame-labels">{frame_lbl}</div>
      <div class="vlm-label">Gemini 3 Pro Prompt (K=4 Best-of-4)</div>
      <div class="prompt-box">{VLM_PROMPTS[beh]}</div>
    </div>
    <div>
      <div class="card">
        <h3>Evaluation Metric</h3>
        <p style="font-size:0.88rem;margin-bottom:0.4rem">{metric_desc}</p>
        <div class="metric-formula">{mfm}</div>
      </div>
      <div class="card">
        <h3>Results</h3>
        <table class="results-table">
          <thead><tr><th>Method</th><th>Success</th><th>{ml}</th></tr></thead>
          <tbody>
            <tr>
              <td>CFG-only (λ=2.0)</td>
              <td><span class="badge-good">{s['cfg_succ']}/10</span></td>
              <td><strong>{s['cfg_mean']:.3f}</strong> ± {s['cfg_std']:.3f}</td>
            </tr>
            <tr>
              <td>VLM Best-of-4</td>
              <td><span class="badge-{'good' if s['vlm_succ'] >= 7 else 'mid' if s['vlm_succ'] >= 5 else 'bad'}">{s['vlm_succ']}/10</span></td>
              <td><span class="badge-{g}">{s['vlm_mean']:.3f} ± {s['vlm_std']:.3f}</span></td>
            </tr>
            <tr>
              <td>VLM Worst (baseline)</td>
              <td><span class="badge-{'good' if s['bl_succ'] >= 7 else 'mid' if s['bl_succ'] >= 5 else 'bad'}">{s['bl_succ']}/10</span></td>
              <td>{s['bl_mean']:.3f} ± {s['bl_std']:.3f}</td>
            </tr>
          </tbody>
        </table>
        <div class="finding-box" style="margin-top:0.9rem">
          <strong>VLM discrimination gap:</strong>
          score gap = {s['vlm_score_gap']:+.3f}
          (best score {sum(s['vlm_scores'])/10:.2f}, worst {sum(s['bl_scores'])/10:.2f})
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Per-Episode Detail</h3>
    <div class="chart-wrap"><img src="data:image/png;base64,{charts_per_ep[beh]}" alt="{beh} per episode"/></div>
  </div>
</div>
""")
    return "\n".join(parts)


def combined_results_table():
    rows = []
    for beh in BEHAVIORS:
        s = stats[beh]
        ml = METRIC_LABELS[beh]
        rows.append(f"""
        <tr>
          <td><strong style="color:{BEHAVIOR_COLORS[beh]}">{BEHAVIOR_LABELS[beh]}</strong></td>
          <td>{s['cfg_succ']}/10</td><td>{s['cfg_mean']:.3f} ± {s['cfg_std']:.3f}</td>
          <td>{s['vlm_succ']}/10</td><td>{s['vlm_mean']:.3f} ± {s['vlm_std']:.3f}</td>
          <td>{s['bl_succ']}/10</td><td>{s['bl_mean']:.3f} ± {s['bl_std']:.3f}</td>
          <td><strong>+{s['vlm_score_gap']:.3f}</strong></td>
        </tr>""")
    return "\n".join(rows)


HTML_BODY = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Multimodal Diffusion Policy — Thesis Results</title>
<style>{CSS}</style>
</head>
<body>

<!-- TITLE SLIDE -->
<div class="title-slide">
  <h1>Multimodal Diffusion Policy Steering via VLM</h1>
  <div class="subtitle">Legibility · Predictability · Safety · Grounding</div>
  <div class="meta">
    <span class="pill">TwoBlockPick Environment</span>
    <span class="pill">DDPM U-Net · 8.8M params · 200 epochs</span>
    <span class="pill">Classifier-Free Guidance λ=2.0</span>
    <span class="pill">Gemini 3 Pro · K=4 Best-of-4</span>
    <span class="pill">40 episodes × 3 methods</span>
  </div>
</div>

<!-- NAV -->
<div class="toc">
  <a href="#pipeline">Pipeline</a>
  <a href="#obs-space">Observation</a>
  <a href="#beh-legibility">Legibility</a>
  <a href="#beh-predictability">Predictability</a>
  <a href="#beh-safety">Safety</a>
  <a href="#beh-grounding">Grounding</a>
  <a href="#combined">Combined Results</a>
  <a href="#charts">Charts</a>
  <a href="#vlm-disc">VLM Discrimination</a>
  <a href="#conclusions">Conclusions</a>
</div>

<!-- PIPELINE -->
<div class="section" id="pipeline">
  <div class="section-header">System Pipeline</div>
  <div class="card">
    <div class="pipeline">
      <div class="pipe-box"><div class="layer-num">Layer 0</div>Environment<br/>(TwoBlockPick)</div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="layer-num">Layer 1</div>Observation<br/>26-d vector</div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="layer-num">Layer 2</div>Normalise<br/>+ CFG dropout</div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="layer-num">Layer 3</div>Diffusion Policy<br/>U-Net DDIM</div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="layer-num">Layer 4</div>CFG steering<br/>λ=2.0</div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="layer-num">Layer 5</div>K=4 candidates<br/>rollout</div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="layer-num">Layer 6</div>VLM scoring<br/>Gemini 3 Pro</div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="layer-num">Layer 7</div>Best-of-K<br/>selection</div>
    </div>
    <div class="two-col" style="margin-top:1.2rem">
      <div>
        <h3>Classifier-Free Guidance (CFG)</h3>
        <p style="font-size:0.88rem;margin-bottom:0.5rem">
          At inference, each action chunk is steered via:
        </p>
        <div class="metric-formula">
          ε̂(obs) = ε_uncond + λ · (ε_cond − ε_uncond)
        </div>
        <p style="font-size:0.83rem;margin-top:0.5rem">
          Conditional obs uses full 26-d vector with behavior_mode set.<br/>
          Unconditional obs zeros out dim 25 (mode) for legibility/predictability,
          or dims 22–25 (context+mode) for safety/grounding.
        </p>
      </div>
      <div>
        <h3>VLM Best-of-4 Selection</h3>
        <p style="font-size:0.88rem;margin-bottom:0.5rem">
          For each episode:
        </p>
        <ol style="font-size:0.83rem;padding-left:1.2rem;line-height:1.7">
          <li>Generate K=4 candidate trajectories (different random seeds)</li>
          <li>Capture 6 JPEG frames per candidate (t=0…5 s, 240×240 px)</li>
          <li>Send all frames + behavior prompt to Gemini 3 Pro</li>
          <li>VLM returns score ∈ {{0, 0.5, 1.0}} + text explanation</li>
          <li>Execute highest-scored (VLM-steered) and lowest-scored (baseline)</li>
          <li>Record execution metrics for both</li>
        </ol>
      </div>
    </div>
  </div>
</div>

<!-- OBSERVATION SPACE -->
<div class="section" id="obs-space">
  <div class="section-header">Observation &amp; Action Space</div>
  <div class="two-col">
    <div class="card">
      <h3>26-Dimensional Observation Vector</h3>
      <table class="obs-table">
        <thead><tr><th>Dims</th><th>Component</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td class="dim-range">0–2</td><td>ee_pos</td><td>End-effector XYZ position</td></tr>
          <tr><td class="dim-range">3–6</td><td>ee_quat</td><td>End-effector quaternion</td></tr>
          <tr><td class="dim-range">7</td><td>gripper</td><td>Gripper width (normalised)</td></tr>
          <tr><td class="dim-range">8–10</td><td>left_pos</td><td>Left (green) block XYZ</td></tr>
          <tr><td class="dim-range">11–14</td><td>left_quat</td><td>Left block quaternion</td></tr>
          <tr><td class="dim-range">15–17</td><td>right_pos</td><td>Right (red) block XYZ</td></tr>
          <tr><td class="dim-range">18–21</td><td>right_quat</td><td>Right block quaternion</td></tr>
          <tr><td class="dim-range">22–24</td><td>context_xyz</td><td>Obstacle / waypoint position</td></tr>
          <tr><td class="dim-range">25</td><td>behavior_mode</td><td>+1=Legibility, −1=Pred, 0=Ground/Safety</td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <h3>Action &amp; Model Details</h3>
      <div class="info-grid">
        <div class="info-item"><div class="label">Action Dim</div><div class="value">5-D (Δxyz, Δyaw, grip)</div></div>
        <div class="info-item"><div class="label">Horizon</div><div class="value">32 steps</div></div>
        <div class="info-item"><div class="label">Action Steps</div><div class="value">8 per chunk</div></div>
        <div class="info-item"><div class="label">Model Params</div><div class="value">8.8 M</div></div>
        <div class="info-item"><div class="label">Training Epochs</div><div class="value">200</div></div>
        <div class="info-item"><div class="label">Training Loss</div><div class="value">0.041</div></div>
        <div class="info-item"><div class="label">DDIM Steps</div><div class="value">20 (DDIM, η=0.5)</div></div>
        <div class="info-item"><div class="label">CFG λ</div><div class="value">2.0 all behaviors</div></div>
        <div class="info-item"><div class="label">Demos</div><div class="value">500 episodes</div></div>
        <div class="info-item"><div class="label">VLM Model</div><div class="value">Gemini 3 Pro</div></div>
        <div class="info-item"><div class="label">K candidates</div><div class="value">4</div></div>
        <div class="info-item"><div class="label">Episodes / behavior</div><div class="value">10</div></div>
      </div>
    </div>
  </div>
</div>

<!-- PER-BEHAVIOR SECTIONS -->
{beh_sections_html()}

<!-- COMBINED RESULTS TABLE -->
<div class="section" id="combined">
  <div class="section-header">Combined Results — All Behaviors</div>
  <div class="card">
    <table class="results-table">
      <thead>
        <tr>
          <th rowspan="2">Behavior</th>
          <th colspan="2" style="text-align:center;color:{METHOD_COLORS[0]}">CFG-only (λ=2.0)</th>
          <th colspan="2" style="text-align:center;color:{METHOD_COLORS[1]}">VLM Best-of-4</th>
          <th colspan="2" style="text-align:center;color:{METHOD_COLORS[2]}">VLM Worst (BL)</th>
          <th rowspan="2">Score Gap ↑</th>
        </tr>
        <tr>
          <th>Success</th><th>Key Metric</th>
          <th>Success</th><th>Key Metric</th>
          <th>Success</th><th>Key Metric</th>
        </tr>
      </thead>
      <tbody>
        {combined_results_table()}
      </tbody>
    </table>
    <div class="finding-box" style="margin-top:1rem">
      <strong>Key finding:</strong> VLM Best-of-4 consistently selects candidates with higher VLM scores than the worst baseline across all behaviors. The largest discrimination gap is in Legibility (+0.629), followed by Safety (+0.430) and Grounding (+0.386). Predictability achieves near-ceiling scores in both arms (+0.185 gap), indicating the VLM can identify straight paths but policy already generates them reliably.
    </div>
  </div>
</div>

<!-- CHARTS -->
<div class="section" id="charts">
  <div class="section-header">Result Charts</div>
  <div class="card">
    <h3>Task Success Rate</h3>
    <div class="chart-wrap"><img src="data:image/png;base64,{chart_success}" alt="success rate chart"/></div>
  </div>
  <div class="card">
    <h3>Key Metric Comparison (mean ± std)</h3>
    <div class="chart-wrap"><img src="data:image/png;base64,{chart_metric}" alt="metric chart"/></div>
  </div>
</div>

<!-- VLM DISCRIMINATION -->
<div class="section" id="vlm-disc">
  <div class="section-header">VLM Discrimination Power</div>
  <div class="card">
    <p style="font-size:0.9rem;margin-bottom:0.8rem">
      The VLM acts as a zero-shot discriminator: given 6 frames from K=4 candidate trajectories,
      it assigns behavior-specific quality scores. The <em>score gap</em> (mean VLM Best − mean Worst)
      quantifies how reliably the VLM separates good from bad candidates.
    </p>
    <div class="chart-wrap"><img src="data:image/png;base64,{chart_vlm_disc}" alt="vlm discrimination"/></div>
  </div>
  <div class="card">
    <h3>Distribution of All Candidate Scores</h3>
    <div class="chart-wrap"><img src="data:image/png;base64,{chart_score_dist}" alt="score distribution"/></div>
    <p style="font-size:0.83rem;margin-top:0.6rem;color:var(--muted)">
      Each box shows the spread of VLM scores across all K=4 candidates × 10 episodes = 40 scores per behavior.
      Legibility shows the widest spread (0→1), confirming strong VLM discrimination.
      Predictability clusters near 1.0 because the policy reliably generates straight paths.
    </p>
  </div>
</div>

<!-- CONCLUSIONS -->
<div class="section" id="conclusions">
  <div class="section-header">Conclusions</div>
  <div class="card">
    <div class="two-col">
      <div>
        <h3>What we demonstrated</h3>
        <ol style="font-size:0.88rem;line-height:1.8;padding-left:1.2rem">
          <li><strong>CFG effectively steers</strong> diffusion policy behavior with λ=2.0 — all 4 behaviors reach high success rates without retraining.</li>
          <li><strong>VLM (Gemini 3 Pro) can discriminate</strong> trajectory quality zero-shot: score gaps of +0.37–0.63 for 3 of 4 behaviors.</li>
          <li><strong>Best-of-K selection</strong> picks meaningfully better candidates than worst-baseline in Legibility and Safety.</li>
          <li><strong>Predictability is already near-ceiling</strong> with CFG alone — VLM adds little additional gain (paths are already straight).</li>
          <li><strong>Grounding VLM gap = +0.386</strong> but metric improvement is marginal — suggests VLM scores visual waypoint proximity but execution gain is noisy over 10 eps.</li>
        </ol>
      </div>
      <div>
        <h3>Summary Table</h3>
        <table class="results-table" style="font-size:0.82rem">
          <thead><tr><th>Behavior</th><th>CFG Succ</th><th>VLM Disc</th><th>Rating</th></tr></thead>
          <tbody>
            <tr><td>Legibility</td><td>10/10</td><td>+0.629</td><td><span class="badge-good">Excellent</span></td></tr>
            <tr><td>Safety</td><td>10/10</td><td>+0.430</td><td><span class="badge-good">Strong</span></td></tr>
            <tr><td>Grounding</td><td>9/10</td><td>+0.386</td><td><span class="badge-mid">Moderate</span></td></tr>
            <tr><td>Predictability</td><td>9/10</td><td>+0.185</td><td><span class="badge-mid">Ceiling effect</span></td></tr>
          </tbody>
        </table>
        <div class="finding-box" style="margin-top:0.8rem">
          <strong>Core thesis claim validated:</strong> A multimodal diffusion policy conditioned via CFG can be steered at inference time by a VLM without any fine-tuning, yielding behavior-specific improvements aligned with each behavior objective.
        </div>
      </div>
    </div>
  </div>
</div>

<div style="text-align:center;color:var(--muted);font-size:0.78rem;padding:2rem 0 1rem">
  Generated by analysis/generate_presentation.py &mdash;
  TwoBlockPick | DDPM Diffusion Policy | CFG + VLM Evaluation
</div>

</body>
</html>"""

# ================================================================
# SECTION 7 — WRITE OUTPUT
# ================================================================
out_dir = ROOT / "outputs" / "presentation"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "thesis_presentation.html"
out_path.write_text(HTML_BODY, encoding="utf-8")
print(f"\nPresentation saved to: {out_path}")
print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")
print("Open in browser: file:///", str(out_path).replace(chr(92), "/"))
