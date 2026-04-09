"""
Clean paper-quality architecture figures for all models.
Design rules:
  - All text is dark on light fills — always readable
  - Generous box sizes so labels never clip
  - Pure left-to-right pipeline flow for each figure
  - Arrows have explicit direction labels where helpful
  - Skip connections are dashed; data flow is solid
  - 150 dpi, white background, DejaVu Sans

Run:
    python figures/generate_arch_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'figure.dpi': 150,
})

# ── Colour palette ────────────────────────────────────────────────────
C_IN    = '#D6E4F0'   # light blue  — inputs
C_EMBED = '#D5E8D4'   # light green — embeddings / time
C_UNET  = '#DAE8FC'   # medium blue — UNet enc/dec blocks
C_NECK  = '#F8CECC'   # rose pink   — bottleneck
C_OUT   = '#FFF2CC'   # pale yellow — output
C_GOAL  = '#E1D5E7'   # lavender    — goal/cond token
C_MODE  = '#FFE6CC'   # pale orange — behaviour mode / CFG
C_VLM   = '#E8F5E9'   # mint        — VLM
C_BORDER = '#444444'
C_ARROW  = '#222222'
C_TEXT   = '#111111'
C_SUB    = '#555555'


# ═══════════════════════════════════════════════════════════════════
# Drawing primitives
# ═══════════════════════════════════════════════════════════════════

def box(ax, cx, cy, w, h, label, sub='', color=C_IN,
        fs=11, sfs=8.5, bold=True):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0,rounding_size=0.07',
                       linewidth=1.3, edgecolor=C_BORDER,
                       facecolor=color, zorder=3)
    ax.add_patch(r)
    ldy = 0.10 if sub else 0.0
    ax.text(cx, cy + ldy, label,
            ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal',
            color=C_TEXT, zorder=4)
    if sub:
        ax.text(cx, cy - 0.16, sub,
                ha='center', va='center', fontsize=sfs,
                color=C_SUB, style='italic', zorder=4)


def arr(ax, x0, y0, x1, y1, label='', lw=1.6, color=C_ARROW,
        dash=False, tip=12):
    ls = 'dashed' if dash else 'solid'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=tip, linestyle=ls),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        offx = 0.0
        offy = 0.08
        ax.text(mx + offx, my + offy, label,
                ha='center', va='bottom',
                fontsize=7.5, color='#888888', zorder=6)


def legend(ax, items, loc='lower left', ncol=1):
    handles = [mpatches.Patch(facecolor=c, edgecolor=C_BORDER, label=l)
               for c, l in items]
    ax.legend(handles=handles, loc=loc, fontsize=8,
              framealpha=0.95, edgecolor='#cccccc', ncol=ncol)


def info_box(ax, cx, cy, text, fc='#f8f8f8', ec='#bbbbbb', fs=8):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.45', facecolor=fc,
                      edgecolor=ec, linewidth=0.9))


# ═══════════════════════════════════════════════════════════════════
# FIG 1  Vanilla Diffusion Policy
# ═══════════════════════════════════════════════════════════════════
# Layout (left → right):
#  Column A: three raw inputs (stacked vertically)
#  Column B: embedding modules (one per input)
#  Column C: U-Net encoder (top branch) + time/obs conditioning (add)
#  Column D: Bottleneck
#  Column E: U-Net decoder
#  Column F: Output MLP
#
# Key measurements chosen so boxes absolutely do not overlap.

def fig1_vanilla():
    FW, FH = 16, 6
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle('Vanilla Diffusion Policy  (Chi et al., 2023)',
                 fontsize=14, fontweight='bold', y=0.98, color=C_TEXT)

    # ── Box geometry ────────────────────────────────────────────────
    BH = 0.80   # box height
    BW = 2.20   # box width  (most boxes)
    EW = 2.00   # embedding width

    # Column x-centres
    xA = 1.3      # inputs
    xB = 3.8      # embeddings
    xC_act = 5.8  # action projected (before +)
    xCplus = 6.6  # + add
    xD1 = 8.0     # Enc 1
    xD2 = 10.0    # Enc 2
    xE  = 12.0    # Bottleneck
    xF1 = 10.0    # Dec 1  (same x as Enc2 but different y)
    xF2 = 8.0     # Dec 2
    xG  = 6.3     # Output MLP

    # Row y-centres
    yAct  = 4.8   # noisy action row
    yObs  = 3.4   # observation row
    yTime = 2.0   # timestep row

    yUnet = 4.0   # U-Net main row

    # ── Inputs ──────────────────────────────────────────────────────
    box(ax, xA, yAct,  BW, BH, 'Noisy Action',  r'a(k) in R^{H x 5}',  C_IN)
    box(ax, xA, yObs,  BW, BH, 'Observation',   r'o_t in R^{22}',       C_IN)
    box(ax, xA, yTime, BW, BH, 'Timestep  k',   '{0, 1, ..., 99}',      C_IN)

    # ── Embeddings ──────────────────────────────────────────────────
    box(ax, xB, yAct,  EW, BH, 'Linear Proj',        'act_dim -> 256',  C_UNET)
    box(ax, xB, yObs,  EW, BH, 'Obs MLP',            '22 -> 256 -> 256',C_EMBED)
    box(ax, xB, yTime, EW, BH, 'Sinusoidal Emb.',    '128 -> 256',      C_EMBED)

    # ── Additive fusion (+) ─────────────────────────────────────────
    # We represent fusion as a small circle labelled "+"
    r = 0.25
    circ = plt.Circle((xCplus, yUnet), r,
                       facecolor='white', edgecolor=C_BORDER,
                       linewidth=1.4, zorder=4)
    ax.add_patch(circ)
    ax.text(xCplus, yUnet, '+', ha='center', va='center',
            fontsize=15, fontweight='bold', color=C_TEXT, zorder=5)

    # ── U-Net ────────────────────────────────────────────────────────
    UW = 2.0
    yUhigh = 4.6  # encoder row
    yUlow  = 3.2  # decoder row

    box(ax, xD1, yUhigh, UW, BH, 'Enc Block 1', '256 -> 512',   C_UNET)
    box(ax, xD2, yUhigh, UW, BH, 'Enc Block 2', '512 -> 1024',  C_UNET)
    box(ax, xE,  yUhigh, UW, BH, 'Bottleneck',  '1024 -> 1024', C_NECK)
    box(ax, xF1, yUlow,  UW, BH, 'Dec Block 1', '2048 -> 512',  C_UNET)
    box(ax, xF2, yUlow,  UW, BH, 'Dec Block 2', '1024 -> 256',  C_UNET)
    box(ax, xG,  yUlow,  UW, BH, 'Output MLP',  '256 -> 5',     C_OUT)

    # ── Arrows: inputs -> embeddings ─────────────────────────────────
    arr(ax, xA + BW/2, yAct,  xB - EW/2, yAct)
    arr(ax, xA + BW/2, yObs,  xB - EW/2, yObs)
    arr(ax, xA + BW/2, yTime, xB - EW/2, yTime)

    # ── Arrows: embeddings -> + ──────────────────────────────────────
    arr(ax, xB + EW/2, yAct,  xCplus - r, yUnet)   # action proj -> +
    arr(ax, xB + EW/2, yObs,  xCplus - r*0.7, yUnet - r*0.7)   # obs -> +
    arr(ax, xB + EW/2, yTime, xCplus - r*0.5, yUnet - r)        # time -> +

    # ── Arrow: + -> Enc1 ─────────────────────────────────────────────
    arr(ax, xCplus + r, yUnet, xD1 - UW/2, yUhigh)

    # ── U-Net forward pass ───────────────────────────────────────────
    arr(ax, xD1 + UW/2, yUhigh, xD2 - UW/2, yUhigh)
    arr(ax, xD2 + UW/2, yUhigh, xE  - UW/2, yUhigh)
    # bottleneck down to decoder
    arr(ax, xE,  yUhigh - BH/2, xF1, yUlow + BH/2)

    arr(ax, xF1 - UW/2, yUlow, xF2 + UW/2, yUlow)
    arr(ax, xF2 - UW/2, yUlow, xG  + UW/2, yUlow)

    # ── Skip connections (dashed) ─────────────────────────────────────
    # Enc1 skip -> Dec2
    ax.annotate('', xy=(xF2, yUlow + BH/2 + 0.05), xytext=(xD1, yUhigh - BH/2 - 0.05),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1,
                                linestyle='dashed', mutation_scale=10,
                                connectionstyle='arc3,rad=0.25'), zorder=2)
    ax.text((xD1+xF2)/2 + 0.2, (yUhigh+yUlow)/2 - 0.25, 'skip',
            fontsize=8, color='#999999', ha='center')

    # Enc2 skip -> Dec1
    ax.annotate('', xy=(xF1, yUlow + BH/2 + 0.05), xytext=(xD2, yUhigh - BH/2 - 0.05),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1,
                                linestyle='dashed', mutation_scale=10), zorder=2)
    ax.text((xD2+xF1)/2 + 0.25, (yUhigh+yUlow)/2, 'skip',
            fontsize=8, color='#999999', ha='center')

    # Obs + Time emb also condition all UNet blocks (shown as dotted line)
    for xb in [xD1, xD2, xE, xF1, xF2]:
        ax.plot([xB + EW/2, xb], [yObs, yUhigh - BH/2 - 0.05],
                color='#cccccc', lw=0.8, linestyle='dotted', zorder=1)
    ax.text(xB + 1.0, yObs - 0.55, 'time + obs cond. (all blocks)',
            fontsize=7.5, color='#aaaaaa', style='italic', ha='center')

    # ── Output label — isolated bottom row, directly below Output MLP ─────
    # xPN=6.3 (same x as Output MLP), yPN=1.7  →  x=[5.3,7.3] clears all emb boxes ✓
    # Sinusoidal Emb at x=[2.8,4.8]: gap=0.5 ✓;  Timestep at x=[0.3,2.4]: gap=2.9 ✓
    box(ax, 6.3, 1.7, 1.6, BH, 'Pred. Noise', 'eps in R^{H x 5}', C_OUT)
    # Arrow: straight down from Output MLP bottom to Pred. Noise top
    arr(ax, xG, yUlow - BH/2, 6.3, 1.7 + BH/2)

    # ── Training / inference info ─────────────────────────────────────
    info_box(ax, FW/2, 0.9,
             'Training: DDPM (T=100, beta_0=1e-4, beta_T=0.1)  |  '
             'Loss: MSE(eps_pred, eps)  |  '
             'Inference: DDIM (10 steps, eta=0.3)',
             fc='#f4f8ff', ec='#9999cc')

    # ── Legend ────────────────────────────────────────────────────────
    legend(ax, [(C_IN, 'Input tensor'),
                (C_EMBED, 'Embedding (time / obs)'),
                (C_UNET, 'U-Net block'),
                (C_NECK, 'Bottleneck'),
                (C_OUT, 'Output projection')], loc='lower right')

    fig.savefig('figures/arch_fig1_vanilla_diffusion.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: figures/arch_fig1_vanilla_diffusion.png')


# ═══════════════════════════════════════════════════════════════════
# FIG 2  Legibility Diffuser  (Bronars et al., RA-L 2024)
# ═══════════════════════════════════════════════════════════════════
# Architecture differences from vanilla:
#   1) Goal embedding: Embedding(3, 256) — {left, right, null}
#   2) 1D Conv backbone (Conv1d kernel=5, channels-first)
#   3) FiLM conditioning: cond = time + obs + goal, injected via additive bias
#   4) CFG at inference: eps = eps_uncond + w*(eps_cond - eps_uncond)

def fig2_legdiff():
    # Layout  (all conflicts verified below)
    # FW=22, FH=8.8  — content fills yAct=7.8 to info boxes at y=0.85
    #
    # Inputs:    xIn=1.3, xEmb=3.8
    #   yAct=7.8, yObs=6.7, yTime=5.6, yGoal=4.5  (1.1 apart)
    #
    # FiLM Cond Sum:  xCond=6.4, yCond=(5.6+4.5)/2=5.05, CW=2.2, CH=1.0
    #   FiLM x-range [5.3, 7.5];  y-range [4.55, 5.55]
    #
    # U-Net:  xE1=8.8, xE2=11.2, xN=13.6;  xD1=11.2, xD2=8.8  (UW=2.1)
    #   yEnc=7.8  enc y-range [7.4, 8.2]   (same y as yAct — different x, fine)
    #   yDec=6.4  dec y-range [6.0, 6.8]
    #     → enc bottom (7.4) > dec top (6.8): gap=0.6  ✓
    #     → FiLM top (5.55) < dec bottom (6.0): gap=0.45  ✓
    #     → Dec Block 2 left edge (7.75) > FiLM right (7.5): gap=0.25  ✓
    #
    # Output row (isolated, below all embeds and FiLM):
    #   xOP=8.5, yOP=3.0   OP y-range [2.6, 3.4]
    #     → Goal embed bottom (4.1) > OP top (3.4): gap=0.7  ✓
    #     → FiLM bottom (4.55) > OP top (3.4): gap=1.15  ✓
    #   xNoise=5.0, yNoise=3.0   PN x-range [3.9, 6.1]
    #     → PN right (6.1) < OP left (7.45): gap=1.35  ✓
    FW, FH = 17, 8.5
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(
        'Legibility Diffuser  (Bronars et al., RA-L 2024)  —  Goal-Conditioned CFG',
        fontsize=14, fontweight='bold', y=0.99, color=C_TEXT)

    BH = 0.80; BW = 2.2; EW = 2.0; UW = 2.1; CW = 2.2; CH = 1.0

    yAct = 7.8;  yObs = 6.7;  yTime = 5.6;  yGoal = 4.5
    xIn  = 1.3;  xEmb = 3.8

    xCond = 6.4;  yCond = (yTime + yGoal) / 2   # = 5.05
    # FiLM x-range [5.3, 7.5];  y-range [4.55, 5.55]

    # UNet columns: xD2/xE1=9.0 left edge=7.95 > FiLM right=7.5: gap=0.45 ✓
    # Spacing xE2-xE1=2.6 (=UW+0.5 gap) → adequate room for arrows
    xE1 = 9.0;  xE2 = 11.6;  xN = 14.2
    xD1 = 11.6; xD2 = 9.0
    yEnc = 7.8;  yDec = 6.4

    xOP    = 9.0;  yOP    = 3.0
    xNoise = 5.0;  yNoise = 3.0

    # ── Inputs ──────────────────────────────────────────────────────
    box(ax, xIn, yAct,  BW, BH, 'Noisy Action',  r'a(k) in R^{H x 5}', C_IN)
    box(ax, xIn, yObs,  BW, BH, 'Observation',   r'o in R^{22}',        C_IN)
    box(ax, xIn, yTime, BW, BH, 'Timestep  k',   '{0, ..., 99}',        C_IN)
    box(ax, xIn, yGoal, BW, BH, 'Goal Label  g', '{left, right, null}', C_GOAL)

    # ── Embeddings ──────────────────────────────────────────────────
    box(ax, xEmb, yAct,  EW, BH, 'Conv1d Proj',     'act_dim -> 256',    C_UNET)
    box(ax, xEmb, yObs,  EW, BH, 'Obs MLP',         '22 -> 256 -> 256',  C_EMBED)
    box(ax, xEmb, yTime, EW, BH, 'Sinusoidal Emb.', '128 -> 256',        C_EMBED)
    box(ax, xEmb, yGoal, EW, BH, 'Goal Embedding',  'Embedding(3, 256)', C_GOAL)

    # ── FiLM Cond Sum box ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (xCond - CW/2, yCond - CH/2), CW, CH,
        boxstyle='round,pad=0,rounding_size=0.07',
        linewidth=1.3, edgecolor=C_BORDER, facecolor=C_GOAL, zorder=3))
    ax.text(xCond, yCond + 0.15, 'FiLM Cond Sum',
            ha='center', va='center', fontsize=10, fontweight='bold', color=C_TEXT, zorder=4)
    ax.text(xCond, yCond - 0.20, 'time + obs + goal → 256-d',
            ha='center', va='center', fontsize=7.5, color=C_SUB, style='italic', zorder=4)

    # ── Arrows: inputs -> embeddings ─────────────────────────────────
    for yi in [yAct, yObs, yTime, yGoal]:
        arr(ax, xIn + BW/2, yi, xEmb - EW/2, yi)

    # ── Arrows: obs/time/goal embeddings -> FiLM Cond Sum ────────────
    arr(ax, xEmb + EW/2, yObs,  xCond - CW/2, yCond + 0.22)
    arr(ax, xEmb + EW/2, yTime, xCond - CW/2, yCond)
    arr(ax, xEmb + EW/2, yGoal, xCond - CW/2, yCond - 0.22)

    # ── Arrow: action conv1d proj -> Enc Block 1 (horizontal) ────────
    arr(ax, xEmb + EW/2, yAct, xE1 - UW/2, yEnc)

    # ── Conv1d U-Net blocks ──────────────────────────────────────────
    box(ax, xE1, yEnc, UW, BH, 'Enc Block 1', 'Conv1d ResBlock\n256 -> 256',  C_UNET)
    box(ax, xE2, yEnc, UW, BH, 'Enc Block 2', 'Conv1d ResBlock\n256 -> 512',  C_UNET)
    box(ax, xN,  yEnc, UW, BH, 'Bottleneck',  'Conv1d ResBlock\n512 -> 512',  C_NECK)
    box(ax, xD1, yDec, UW, BH, 'Dec Block 1', 'Conv1d ResBlock\n1024 -> 512', C_UNET)
    box(ax, xD2, yDec, UW, BH, 'Dec Block 2', 'Conv1d ResBlock\n512 -> 256',  C_UNET)
    box(ax, xOP, yOP,  UW, BH, 'Output Proj', 'GN + Mish\nConv1d -> act_dim', C_OUT)

    # ── U-Net forward flow ───────────────────────────────────────────
    arr(ax, xE1 + UW/2, yEnc, xE2 - UW/2, yEnc)
    arr(ax, xE2 + UW/2, yEnc, xN  - UW/2, yEnc)
    arr(ax, xN,  yEnc - BH/2, xD1, yDec + BH/2)   # bottleneck -> dec1
    arr(ax, xD1 - UW/2, yDec, xD2 + UW/2, yDec)   # dec1 -> dec2
    # dec2 -> output proj (straight down: dec2 bottom to OP top)
    arr(ax, xD2, yDec - BH/2, xOP, yOP + BH/2)

    # ── Skip connections ─────────────────────────────────────────────
    ax.annotate('', xy=(xD2, yDec + BH/2 + 0.08),
                xytext=(xE1, yEnc - BH/2 - 0.08),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1,
                                linestyle='dashed', mutation_scale=10,
                                connectionstyle='arc3,rad=0.2'), zorder=2)
    ax.text((xE1+xD2)/2 - 0.6, (yEnc+yDec)/2 + 0.15, 'skip', fontsize=8, color='#999999')

    ax.annotate('', xy=(xD1, yDec + BH/2 + 0.08),
                xytext=(xE2, yEnc - BH/2 - 0.08),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1,
                                linestyle='dashed', mutation_scale=10), zorder=2)
    ax.text((xE2+xD1)/2 + 0.3, (yEnc+yDec)/2 + 0.08, 'skip', fontsize=8, color='#999999')

    # ── FiLM cond -> all ResBlocks (dotted pale arrows) ───────────────
    for xb, yb in [(xE1, yEnc), (xE2, yEnc), (xN, yEnc),
                   (xD1, yDec), (xD2, yDec)]:
        ax.annotate('', xy=(xb, yb - BH/2 - 0.04), xytext=(xCond + CW/2, yCond + 0.1),
                    arrowprops=dict(arrowstyle='->', color='#bbbbbb', lw=0.8,
                                    linestyle='dotted', mutation_scale=8), zorder=1)
    ax.text(xCond + CW/2 + 0.2, yCond - 0.65,
            'FiLM cond injected\ninto all ResBlocks',
            fontsize=7.5, color='#aaaaaa', style='italic', ha='left')

    # ── Predicted noise output (isolated bottom row) ───────────────────
    box(ax, xNoise, yNoise, 2.2, BH, 'Pred. Noise', 'eps in R^{H x 5}', C_OUT)
    # Output Proj -> Pred. Noise (left-going horizontal arrow)
    arr(ax, xOP - UW/2, yOP, xNoise + 1.1 + 0.05, yNoise)

    # ── Training / inference notes ────────────────────────────────────
    info_box(ax, FW/2, 1.7,
             'Training: 85% goal-conditioned  +  15% unconditional (null goal token)'
             '  |  Loss: MSE(eps_pred, eps)',
             fc='#f8f8f8', ec='#cccccc')

    info_box(ax, FW/2, 0.85,
             'Inference (CFG):  eps_hat = eps_uncond + w * (eps_cond - eps_uncond)'
             '     w = 3.0  ->  amplifies goal commitment  ->  higher early legibility',
             fc='#fef9f0', ec='#e0c080', fs=9)

    # ── Legend ─────────────────────────────────────────────────────────
    legend(ax, [(C_IN,    'Input'),
                (C_GOAL,  'Goal condition / FiLM vector'),
                (C_EMBED, 'Time / obs embedding'),
                (C_UNET,  'Conv1d ResBlock'),
                (C_NECK,  'Bottleneck'),
                (C_OUT,   'Output')],
           loc='lower left', ncol=2)

    fig.savefig('figures/arch_fig2_legdiff.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: figures/arch_fig2_legdiff.png')


# ═══════════════════════════════════════════════════════════════════
# FIG 3  CFG Multimodal Diffusion Policy (ours)
# ═══════════════════════════════════════════════════════════════════
# Key differences from vanilla:
#   1) obs_dim = 26  (22 base + 3 context_pos + 1 behaviour_mode)
#   2) CFG dropout: during training zeros indices [22:26] with p=0.15
#   3) Same MLP-based U-Net as vanilla (not Conv1d)
#   4) lambda = 2.0 at inference

def fig3_cfg():
    # Layout (verified non-overlapping):
    #   Figure 20 x 8.5
    #   5 inputs stacked left at x=1.5: y = 7.2, 6.1, 5.0, 3.7, 2.6
    #     (base obs, context pos, behaviour mode, noisy action, timestep)
    #   CFG Dropout box at x=4.3, y=5.55 — between ctx (6.1) and mode (5.0) midpoint
    #     height=1.3 => top=6.2, bot=4.9 — just below ctx embed at 6.1+0.4=6.5 ✓
    #     Wait: ctx input is at y=6.1, box top=6.1+BH/2=6.5; dropout top=5.55+0.65=6.2
    #     So dropout box overlaps ctx input bounding box. Shift dropout to x=4.3 (right of inputs x_max=1.5+1.2=2.7)
    #     But xEmb is at 7.0, so range 2.7->7.0 is clear. Place dropout at x=4.7.
    #   Obs MLP at x=7.0, y=5.7 (receives base obs + dropout vector)
    #   Linear Proj (action) at x=7.0, y=3.7
    #   Sinusoidal Emb. (timestep) at x=7.0, y=2.6
    #   Plus "+ " circle at x=8.9, y=4.15 (between ObsMLP=5.7 and Sinusoidal=2.6)
    #   U-Net encoder row at y=6.5, x: 10.5, 13.0, 15.5
    #   U-Net decoder row at y=4.0, x: 13.0, 10.5
    #   Output MLP at x=8.5, y=4.0
    #   Pred. Noise at x=6.5, y=4.0
    FW, FH = 20, 9.0
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(
        'CFG Multimodal Diffusion Policy  (ours)  —  Behaviour-Conditioned  [obs=26-d]',
        fontsize=14, fontweight='bold', y=0.99, color=C_TEXT)

    BH = 0.80
    BW = 2.4
    EW = 2.2
    UW = 2.2
    C_WARN = '#FFF0F0'

    # Input y positions (spaced 1.1 apart)
    xObs  = 1.5
    yBase = 7.5
    yCtx  = 6.4
    yMode = 5.3
    yAct  = 4.0
    yTime = 2.9

    box(ax, xObs, yBase, BW, BH, 'Base Observation', '22-d: EE pos/vel, cube poses', C_IN)
    box(ax, xObs, yCtx,  BW, BH, 'Context Position',  '3-d: target cube (x,y,z)',    C_MODE)
    box(ax, xObs, yMode, BW, BH, 'Behaviour Mode',
        '1-d: {legib, pred, safe, grnd}',                                              C_MODE)
    box(ax, xObs, yAct,  BW, BH, 'Noisy Action', r'a(k) in R^{32 x 5}',              C_IN)
    box(ax, xObs, yTime, BW, BH, 'Timestep  k',  '{0, ..., 99}',                      C_IN)

    # Brace: ctx+mode -> "obs_26d"
    y_brace_top = yCtx + BH/2
    y_brace_bot = yMode - BH/2
    xbrace = xObs + BW/2 + 0.18
    ax.annotate('', xy=(xbrace, y_brace_bot), xytext=(xbrace, y_brace_top),
                arrowprops=dict(arrowstyle='<->', color=C_BORDER, lw=1.2))
    ax.text(xbrace + 0.22, (y_brace_top + y_brace_bot)/2,
            'concat\nobs_26d', fontsize=8.5, color=C_SUB, va='center', style='italic')

    # CFG Dropout box: x=4.7, y=(yCtx+yMode)/2 = 5.85; height=1.4 => top=6.55, bot=5.15
    # ctx input top = yCtx + BH/2 = 6.4+0.4 = 6.8 — dropout top 6.55 < 6.8 OK (no horizontal overlap since xDrop=4.7 > xObs + BW/2 = 2.7)
    xDrop = 4.8
    yDrop = (yCtx + yMode) / 2   # 5.85
    DW, DH = 2.2, 1.4
    r_drop = FancyBboxPatch((xDrop - DW/2, yDrop - DH/2), DW, DH,
                             boxstyle='round,pad=0,rounding_size=0.07',
                             linewidth=1.3, edgecolor='#cc6666', facecolor=C_WARN, zorder=3)
    ax.add_patch(r_drop)
    ax.text(xDrop, yDrop + 0.25, 'CFG Dropout', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#aa2222', zorder=4)
    ax.text(xDrop, yDrop + 0.00, '(training only)', ha='center', va='center',
            fontsize=8.5, color='#aa2222', zorder=4)
    ax.text(xDrop, yDrop - 0.25, 'p=0.15  zero dims [22:26]', ha='center', va='center',
            fontsize=8, color=C_SUB, style='italic', zorder=4)
    ax.text(xDrop, yDrop + DH/2 + 0.12, '15% of training steps',
            fontsize=7.5, color='#cc4444', ha='center', style='italic')

    # Arrows: ctx+mode -> dropout
    arr(ax, xObs + BW/2, yCtx,  xDrop - DW/2, yDrop + 0.28)
    arr(ax, xObs + BW/2, yMode, xDrop - DW/2, yDrop - 0.28)

    # Embedding column
    xEmb = 7.5
    yObsMlp = 5.85   # Obs MLP (receives base obs + 26d concat)
    box(ax, xEmb, yObsMlp, EW, BH, 'Obs MLP', '26 -> 256 -> 256', C_EMBED)
    box(ax, xEmb, yAct,    EW, BH, 'Linear Proj', 'act_dim -> 256', C_UNET)
    box(ax, xEmb, yTime,   EW, BH, 'Sinusoidal Emb.', '128 -> 256', C_EMBED)

    # Base obs -> ObsMLP  (top route)
    arr(ax, xObs + BW/2, yBase, xEmb - EW/2, yObsMlp)
    # Dropout -> ObsMLP
    arr(ax, xDrop + DW/2, yDrop, xEmb - EW/2, yObsMlp)
    ax.text((xDrop + xEmb)/2 + 0.2, yDrop - 0.1, 'obs_26d',
            fontsize=7.5, color=C_SUB, ha='center', style='italic')
    # Action, Timestep
    arr(ax, xObs + BW/2, yAct,  xEmb - EW/2, yAct)
    arr(ax, xObs + BW/2, yTime, xEmb - EW/2, yTime)

    # Plus (+) circle: y shifted to y=5.3 so bottom (5.05) clears Output MLP top (4.9)
    xPlus = 9.9
    yPlus = 5.3
    r = 0.25
    circ = plt.Circle((xPlus, yPlus), r,
                       facecolor='white', edgecolor=C_BORDER,
                       linewidth=1.4, zorder=4)
    ax.add_patch(circ)
    ax.text(xPlus, yPlus, '+', ha='center', va='center',
            fontsize=15, fontweight='bold', color=C_TEXT, zorder=5)
    ax.text(xPlus + 0.35, yPlus - 0.65, 'obs_emb\n+ time_emb',
            fontsize=7.5, color=C_SUB, ha='left')

    arr(ax, xEmb + EW/2, yObsMlp, xPlus - r*0.7, yPlus + r*0.7)
    arr(ax, xEmb + EW/2, yAct,    xPlus - r,      yPlus)
    arr(ax, xEmb + EW/2, yTime,   xPlus - r*0.7,  yPlus - r*0.7)

    # U-Net (MLP-based same as vanilla)
    xU1 = 11.8
    xU2 = 14.3
    xUB = 16.8
    xBD1 = 14.3
    xBD2 = 11.8
    yEncR = 6.8   # encoder row
    yDecR = 4.5   # decoder row (well above yPlus+r=5.175)

    box(ax, xU1, yEncR, UW, BH, 'Enc Block 1', '256 -> 512',   C_UNET)
    box(ax, xU2, yEncR, UW, BH, 'Enc Block 2', '512 -> 1024',  C_UNET)
    box(ax, xUB, yEncR, UW, BH, 'Bottleneck',  '1024 -> 1024', C_NECK)
    box(ax, xBD1, yDecR, UW, BH, 'Dec Block 1', '2048 -> 512', C_UNET)
    box(ax, xBD2, yDecR, UW, BH, 'Dec Block 2', '1024 -> 256', C_UNET)

    # Plus -> Enc1
    arr(ax, xPlus + r, yPlus, xU1 - UW/2, yEncR)

    # Encoder chain
    arr(ax, xU1 + UW/2, yEncR, xU2 - UW/2, yEncR)
    arr(ax, xU2 + UW/2, yEncR, xUB - UW/2, yEncR)
    # Bottleneck down to Dec1
    arr(ax, xUB, yEncR - BH/2, xBD1, yDecR + BH/2)
    # Decoder chain (left)
    arr(ax, xBD1 - UW/2, yDecR, xBD2 + UW/2, yDecR)

    # Output MLP
    xOutMlp = 9.6
    NW = 2.0
    box(ax, xOutMlp, yDecR, NW, BH, 'Output MLP', '256 -> 5', C_OUT)
    arr(ax, xBD2 - UW/2, yDecR, xOutMlp + NW/2 + 0.05, yDecR)

    # Pred. Noise: isolated below Output MLP (straight-down arrow)
    # x=[8.6,10.6] clears Linear Proj x=[6.4,8.6] (touching boundary ✓)
    # y=2.5 clears Linear Proj bottom (3.6) by 1.1 ✓
    box(ax, 9.6, 2.5, NW, BH, 'Pred. Noise', 'eps in R^{H x 5}', C_OUT)
    arr(ax, xOutMlp, yDecR - BH/2, 9.6, 2.5 + BH/2)

    # Skip connections
    ax.annotate('', xy=(xBD2, yDecR + BH/2 + 0.08),
                xytext=(xU1, yEncR - BH/2 - 0.08),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1,
                                linestyle='dashed', mutation_scale=10,
                                connectionstyle='arc3,rad=0.2'), zorder=2)
    ax.text((xU1+xBD2)/2 - 0.3, (yEncR+yDecR)/2 + 0.15, 'skip', fontsize=8, color='#999999')

    ax.annotate('', xy=(xBD1, yDecR + BH/2 + 0.08),
                xytext=(xU2, yEncR - BH/2 - 0.08),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1,
                                linestyle='dashed', mutation_scale=10), zorder=2)
    ax.text((xU2+xBD1)/2 + 0.3, (yEncR+yDecR)/2 + 0.05, 'skip', fontsize=8, color='#999999')

    # Time+obs cond dotted lines
    for xb in [xU1, xU2, xUB, xBD1, xBD2]:
        ax.plot([xPlus + r, xb], [yPlus, yEncR - BH/2 - 0.05],
                color='#dddddd', lw=0.8, linestyle='dotted', zorder=1)

    # ── Notes at bottom ───────────────────────────────────────────────
    info_box(ax, FW/2, 1.6,
             'obs_full = [o_base (22) | context_xyz (3) | behaviour_mode (1)] in R^{26}'
             '  |  Training: 15% dropout on dims [22:26]'
             '  |  Inference (CFG): eps_hat = eps_uncond + lambda*(eps_cond - eps_uncond)  lambda=2.0',
             fc='#fef9f0', ec='#e0c080', fs=8.5)

    # ── Legend at lower right (clear of input boxes on left) ──────────
    legend(ax, [(C_IN,    'Input'),
                (C_MODE,  'Behaviour conditioning (26-d)'),
                (C_WARN,  'CFG dropout (train only)'),
                (C_EMBED, 'Time / obs embedding'),
                (C_UNET,  'U-Net block (MLP-based)'),
                (C_NECK,  'Bottleneck'),
                (C_OUT,   'Output')],
           loc='lower right', ncol=2)

    fig.savefig('figures/arch_fig3_cfg_multimodal.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: figures/arch_fig3_cfg_multimodal.png')


# ═══════════════════════════════════════════════════════════════════
# FIG 4  Classifier Guidance — Legibility via Gradient Guidance
# ═══════════════════════════════════════════════════════════════════

def fig4_classifier_guidance():
    FW, FH = 17, 5.5
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(
        'Classifier Guidance — Legibility via Gradient Guidance  (inference-time, no retraining)',
        fontsize=13, fontweight='bold', y=0.99, color=C_TEXT)

    BH = 0.90
    BW = 2.5
    yMain = 4.0   # main pipeline row

    # Five pipeline steps
    xs = [1.4, 4.1, 6.8, 9.5, 12.2, 15.0]

    box(ax, xs[0], yMain, BW, BH, 'Observation',
        'o_t in R^{22}', C_IN)
    box(ax, xs[1], yMain, BW, BH, 'DDIM Denoising\nStep  k',
        'x_(k-1) ~ p(x | x_k, o)', C_UNET)
    box(ax, xs[2], yMain, BW, BH, 'Legibility\nScorer',
        'L_early(a_1:T0)', C_EMBED)
    box(ax, xs[3], yMain, BW, BH, 'Gradient\nGuidance',
        'grad_{a(k)} L_early', C_GOAL)
    box(ax, xs[4], yMain, BW, BH, 'Guided Update',
        'a(k-1) = a(k) + w * grad L', C_MODE)
    box(ax, xs[5], yMain, 1.6, BH, 'Execute', 'best chunk', C_OUT, fs=10)

    # Arrows between steps
    for i in range(len(xs)-1):
        x0 = xs[i] + (BW/2 if i < len(xs)-2 else 0.8)
        x1 = xs[i+1] - (BW/2 if i < len(xs)-3 else 0.8)
        arr(ax, x0, yMain, x1, yMain)

    # Step labels above
    labels = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', '']
    for xi, lbl in zip(xs, labels):
        if lbl:
            ax.text(xi, yMain + BH/2 + 0.15, lbl,
                    ha='center', fontsize=8, color='#888888', style='italic')

    # Feedback loop arc: from "Guided Update" back to "DDIM step"
    ax.annotate('', xy=(xs[1], yMain - BH/2 - 0.05),
                xytext=(xs[4], yMain - BH/2 - 0.05),
                arrowprops=dict(arrowstyle='<-', color='#d06020', lw=1.7,
                                mutation_scale=13,
                                connectionstyle='arc3,rad=0.0'), zorder=3)
    ax.text((xs[1]+xs[4])/2, yMain - BH/2 - 0.38,
            'Iterate for k = T, T-1, ..., 0  (each DDIM step)',
            ha='center', fontsize=9, color='#d06020',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7f0',
                      edgecolor='#e0a060', linewidth=0.9))

    # L_early formula explanation
    info_box(ax, FW/2, 2.0,
             'L_early(a) = P(goal | a_{1:T0})  proportional to  exp( -dist(a_{T0}, goal) )'
             '     T0 = first 30% of horizon     w = guidance weight (sweep: 2 to 20)',
             fc='#f0f4ff', ec='#aabbdd', fs=9)

    info_box(ax, FW/2, 1.2,
             'No model retraining required  |  '
             'Applied to vanilla diffusion policy (runs/diffusion_20260222_195530)'
             '  |  n=20 per weight  |  L_early: baseline=0.906, w=5->0.946, w=10->0.952',
             fc='#f5f5f5', ec='#bbbbbb', fs=8)

    legend(ax, [(C_IN,    'Input'),
                (C_UNET,  'Diffusion step'),
                (C_EMBED, 'Scorer'),
                (C_GOAL,  'Gradient computation'),
                (C_MODE,  'Guided update'),
                (C_OUT,   'Execution')], loc='upper left', ncol=2)

    fig.savefig('figures/arch_fig4_classifier_guidance.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: figures/arch_fig4_classifier_guidance.png')


# ═══════════════════════════════════════════════════════════════════
# FIG 5  VLM Steering Pipeline
# ═══════════════════════════════════════════════════════════════════

def fig5_vlm():
    # Layout (verified non-overlapping, v3):
    #   FW=24 gives 1.2-unit gaps between all boxes for fan labels
    #   xs=[1.7, 5.5, 9.5, 13.5, 17.5, 21.8]; BW=2.8
    #   Fan labels placed at midpoint of the 1.2-unit gap above each arrow
    #   'K frames'/'K scores' placed explicitly with ax.text above arrow midpoints
    FW, FH = 24, 6.5
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(
        'VLM Steering Pipeline  —  Best-of-K with Gemini Flash',
        fontsize=14, fontweight='bold', y=0.99, color=C_TEXT)

    BH = 0.90
    BW = 2.8
    yMain = 4.5

    # gaps: c1_right=3.1, c2_left=4.1 -> 1.0 unit gap (obs->diffpol)
    # c2_right=6.9, c3_left=8.1 -> 1.2 unit gap (fan-out region)
    # c3_right=10.9, c4_left=12.1 -> 1.2 unit gap (K frames)
    # c4_right=14.9, c5_left=16.1 -> 1.2 unit gap (K scores)
    # c5_right=18.9, c6_left=20.6 -> 1.7 unit gap (select->execute)
    xs = [1.7, 5.5, 9.5, 13.5, 17.5, 21.8]

    box(ax, xs[0], yMain, BW, BH, 'Observation',
        'o_t in R^{22}', C_IN)
    box(ax, xs[1], yMain, BW, BH, 'Diffusion Policy',
        'Sample K=3 chunks\n(DDIM, 10 steps)', C_UNET)
    box(ax, xs[2], yMain, BW, BH, 'Render Trajectories',
        'Overlay arm path on\nRGB camera frame', C_EMBED)
    box(ax, xs[3], yMain, BW, BH, 'Gemini Flash\n(VLM Scorer)',
        'Score each candidate\n[0, 1] legibility', C_VLM)
    box(ax, xs[4], yMain, BW, BH, 'Select Best',
        'argmax( VLM score )\nover K candidates', C_MODE)
    box(ax, xs[5], yMain, 2.4, BH, 'Execute', 'best chunk', C_OUT, fs=10)

    # Arrow step 1->2
    arr(ax, xs[0]+BW/2, yMain, xs[1]-BW/2, yMain)

    # Fan-out arrows box2->box3 (K=3 dashed lines) with labels in 1.2-unit gap
    x_start = xs[1]+BW/2   # 6.9
    x_end   = xs[2]-BW/2   # 8.1
    x_mid   = (x_start + x_end) / 2  # 7.5
    for dy, lbl in [(-0.42, 'chunk 1'), (0.0, 'chunk 2'), (0.42, 'chunk 3')]:
        ax.annotate('', xy=(x_end, yMain+dy), xytext=(x_start, yMain),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=0.9,
                                    mutation_scale=9, linestyle='dashed'), zorder=2)
        ax.text(x_mid, yMain+dy+0.13, lbl,
                fontsize=8, color='#777777', ha='center', va='bottom')

    # Arrow 3->4 with 'K frames' label above midpoint (1.2-unit gap: 10.9->12.1, mid=11.5)
    arr(ax, xs[2]+BW/2, yMain, xs[3]-BW/2, yMain)
    ax.text((xs[2]+BW/2 + xs[3]-BW/2)/2, yMain+0.22, 'K frames',
            ha='center', fontsize=9, color='#666666')

    # Arrow 4->5 with 'K scores' label above midpoint (1.2-unit gap: 14.9->16.1, mid=15.5)
    arr(ax, xs[3]+BW/2, yMain, xs[4]-BW/2, yMain)
    ax.text((xs[3]+BW/2 + xs[4]-BW/2)/2, yMain+0.22, 'K scores',
            ha='center', fontsize=9, color='#666666')

    # Arrow 5->6
    arr(ax, xs[4]+BW/2, yMain, xs[5]-1.0, yMain)

    # Step labels above each box
    for xi, lbl in zip(xs, ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', '']):
        if lbl:
            ax.text(xi, yMain+BH/2+0.22, lbl, ha='center',
                    fontsize=9, color='#888888', style='italic')

    info_box(ax, FW/2, 2.4,
             'Prompt: "Given these K robot arm trajectories, which one most clearly indicates '
             'intent towards the target block?  Score each 0 (ambiguous) to 1 (clearly legible)."',
             fc='#f0f4ff', ec='#9999cc', fs=8.5)

    info_box(ax, FW/2, 1.4,
             'Applied every H=32 action steps  |  '
             'K=3 candidates per decision point  |  '
             'LegDiff checkpoint: legdiff_20260331_021740  |  '
             'L_early improves: 0.951 -> 0.955  (n=10)',
             fc='#f5f5f5', ec='#bbbbbb', fs=8)

    legend(ax, [(C_IN,    'Input'),
                (C_UNET,  'Diffusion policy'),
                (C_EMBED, 'Rendering'),
                (C_VLM,   'VLM scorer (Gemini Flash)'),
                (C_MODE,  'Candidate selection'),
                (C_OUT,   'Execution')], loc='lower left', ncol=2)

    fig.savefig('figures/arch_fig5_vlm_steering.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: figures/arch_fig5_vlm_steering.png')


# ═══════════════════════════════════════════════════════════════════
# FIG 6  Side-by-side comparison (4 models, 1 page)
# ═══════════════════════════════════════════════════════════════════

def fig6_comparison():
    # ── Layout constants ─────────────────────────────────────────────
    ROW_H = 1.50   # vertical pitch per row slot (box + gap)
    BHALF = 0.42   # half box height  →  box height = 0.84 units
    BH    = BHALF * 2
    GAP   = ROW_H - BH        # = 0.66 units clear between consecutive boxes
    ARR_M = 0.09               # margin inside gap at each arrow endpoint

    FS_L  = 13.5   # bold label font size
    FS_D  = 11.0   # detail font size
    FS_T  = 13.0   # column header font size
    C_DET = '#111111'          # near-black for detail text (was faint #555555)

    def row_y(slot):
        """Y-centre of global row slot; slot=0 → bottom, slot increases upward."""
        return ROW_H * slot + ROW_H / 2

    # ── No empty-label rows — fold annotations into detail text ──────
    specs = [
        dict(
            title='(1) Vanilla\nDiffusion Policy',
            header_color=C_UNET,
            rows=[
                ('Inputs',        C_IN,     'o in R^22  |  a(k) in R^{Hx5}  |  k'),
                ('Time Embed.',   C_EMBED,  'Sinusoidal -> Linear -> 256-d'),
                ('Obs Embed.',    C_EMBED,  'MLP  22 -> 256 -> 256'),
                ('Action Proj.',  C_UNET,   'Linear  act_dim -> 256'),
                ('Add (+)',       '#E8E8E8', 'obs_emb + time_emb'),
                ('Enc Block 1',  C_UNET,   'Residual MLP  256 -> 512'),
                ('Enc Block 2',  C_UNET,   'Residual MLP  512 -> 1024'),
                ('Bottleneck',    C_NECK,   '1024 -> 1024'),
                ('Dec Block 1',  C_UNET,   'Skip-cat  2048 -> 512'),
                ('Dec Block 2',  C_UNET,   'Skip-cat  1024 -> 256'),
                ('Output MLP',   C_OUT,    '256 -> 5  =  eps_hat'),
            ],
            note='No behaviour conditioning\nUnconstrained multimodal output\nn=22 obs',
        ),
        dict(
            title='(2) Legibility\nDiffuser (CFG)',
            header_color=C_GOAL,
            rows=[
                ('Inputs',        C_IN,     'o in R^22  |  a(k) in R^{Hx5}  |  k  |  g'),
                ('Time Embed.',   C_EMBED,  'Sinusoidal -> Linear -> 256-d'),
                ('Obs Embed.',    C_EMBED,  'MLP  22 -> 256 -> 256'),
                ('Goal Embed.',   C_GOAL,   'nn.Embedding(3, 256)'),
                ('FiLM Cond.',    C_GOAL,   'cond = time + obs + goal'),
                ('Conv1d Proj.',  C_UNET,   'act_dim -> 256  (channels-first)'),
                ('Enc Block 1',  C_UNET,   'Conv1d ResBlock  256->256  k=5'),
                ('Enc Block 2',  C_UNET,   'Conv1d ResBlock  256->512  k=5'),
                ('Bottleneck',    C_NECK,   'Conv1d ResBlock  512 -> 512'),
                ('Dec Block 1',  C_UNET,   'Skip-cat  1024 -> 512'),
                ('Dec Block 2',  C_UNET,   'Skip-cat  512 -> 256'),
                ('Output Proj.',  C_OUT,    'GN + Mish + Conv1d -> act_dim'),
            ],
            note='Goal label g in {left, right, null}\n15% unconditional training\nCFG w=3.0  (Bronars RA-L 2024)',
        ),
        dict(
            title='(3) CFG Multimodal\n(ours)',
            header_color=C_MODE,
            # Empty-label annotation row removed — folded into Inputs detail
            rows=[
                ('Inputs',        C_IN,     'o in R^26  |  a(k) in R^{Hx5}  |  k\n[base 22 | ctx 3 | mode 1]'),
                ('CFG Dropout',   '#FFE8E8', '15% prob: zero dims [22:26]'),
                ('Time Embed.',   C_EMBED,  'Sinusoidal -> Linear -> 256-d'),
                ('Obs Embed.',    C_EMBED,  'MLP  26->256->256  (incl. mode)'),
                ('Action Proj.',  C_UNET,   'Linear  act_dim -> 256'),
                ('Add (+)',       '#E8E8E8', 'obs_emb + time_emb'),
                ('Enc Block 1',  C_UNET,   'Residual MLP  256 -> 512'),
                ('Enc Block 2',  C_UNET,   'Residual MLP  512 -> 1024'),
                ('Bottleneck',    C_NECK,   '1024 -> 1024'),
                ('Dec Block 1',  C_UNET,   'Skip-cat  2048 -> 512'),
                ('Dec Block 2',  C_UNET,   'Skip-cat  1024 -> 256'),
                ('Output MLP',   C_OUT,    '256 -> 5  =  eps_hat'),
            ],
            note='mode in {legib, pred, safe, grnd}\nlambda=2.0 at inference\n26-d obs (same UNet backbone)',
        ),
        dict(
            title='(4) Inference-Time\nCandidate Scoring',
            header_color=C_VLM,
            rows=[
                ('Observation',      C_IN,    'o_t in R^{22}'),
                ('Diffusion Policy', C_UNET,  'DDIM sample: K=3 action chunks'),
                ('Render Frames',    C_EMBED, 'Render each chunk on RGB frame'),
                ('Leg. Scoring',     C_VLM,   'Gemini Flash: legibility score [0,1]'),
                ('Select Best',      C_MODE,  'argmax(score) over K candidates'),
                ('Execute',          C_OUT,   'Execute best chunk for H steps'),
            ],
            note='No model retraining\nK=3 candidates per step\nPolicy: CFG Multimodal',
        ),
    ]

    N_MAX  = max(len(s['rows']) for s in specs)   # = 12

    # Fixed y extents shared by all 4 axes
    HDR_H    = 1.60
    HDR_BOT  = row_y(N_MAX - 1) + BHALF + 0.25   # just above topmost box
    YLIM_TOP = HDR_BOT + HDR_H + 0.50
    YLIM_BOT = -2.20

    fig, axes = plt.subplots(1, 4, figsize=(30, 20))
    fig.suptitle('Architecture Comparison Across All Methods',
                 fontsize=20, fontweight='bold', y=1.01, color=C_TEXT)

    for ax, sp in zip(axes, specs):
        rows  = sp['rows']
        N     = len(rows)
        i_off = N_MAX - N   # top-align all columns

        ax.set_xlim(0, 4)
        ax.set_ylim(YLIM_BOT, YLIM_TOP)
        ax.axis('off')

        # ── Column header band ─────────────────────────────────────
        hdr = FancyBboxPatch(
            (0.0, HDR_BOT), 4.0, HDR_H,
            boxstyle='round,pad=0,rounding_size=0.13',
            linewidth=1.8, edgecolor=C_BORDER,
            facecolor=sp['header_color'], zorder=3)
        ax.add_patch(hdr)
        ax.text(2.0, HDR_BOT + HDR_H / 2, sp['title'],
                ha='center', va='center',
                fontsize=FS_T, fontweight='bold', color=C_TEXT,
                zorder=4, linespacing=1.55)

        # ── Row boxes — reversed(rows) so j=0 is the BOTTOM box ────
        for j, (lbl, col, detail) in enumerate(reversed(rows)):
            slot = j + i_off          # global slot (0 = very bottom)
            y    = row_y(slot)

            # Draw box
            r = FancyBboxPatch(
                (0.06, y - BHALF), 3.88, BH,
                boxstyle='round,pad=0,rounding_size=0.10',
                linewidth=1.3, edgecolor=C_BORDER,
                facecolor=col, zorder=3)
            ax.add_patch(r)

            # Bold label — upper portion of box
            ax.text(2.0, y + 0.17, lbl,
                    ha='center', va='center',
                    fontsize=FS_L, fontweight='bold',
                    color=C_TEXT, zorder=4)

            # Detail text — lower portion, dark colour, no italic
            n_lines = detail.count('\n') + 1
            det_fs  = FS_D if n_lines == 1 else FS_D - 1.5   # 9.5 for 2-line
            ax.text(2.0, y - 0.17, detail,
                    ha='center', va='center',
                    fontsize=det_fs, color=C_DET,
                    style='normal', zorder=4, linespacing=1.25)

            # ── Arrow: in the GAP between this box and the one above ──
            # tail  = just below the bottom of the box above  (higher y)
            # head  = just above the top  of this box         (lower  y → arrowhead)
            # With arrowstyle '->' the arrowhead is at xy (lower y) → points DOWN ✓
            if j < N - 1:
                y_head = row_y(slot)     + BHALF + ARR_M   # lower end, arrowhead
                y_tail = row_y(slot + 1) - BHALF - ARR_M   # upper end, tail
                ax.annotate('',
                    xy     =(2.0, y_head),
                    xytext =(2.0, y_tail),
                    arrowprops=dict(
                        arrowstyle='->', color=C_ARROW,
                        lw=1.8, mutation_scale=18),
                    zorder=5)

        # ── Bottom note (just below this column's lowest box) ──────
        y_note_top = row_y(i_off) - BHALF - 0.35
        ax.text(2.0, y_note_top, sp['note'],
                ha='center', va='top',
                fontsize=10.0, color='#1a1a1a', style='italic',
                linespacing=1.5,
                bbox=dict(boxstyle='round,pad=0.45',
                          facecolor='#f0f0f0',
                          edgecolor='#aaaaaa', linewidth=0.9))

    plt.tight_layout(pad=1.5)
    out = 'figures/arch_fig6_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')


# ═══════════════════════════════════════════════════════════════════
# Individual PPT slides — horizontal left-to-right flow
# ═══════════════════════════════════════════════════════════════════

def _ppt_horiz(title, header_color, rows, note, out_path):
    """
    Horizontal pipeline diagram for PowerPoint.
    Flow is strictly left → right.
    figsize is set so 1 data unit ≈ 1 physical inch → fonts are predictable.
    """
    N      = len(rows)
    COL_W  = 3.6    # inches per column slot
    BW     = 3.0    # box width (inches / data units)
    BHALF  = 1.05   # half box height  →  BH = 2.1 inches
    yMid   = 3.0    # y centre of all boxes (lowered to remove top blank space)
    MARG   = 1.5    # left + right margin (each side)

    total_w = N * COL_W + 2 * MARG
    # Compute tight height: header top + padding above, note bottom + padding below
    hdr_bot  = yMid + BHALF + 0.45
    hdr_h    = 1.25
    hdr_top  = hdr_bot + hdr_h
    note_y   = yMid - BHALF - 0.40   # top of note text
    total_h  = hdr_top + 0.65        # 0.65 padding above header

    # Box centres
    xs = [MARG + COL_W / 2 + i * COL_W for i in range(N)]

    fig, ax = plt.subplots(figsize=(total_w, total_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    # ── Header band ────────────────────────────────────────────────
    hdr = FancyBboxPatch(
        (0.2, hdr_bot), total_w - 0.4, hdr_h,
        boxstyle='round,pad=0,rounding_size=0.15',
        linewidth=2.0, edgecolor=C_BORDER,
        facecolor=header_color, zorder=3)
    ax.add_patch(hdr)
    ax.text(total_w / 2, hdr_bot + hdr_h / 2,
            title,
            ha='center', va='center',
            fontsize=24, fontweight='bold', color=C_TEXT,
            zorder=4, linespacing=1.5)

    # ── Boxes and arrows ───────────────────────────────────────────
    for i, (lbl, col, detail) in enumerate(rows):
        cx = xs[i]
        n_det_lines = detail.count('\n') + 1
        lbl_y  = yMid + (0.38 if n_det_lines == 1 else 0.45)
        det_y  = yMid - (0.32 if n_det_lines == 1 else 0.20)
        det_fs = 12.5 if n_det_lines == 1 else 11.0

        # Box
        r = FancyBboxPatch(
            (cx - BW / 2, yMid - BHALF), BW, BHALF * 2,
            boxstyle='round,pad=0,rounding_size=0.13',
            linewidth=1.8, edgecolor=C_BORDER,
            facecolor=col, zorder=3)
        ax.add_patch(r)

        # Bold label — solid black, easy to read
        ax.text(cx, lbl_y, lbl,
                ha='center', va='center',
                fontsize=19, fontweight='bold',
                color=C_TEXT, zorder=4)

        # Detail text — near-black, normal weight, readable
        ax.text(cx, det_y, detail,
                ha='center', va='center',
                fontsize=det_fs, color='#0d0d0d',
                style='normal', zorder=4, linespacing=1.30)

        # Right-pointing arrow to next box (strictly in the gap)
        if i < N - 1:
            gap    = COL_W - BW
            x_tail = cx + BW / 2 + gap * 0.12
            x_head = xs[i + 1] - BW / 2 - gap * 0.12
            ax.annotate('',
                xy=(x_head, yMid),
                xytext=(x_tail, yMid),
                arrowprops=dict(
                    arrowstyle='->', color=C_ARROW,
                    lw=2.2, mutation_scale=22),
                zorder=5)

    # ── Note box below the pipeline ────────────────────────────────
    ax.text(total_w / 2, note_y, note,
            ha='center', va='top',
            fontsize=12.5, color='#111111', style='italic',
            linespacing=1.55,
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#f0f0f0',
                      edgecolor='#aaaaaa', linewidth=1.0))

    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out_path}')


def fig_ppt1_vanilla():
    _ppt_horiz(
        title        = '(1) Vanilla Diffusion Policy',
        header_color = C_UNET,
        rows=[
            ('Inputs',       C_IN,     'obs(22d)  |  chunk(Hx5d)  |  step k'),

            ('Time Embed.',  C_EMBED,  'Sinusoidal → Linear → 256-d'),
            ('Obs Embed.',   C_EMBED,  'MLP  22 → 256 → 256'),
            ('Action Proj.', C_UNET,   'Linear  act_dim → 256'),
            ('Add (+)',      '#E8E8E8', 'obs_emb + time_emb'),
            ('Enc Block 1',  C_UNET,   'Residual MLP  256 → 512'),
            ('Enc Block 2',  C_UNET,   'Residual MLP  512 → 1024'),
            ('Bottleneck',   C_NECK,   '1024 → 1024'),
            ('Dec Block 1',  C_UNET,   'Skip-cat  2048 → 512'),
            ('Dec Block 2',  C_UNET,   'Skip-cat  1024 → 256'),
            ('Output MLP',   C_OUT,    '256 → 5  =  eps_hat'),
        ],
        note   = 'No behaviour conditioning  |  Unconstrained multimodal output  |  n=22 obs',
        out_path = 'figures/ppt1_vanilla.png',
    )


def fig_ppt2_legdiff():
    _ppt_horiz(
        title        = '(2) Legibility Diffuser  (Bronars RA-L 2024)',
        header_color = C_GOAL,
        rows=[
            ('Inputs',        C_IN,    'obs(22d)  |  chunk(Hx5d)  |  k  |  g'),

            ('Time Embed.',   C_EMBED, 'Sinusoidal → Linear → 256-d'),
            ('Obs Embed.',    C_EMBED, 'MLP  22 → 256 → 256'),
            ('Goal Embed.',   C_GOAL,  'nn.Embedding(3, 256)'),
            ('FiLM Cond.',    C_GOAL,  'cond = time + obs + goal'),
            ('Conv1d Proj.',  C_UNET,  'act_dim → 256  (channels-first)'),
            ('Enc Block 1',   C_UNET,  'Conv1d ResBlock  256→256  k=5'),
            ('Enc Block 2',   C_UNET,  'Conv1d ResBlock  256→512  k=5'),
            ('Bottleneck',    C_NECK,  'Conv1d ResBlock  512 → 512'),
            ('Dec Block 1',   C_UNET,  'Skip-cat  1024 → 512'),
            ('Dec Block 2',   C_UNET,  'Skip-cat  512 → 256'),
            ('Output Proj.',  C_OUT,   'GN + Mish + Conv1d → act_dim'),
        ],
        note   = 'Goal label g in {left, right, null}  |  15% unconditional training  |  CFG w=3.0',
        out_path = 'figures/ppt2_legdiff.png',
    )


def fig_ppt3_cfg():
    _ppt_horiz(
        title        = '(3) CFG Multimodal  (ours)',
        header_color = C_MODE,
        rows=[
            ('Inputs',        C_IN,     'o in R^26  |  a(k) in R^{Hx5}  |  k\n[base 22 | context 3 | mode 1]'),
            ('CFG Dropout',   '#FFE8E8', '15% prob: zero out dims [22:26]'),
            ('Time Embed.',   C_EMBED,  'Sinusoidal → Linear → 256-d'),
            ('Obs Embed.',    C_EMBED,  'MLP  26→256→256  (incl. mode)'),
            ('Action Proj.',  C_UNET,   'Linear  act_dim → 256'),
            ('Add (+)',       '#E8E8E8', 'obs_emb + time_emb'),
            ('Enc Block 1',   C_UNET,   'Residual MLP  256 → 512'),
            ('Enc Block 2',   C_UNET,   'Residual MLP  512 → 1024'),
            ('Bottleneck',    C_NECK,   '1024 → 1024'),
            ('Dec Block 1',   C_UNET,   'Skip-cat  2048 → 512'),
            ('Dec Block 2',   C_UNET,   'Skip-cat  1024 → 256'),
            ('Output MLP',    C_OUT,    '256 → 5  =  eps_hat'),
        ],
        note   = 'mode in {legib, pred, safe, grnd}  |  lambda=2.0 at inference  |  26-d obs (same UNet backbone)',
        out_path = 'figures/ppt3_cfg_multimodal.png',
    )


def fig_ppt4_inference():
    _ppt_horiz(
        title        = '(4) Inference-Time Candidate Scoring',
        header_color = C_VLM,
        rows=[
            ('Observation',        C_IN,    'o_t  in  R^{22}'),
            ('Diffusion Policy',   C_UNET,  'DDIM sample: K=3 action chunks'),
            ('Render Frames',      C_EMBED, 'Render each chunk on RGB frame'),
            ('Legibility Scoring', C_VLM,   'Gemini Flash: score in [0, 1]'),
            ('Select Best',        C_MODE,  'argmax(score) over K candidates'),
            ('Execute',            C_OUT,   'Execute best chunk for H steps'),
        ],
        note   = 'No model retraining  |  K=3 candidates per step  |  Policy: CFG Multimodal',
        out_path = 'figures/ppt4_inference_scoring.png',
    )


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating architecture figures...')
    fig_ppt1_vanilla()
    fig_ppt2_legdiff()
    fig_ppt3_cfg()
    fig_ppt4_inference()
    print()
    fig6_comparison()
    print('\nAll done.')
