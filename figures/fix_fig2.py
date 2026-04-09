"""Fig2 LegDiff clean rewrite.
Key layout principle: encoder and decoder rows at DIFFERENT x-columns so
they can be close vertically without box collision.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

C_IN='#D6E4F0'; C_EMBED='#D5E8D4'; C_UNET='#DAE8FC'; C_NECK='#F8CECC'
C_OUT='#FFF2CC'; C_GOAL='#E1D5E7'; C_BORDER='#444444'; C_TEXT='#111111'; C_SUB='#555555'

def box(ax, cx, cy, w, h, label, sub='', color=C_IN, fs=10.5):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle='round,pad=0,rounding_size=0.07',
        linewidth=1.3, edgecolor=C_BORDER, facecolor=color, zorder=3))
    dy = 0.10 if sub else 0.0
    ax.text(cx, cy+dy, label, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=C_TEXT, zorder=4, wrap=False)
    if sub:
        ax.text(cx, cy-0.18, sub, ha='center', va='center', fontsize=8,
                color=C_SUB, style='italic', zorder=4)

def arr(ax, x0, y0, x1, y1, lw=1.5, color='#222222', dash=False):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=11, linestyle='dashed' if dash else 'solid'), zorder=5)

def info_box(ax, cx, cy, text, fc='#f8f8f8', ec='#bbbbbb', fs=8.5):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.45', facecolor=fc, edgecolor=ec, lw=0.9))

# ─── coordinate plan ───────────────────────────────────────────────────────────
# FW=24, FH=9.5  (wide to give encoder/decoder room side by side without crowding)
#
# Inputs (left half):
#   xIn=1.4   Inp boxes  BW=2.2
#   xEmb=4.2  Emb boxes  EW=2.2
#   yAct=8.0, yObs=6.9, yTime=5.8, yGoal=4.7  (1.1 apart)
#
# FiLM box:  xCond=7.0  yCond=(5.8+4.7)/2=5.25  CW=2.4  CH=1.0
#   FiLM top=5.75, bottom=4.75, left=5.8, right=8.2
#
# U-Net encoder row: y=7.5  x: xE1=9.5 xE2=12.1 xN=14.7  UW=2.2
#   Encoder bottom = 7.5-0.4=7.1  (clear of FiLM top 5.75 by 1.35 ✓)
#
# U-Net decoder row: y=6.2  x: xD1=12.1 xD2=9.5  (SAME x as encoder)
#   Dec top=6.6, encoder bottom=7.1 → gap=0.5  (labels won't overlap since BH=0.8 gap=0.5)
#   Actually at same x, gap of 0.5 means label sub-text will collide. Use yDec=5.9?
#   Dec top=6.3, enc bottom=7.1 → gap=0.8 ✓ ... now FiLM top=5.75 and Dec bottom=5.9-0.4=5.5
#   FiLM top 5.75 > Dec bottom 5.5 — overlap! Use yDec=6.5: Dec bottom=6.1 > FiLM top=5.75 ✓ gap=0.35
#   Enc bottom=7.1, Dec top=6.9 → gap=0.2 — too tight. Use yEnc=8.0, yDec=6.5:
#   enc bottom=7.6, dec top=6.9 → gap=0.7 ✓
#   FiLM top=5.75, dec bottom=6.1 → gap=0.35 ... still close but OK (different x for FiLM vs dec)
#   FiLM x-range=[5.8, 8.2]; dec xD2=9.5-1.1=8.4 left edge: gap from FiLM right=0.2 at diff y=6.5 vs 5.25
#
# Output row: y=3.6  (yGoal bottom=4.3; out top=4.0 → gap=0.3)
#   Use yOut=3.3: top=3.7, yGoal bottom=4.3 → gap=0.6 ✓
#   xOP=8.5 (Output Proj), xNoise=5.5 (Pred.Noise)
#   xOP left=7.4, FiLM right=8.2 → 0.8 gap at diff y (3.3 vs 5.25) ✓
#   xNoise=5.5 right=6.5, FiLM left=5.8 → minor overlap in x but y gap=5.25-3.3=1.95 >> BH ✓

FW, FH = 24, 9.5
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('Legibility Diffuser  (Bronars et al., RA-L 2024)  —  Goal-Conditioned CFG',
             fontsize=14, fontweight='bold', y=0.99, color=C_TEXT)

BH=0.80; BW=2.2; EW=2.2; UW=2.2; CW=2.4; CH=1.0

xIn=1.4; xEmb=4.2
yAct=8.0; yObs=6.9; yTime=5.8; yGoal=4.7

xCond=7.0; yCond=(yTime+yGoal)/2  # = 5.25

xE1=9.5; xE2=12.1; xN=14.7
xD1=12.1; xD2=9.5
yEnc=8.0; yDec=6.5

xOP=8.5; xNoise=5.5; yOut=3.3

# ── Inputs ────────────────────────────────────────────────────────────────────
box(ax, xIn, yAct,  BW, BH, 'Noisy Action',  r'a(k) in R^{H x 5}', C_IN)
box(ax, xIn, yObs,  BW, BH, 'Observation',   r'o in R^{22}',        C_IN)
box(ax, xIn, yTime, BW, BH, 'Timestep  k',   '{0, ..., 99}',        C_IN)
box(ax, xIn, yGoal, BW, BH, 'Goal Label  g', '{left, right, null}', C_GOAL)

# ── Embeddings ────────────────────────────────────────────────────────────────
box(ax, xEmb, yAct,  EW, BH, 'Conv1d Proj',     'act_dim -> 256',    C_UNET)
box(ax, xEmb, yObs,  EW, BH, 'Obs MLP',         '22 -> 256 -> 256',  C_EMBED)
box(ax, xEmb, yTime, EW, BH, 'Sinusoidal Emb.', '128 -> 256',        C_EMBED)
box(ax, xEmb, yGoal, EW, BH, 'Goal Embedding',  'Embedding(3, 256)', C_GOAL)

# ── FiLM Cond Sum ─────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((xCond-CW/2, yCond-CH/2), CW, CH,
    boxstyle='round,pad=0,rounding_size=0.07',
    linewidth=1.3, edgecolor=C_BORDER, facecolor=C_GOAL, zorder=3))
ax.text(xCond, yCond+0.15, 'FiLM Cond Sum', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C_TEXT, zorder=4)
ax.text(xCond, yCond-0.20, 'time + obs + goal -> R^{256}', ha='center', va='center',
        fontsize=8, color=C_SUB, style='italic', zorder=4)

# ── arrows inputs -> embeds ───────────────────────────────────────────────────
for yi in [yAct, yObs, yTime, yGoal]:
    arr(ax, xIn+BW/2, yi, xEmb-EW/2, yi)

# ── arrows embeds -> FiLM ─────────────────────────────────────────────────────
arr(ax, xEmb+EW/2, yObs,  xCond-CW/2, yCond+0.22)
arr(ax, xEmb+EW/2, yTime, xCond-CW/2, yCond)
arr(ax, xEmb+EW/2, yGoal, xCond-CW/2, yCond-0.22)

# ── arrow Conv1d Proj -> Enc Block 1 ──────────────────────────────────────────
arr(ax, xEmb+EW/2, yAct, xE1-UW/2, yEnc)

# ── U-Net blocks ──────────────────────────────────────────────────────────────
box(ax, xE1, yEnc, UW, BH, 'Enc Block 1', 'Conv1d ResBlock\n256->256', C_UNET)
box(ax, xE2, yEnc, UW, BH, 'Enc Block 2', 'Conv1d ResBlock\n256->512', C_UNET)
box(ax, xN,  yEnc, UW, BH, 'Bottleneck',  'Conv1d ResBlock\n512->512', C_NECK)
box(ax, xD1, yDec, UW, BH, 'Dec Block 1', 'Conv1d ResBlock\n1024->512', C_UNET)
box(ax, xD2, yDec, UW, BH, 'Dec Block 2', 'Conv1d ResBlock\n512->256',  C_UNET)
box(ax, xOP, yOut, UW, BH, 'Output Proj', 'GN + Mish\nConv1d->act_dim', C_OUT)

# ── U-Net flow ────────────────────────────────────────────────────────────────
arr(ax, xE1+UW/2, yEnc, xE2-UW/2, yEnc)
arr(ax, xE2+UW/2, yEnc, xN -UW/2, yEnc)
arr(ax, xN,  yEnc-BH/2, xD1, yDec+BH/2)
arr(ax, xD1-UW/2, yDec, xD2+UW/2, yDec)
arr(ax, xD2-UW/2, yDec, xOP+UW/2+0.05, yOut)

# ── skip connections ──────────────────────────────────────────────────────────
ax.annotate('', xy=(xD2, yDec+BH/2+0.05), xytext=(xE1, yEnc-BH/2-0.05),
    arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.0,
                    linestyle='dashed', mutation_scale=9,
                    connectionstyle='arc3,rad=0.2'), zorder=2)
ax.text((xE1+xD2)/2-0.8, (yEnc+yDec)/2+0.1, 'skip', fontsize=7.5, color='#aaaaaa')

ax.annotate('', xy=(xD1, yDec+BH/2+0.05), xytext=(xE2, yEnc-BH/2-0.05),
    arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.0,
                    linestyle='dashed', mutation_scale=9), zorder=2)
ax.text((xE2+xD1)/2+0.4, (yEnc+yDec)/2+0.05, 'skip', fontsize=7.5, color='#aaaaaa')

# ── FiLM -> ResBlocks (dotted pale arrows) ────────────────────────────────────
for xb, yb in [(xE1,yEnc),(xE2,yEnc),(xN,yEnc),(xD1,yDec),(xD2,yDec)]:
    ax.annotate('', xy=(xb, yb-BH/2-0.04), xytext=(xCond+CW/2, yCond+0.1),
        arrowprops=dict(arrowstyle='->', color='#cccccc', lw=0.8,
                        linestyle='dotted', mutation_scale=8), zorder=1)
ax.text(xCond+CW/2+0.15, yCond-0.75,
        'FiLM injected\ninto ResBlocks', fontsize=7.5, color='#aaaaaa', style='italic')

# ── Pred. Noise output ────────────────────────────────────────────────────────
# at xNoise=5.5, yOut=3.3: right edge=6.6 < xOP left=7.4 → 0.8 gap ✓
# yOut=3.3 vs yGoal=4.7: gap=1.4 >> BH ✓
box(ax, xNoise, yOut, 2.2, BH, 'Pred. Noise', 'eps in R^{H x 5}', C_OUT)
arr(ax, xOP-UW/2, yOut, xNoise+1.1+0.05, yOut)

# ── info boxes ────────────────────────────────────────────────────────────────
info_box(ax, FW/2, 1.9,
         'Training: 85% goal-conditioned  +  15% unconditional (null goal token)'
         '  |  Loss: MSE(eps_pred, eps)')
info_box(ax, FW/2, 1.0,
         'Inference (CFG):  eps_hat = eps_uncond + w*(eps_cond - eps_uncond)'
         '     w = 3.0  ->  amplifies goal commitment  ->  higher early legibility',
         fc='#fef9f0', ec='#e0c080')

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor=C_BORDER, label=l) for c,l in [
    (C_IN,'Input'),(C_GOAL,'Goal / FiLM vector'),(C_EMBED,'Time / obs embed'),
    (C_UNET,'Conv1d ResBlock'),(C_NECK,'Bottleneck'),(C_OUT,'Output')]],
    loc='lower left', fontsize=8.5, framealpha=0.95, edgecolor='#cccccc', ncol=2)

fig.savefig('figures/arch_fig2_legdiff.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('OK fig2 v5')
