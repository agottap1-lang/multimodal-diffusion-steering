"""Standalone Fig5 rewrite with fixed fan-out label positions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

C_IN='#D6E4F0'; C_EMBED='#D5E8D4'; C_UNET='#DAE8FC'; C_NECK='#F8CECC'
C_OUT='#FFF2CC'; C_GOAL='#E1D5E7'; C_MODE='#FFE6CC'; C_VLM='#E8F5E9'
C_BORDER='#444444'; C_TEXT='#111111'; C_SUB='#555555'


def box(ax, cx, cy, w, h, label, sub='', color=C_IN, fs=10.5, sfs=8.5):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle='round,pad=0,rounding_size=0.07',
                       linewidth=1.3, edgecolor=C_BORDER, facecolor=color, zorder=3)
    ax.add_patch(r)
    ldy = 0.10 if sub else 0.0
    ax.text(cx, cy+ldy, label, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=C_TEXT, zorder=4)
    if sub:
        ax.text(cx, cy-0.18, sub, ha='center', va='center', fontsize=8.0,
                color=C_SUB, style='italic', zorder=4)


def arr(ax, x0, y0, x1, y1, lw=1.5, color='#222222', dash=False):
    ls = 'dashed' if dash else 'solid'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=11, linestyle=ls), zorder=5)


def info_box(ax, cx, cy, text, fc='#f8f8f8', ec='#bbbbbb', fs=8):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.45', facecolor=fc, edgecolor=ec, linewidth=0.9))


FW = 24; FH = 6.5
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('VLM Steering Pipeline  —  Best-of-K with Gemini Flash',
             fontsize=14, fontweight='bold', y=0.99, color=C_TEXT)

BH = 0.9; BW = 2.8; yMain = 4.5

# Box centres: 6 boxes in FW=24
# BW=2.8 each (x5) + 2.0 for Execute = 16.0 total box width
# remaining = 24 - 16 = 8 units for 5+2 margins = ~1.1 gap between boxes + 0.8 margin each side
# Manually: c1=1.7, c2=5.5, c3=9.5, c4=13.5, c5=17.5, c6=21.5
# gaps: c2 left - c1 right = 5.5-1.4-1.7-1.4 = 5.5-3.1=wait: c1_right=1.7+1.4=3.1, c2_left=5.5-1.4=4.1: gap=1.0
# c2_right=5.5+1.4=6.9, c3_left=9.5-1.4=8.1: gap=1.2  (enough for fan labels)
xs = [1.7, 5.5, 9.5, 13.5, 17.5, 21.8]

box(ax, xs[0], yMain, BW, BH, 'Observation',         'o_t in R^{22}',                C_IN)
box(ax, xs[1], yMain, BW, BH, 'Diffusion Policy',    'Sample K=3 chunks\n(DDIM, 10 steps)', C_UNET)
box(ax, xs[2], yMain, BW, BH, 'Render Trajectories', 'Overlay arm path on\nRGB camera frame', C_EMBED)
box(ax, xs[3], yMain, BW, BH, 'Gemini Flash\n(VLM Scorer)', 'Score each candidate\n[0, 1] legibility', C_VLM)
box(ax, xs[4], yMain, BW, BH, 'Select Best',         'argmax( VLM score )\nover K candidates', C_MODE)
box(ax, xs[5], yMain, 2.4, BH, 'Execute',            'best chunk', C_OUT, fs=10)

# Arrow 1->2
arr(ax, xs[0]+BW/2, yMain, xs[1]-BW/2, yMain)

# Fan-out arrows box2->box3 (3 dashed lines)
# box2 right = xs[1]+BW/2 = 5.5+1.4 = 6.9
# box3 left  = xs[2]-BW/2 = 9.5-1.4 = 8.1   gap = 1.2
x_start = xs[1]+BW/2
x_end   = xs[2]-BW/2
x_mid   = (x_start + x_end) / 2  # = 7.5
for dy, lbl in [(-0.42, 'chunk 1'), (0.0, 'chunk 2'), (0.42, 'chunk 3')]:
    ax.annotate('', xy=(x_end, yMain+dy),
                xytext=(x_start, yMain),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.9,
                                mutation_scale=9, linestyle='dashed'), zorder=2)
    # Label at midpoint of the fan arrow, offset up slightly
    ax.text(x_mid, yMain+dy+0.13, lbl,
            fontsize=8, color='#777777', ha='center', va='bottom')

# Arrow 3->4  'K frames' label above midpoint (in the 1.2-unit gap)
# gap: box3 right = 9.5+1.4=10.9, box4 left = 13.5-1.4=12.1; mid = 11.5
arr(ax, xs[2]+BW/2, yMain, xs[3]-BW/2, yMain)
ax.text((xs[2]+BW/2 + xs[3]-BW/2)/2, yMain+0.22, 'K frames',
        ha='center', fontsize=9, color='#666666')

# Arrow 4->5 'K scores' label above midpoint
# gap: box4 right=14.9, box5 left=16.1; mid=15.5
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

prompt = ('Prompt: "Given these K robot arm trajectories, which one most clearly indicates '
          'intent towards the target block?  Score each 0 (ambiguous) to 1 (clearly legible)."')
info_box(ax, FW/2, 2.4, prompt, fc='#f0f4ff', ec='#9999cc', fs=8.5)

impl = ('Applied every H=32 action steps  |  K=3 candidates per decision point  |  '
        'LegDiff checkpoint: legdiff_20260331_021740  |  L_early improves: 0.951 -> 0.955  (n=10)')
info_box(ax, FW/2, 1.4, impl, fc='#f5f5f5', ec='#bbbbbb', fs=8)

handles = [mpatches.Patch(facecolor=c, edgecolor=C_BORDER, label=l) for c, l in [
    (C_IN, 'Input'), (C_UNET, 'Diffusion policy'),
    (C_EMBED, 'Rendering'), (C_VLM, 'VLM scorer (Gemini Flash)'),
    (C_MODE, 'Candidate selection'), (C_OUT, 'Execution')]]
ax.legend(handles=handles, loc='lower left', fontsize=8.5,
          framealpha=0.95, edgecolor='#cccccc', ncol=2)

fig.savefig('figures/arch_fig5_vlm_steering.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print('OK fig5 v3')
