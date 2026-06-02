import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 3.5, 6.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor('white')

TEAL    = '#1A7A82'
ORANGE  = '#C05818'
PURPLE  = '#52288A'
DARK    = '#1A2E38'
GRAY    = '#AAAAAA'
BG      = '#F4F6F7'
WHITE   = '#FFFFFF'
C_S = TEAL
C_V = PURPLE
C_E = ORANGE
LW  = 0.85

def rbox(cx, cy, w, h, fc, ec=None, lw=LW, ls='-', radius=0.07, zorder=3):
    ec = ec or fc
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad=0.01,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(p)

def txt(x, y, s, fs=7, color='#1A1A1A', ha='center', va='center',
        bold=False, italic=False, zorder=6):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            fontfamily='DejaVu Sans', zorder=zorder)

def arr(x1, y1, x2, y2, color='#333', lw=LW, ls='-', ms=6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, mutation_scale=ms), zorder=5)

def hv(pts, color, lw=LW, ls='-'):
    """Draw a path through a list of (x,y) points as horizontal/vertical segments,
    with an arrowhead on the last segment."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, lw=lw, ls=ls, zorder=5)
    # last segment with arrow
    arr(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, ls=ls)

# ── Y positions (bottom→top) ────────────────────────────────────────────
Y_eb     = 0.30
Y_sum    = 0.78
Y_rmlp   = 1.22
Y_lb     = 1.56   # loop box bottom
Y_upd    = 2.08
Y_msg    = 2.88
Y_lt     = 3.38   # loop box top
Y_emb    = 3.80
Y_rbf    = 4.40
Y_ee     = 4.08   # EdgeEmbed
Y_hdr    = 4.78
Y_graph  = 5.35
Y_title  = 5.82

xL, xC, xR = 0.68, 1.75, 2.82

# ── Title & graph icon ─────────────────────────────────────────────────
txt(xC, Y_title, 'AdsNet Architecture', fs=8.5, bold=True, color=DARK)

nodes = [(1.75, Y_graph+0.12),(1.50, Y_graph-0.06),
         (2.00, Y_graph-0.06),(1.63, Y_graph-0.24),(1.87, Y_graph-0.24)]
for e in [(0,1),(0,2),(1,3),(2,4),(1,4),(3,4)]:
    ax.plot([nodes[e[0]][0],nodes[e[1]][0]],[nodes[e[0]][1],nodes[e[1]][1]],
            color='#BBBBBB', lw=0.6, zorder=1)
for i,(nx,ny) in enumerate(nodes):
    ax.plot(nx,ny,'o',ms=4.5,color=[TEAL,ORANGE,ORANGE,'#2AABB6','#2AABB6'][i],
            zorder=2, mew=0.5, mec='white')
txt(xC, Y_graph-0.40, 'Structure  →  Graph', fs=6.2, color='#555', italic=True)

# ── Input headers ──────────────────────────────────────────────────────
txt(xL, Y_hdr, r'$\mathbf{v}_0$', fs=9, color=C_V)
txt(xC, Y_hdr, r'$Z_i$',          fs=9, color=C_S)
txt(xR, Y_hdr, r'$r_{ij}$',       fs=9, color=C_E)

# ── v₀ lane: dashed arrow straight down into loop ──────────────────────
hv([(xL, Y_hdr-0.14), (xL, Y_lt+0.02)], C_V, ls='--')

# ── Z_i → Embedding ────────────────────────────────────────────────────
arr(xC, Y_hdr-0.14, xC, Y_emb+0.115, C_S)
rbox(xC, Y_emb, 0.82, 0.23, TEAL)
txt(xC, Y_emb, 'Embedding', fs=6.8, color=WHITE)
arr(xC, Y_emb-0.115, xC, Y_lt+0.02, C_S)

# ── r_ij → RBF → EdgeEmbed ─────────────────────────────────────────────
arr(xR, Y_hdr-0.14, xR, Y_rbf+0.115, C_E)
rbox(xR, Y_rbf, 0.80, 0.23, ORANGE)
txt(xR, Y_rbf, 'RBF', fs=6.8, color=WHITE)
arr(xR, Y_rbf-0.115, xR, Y_ee+0.115, C_E)
rbox(xR, Y_ee, 0.80, 0.23, ORANGE)
txt(xR, Y_ee, 'EdgeEmbed', fs=6.8, color=WHITE)
# route edge: down to loop top, then enter from right side of Message
hv([(xR, Y_ee-0.115), (xR, Y_lt+0.02)], C_E)

# ── Interaction dashed box ─────────────────────────────────────────────
lm = 0.18
rbox(xC, (Y_lb+Y_lt)/2, W-2*lm, Y_lt-Y_lb,
     BG, ec=GRAY, lw=0.9, ls='--', radius=0.10, zorder=1)
txt(W-lm-0.06, Y_lt-0.11, '× N interactions', fs=5.5, color=GRAY, ha='right')

# ── Message block ──────────────────────────────────────────────────────
mw, mh = 2.56, 0.40
rbox(xC, Y_msg, mw, mh, TEAL)
txt(xC, Y_msg+0.07, 'Message Aggregation', fs=7.2, bold=True, color=WHITE)
txt(xC, Y_msg-0.09,
    r's: filter(RBF) × scalar     v: gate × v + $\hat{r}_{ij}$',
    fs=5.0, color='#B8E8EC')

# ── Update block ───────────────────────────────────────────────────────
uw, uh = 2.56, 0.40
rbox(xC, Y_upd, uw, uh, PURPLE)
txt(xC, Y_upd+0.07, 'Equivariant Update', fs=7.2, bold=True, color=WHITE)
txt(xC, Y_upd-0.09,
    r's: $a_{sv}$·⟨Uv,Vv⟩ + $a_{ss}$     v: $a_{vv}$·Uv',
    fs=5.0, color='#C8B8E8')

# ── Arrows into Message (from above the loop box) ──────────────────────
arr(xC, Y_lt, xC, Y_msg+mh/2, C_S)                    # scalar center
arr(xL, Y_lt, xL, Y_msg+mh/2, C_V, ls='--')           # vector left

# Edge enters from the right side of Message horizontally
xR_msg = xC + mw/2
hv([(xR, Y_lt), (xR, Y_msg), (xR_msg, Y_msg)], C_E)

# ── Message → Update ───────────────────────────────────────────────────
arr(xC, Y_msg-mh/2, xC, Y_upd+uh/2, C_S)
arr(xL, Y_msg-mh/2, xL, Y_upd+uh/2, C_V, ls='--')

# ── Loop-back (left margin) ───────────────────────────────────────────
# two separate vertical lines close together on the left
xs = lm + 0.07   # scalar loop-back
xv = lm + 0.22   # vector loop-back
ytop = Y_lt - 0.06

# scalar
ax.plot([xs,xs],[Y_upd-uh/2, ytop], color=C_S, lw=LW, zorder=5)
ax.plot([xs,xC],[ytop, ytop],        color=C_S, lw=LW, zorder=5)

# vector
ax.plot([xv,xv],[Y_upd-uh/2, ytop+0.06], color=C_V, lw=LW, ls='--', zorder=5)
ax.plot([xv,xL],[ytop+0.06, ytop+0.06],  color=C_V, lw=LW, ls='--', zorder=5)

# ── Readout ────────────────────────────────────────────────────────────
arr(xC, Y_lb, xC, Y_rmlp+0.115, C_S)
rbox(xC, Y_rmlp, 1.10, 0.23, TEAL)
txt(xC, Y_rmlp, 'Readout MLP', fs=6.8, color=WHITE)
arr(xC, Y_rmlp-0.115, xC, Y_sum+0.09, C_S)
txt(xC, Y_sum, r'$\sum_i\,\varepsilon_i$  (atom energies)', fs=6.5, color=DARK)
arr(xC, Y_sum-0.09, xC, Y_eb+0.115, C_S)
rbox(xC, Y_eb, 1.90, 0.23, DARK)
txt(xC, Y_eb, r'$E_b$ — adsorption energy', fs=6.8, color=WHITE)

# ── Legend (bottom-left, below loop) ──────────────────────────────────
lx0, ly0, gap = 0.22, 1.02, 0.16
for i,(col, lbl, ls) in enumerate([
        (C_S, 'scalar  s',  '-'),
        (C_V, 'vector  v',  '--'),
        (C_E, 'edge / RBF', '-'),
]):
    y = ly0 - i*gap
    ax.plot([lx0, lx0+0.20], [y,y], color=col, lw=1.0, ls=ls, zorder=5)
    ax.annotate('', xy=(lx0+0.20, y), xytext=(lx0+0.14, y),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.0,
                                linestyle=ls, mutation_scale=5), zorder=5)
    txt(lx0+0.26, y, lbl, fs=5.5, color='#444', ha='left')

plt.tight_layout(pad=0.15)
base = '/Users/changzhi/Library/CloudStorage/OneDrive-SLACNationalAcceleratorLaboratory/SLAC/AdsNet/adsnet_architecture'
fig.savefig(base+'.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(base+'.png', dpi=300, bbox_inches='tight', facecolor='white')
print('done')
