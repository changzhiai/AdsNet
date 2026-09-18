"""Generate the AdsNet logo: horizontal layout — graph icon + text side by side."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL = "#0D8A8A"
ORANGE = "#E8763A"
DARK = "#1C2D3A"
WHITE = "#FFFFFF"
LIGHT_TEAL = "#B2DFDF"

fig, ax = plt.subplots(figsize=(3.2, 1.2))
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("none")

# -- Graph icon (left side) --
scale = 0.38
ox, oy = 0.20, 0.12

surface = [(-0.9*scale+ox, -0.15*scale+oy),
           (0.0*scale+ox, -0.15*scale+oy),
           (0.9*scale+ox, -0.15*scale+oy)]
surface2 = [(-0.45*scale+ox, 0.63*scale+oy),
            (0.45*scale+ox, 0.63*scale+oy)]
ads = (0.0*scale+ox, 1.25*scale+oy)

# Edges: adsorbate to neighbors
for sx, sy in surface2:
    ax.plot([ads[0], sx], [ads[1], sy], color=TEAL, lw=2.0,
            solid_capstyle="round", zorder=3)
for sx, sy in surface:
    ax.plot([ads[0], sx], [ads[1], sy], color=TEAL, lw=1.4,
            alpha=0.35, solid_capstyle="round", zorder=3)

# Edges between surface atoms
for x1, y1 in surface:
    for x2, y2 in surface2:
        if abs(x1 - x2) < 0.6 * scale:
            ax.plot([x1, x2], [y1, y2], color=LIGHT_TEAL, lw=1.4,
                    solid_capstyle="round", zorder=3)
for i in range(len(surface) - 1):
    ax.plot([surface[i][0], surface[i+1][0]],
            [surface[i][1], surface[i+1][1]],
            color=LIGHT_TEAL, lw=1.4, solid_capstyle="round", zorder=3)
ax.plot([surface2[0][0], surface2[1][0]],
        [surface2[0][1], surface2[1][1]],
        color=LIGHT_TEAL, lw=1.4, solid_capstyle="round", zorder=3)

# Surface atoms
for sx, sy in surface:
    ax.scatter(sx, sy, s=120, c=TEAL, edgecolors=DARK, linewidths=0.7,
               zorder=5, alpha=0.4)
for sx, sy in surface2:
    ax.scatter(sx, sy, s=150, c=TEAL, edgecolors=DARK, linewidths=0.9,
               zorder=5)

# Adsorbate
ax.scatter(*ads, s=200, c=ORANGE, edgecolors=DARK, linewidths=1.0, zorder=6)

# -- Text: centered with graph --
graph_top = ads[1]
graph_bottom = surface[0][1]
graph_mid = (graph_top + graph_bottom) / 2

tx = 1.55

ax.text(tx, graph_bottom + 0.40, "AdsNet", ha="center", va="center",
        fontsize=34, fontweight="bold", color=DARK,
        fontfamily="DejaVu Sans", zorder=2)

# Orange accent line (same width as text)
ax.plot([tx - 0.88, tx + 0.88], [graph_bottom + 0.15, graph_bottom + 0.15],
        color=ORANGE, lw=2.0, solid_capstyle="round", zorder=2)

ax.text(tx, graph_bottom - 0.07, "Adsorption Neural Network", ha="center", va="bottom",
        fontsize=10, color="#777777", fontfamily="DejaVu Sans",
        fontweight="light", zorder=2)

# Limits
margin = 0.06
ax.set_xlim(surface[0][0] - 0.1, 2.50)
ax.set_ylim(graph_bottom - 0.12, graph_top + 0.12)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

out = "/Users/changzhi/Library/CloudStorage/OneDrive-SLACNationalAcceleratorLaboratory/SLAC/AdsNet/docs/source/_static/images/"
fig.savefig(out + "logo.svg", bbox_inches="tight", facecolor="none", pad_inches=0, transparent=True)
fig.savefig(out + "logo.png", dpi=300, bbox_inches="tight", facecolor="none", pad_inches=0, transparent=True)
fig.savefig(out + "logo_white.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05)
fig.savefig(out + "logo_white.svg", bbox_inches="tight", facecolor="white", pad_inches=0.05)
print("Saved logo.svg, logo.png, logo_white.png, logo_white.svg")
