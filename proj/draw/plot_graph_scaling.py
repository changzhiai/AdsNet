"""
plot_graph_scaling.py
======================
Shows the computational cost saving of AdsNet's adsorbate-centered graph
vs. a full-graph neighbor list, as a function of slab size.

Method: tile the base Pt+CO slab (3×3, 27 Pt + CO) into supercells
1×1 → 4×4 in XY. The adsorbate count stays fixed (1 CO = 2 atoms);
only the slab grows. We count neighbor pairs for:
  - Full graph  : neighbor_list over all atoms → O(N)
  - AdsNet graph: pairs centered on ads_idx only → O(1) w.r.t. slab size
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import asap3
from ase.io import read
from ase.neighborlist import neighbor_list

HERE   = os.path.dirname(os.path.abspath(__file__))
TRAJ   = f'{HERE}/../proj_metals/dataset/images_refined.traj'
IDX    = 26000
CUTOFF = 5.5

# ── load base structure ───────────────────────────────────────────────────────
images = read(TRAJ, ':')
base   = images[IDX]
ads_idx_base = list(base.info['data']['ads_idx'])  # [27, 28] = C, O

# ── supercell sweep ───────────────────────────────────────────────────────────
tiles  = [(1,1),(1,2),(2,2),(2,3),(3,3),(3,4),(4,4)]

n_atoms_list, full_pairs, ads_pairs = [], [], []

for (nx, ny) in tiles:
    slab = base.repeat([nx, ny, 1])
    n    = len(slab)

    # --- full graph pairs (all atoms, undirected) ---
    ii, jj = neighbor_list('ij', slab, CUTOFF)
    n_full = len(ii) // 2      # undirected: divide by 2

    # --- AdsNet pairs (adsorbate-centered, bidirectional as data.py) ---
    nl = asap3.FullNeighborList(CUTOFF, slab)
    pi, pj = [], []
    for i in ads_idx_base:     # adsorbate indices don't change with repeat
        idxs, _, _ = nl.get_neighbors(i)
        pi += [i] * len(idxs)
        pj += list(idxs)
    # reverse pairs
    rev_i = [j for j, i, in sorted(zip(pj, pi))]
    rev_j = [i for j, i  in sorted(zip(pj, pi))]
    n_ads  = len(pi) + len(rev_i)

    n_atoms_list.append(n)
    full_pairs.append(n_full)
    ads_pairs.append(n_ads)

    print(f"  {nx}×{ny}: {n:4d} atoms  |  full={n_full:5d}  ads={n_ads:4d}")

n_atoms_list = np.array(n_atoms_list)
full_pairs   = np.array(full_pairs)
ads_pairs    = np.array(ads_pairs)

# ── linear fit for full graph ─────────────────────────────────────────────────
coeffs = np.polyfit(n_atoms_list, full_pairs, 1)
x_fit  = np.linspace(n_atoms_list[0] * 0.9, n_atoms_list[-1] * 1.05, 200)
y_fit  = np.polyval(coeffs, x_fit)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.8))

ax.plot(x_fit, y_fit,
        color='#E74C3C', lw=1.5, ls='--', label='Full graph (fit)')
ax.scatter(n_atoms_list, full_pairs,
           color='#E74C3C', s=55, zorder=4, label='Full graph')

ax.plot(n_atoms_list, ads_pairs,
        color='#2ECC40', lw=1.5, ls='-')
ax.scatter(n_atoms_list, ads_pairs,
           color='#2ECC40', s=55, zorder=4, label='AdsNet graph (const.)')

# shade the saving region
ax.fill_between(n_atoms_list, ads_pairs, full_pairs,
                color='#E74C3C', alpha=0.10, label='Cost saved')

# tile labels on full-graph points
for n, fp, tile in zip(n_atoms_list, full_pairs, tiles):
    ax.annotate(f'{tile[0]}×{tile[1]}', xy=(n, fp),
                xytext=(4, 4), textcoords='offset points',
                fontsize=6.5, color='#555555')

ax.set_xlabel('Number of atoms in slab', fontsize=9)
ax.set_ylabel('Number of neighbor pairs', fontsize=9)
ax.set_title('Graph size scaling: AdsNet vs. full neighbor list\n'
             f'(Pt slab + CO,  cutoff = {CUTOFF} Å)',
             fontsize=9, pad=7)
ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#CCCCCC')
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis='both', labelsize=8)
ax.grid(axis='y', lw=0.5, color='#DDDDDD')

plt.tight_layout(pad=0.5)
out = f'{HERE}/graph_scaling.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nSaved → {out}')
plt.show()
