"""Visualize the proj_metals adsorption database."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
from collections import defaultdict

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
traj_path = f'{path}/dataset/images_refined.traj'
out_dir   = f'{path}/dataset'

# ── load trajectory into a DataFrame ─────────────────────────────────────────
print('Loading trajectory...')
traj = read(traj_path, ':')

rows = []
for atoms in traj:
    syms     = atoms.get_chemical_symbols()
    ads_idx  = atoms.info['data']['ads_idx']
    ads_syms = [syms[i] for i in ads_idx]
    surf_syms = [s for i, s in enumerate(syms) if i not in ads_idx]
    metal = list(set(surf_syms))[0] if len(set(surf_syms)) == 1 else 'mixed'
    ads   = ''.join(sorted(set(ads_syms)))
    energy = atoms.info['data']['ads_energy']
    rows.append({'metal': metal, 'adsorbate': ads, 'energy': energy})

df = pd.DataFrame(rows)
print(f'Loaded {len(df)} structures')

# keep only the 4 primary adsorbates
primary_ads = ['H', 'CO', 'HO', 'CHO']
df_main = df[df['adsorbate'].isin(primary_ads)].copy()
ads_labels = {'H': 'H', 'CO': 'CO', 'HO': 'OH', 'CHO': 'HOCO'}
df_main['adsorbate'] = df_main['adsorbate'].map(ads_labels)

METALS = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
          'Y',  'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Ag', 'Cd',
          'Hf', 'Ta', 'W',  'Re', 'Ir', 'Pt', 'Au']
ADS_ORDER = ['H', 'CO', 'OH', 'HOCO']


# ── Figure 1: heatmap of mean binding energy ─────────────────────────────────
pivot_min = df_main.groupby(['metal', 'adsorbate'])['energy'].min().unstack()
pivot_min = pivot_min.reindex(index=METALS, columns=ADS_ORDER)

fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(pivot_min.values, cmap='RdYlBu_r', aspect='auto',
               vmin=-2.5, vmax=2.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Min $E_{ads}$ (eV)', fontsize=11)

ax.set_xticks(range(len(ADS_ORDER)))
ax.set_xticklabels(ADS_ORDER, fontsize=11)
ax.set_yticks(range(len(METALS)))
ax.set_yticklabels(METALS, fontsize=9)
ax.set_xlabel('Adsorbate', fontsize=11)
ax.set_ylabel('Metal', fontsize=11)
ax.set_title('Minimum Adsorption Energy (eV)', fontsize=12)

# annotate cells
for i, metal in enumerate(METALS):
    for j, ads in enumerate(ADS_ORDER):
        val = pivot_min.loc[metal, ads] if (metal in pivot_min.index and ads in pivot_min.columns) else np.nan
        if not np.isnan(val):
            color = 'white' if abs(val) > 1.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

plt.tight_layout()
fig.savefig(f'{out_dir}/db_heatmap.png', dpi=150, bbox_inches='tight')
print(f'Saved {out_dir}/db_heatmap.png')
plt.close()


# ── Figure 2: violin plot — energy distribution per adsorbate ────────────────
fig, ax = plt.subplots(figsize=(7, 4))
data_by_ads = [df_main[df_main['adsorbate'] == ads]['energy'].values
               for ads in ADS_ORDER]
parts = ax.violinplot(data_by_ads, positions=range(len(ADS_ORDER)),
                      showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor('#6096BA')
    pc.set_alpha(0.7)
ax.set_xticks(range(len(ADS_ORDER)))
ax.set_xticklabels(ADS_ORDER, fontsize=11)
ax.set_ylabel('$E_{ads}$ (eV)', fontsize=11)
ax.set_xlabel('Adsorbate', fontsize=11)
ax.set_title('Binding Energy Distribution per Adsorbate', fontsize=12)
ax.axhline(0, color='k', lw=0.8, ls='--')
plt.tight_layout()
fig.savefig(f'{out_dir}/db_violin.png', dpi=150, bbox_inches='tight')
print(f'Saved {out_dir}/db_violin.png')
plt.close()


# ── Figure 3: bar chart — number of structures per metal, coloured by ads ────
counts = df_main.groupby(['metal', 'adsorbate']).size().unstack(fill_value=0)
counts = counts.reindex(index=METALS, columns=ADS_ORDER, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 4))
bottom = np.zeros(len(METALS))
colors = ['#6096BA', '#E07A5F', '#81B29A', '#F2CC8F']
for ads, color in zip(ADS_ORDER, colors):
    vals = counts[ads].values
    ax.bar(range(len(METALS)), vals, bottom=bottom, label=ads,
           color=color, edgecolor='white', linewidth=0.5)
    bottom += vals
ax.set_xticks(range(len(METALS)))
ax.set_xticklabels(METALS, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of structures', fontsize=11)
ax.set_title('Database Size per Metal', fontsize=12)
ax.legend(title='Adsorbate', fontsize=9, title_fontsize=9)
plt.tight_layout()
fig.savefig(f'{out_dir}/db_counts.png', dpi=150, bbox_inches='tight')
print(f'Saved {out_dir}/db_counts.png')
plt.close()


# ── Figure 4: periodic table coloured by number of structures ────────────────
# Element positions: (row, col) in the periodic table grid (1-indexed)
PTABLE = {
    'H':  (1,1),  'He': (1,18),
    'Li': (2,1),  'Be': (2,2),  'B':  (2,13), 'C':  (2,14), 'N':  (2,15), 'O':  (2,16), 'F':  (2,17), 'Ne': (2,18),
    'Na': (3,1),  'Mg': (3,2),  'Al': (3,13), 'Si': (3,14), 'P':  (3,15), 'S':  (3,16), 'Cl': (3,17), 'Ar': (3,18),
    'K':  (4,1),  'Ca': (4,2),
    'Sc': (4,3),  'Ti': (4,4),  'V':  (4,5),  'Cr': (4,6),  'Mn': (4,7),  'Fe': (4,8),  'Co': (4,9),  'Ni': (4,10), 'Cu': (4,11), 'Zn': (4,12),
    'Ga': (4,13), 'Ge': (4,14), 'As': (4,15), 'Se': (4,16), 'Br': (4,17), 'Kr': (4,18),
    'Rb': (5,1),  'Sr': (5,2),
    'Y':  (5,3),  'Zr': (5,4),  'Nb': (5,5),  'Mo': (5,6),  'Tc': (5,7),  'Ru': (5,8),  'Rh': (5,9),  'Pd': (5,10), 'Ag': (5,11), 'Cd': (5,12),
    'In': (5,13), 'Sn': (5,14), 'Sb': (5,15), 'Te': (5,16), 'I':  (5,17), 'Xe': (5,18),
    'Cs': (6,1),  'Ba': (6,2),
    'Hf': (6,4),  'Ta': (6,5),  'W':  (6,6),  'Re': (6,7),  'Os': (6,8),  'Ir': (6,9),  'Pt': (6,10), 'Au': (6,11), 'Hg': (6,12),
    'Tl': (6,13), 'Pb': (6,14), 'Bi': (6,15), 'Po': (6,16), 'At': (6,17), 'Rn': (6,18),
}

counts_per_metal = df['metal'].value_counts().to_dict()
vmax = max(counts_per_metal.values())
cmap = plt.cm.YlOrRd

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 18.5)
ax.set_ylim(0, 7.5)
ax.invert_yaxis()
ax.axis('off')
ax.set_title('Number of Structures per Element', fontsize=13, pad=10)

for elem, (row, col) in PTABLE.items():
    x, y = col - 1, row - 1
    count = counts_per_metal.get(elem, 0)
    color = cmap(count / vmax) if count > 0 else '#EEEEEE'
    rect = plt.Rectangle((x, y), 0.92, 0.92, color=color,
                          ec='white', lw=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + 0.46, y + 0.38, elem, ha='center', va='center',
            fontsize=7, fontweight='bold',
            color='white' if count > vmax * 0.5 else '#333333', zorder=3)
    if count > 0:
        ax.text(x + 0.46, y + 0.70, f'{count}', ha='center', va='center',
                fontsize=6, color='white' if count > vmax * 0.5 else '#555555', zorder=3)

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, aspect=20)
cbar.set_label('N structures', fontsize=9)

plt.tight_layout()
fig.savefig(f'{out_dir}/db_periodic_table.png', dpi=150, bbox_inches='tight')
print(f'Saved {out_dir}/db_periodic_table.png')
plt.close()
