"""
plot_energy_dist.py
====================
Energy distribution of adsorption energies in the AdsNet dataset.

Three panels:
  1. Overall histogram with KDE
  2. Violin plot split by adsorbate  (H*, CO*, OH*, HOCO*)
  3. Box plot split by metal (ordered as in the periodic table)

Adsorbate assignment matches build/get_adsorbate_db.py:
  sorted(set(ads_syms)) → 'H'→H*, 'CO'→CO*, 'HO'→OH*, 'CHO'→HOCO*

Usage
-----
    python plot_energy_dist.py
    python plot_energy_dist.py --traj my.traj --out dist.png --dpi 300
    python plot_energy_dist.py --no-show
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# ── dataset constants (mirror plot_dataset.py) ────────────────────────────────
METALS = ['Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
          'Y',  'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Ag', 'Cd',
          'Hf', 'Ta', 'W',  'Re', 'Ir', 'Pt', 'Au']

ADS_ORDER  = ['H*', 'CO*', 'OH*', 'HOCO*']
ADS_COLORS = {
    'H*':    '#F39C12',
    'CO*':   '#E74C3C',
    'OH*':   '#2ECC40',
    'HOCO*': '#3498DB',
}

# sorted(set(ads_syms)) → label  (matches build.py adsorbate set)
_ADS_MAP = {
    'H':   'H*',
    'CO':  'CO*',
    'HO':  'OH*',
    'CHO': 'HOCO*',
}


def _ads_label(ads_syms: list) -> str:
    key = ''.join(sorted(set(ads_syms)))
    return _ADS_MAP.get(key, key + '*')


def load_dataset(traj_path: str):
    """Return (energies, metals, ads_labels) as parallel lists."""
    from ase.io import read
    print(f"Loading {traj_path} ...")
    images = read(traj_path, ':')
    print(f"  {len(images)} structures")

    energies, metals, ads_labels = [], [], []
    for atoms in images:
        data    = atoms.info['data']
        syms    = atoms.get_chemical_symbols()
        ads_idx = set(data['ads_idx'])

        ads_syms   = [syms[i] for i in ads_idx]
        metal_syms = [s for i, s in enumerate(syms) if i not in ads_idx]
        metal      = Counter(metal_syms).most_common(1)[0][0]
        label      = _ads_label(ads_syms)

        energies.append(float(data['ads_energy']))
        metals.append(metal)
        ads_labels.append(label)

    return energies, metals, ads_labels


def plot_energy_dist(traj_path: str,
                     save_path: str = None,
                     dpi: int = 300,
                     show: bool = True):

    energies, metals, ads_labels = load_dataset(traj_path)
    energies = np.array(energies)

    # ── figure: 3 panels ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 4.5))
    fig.patch.set_facecolor('white')
    gs  = fig.add_gridspec(1, 3, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel 1: overall histogram + KDE ─────────────────────────────────────
    bins = np.linspace(energies.min() - 0.15, energies.max() + 0.15, 60)
    ax1.hist(energies, bins=bins, color='#5B84C4', alpha=0.70,
             edgecolor='white', linewidth=0.4, density=True)

    try:
        from scipy.stats import gaussian_kde
        kde   = gaussian_kde(energies, bw_method=0.12)
        x_kde = np.linspace(bins[0], bins[-1], 400)
        ax1.plot(x_kde, kde(x_kde), color='#1A3A6B', lw=1.8)
    except ImportError:
        pass

    ax1.axvline(np.mean(energies),   color='#E74C3C', lw=1.2, ls='--',
                label=f'Mean = {np.mean(energies):.2f} eV')
    ax1.axvline(np.median(energies), color='#2ECC40', lw=1.2, ls=':',
                label=f'Median = {np.median(energies):.2f} eV')

    ax1.set_xlabel('Adsorption energy (eV)', fontsize=9)
    ax1.set_ylabel('Density', fontsize=9)
    ax1.set_title('Overall distribution', fontsize=9, pad=5)
    ax1.legend(fontsize=7.5, framealpha=0.85, edgecolor='#CCCCCC')
    ax1.tick_params(labelsize=8)
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid(axis='y', lw=0.4, color='#EEEEEE')

    # ── Panel 2: violin by adsorbate ─────────────────────────────────────────
    # only the 4 primary adsorbates, in ADS_ORDER
    grp_data = [np.array([e for e, g in zip(energies, ads_labels) if g == ads])
                for ads in ADS_ORDER]

    parts = ax2.violinplot(grp_data, positions=range(len(ADS_ORDER)),
                           showmedians=True, showextrema=False)
    for pc, ads in zip(parts['bodies'], ADS_ORDER):
        pc.set_facecolor(ADS_COLORS[ads])
        pc.set_alpha(0.72)
        pc.set_edgecolor('none')
    parts['cmedians'].set_color('#222222')
    parts['cmedians'].set_linewidth(1.5)

    ax2.set_xticks(range(len(ADS_ORDER)))
    ax2.set_xticklabels(ADS_ORDER, fontsize=9)
    ax2.set_ylabel('Adsorption energy (eV)', fontsize=9)
    ax2.set_title('By adsorbate', fontsize=9, pad=5)
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.axhline(0, color='#AAAAAA', lw=0.8, ls='--')
    ax2.grid(axis='y', lw=0.4, color='#EEEEEE')

    # count labels above each violin
    y_top = max(energies) + 0.15
    for i, (d, ads) in enumerate(zip(grp_data, ADS_ORDER)):
        ax2.text(i, y_top, f'n={len(d)}',
                 ha='center', va='bottom', fontsize=6.5, color='#555555')

    # ── Panel 3: box plot by metal (periodic-table order) ────────────────────
    # only metals present in dataset, keeping METALS order
    present = set(metals)
    metal_list = [m for m in METALS if m in present]
    metal_data = [np.array([e for e, m in zip(energies, metals) if m == metal])
                  for metal in metal_list]

    bp = ax3.boxplot(
        metal_data,
        positions=range(len(metal_list)),
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color='#222222', linewidth=1.4),
        whiskerprops=dict(color='#888888', linewidth=0.8),
        capprops=dict(color='#888888', linewidth=0.8),
        flierprops=dict(marker='o', markersize=1.2,
                        markerfacecolor='#BBBBBB',
                        markeredgecolor='none', alpha=0.4),
    )

    # color each box by its median (blue = strong binding, red = weak)
    medians = np.array([np.median(d) for d in metal_data])
    norm    = plt.Normalize(medians.min(), medians.max())
    cmap    = plt.cm.RdYlBu_r
    for patch, med in zip(bp['boxes'], medians):
        patch.set_facecolor(cmap(norm(med)))
        patch.set_alpha(0.82)

    ax3.set_xticks(range(len(metal_list)))
    ax3.set_xticklabels(metal_list, fontsize=7, rotation=45, ha='right')
    ax3.set_ylabel('Adsorption energy (eV)', fontsize=9)
    ax3.set_title('By metal', fontsize=9, pad=5)
    ax3.tick_params(labelsize=8)
    ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax3.axhline(0, color='#AAAAAA', lw=0.8, ls='--')
    ax3.grid(axis='y', lw=0.4, color='#EEEEEE')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, pad=0.02, fraction=0.025)
    cbar.set_label('Median $E_{ads}$ (eV)', fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    fig.suptitle(
        f'AdsNet — adsorption energy distributions  (N = {len(energies):,})',
        fontsize=10, y=1.02, color='#222222',
    )
    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot AdsNet energy distributions.')
    parser.add_argument('--traj',    default=f'{HERE}/../proj_metals/dataset/images_refined.traj')
    parser.add_argument('--out',     default=f'{HERE}/energy_dist.png')
    parser.add_argument('--dpi',     type=int, default=300)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    plot_energy_dist(
        traj_path = args.traj,
        save_path = args.out,
        dpi       = args.dpi,
        show      = not args.no_show,
    )
