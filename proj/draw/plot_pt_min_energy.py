"""
plot_pt_min_energy.py
=====================
For each of H*, CO*, OH*, HOCO* on Pt, find the structure with the minimum
DFT adsorption energy in total.traj, run AdsNet inference, and plot per-atom
energy contributions in a 2×2 grid.

Usage
-----
    python plot_pt_min_energy.py
    python plot_pt_min_energy.py --model path/to/model.pt
    python plot_pt_min_energy.py --rotation 30x,10y,0z
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ase.io import read
from ase.visualize.plot import plot_atoms

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = f'{HERE}/../proj_metals/output/model.pt'
# DEFAULT_MODEL = f'{HERE}/../proj_metals/legacy2_output/model.pt'
# DEFAULT_TRAJ  = f'{HERE}/../proj_metals/dataset/total.traj'
DEFAULT_TRAJ  = f'{HERE}/../proj_metals/legacy_dataset/images_refined.traj'

# sorted-unique-symbol keys → display label
_ADS_KEY = {'H': 'H*', 'CO': 'CO*', 'HO': 'OH*', 'CHO': 'HOCO*'}
ADS_ORDER = ['H*', 'CO*', 'OH*', 'HOCO*']


def _ads_key(atoms):
    syms     = atoms.get_chemical_symbols()
    ads_idx  = atoms.info['data']['ads_idx']
    ads_syms = [syms[i] for i in ads_idx]
    return ''.join(sorted(set(ads_syms)))


def _is_pt(atoms):
    syms    = atoms.get_chemical_symbols()
    ads_idx = set(atoms.info['data']['ads_idx'])
    surf    = [s for i, s in enumerate(syms) if i not in ads_idx]
    return len(set(surf)) == 1 and set(surf) == {'Pt'}


def find_min_energy_structures(traj_path):
    """Return {label: (atoms, traj_idx)} for the min-DFT-energy Pt structure per adsorbate."""
    print(f'Scanning {traj_path} for Pt structures...')
    best = {k: (None, float('inf'), -1) for k in ADS_ORDER}

    for traj_idx, atoms in enumerate(read(traj_path, ':')):
        if not _is_pt(atoms):
            continue
        key   = _ads_key(atoms)
        label = _ADS_KEY.get(key)
        if label is None:
            continue
        energy = atoms.info['data']['ads_energy']
        if energy < best[label][1]:
            best[label] = (atoms, energy, traj_idx)

    return {label: (atoms, traj_idx)
            for label, (atoms, _, traj_idx) in best.items() if atoms is not None}


def load_model(model_path):
    import torch
    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.models.adsnet import AdsNet
    ck = torch.load(model_path, map_location='cpu')
    model = AdsNet(
        num_layers=ck['num_layers'],
        num_channels=ck['num_channels'],
        cutoff=ck['cutoff'],
        rbf_type=ck.get('rbf_type', 'sinc'),
        edge_embed=ck.get('edge_embed', 'linear'),
    )
    model.load_state_dict(ck['model'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Plot per-atom energy for min-E Pt structures.')
    parser.add_argument('--model',    default=DEFAULT_MODEL)
    parser.add_argument('--traj',     default=DEFAULT_TRAJ)
    parser.add_argument('--rotation', default='0x,0y,0z')
    parser.add_argument('--cmap',     default='RdYlBu_r')
    parser.add_argument('--out',      default=f'{HERE}/pt_min_energy.png')
    parser.add_argument('--dpi',      type=int, default=300)
    parser.add_argument('--no-show',  action='store_true')
    args = parser.parse_args()

    # ── find min-energy structures ────────────────────────────────────────────
    best = find_min_energy_structures(args.traj)
    for label in ADS_ORDER:
        if label not in best:
            print(f'WARNING: no Pt+{label} structure found in traj')

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading model: {args.model}')
    model = load_model(args.model)
    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.calculators.calculator import MLCalculator
    calc = MLCalculator(model)

    # ── collect inference results ─────────────────────────────────────────────
    from ase.visualize import view

    results = {}
    for label in ADS_ORDER:
        if label not in best:
            continue
        atoms, traj_idx = best[label]
        local_idx, local_energies = calc.get_be_distributions(atoms)
        E_total = calc.results['energy']
        E_dft   = atoms.info['data']['ads_energy']
        syms    = atoms.get_chemical_symbols()

        print(f'\n─── {label} ───────────────────────────────')
        print(f'  DFT={E_dft:.4f} eV   NNP={E_total:.4f} eV')
        print(f'  {"idx":>5}  {"sym":>3}  {"energy (eV)":>12}')
        for idx, e in zip(np.asarray(local_idx).flatten(),
                          np.asarray(local_energies).flatten()):
            print(f'  {int(idx):>5}  {syms[int(idx)]:>3}  {e:>12.4f}')

        view(atoms)
        results[label] = (atoms, local_idx, local_energies, E_total, E_dft)

    cm = plt.get_cmap(args.cmap)

    # ── 2×2 figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.patch.set_facecolor('white')

    for ax, label in zip(axes.flat, ADS_ORDER):
        ax.set_axis_off()
        if label not in results:
            ax.set_title(f'{label} — not found', fontsize=9)
            continue

        atoms, local_idx, local_energies, E_nnp, E_dft = results[label]
        local_idx      = np.asarray(local_idx).flatten()
        local_energies = np.asarray(local_energies).flatten()

        abs_max = float(np.max(np.abs(local_energies)))
        norm    = plt.Normalize(vmin=-abs_max, vmax=abs_max)
        e_dict  = dict(zip(local_idx.tolist(), local_energies.tolist()))

        colors = []
        for i in range(len(atoms)):
            colors.append(cm(norm(e_dict[i])) if i in e_dict else '#CCCCCC')

        plot_atoms(atoms, ax=ax, colors=colors, rotation=args.rotation)
        ax.set_title(
            f'{label}   DFT={E_dft:.3f} eV  NNP={E_nnp:.3f} eV',
            fontsize=9, pad=5, color='#222222',
        )
        ax.legend(
            handles=[mpatches.Patch(fc='#CCCCCC', ec='#888888', lw=0.8, label='Slab')],
            loc='lower left', fontsize=7, framealpha=0.85, edgecolor='#CCCCCC',
        )

        # per-subplot colorbar
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, shrink=0.7)
        cbar.set_label('Atom energy (eV)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.8)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved → {args.out}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
