"""
plot_energy_waterfall.py
========================
Waterfall chart of per-atom binding energy contributions from AdsNet.

Each adsorbate atom is a floating bar showing its individual contribution;
a final bar lands on the total predicted adsorption energy.

Usage
-----
    # default: traj idx 26000, proj_metals model
    python plot_energy_waterfall.py

    # custom structure index
    python plot_energy_waterfall.py --idx 100

    # custom model checkpoint
    python plot_energy_waterfall.py --model path/to/model.pt --idx 26000

    # built-in Pt(111)+CO slab
    python plot_energy_waterfall.py --custom --size 4,4,4
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = f'{HERE}/../proj_metals/output/model.pt'
DEFAULT_TRAJ  = f'{HERE}/../proj_metals/dataset/total.traj'

# colors for binding (negative) vs. repulsive (positive) contributions
_COLOR_NEG   = '#2166ac'   # blue  – stabilizing
_COLOR_POS   = '#d6604d'   # red   – destabilizing
_COLOR_TOTAL = '#444444'   # dark  – total bar


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (same helper as plot_atoms_energy.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    import torch
    import sys
    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.models.adsnet import AdsNet

    ck = torch.load(model_path, map_location='cpu')
    model = AdsNet(
        num_interactions=ck['num_layer'],
        hidden_state_size=ck['node_size'],
        cutoff=ck['cutoff'],
    )
    model.load_state_dict(ck['model'])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_waterfall(atoms,
                   local_idx,
                   local_energies,
                   E_total: float = None,
                   title: str = None,
                   figsize=(6, 4),
                   save_path: str = None,
                   dpi: int = 300,
                   show: bool = True):
    """
    Waterfall chart of per-atom binding energy contributions.

    Parameters
    ----------
    atoms          : ASE Atoms
    local_idx      : 1-D array, adsorbate atom indices (model's local_idx)
    local_energies : 1-D array, per-atom energy contributions (eV)
    E_total        : total predicted adsorption energy (eV)
    """
    local_idx      = np.asarray(local_idx).flatten()
    local_energies = np.asarray(local_energies).flatten()
    symbols        = atoms.get_chemical_symbols()

    n = len(local_idx)
    labels = [f"{symbols[i]}{i}" for i in local_idx]

    # ── build waterfall geometry ──────────────────────────────────────────────
    bottoms  = np.zeros(n)
    running  = 0.0
    for k, e in enumerate(local_energies):
        bottoms[k] = running
        running   += e

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    bar_colors = [_COLOR_NEG if e < 0 else _COLOR_POS for e in local_energies]

    bars = ax.bar(range(n), local_energies, bottom=bottoms,
                  color=bar_colors, edgecolor='white', linewidth=0.6,
                  width=0.55, zorder=3)

    # connector lines between consecutive bars
    for k in range(n - 1):
        top_k = bottoms[k] + local_energies[k]
        ax.plot([k + 0.275, k + 0.725], [top_k, top_k],
                color='#888888', lw=0.8, ls='--', zorder=2)

    # total bar
    if E_total is not None:
        ax.bar(n, E_total, bottom=0,
               color=_COLOR_TOTAL, edgecolor='white', linewidth=0.6,
               width=0.55, zorder=3)
        labels.append(f'Total\n({E_total:+.3f} eV)')
    else:
        labels.append('Total')

    # value annotations on each contribution bar
    for k, (bot, e) in enumerate(zip(bottoms, local_energies)):
        va   = 'bottom' if e >= 0 else 'top'
        ypos = bot + e + (0.01 if e >= 0 else -0.01)
        ax.text(k, ypos, f'{e:+.3f}', ha='center', va=va,
                fontsize=7.5, color='#222222')

    if E_total is not None:
        va   = 'bottom' if E_total >= 0 else 'top'
        ypos = E_total + (0.01 if E_total >= 0 else -0.01)
        ax.text(n, ypos, f'{E_total:+.3f}', ha='center', va=va,
                fontsize=7.5, color='#222222')

    ax.axhline(0, color='#222222', lw=0.8, zorder=1)

    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Energy contribution (eV)', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    legend_patches = [
        mpatches.Patch(fc=_COLOR_NEG, ec='white', lw=0.6, label='Stabilizing (< 0)'),
        mpatches.Patch(fc=_COLOR_POS, ec='white', lw=0.6, label='Destabilizing (> 0)'),
        mpatches.Patch(fc=_COLOR_TOTAL, ec='white', lw=0.6, label='Total $E_{ads}$'),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, framealpha=0.85,
              edgecolor='#CCCCCC', loc='best')

    if title is None:
        from collections import Counter
        sym_str = '  +  '.join(f'{v}×{k}' for k, v in Counter(symbols).items())
        title   = sym_str
    ax.set_title(title, fontsize=9, pad=6, color='#222222')

    plt.tight_layout(pad=0.5)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Waterfall chart of per-atom binding energy.')
    parser.add_argument('--model',   default=DEFAULT_MODEL, help='Model checkpoint (.pt)')
    parser.add_argument('--traj',    default=DEFAULT_TRAJ)
    parser.add_argument('--idx',     type=int, default=26000, help='Structure index in traj')
    parser.add_argument('--custom',  action='store_true',
                        help='Use built-in Pt(111)+CO slab instead of traj')
    parser.add_argument('--size',    default='4,4,4',
                        help='Slab size for --custom, e.g. "4,4,4"')
    parser.add_argument('--out',     default=f'{HERE}/energy_waterfall.png')
    parser.add_argument('--dpi',     type=int, default=300)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    # ── load structure ────────────────────────────────────────────────────────
    if args.custom:
        import sys
        sys.path.insert(0, HERE)
        from draw_atoms2graph import make_pt_co_slab
        size  = tuple(int(x) for x in args.size.split(','))
        atoms = make_pt_co_slab(size=size)
        print(f"Custom Pt(111)+CO slab: {len(atoms)} atoms, size={size}")
    else:
        from ase.io import read
        images = read(args.traj, ':')
        atoms  = images[args.idx]
        print(f"Loaded structure {args.idx}: {len(atoms)} atoms")

    # ── load model + run inference ────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    import sys
    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.calculators.calculator import MLCalculator

    calc = MLCalculator(model)
    local_idx, local_energies = calc.get_be_distributions(atoms)
    E_total = calc.results['energy']

    print(f"E_ads = {E_total:.4f} eV")
    print(f"Adsorbate atoms ({len(local_idx)}): {local_idx}")
    print(f"Local energies:  {local_energies}")

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_waterfall(
        atoms, local_idx, local_energies,
        E_total   = E_total,
        save_path = args.out,
        dpi       = args.dpi,
        show      = not args.no_show,
    )


if __name__ == '__main__':
    main()
