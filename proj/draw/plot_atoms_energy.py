"""
plot_atoms_energy.py
=====================
Run AdsNet inference on a structure and visualize per-atom energy
contributions using plot_atoms, colored on a diverging colormap.

The adsorbate atoms are colored by their local_energies (eV) output
by the model. Slab atoms are gray. A colorbar shows the energy scale.

Usage
-----
    # default: traj idx 26000, proj_metals model
    python plot_atoms_energy.py

    # custom structure index
    python plot_atoms_energy.py --idx 100

    # custom model checkpoint
    python plot_atoms_energy.py --model path/to/model.pt --idx 26000

    # built-in Pt(111)+CO slab
    python plot_atoms_energy.py --custom --size 4,4,4

    # change view rotation
    python plot_atoms_energy.py --rotation 30x,10y,0z
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ase.visualize import view

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = f'{HERE}/../proj_metals/output/model.pt'
DEFAULT_TRAJ  = f'{HERE}/../proj_metals/legacy_dataset/images_refined.traj'


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    import torch
    import sys
    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.models.adsnet import AdsNet

    ck = torch.load(model_path, map_location='cpu')
    model = AdsNet(
        num_layers=ck['num_layers'],
        num_channels=ck['num_channels'],
        cutoff=5.,
        rbf_type=ck.get('rbf_type', 'sinc'),
        edge_embed=ck.get('edge_embed', 'linear'),
    )
    model.load_state_dict(ck['model'])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_atoms_energy(atoms,
                      local_idx,
                      local_energies,
                      E_total: float = None,
                      title: str = None,
                      figsize=(5, 4.5),
                      cmap: str = 'RdYlBu_r',
                      vmin: float = None,
                      vmax: float = None,
                      rotation: str = '0x,0y,0z',
                      save_path: str = None,
                      dpi: int = 300,
                      show: bool = True):
    """
    Render atoms colored by per-atom predicted energy (local_energies).

    Parameters
    ----------
    atoms          : ASE Atoms
    local_idx      : 1-D array, adsorbate atom indices (model's local_idx)
    local_energies : 1-D array, per-atom energy contributions (eV)
    E_total        : total predicted adsorption energy (eV), for title
    cmap           : diverging colormap; default 'RdYlBu_r'
                     (blue = binding/negative, red = repulsive/positive)
    vmin/vmax      : colormap limits; default symmetric ±max(|E|)
    rotation       : ASE rotation string, e.g. '30x,10y,0z'
    """
    from ase.visualize.plot import plot_atoms

    local_idx      = np.asarray(local_idx).flatten()
    local_energies = np.asarray(local_energies).flatten()

    # ── colormap range (symmetric around zero by default) ────────────────────
    abs_max = float(np.max(np.abs(local_energies)))
    if vmin is None:
        vmin = -abs_max
    if vmax is None:
        vmax =  abs_max

    norm   = plt.Normalize(vmin=vmin, vmax=vmax)
    cm     = plt.get_cmap(cmap)
    e_dict = dict(zip(local_idx.tolist(), local_energies.tolist()))

    # ── per-atom color list ───────────────────────────────────────────────────
    colors = []
    for i in range(len(atoms)):
        if i in e_dict:
            colors.append(cm(norm(e_dict[i])))
        else:
            colors.append('#CCCCCC')   # slab atoms

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    plot_atoms(atoms, ax=ax, colors=colors, rotation=rotation)
    ax.set_axis_off()

    # title
    if title is None:
        syms = atoms.get_chemical_symbols()
        from collections import Counter
        sym_str = '  +  '.join(f'{v}×{k}' for k, v in Counter(syms).items())
        e_str   = f'   $E_{{ads}}$ = {E_total:.3f} eV' if E_total is not None else ''
        title   = sym_str + e_str
    ax.set_title(title, fontsize=9, pad=6, color='#222222')

    # ── colorbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, shrink=0.7)
    cbar.set_label('Atom energy contribution (eV)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # slab legend patch
    ax.legend(
        handles=[mpatches.Patch(fc='#CCCCCC', ec='#888888', lw=0.8, label='Slab')],
        loc='lower left', fontsize=7, framealpha=0.85, edgecolor='#CCCCCC',
    )

    plt.tight_layout(pad=0.4)
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
    parser = argparse.ArgumentParser(description='Plot per-atom energy from AdsNet.')
    parser.add_argument('--model',    default=DEFAULT_MODEL, help='Model checkpoint (.pt)')
    parser.add_argument('--traj',     default=DEFAULT_TRAJ)
    parser.add_argument('--idx',      type=int, default=23481, help='Structure index in traj')
    parser.add_argument('--custom',   action='store_true', default=True,
                        help='Use built-in Pt(111)+CO slab instead of traj')
    parser.add_argument('--size',     default='4,4,4',
                        help='Slab size for --custom, e.g. "4,4,4"')
    parser.add_argument('--rotation', default='0x,0y,0z',
                        help='ASE rotation string, e.g. "30x,10y,0z"')
    parser.add_argument('--cmap',     default='RdYlBu_r')
    parser.add_argument('--out',      default=f'{HERE}/atoms_energy.png')
    parser.add_argument('--dpi',      type=int, default=300)
    parser.add_argument('--no-show',  action='store_true')
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

    view(atoms)

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

    # ── plot: atoms view ─────────────────────────────────────────────────────
    plot_atoms_energy(
        atoms, local_idx, local_energies,
        E_total   = E_total,
        cmap      = args.cmap,
        rotation  = args.rotation,
        save_path = args.out,
        dpi       = args.dpi,
        show      = not args.no_show,
    )



if __name__ == '__main__':
    main()
