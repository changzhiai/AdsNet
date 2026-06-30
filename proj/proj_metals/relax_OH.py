"""
plot_relax.py
=============
Paper-quality figure for a single AdsNet relaxation:

  Left  : binding energy vs. relaxation step
  Middle: initial structure colored by per-atom energy (plot_atoms)
  Right : relaxed structure colored by per-atom energy (plot_atoms)

Usage
-----
    # run relaxation and plot
    python plot_relax.py

    # load a saved trajectory and plot (skip inference)
    python plot_relax.py --traj relax_output/relax.traj

    # pick a different structure
    python plot_relax.py --idx 2000
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ase.io import read, write
from ase.visualize.plot import plot_atoms

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = f'{HERE}/../proj_metals/output/model.pt'
DEFAULT_TRAJ  = f'{HERE}/relax/start.traj'


def load_model(model_path):
    import torch
    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.models.adsnet import AdsNet
    ck = torch.load(model_path, map_location='cpu')
    model = AdsNet(
        num_layers=ck['num_layers'],
        num_channels=ck['num_channels'],
        cutoff=ck['cutoff'],
        compute_forces=True,
        rbf_type=ck.get('rbf_type', 'sinc'),
        edge_embed=ck.get('edge_embed', 'linear'),
    )
    model.load_state_dict(ck['model'])
    model.eval()
    return model


def get_atom_energies(atoms, calc):
    """Run inference and return (local_idx, local_energies, E_total)."""
    local_idx, local_energies = calc.get_be_distributions(atoms)
    return (np.asarray(local_idx).flatten(),
            np.asarray(local_energies).flatten(),
            calc.results['energy'])


def atoms_energy_colors(atoms, local_idx, local_energies, cm, norm):
    e_dict = dict(zip(local_idx.tolist(), local_energies.tolist()))
    return [cm(norm(e_dict[i])) if i in e_dict else '#CCCCCC'
            for i in range(len(atoms))]


def plot_atoms_panel(ax, atoms, colors, title, fig, cmap, norm):
    plot_atoms(atoms, ax=ax, colors=colors, rotation='0x,0y,0z')
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=5, color='#222222')
    ax.legend(
        handles=[mpatches.Patch(fc='#CCCCCC', ec='#888888', lw=0.8, label='Slab')],
        loc='lower left', fontsize=7, framealpha=0.85, edgecolor='#CCCCCC',
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, shrink=0.7)
    cbar.set_label('Atomic binding energy (eV)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser(description='Plot AdsNet relaxation for a paper figure.')
    parser.add_argument('--model',    default=DEFAULT_MODEL)
    parser.add_argument('--traj',     default=DEFAULT_TRAJ)
    parser.add_argument('--idx',      type=int, default=2000,
                        help='Structure index in traj')
    parser.add_argument('--relax-traj', default=None,
                        help='Load saved relax trajectory (.traj) instead of running relaxation')
    parser.add_argument('--cmap',     default='RdYlBu_r')
    parser.add_argument('--out',      default=f'{HERE}/relax_figure.png')
    parser.add_argument('--dpi',      type=int, default=300)
    parser.add_argument('--no-show',  action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.calculators.calculator import MLCalculator

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading model: {args.model}')
    model = load_model(args.model)
    calc  = MLCalculator(model)

    # ── get initial structure ─────────────────────────────────────────────────
    print(f'Loading structure idx={args.idx} from {args.traj}')
    # images = read(args.traj, ':')
    # atoms_init = images[args.idx]
    atoms_init = read(args.traj)
    atoms_init.info['data'] = {}
    atoms_init.info['data']['ads_energy'] = 0.4090127899999949
    atoms_init.info['data']['ads_idx'] = [27, 28]
    E_dft = atoms_init.info['data']['ads_energy']
    print(f'DFT E_ads = {E_dft:.4f} eV')

    # ── run or load relaxation ────────────────────────────────────────────────
    if args.relax_traj:
        print(f'Loading relaxation trajectory from {args.relax_traj}')
        traj = read(args.relax_traj, ':')
        energies = [calc.get_binding_energy(a) for a in traj]
        atoms_relaxed = traj[-1]
    else:
        from adsnet.tools import relax_structure
        print('Running relaxation...')
        result = relax_structure(
            atoms_init, model,
            method='pgd',
            max_step_xy=0.5,
            max_step_z=1.0,
            force_tol=0.05,
            max_iter=200,
            maxfun=200,
            energy_plateau_tol=1e-4,
            energy_plateau_patience=10,
        )
        print(f"success={result['success']}  message={result['message']}")
        print(f"E_relaxed={result['energy']:.4f} eV  steps={result['n_steps']}")

        traj    = result['trajectory']
        energies = [calc.get_binding_energy(a) for a in traj]
        atoms_relaxed = result['atoms']

        out_traj = f'{HERE}/../proj_metals/relax_output/relax.traj'
        os.makedirs(os.path.dirname(out_traj), exist_ok=True)
        write(out_traj, traj)
        print(f'Saved trajectory → {out_traj}')

    # ── DFT reference energies and relaxed structure ────────────────────────
    E_dft_init    = atoms_init.info['data']['ads_energy']
    E_dft_relaxed = 0.25761523999999625

    atoms_dft_relaxed = read(f'{HERE}/relax/final.traj')
    atoms_dft_relaxed.info['data'] = {}
    atoms_dft_relaxed.info['data']['ads_idx'] = atoms_init.info['data']['ads_idx']
    atoms_dft_relaxed.info['data']['ads_energy'] = E_dft_relaxed

    # ── per-atom energies for all structures ──────────────────────────────────
    idx_i, e_i, E_i = get_atom_energies(atoms_init, calc)
    idx_r, e_r, E_r = get_atom_energies(atoms_relaxed, calc)
    idx_d, e_d, E_d = get_atom_energies(atoms_dft_relaxed, calc)

    # shared colormap range across all structures
    all_e   = np.concatenate([e_i, e_r, e_d])
    abs_max = float(np.max(np.abs(all_e)))
    norm    = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    cm      = plt.get_cmap(args.cmap)

    colors_i = atoms_energy_colors(atoms_init,        idx_i, e_i, cm, norm)
    colors_r = atoms_energy_colors(atoms_relaxed,     idx_r, e_r, cm, norm)
    colors_d = atoms_energy_colors(atoms_dft_relaxed, idx_d, e_d, cm, norm)

    # ── figure: energy curve (left) + 2x2 structures (right) ─────────────────
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1],
                          height_ratios=[1, 1], hspace=0.1, wspace=0.3)

    # left: energy curve spanning both rows
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_curve.set_aspect('auto')
    ax_curve.set_box_aspect(0.9)

    steps = np.arange(1, len(energies) + 1)
    ax_curve.plot(steps, energies, color='#2166ac', lw=2, marker='o', ms=3,
                  label='NNP relaxation')
    # ax_curve.axhline(E_dft_init, color='#888888', lw=1.2, ls='--',
    #                  label=f'DFT initial: {E_dft_init:.4f} eV')
    # ax_curve.axhline(E_dft_relaxed, color='#d6604d', lw=1.2, ls='--',
    #                  label=f'DFT relaxed: {E_dft_relaxed:.4f} eV')
    ax_curve.axhline(energies[-1], color='#2166ac', lw=1.0, ls=':',
                     label=f'NNP relaxed: {energies[-1]:.4f} eV')
    ax_curve.set_xlabel('Relaxation step', fontsize=11)
    ax_curve.set_ylabel('$E_\\mathrm{ads}$ (eV)', fontsize=11)
    ax_curve.legend(fontsize=9, framealpha=0.85, loc='upper right')
    ax_curve.spines[['top', 'right']].set_visible(False)
    ax_curve.tick_params(labelsize=9)

    # right 2x2: top row = initial structures, bottom row = relaxed structures
    ax_init_dft  = fig.add_subplot(gs[0, 1])
    ax_init_nnp  = fig.add_subplot(gs[0, 2])
    ax_relax_dft = fig.add_subplot(gs[1, 1])
    ax_relax_nnp = fig.add_subplot(gs[1, 2])

    # top-left: initial (DFT label)
    plot_atoms_panel(ax_init_dft, atoms_init, colors_i,
                     f'Initial (DFT)  $E_{{ads}}$={E_dft_init:.4f} eV',
                     fig, cm, norm)
    # top-right: initial (NNP label)
    plot_atoms_panel(ax_init_nnp, atoms_init, colors_i,
                     f'Initial (NNP)  $E_{{ads}}$={E_dft_init:.4f} eV',
                     fig, cm, norm)
    # bottom-left: DFT relaxed structure
    plot_atoms_panel(ax_relax_dft, atoms_dft_relaxed, colors_d,
                     f'Relaxed (DFT)  $E_{{ads}}$={E_dft_relaxed:.4f} eV',
                     fig, cm, norm)
    # bottom-right: NNP relaxed structure
    plot_atoms_panel(ax_relax_nnp, atoms_relaxed, colors_r,
                     f'Relaxed (NNP)  $E_{{ads}}$={E_r:.4f} eV',
                     fig, cm, norm)

    plt.tight_layout(pad=0.8)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved → {args.out}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
