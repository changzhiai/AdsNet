"""
plot_graph_energy.py
====================
Run AdsNet inference on a structure and visualize per-atom binding energy
contributions on the adsorbate-centered graph.

  - Adsorbate nodes : colored by per-atom predicted energy (diverging cmap).
  - Metal neighbor nodes : same red style as draw_atoms2graph.py.
  - Energy values (eV, 2 decimal) shown below each adsorbate node label.

Usage
-----
    # default: traj idx 26000, proj_metals model
    python plot_graph_energy.py

    # custom structure index
    python plot_graph_energy.py --idx 100

    # custom model checkpoint
    python plot_graph_energy.py --model path/to/model.pt --idx 26000

    # built-in Pt(111)+CO slab
    python plot_graph_energy.py --custom --size 4,4,4

    # change graph layout
    python plot_graph_energy.py --layout star
"""

import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = f'{HERE}/../proj_metals/output/model.pt'


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

def plot_graph_energy(atoms,
                      local_idx,
                      local_energies,
                      E_total: float = None,
                      title: str = None,
                      figsize=(5, 4.5),
                      cmap: str = 'RdYlBu_r',
                      vmin: float = None,
                      vmax: float = None,
                      node_size: int = 300,
                      cutoff: float = 5.5,
                      layout: str = 'auto',
                      save_path: str = None,
                      dpi: int = 300,
                      show: bool = True) -> None:
    """
    Build the AdsNet graph from atoms, then draw it with energy/distance
    coloring on every node.

    Parameters
    ----------
    atoms          : ASE Atoms
    local_idx      : 1-D array, adsorbate atom indices (model output)
    local_energies : 1-D array, per-atom energy contributions (eV)
    E_total        : total predicted adsorption energy (eV), for title
    cmap           : diverging colormap for adsorbate energy (default 'RdYlBu_r')
    vmin/vmax      : colormap range for energy; defaults to symmetric ±max(|E|)
    cutoff         : neighbor cutoff used to build the graph (Å)
    layout         : 'auto' | 'star' | 'kamada' | 'spring' | 'xy'
    """
    import sys
    sys.path.insert(0, HERE)
    from draw_atoms2graph import build_graph, node_roles, _make_layout, _ROLE_STYLE

    local_idx      = np.asarray(local_idx).flatten()
    local_energies = np.asarray(local_energies).flatten()

    G     = build_graph(atoms, cutoff=cutoff)
    roles = node_roles(G)

    # ── energy colormap ───────────────────────────────────────────────────────
    abs_max = float(np.max(np.abs(local_energies)))
    if vmin is None:
        vmin = -abs_max
    if vmax is None:
        vmax =  abs_max

    e_norm = plt.Normalize(vmin=vmin, vmax=vmax)
    e_cm   = plt.get_cmap(cmap)
    e_dict = dict(zip(local_idx.tolist(), local_energies.tolist()))

    ads_nodes   = [n for n, r in roles.items() if r == 'adsorbate']
    metal_nodes = [n for n, r in roles.items() if r == 'ads_neighbor']

    # ── 2D layout ─────────────────────────────────────────────────────────────
    pos2d = _make_layout(G, layout)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    nx.draw_networkx_edges(G, pos2d, ax=ax,
                           edge_color='#888888', width=1.3, alpha=0.75)

    # metal neighbor nodes: colored by local energy (same cmap as adsorbate)
    if metal_nodes:
        _, ec, _ = _ROLE_STYLE['ads_neighbor']
        metal_colors = [e_cm(e_norm(e_dict[n])) if n in e_dict else '#AAAAAA'
                        for n in metal_nodes]
        nx.draw_networkx_nodes(G, pos2d, nodelist=metal_nodes, ax=ax,
                               node_color=metal_colors, edgecolors=ec,
                               linewidths=1.4, node_size=node_size)
        nx.draw_networkx_labels(G, pos2d,
                                labels={n: G.nodes[n]['symbol'] for n in metal_nodes},
                                ax=ax, font_size=5, font_color='white')
        metal_offset = -(np.sqrt(node_size / np.pi) + 4)
        for n in metal_nodes:
            x, y = pos2d[n]
            val = f'{e_dict[n]:.2f}' if n in e_dict else 'N/A'
            ax.annotate(val, xy=(x, y),
                        xytext=(0, metal_offset), textcoords='offset points',
                        ha='center', va='top', fontsize=5.5, color='#222222')

    # adsorbate nodes: colored by local energy, energy value below label
    if ads_nodes:
        node_colors = [e_cm(e_norm(e_dict[n])) if n in e_dict else '#AAAAAA'
                       for n in ads_nodes]
        nx.draw_networkx_nodes(G, pos2d, nodelist=ads_nodes, ax=ax,
                               node_color=node_colors, edgecolors='#333333',
                               linewidths=1.4, node_size=node_size * 1.15)
        nx.draw_networkx_labels(G, pos2d,
                                labels={n: G.nodes[n]['symbol'] for n in ads_nodes},
                                ax=ax, font_size=6, font_color='#111111')
        ads_offset = -(np.sqrt(node_size * 1.15 / np.pi) + 4)
        for n in ads_nodes:
            x, y = pos2d[n]
            val = f'{e_dict[n]:.2f}' if n in e_dict else 'N/A'
            ax.annotate(val, xy=(x, y),
                        xytext=(0, ads_offset), textcoords='offset points',
                        ha='center', va='top', fontsize=5.5, color='#111111')

    # colorbar: adsorbate energy
    sm_e = plt.cm.ScalarMappable(cmap=e_cm, norm=e_norm)
    sm_e.set_array([])
    cbar = fig.colorbar(sm_e, ax=ax, fraction=0.030, pad=0.02, shrink=0.65,
                        location='right')
    cbar.set_label('Ads. energy (eV)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if title is None:
        from collections import Counter
        sym_str = '  +  '.join(f'{v}×{k}' for k, v in Counter(atoms.get_chemical_symbols()).items())
        e_str   = f'   $E_{{ads}}$ = {E_total:.3f} eV' if E_total is not None else ''
        title   = sym_str + e_str
    ax.set_title(title, fontsize=9, pad=6, color='#222222')

    plt.tight_layout(pad=0.3)
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
    parser = argparse.ArgumentParser(description='Graph view of per-atom binding energy.')
    parser.add_argument('--traj',     default=f'{HERE}/../proj_metals/dataset/images_refined.traj')
    parser.add_argument('--idx',      type=int, default=26000, help='Structure index in traj')
    parser.add_argument('--cutoff',   type=float, default=5, help='Bond cutoff distance (Å)')
    parser.add_argument('--custom',   action='store_true', default=True,
                        help='Use built-in Pt(111)+CO slab instead of traj file')
    parser.add_argument('--size',     default='4,4,4',
                        help='Slab size for --custom, e.g. "4,4,4"')
    parser.add_argument('--layout',   default='kamada',
                        choices=['auto','star','kamada','spring','xy'])
    parser.add_argument('--out',      default=f'{HERE}/graph_energy.png')
    parser.add_argument('--dpi',      type=int, default=300)
    parser.add_argument('--no-show',  action='store_true')
    parser.add_argument('--model',    default=DEFAULT_MODEL, help='Model checkpoint (.pt)')
    parser.add_argument('--cmap',     default='RdYlBu_r')
    args = parser.parse_args()

    # ── load structure (mirrors draw_atoms2graph.py) ──────────────────────────
    import sys
    sys.path.insert(0, HERE)
    from draw_atoms2graph import make_pt_co_slab
    if args.custom:
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

    sys.path.insert(0, os.path.join(HERE, '../..'))
    from adsnet.calculators.calculator import MLCalculator

    calc = MLCalculator(model)
    local_idx, local_energies = calc.get_be_distributions(atoms)
    E_total = calc.results['energy']

    print(f"E_ads = {E_total:.4f} eV")
    print(f"Adsorbate atoms ({len(local_idx)}): {local_idx}")
    print(f"Local energies:  {local_energies}")

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_graph_energy(
        atoms, local_idx, local_energies,
        E_total   = E_total,
        cmap      = args.cmap,
        cutoff    = args.cutoff,
        layout    = args.layout,
        save_path = args.out,
        dpi       = args.dpi,
        show      = not args.no_show,
    )


if __name__ == '__main__':
    main()
