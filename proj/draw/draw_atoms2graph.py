"""
draw_atoms2graph.py
====================
Visualize the AdsNet graph for an adsorbate+slab structure.

Graph construction mirrors AdsNet data.py exactly:
  - uses asap3.FullNeighborList with the same cutoff (default 5.5 Å)
  - loops over ads_idx only → adsorbate-centered pairs, no metal-metal edges
  - adds reverse pairs (metal→adsorbate) so message passing is bidirectional

Node coloring:
  - Green  : adsorbate atoms (non-metals: C, O, H, N, …)
  - Red    : metal atoms within cutoff of any adsorbate (ads. neighbors)
  - White  : metal atoms outside cutoff (non-neighbors, full-slab view only)

Outputs (per run):
  graph_style.png         — AdsNet graph (neighbors + adsorbate only)
  graph_style_compare.png — side-by-side: AdsNet graph vs. full slab
  graph_style_atoms.png   — real atomic structure via ase.visualize.plot
  ASE viewer              — interactive 3D view of adsorbate + neighbors

Usage
-----
    python draw_atoms2graph.py                        # default traj, idx=26000 (Pt+CO)
    python draw_atoms2graph.py --idx 42
    python draw_atoms2graph.py --traj my.traj --idx 0 --cutoff 6.0
    python draw_atoms2graph.py --out result.png --dpi 300 --no-show
"""

import argparse
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ase.io import read
from ase.data import chemical_symbols
# metals by atomic number (Z=3..83 minus non-metals/metalloids — simple set)
_NONMETALS = {"H","He","C","N","O","F","Ne","P","S","Cl","Ar",
              "Se","Br","Kr","I","Xe","At","Rn"}

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction (subset: top layers + adsorbate)
# ─────────────────────────────────────────────────────────────────────────────

def _is_adsorbate(symbol: str) -> bool:
    return symbol in _NONMETALS


def build_graph(atoms, cutoff: float = 5.5):
    """
    Build a graph mirroring AdsNet data.py's _get_neighborlist logic:
      - uses asap3.FullNeighborList with PBC support
      - loops only over ads_idx atoms (adsorbate-centered pairs)
      - adds reverse pairs so every metal neighbor is also included
    Nodes: all adsorbate atoms + all metal atoms that appear as neighbors.
    """
    import asap3

    symbols  = atoms.get_chemical_symbols()
    ads_idx  = atoms.info['data']['ads_idx']   # same source as data.py

    # ── build neighbor list (mirrors data.py _get_neighborlist) ──────────
    nl = asap3.FullNeighborList(cutoff, atoms)

    pair_i, pair_j, edge_diff = [], [], []
    for i in ads_idx:
        indices, diff, _ = nl.get_neighbors(i)
        pair_i    += [i] * len(indices)
        pair_j    += list(indices)
        edge_diff += diff.tolist()

    # reverse pairs (neighbor → adsorbate), exactly as data.py does
    rev_i, rev_j, rev_diff = [], [], []
    for j, i, d in sorted(zip(pair_j, pair_i, edge_diff)):
        rev_i.append(j); rev_j.append(i); rev_diff.append([-x for x in d])

    pair_i    += rev_i
    pair_j    += rev_j
    edge_diff += rev_diff

    # ── build NetworkX graph ─────────────────────────────────────────────
    involved = set(pair_i) | set(pair_j)   # all atoms that appear in pairs

    G = nx.Graph()
    for idx in involved:
        G.add_node(idx,
                   symbol=symbols[idx],
                   pos=atoms.positions[idx].copy(),
                   z=float(atoms.positions[idx, 2]))

    for i, j, d in zip(pair_i, pair_j, edge_diff):
        if not G.has_edge(i, j):
            dist = float(np.linalg.norm(d))
            G.add_edge(i, j, distance=round(dist, 4))

    return G


def build_full_graph(atoms, cutoff: float = 5.5):
    """
    Build a graph of ALL atoms within cutoff of each other (full slab view).
    Non-neighbors (atoms not connected to any adsorbate) are included as nodes.
    """
    from ase.neighborlist import neighbor_list

    symbols = atoms.get_chemical_symbols()
    idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff)

    G = nx.Graph()
    for i in range(len(atoms)):
        G.add_node(i,
                   symbol=symbols[i],
                   pos=atoms.positions[i].copy(),
                   z=float(atoms.positions[i, 2]))

    for i, j, d in zip(idx_i, idx_j, dists):
        if not G.has_edge(i, j):
            G.add_edge(i, j, distance=round(float(d), 4))

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Node role → colour
# ─────────────────────────────────────────────────────────────────────────────

def node_roles(G: nx.Graph):
    """
    Returns a dict  node_id → 'adsorbate' | 'ads_neighbor' | 'non_neighbor'

    - adsorbate    : non-metal atoms (C, O, H, …)
    - ads_neighbor : metal atoms directly connected to any adsorbate atom
    - non_neighbor : all other metal atoms
    """
    ads_nodes = {n for n, d in G.nodes(data=True) if _is_adsorbate(d['symbol'])}
    ads_neighbors = set()
    for n in ads_nodes:
        for nb in G.neighbors(n):
            if nb not in ads_nodes:
                ads_neighbors.add(nb)
    roles = {}
    for n in G.nodes():
        if n in ads_nodes:
            roles[n] = 'adsorbate'
        elif n in ads_neighbors:
            roles[n] = 'ads_neighbor'
        else:
            roles[n] = 'non_neighbor'
    return roles


_ROLE_STYLE = {
    # (facecolor, edgecolor, zorder)
    'adsorbate':    ('#2ECC40', '#1A5C28', 4),   # green
    'ads_neighbor': ('#E74C3C', '#7B1010', 3),   # red
    'non_neighbor': ('#F8F8F8', '#333333', 2),   # white
}


# ─────────────────────────────────────────────────────────────────────────────
# Layout helper
# ─────────────────────────────────────────────────────────────────────────────

def _star_layout(G: nx.Graph, ads_nodes: list, radius: float = 1.0) -> dict:
    """
    Place adsorbate atoms clustered at the center, metal neighbors
    evenly distributed on a circle of the given radius.
    Neighbors are sorted by angle from their real XY positions so the
    ring order matches the physical geometry.
    """
    non_ads = [n for n in G.nodes() if n not in ads_nodes]
    pos = {}

    # ── adsorbate atoms: small cluster at center ─────────────────────────
    n_ads = len(ads_nodes)
    if n_ads == 1:
        pos[ads_nodes[0]] = np.array([0.0, 0.0])
    else:
        # place them on a tiny inner circle so they don't overlap
        gap = 0.18
        for i, n in enumerate(ads_nodes):
            angle = 2 * np.pi * i / n_ads
            pos[n] = np.array([gap * np.cos(angle), gap * np.sin(angle)])

    # ── metal neighbors: evenly spaced on outer circle ───────────────────
    # sort by real XY angle so the layout reflects the physical arrangement
    if non_ads:
        cx = np.mean([G.nodes[n]['pos'][0] for n in non_ads])
        cy = np.mean([G.nodes[n]['pos'][1] for n in non_ads])
        non_ads_sorted = sorted(
            non_ads,
            key=lambda n: np.arctan2(G.nodes[n]['pos'][1] - cy,
                                     G.nodes[n]['pos'][0] - cx)
        )
        n_nb = len(non_ads_sorted)
        for i, n in enumerate(non_ads_sorted):
            angle = 2 * np.pi * i / n_nb
            pos[n] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    return pos


def _make_layout(G: nx.Graph, layout: str = 'auto') -> dict:
    """
    Choose a 2D layout that avoids node overlap.

    layout='auto'   : star layout if the graph is star-like (AdsNet graph),
                      kamada otherwise (full slab graph)
    layout='star'   : adsorbate at centre, neighbors on a circle (clean star)
    layout='kamada' : Kamada-Kawai weighted by 1/distance
    layout='spring' : force-directed
    layout='xy'     : project real XY coordinates
    """
    ads = [n for n, d in G.nodes(data=True) if _is_adsorbate(d['symbol'])]
    non_ads = [n for n in G.nodes() if n not in ads]

    if layout == 'auto':
        # star-like: no edges among non-adsorbate nodes → use star layout
        metal_edges = sum(1 for u, v in G.edges()
                          if u in non_ads and v in non_ads)
        layout = 'star' if metal_edges == 0 else 'kamada'

    if layout == 'star':
        return _star_layout(G, ads)

    if layout == 'kamada':
        for u, v, d in G.edges(data=True):
            G[u][v]['inv_dist'] = 1.0 / max(d['distance'], 1e-6)
        init  = nx.spring_layout(G, seed=7, weight='inv_dist', k=0.5, iterations=80)
        return nx.kamada_kawai_layout(G, weight='inv_dist', pos=init)

    if layout == 'spring':
        return nx.spring_layout(G, seed=0, weight=None, k=0.55, iterations=120)

    if layout == 'xy':
        return {n: G.nodes[n]['pos'][:2] for n in G.nodes()}

    raise ValueError(f"Unknown layout '{layout}'")


# ─────────────────────────────────────────────────────────────────────────────
# Draw
# ─────────────────────────────────────────────────────────────────────────────

def draw_graph(G: nx.Graph, roles: dict,
               title: str = None,
               figsize=(4, 4),
               node_size: int = 280,
               layout: str = 'kamada',
               save_path: str = None,
               dpi: int = 200,
               show: bool = True) -> None:
    """
    Draw the graph with the publication-quality style.

    layout : 'kamada'  – Kamada-Kawai (bond-distance weighted, recommended)
             'spring'  – force-directed (faster for large graphs)
             'xy'      – project real XY coordinates directly
    """
    # ── 2D layout ──────────────────────────────────────────────────────────
    pos2d = _make_layout(G, layout)

    # ── figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── edges ──────────────────────────────────────────────────────────────
    nx.draw_networkx_edges(
        G, pos2d, ax=ax,
        edge_color='#888888',
        width=1.3,
        alpha=0.75,
    )

    # ── nodes, drawn back-to-front by zorder ───────────────────────────────
    for role in ('non_neighbor', 'ads_neighbor', 'adsorbate'):
        nodes = [n for n, r in roles.items() if r == role]
        if not nodes:
            continue
        fc, ec, zo = _ROLE_STYLE[role]
        nx.draw_networkx_nodes(
            G, pos2d, nodelist=nodes, ax=ax,
            node_color=fc,
            edgecolors=ec,
            linewidths=1.4,
            node_size=node_size,
        )

    # ── legend ──────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(fc='#2ECC40', ec='#1A5C28', lw=1.2, label='Adsorbate'),
        mpatches.Patch(fc='#E74C3C', ec='#7B1010', lw=1.2, label='Ads. neighbor'),
        mpatches.Patch(fc='#F8F8F8', ec='#333333', lw=1.2, label='Non-neighbor'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              fontsize=7, framealpha=0.85, edgecolor='#CCCCCC',
              handlelength=1.2, handleheight=1.0)

    if title:
        ax.set_title(title, fontsize=9, pad=6, color='#222222')

    plt.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ASE structure view — adsorbate + neighbors highlighted
# ─────────────────────────────────────────────────────────────────────────────

def draw_atoms_structure(atoms, roles: dict,
                         title: str = None,
                         figsize=(4, 4),
                         save_path: str = None,
                         dpi: int = 300,
                         show: bool = True) -> None:
    """
    Render the real atomic structure using ase.visualize.plot.plot_atoms,
    coloring each atom by its graph role:
      green  – adsorbate
      red    – adsorbate neighbor (within cutoff)
      white  – rest of the slab (not in graph)
    """
    from ase.visualize.plot import plot_atoms

    # build per-atom color list (all atoms, not just graph nodes)
    node_role = {n: r for n, r in roles.items()}
    fc_map = {
        'adsorbate':    '#2ECC40',
        'ads_neighbor': '#E74C3C',
        'non_neighbor': '#AAAAAA',   # shouldn't appear — not in graph
    }
    colors = []
    for i in range(len(atoms)):
        if i in node_role:
            colors.append(fc_map[node_role[i]])
        else:
            colors.append('#DDDDDD')   # slab atoms outside the cutoff sphere

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    plot_atoms(atoms, ax=ax, colors=colors, rotation='0x,0y,0z')

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, pad=6, color='#222222')

    # legend
    legend_items = [
        mpatches.Patch(fc='#2ECC40', ec='#1A5C28', lw=1.0, label='Adsorbate'),
        mpatches.Patch(fc='#E74C3C', ec='#7B1010', lw=1.0, label='Ads. neighbor'),
        mpatches.Patch(fc='#DDDDDD', ec='#888888', lw=1.0, label='Slab'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              fontsize=7, framealpha=0.85, edgecolor='#CCCCCC',
              handlelength=1.2, handleheight=1.0)

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
    parser = argparse.ArgumentParser(description='Draw adsorbate graph.')
    parser.add_argument('--traj',   default=f'{HERE}/../proj_metals/dataset/images_refined.traj')
    parser.add_argument('--idx',    type=int, default=26000, help='Structure index in traj')
    parser.add_argument('--cutoff', type=float, default=5, help='Bond cutoff distance (Å)')

    parser.add_argument('--layout', default='kamada', choices=['auto','star','kamada','spring','xy'])
    parser.add_argument('--out',    default=f'{HERE}/graph_style.png', help='Output file')
    parser.add_argument('--dpi',    type=int, default=300)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    images = read(args.traj, ':')
    atoms  = images[args.idx]

    from collections import Counter
    syms = Counter(atoms.get_chemical_symbols())
    title = '  +  '.join(f'{v}×{k}' for k, v in syms.items())

    G = build_graph(atoms, cutoff=args.cutoff)
    roles = node_roles(G)

    from ase.visualize import view
    from ase import Atoms
    involved = sorted(roles.keys())
    view(atoms[involved])

    from collections import Counter
    print(f"Graph (neighbors only): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Roles:", Counter(roles.values()))

    # full graph — all slab atoms, same role coloring
    G_full  = build_full_graph(atoms, cutoff=args.cutoff)
    roles_full = node_roles(G_full)   # non_neighbor nodes now appear
    print(f"Graph (full slab):       {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

    # ── side-by-side comparison plot ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('white')

    def _draw_on_ax_auto(ax, Gg, rroles, subtitle, layout='auto'):
        pos2d = _make_layout(Gg, layout)
        nx.draw_networkx_edges(Gg, pos2d, ax=ax, edge_color='#888888', width=1.1, alpha=0.6)
        for role in ('non_neighbor', 'ads_neighbor', 'adsorbate'):
            nodes = [n for n, r in rroles.items() if r == role]
            if not nodes:
                continue
            fc, ec, _ = _ROLE_STYLE[role]
            nx.draw_networkx_nodes(Gg, pos2d, nodelist=nodes, ax=ax,
                                   node_color=fc, edgecolors=ec,
                                   linewidths=1.3, node_size=220)
        ax.set_title(subtitle, fontsize=9, pad=5, color='#222222')
        ax.set_aspect('equal'); ax.axis('off')

    _draw_on_ax_auto(axes[0], G_full, roles_full, 'Full slab  (with non-neighbors)', layout='kamada')
    _draw_on_ax_auto(axes[1], G,      roles,      'AdsNet graph  (neighbors only)', layout='kamada')

    legend_items = [
        mpatches.Patch(fc='#2ECC40', ec='#1A5C28', lw=1.0, label='Adsorbate'),
        mpatches.Patch(fc='#E74C3C', ec='#7B1010', lw=1.0, label='Ads. neighbor'),
        mpatches.Patch(fc='#F8F8F8', ec='#333333', lw=1.0, label='Non-neighbor'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, edgecolor='#CCCCCC')
    fig.suptitle(title, fontsize=9, y=1.01, color='#333333')
    plt.tight_layout(pad=0.4)

    compare_out = args.out.replace('.png', '_compare.png')
    fig.savefig(compare_out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved → {compare_out}")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # ── individual saves ─────────────────────────────────────────────────
    atoms_out = args.out.replace('.png', '_atoms.png')
    draw_atoms_structure(atoms, roles,
                         title=title,
                         save_path=atoms_out,
                         dpi=args.dpi,
                         show=not args.no_show)

    draw_graph(G, roles,
               title=title,
               layout=args.layout,
               save_path=args.out,
               dpi=args.dpi,
               show=not args.no_show)


if __name__ == '__main__':
    main()
