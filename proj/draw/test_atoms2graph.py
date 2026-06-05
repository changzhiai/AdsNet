"""
ASE Atoms → NetworkX Graph
===========================
Converts any ASE Atoms object (molecule or periodic crystal) into a
NetworkX graph with node features and a labelled adjacency matrix.

Edges are formed between atoms whose distance is within
  scale_factor × (r_cov_i + r_cov_j)
using ASE covalent radii. For periodic structures all minimum-image
distances are used automatically via ASE's neighbor list.

Usage
-----
    from geometry_graph import atoms_to_graph, plot_graph_3d
    import networkx as nx

    # --- molecule example ---
    from ase.build import molecule
    atoms = molecule("H2O")

    # --- bulk crystal example ---
    from ase.build import bulk
    atoms = bulk("Cu", "fcc", a=3.6).repeat([2, 2, 2])

    # --- read from file (CIF / XYZ / POSCAR / etc.) ---
    from ase.io import read
    atoms = read("structure.cif")

    G = atoms_to_graph(atoms)
    plot_graph_3d(G)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ase import Atoms
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.neighborlist import NeighborList, natural_cutoffs
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────────────────────

def atoms_to_graph(atoms: Atoms, scale_factor: float = 1.1) -> nx.Graph:
    """
    Convert an ASE Atoms object to a NetworkX graph.

    Parameters
    ----------
    atoms        : ASE Atoms object (molecule or periodic crystal)
    scale_factor : multiplier on covalent-radius cutoff for bond detection
                   1.1 is a reasonable default; increase for loosely bonded
                   systems, decrease to remove weak/long contacts.

    Returns
    -------
    G : nx.Graph
        Node attributes
          - symbol      : str   chemical symbol, e.g. "Fe"
          - atomic_num  : int   atomic number
          - mass        : float atomic mass (a.m.u.)
          - pos         : np.ndarray  shape (3,) Cartesian coordinates (Å)
          - cov_radius  : float covalent radius (Å)

        Edge attributes
          - distance    : float interatomic distance (Å)
          - offset      : np.ndarray  periodic image offset vector (integers)
                          [0,0,0] for molecules / non-periodic pairs
    """
    cutoffs = natural_cutoffs(atoms, mult=scale_factor)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    G = nx.Graph()

    # ── nodes ────────────────────────────────────────────────────────────────
    for i, atom in enumerate(atoms):
        G.add_node(
            i,
            symbol     = atom.symbol,
            atomic_num = int(atom.number),
            mass       = float(atom.mass),
            pos        = atom.position.copy(),          # Å
            cov_radius = float(covalent_radii[atom.number]),
        )

    # ── edges ────────────────────────────────────────────────────────────────
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            if G.has_edge(i, j):
                continue  # undirected: skip duplicate
            # distance accounting for periodic images
            disp = (atoms.positions[j]
                    + offset @ atoms.get_cell()
                    - atoms.positions[i])
            dist = float(np.linalg.norm(disp))
            G.add_edge(i, j, distance=round(dist, 4), offset=offset)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(G: nx.Graph) -> None:
    """Print node features, adjacency matrix, and edge list."""
    print("=" * 60)
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}")
    print("=" * 60)

    print("\nNode features:")
    print(f"  {'ID':>4}  {'Symbol':6}  {'Z':>3}  {'Mass':>8}  {'Cov_r':>6}  pos (Å)")
    for n, d in G.nodes(data=True):
        p = d["pos"]
        print(f"  {n:>4}  {d['symbol']:6}  {d['atomic_num']:>3}  "
              f"{d['mass']:>8.3f}  {d['cov_radius']:>6.3f}  "
              f"({p[0]:6.3f}, {p[1]:6.3f}, {p[2]:6.3f})")

    print("\nAdjacency matrix:")
    A = nx.to_numpy_array(G, dtype=int)
    print(A)

    print("\nEdge list (with bond distances):")
    for u, v, d in G.edges(data=True):
        print(f"  ({u:>3}, {v:>3})  {d['distance']:.4f} Å  offset={d['offset']}")


def node_feature_matrix(G: nx.Graph) -> np.ndarray:
    """
    Build a node feature matrix X of shape (N, F).

    Features per node (F = 3 + 3 = 6):
      [atomic_num, mass, cov_radius,  x, y, z]

    You can extend this with one-hot encoding, electronegativity, etc.
    """
    rows = []
    for n in sorted(G.nodes()):
        d = G.nodes[n]
        rows.append([
            d["atomic_num"],
            d["mass"],
            d["cov_radius"],
            *d["pos"],
        ])
    return np.array(rows, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

# CPK-like colour palette (extend as needed)
_CPK = {
    "H":  "#FFFFFF", "C":  "#404040", "N":  "#3050F8",
    "O":  "#FF0D0D", "F":  "#90E050", "P":  "#FF8000",
    "S":  "#FFFF30", "Cl": "#1FF01F", "Fe": "#E06633",
    "Cu": "#C88033", "Zn": "#7D80B0", "default": "#BB88BB",
}

def _node_color(symbol: str) -> str:
    return _CPK.get(symbol, _CPK["default"])


def plot_graph_3d(G: nx.Graph, title: str = "Geometry Graph",
                  save_path: str = None) -> None:
    """3-D scatter + bond plot of the graph."""
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")

    pos = {n: G.nodes[n]["pos"] for n in G.nodes()}

    # edges
    for u, v in G.edges():
        xs = [pos[u][0], pos[v][0]]
        ys = [pos[u][1], pos[v][1]]
        zs = [pos[u][2], pos[v][2]]
        ax.plot(xs, ys, zs, color="#888888", linewidth=1.4, zorder=1)

    # nodes
    seen_labels = set()
    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        sym = d["symbol"]
        col = _node_color(sym)
        label = sym if sym not in seen_labels else None
        seen_labels.add(sym)
        ax.scatter(x, y, z, s=200, c=col, edgecolors="black",
                   linewidths=0.8, zorder=5, label=label)
        ax.text(x, y, z, f" {n}", fontsize=7, color="#222222")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.legend(loc="upper left", fontsize=8, title="Elements")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved → {save_path}")
    plt.show()


def plot_graph_2d(G: nx.Graph, title: str = "Geometry Graph (2D)",
                  layout: str = "spring", projection: str = "xy",
                  save_path: str = None) -> None:
    """
    2-D graph plot with three layout options:

    layout="coords"  – project actual 3-D atomic coordinates onto a 2-D plane
                       (use `projection` to pick "xy", "xz", or "yz")
    layout="spring"  – force-directed spring layout (good for topology view)
    layout="kamada"  – Kamada-Kawai layout weighted by bond distance
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # ── choose 2-D positions ─────────────────────────────────────────────────
    if layout == "coords":
        plane = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[projection]
        pos2d = {n: G.nodes[n]["pos"][list(plane)] for n in G.nodes()}
        xlabel, ylabel = [s + " (Å)" for s in projection]
    elif layout == "spring":
        pos2d = nx.spring_layout(G, seed=42, weight=None)
        xlabel = ylabel = ""
    elif layout == "kamada":
        # weight = 1/distance so shorter bonds pull harder
        for u, v, d in G.edges(data=True):
            G[u][v]["inv_dist"] = 1.0 / max(d["distance"], 1e-6)
        pos2d  = nx.kamada_kawai_layout(G, weight="inv_dist")
        xlabel = ylabel = ""
    else:
        raise ValueError(f"Unknown layout '{layout}'. Choose coords/spring/kamada.")

    # ── draw edges with distance labels ─────────────────────────────────────
    nx.draw_networkx_edges(G, pos2d, ax=ax,
                           edge_color="#AAAAAA", width=1.8, alpha=0.8)
    edge_labels = {(u, v): f"{d['distance']:.2f}Å"
                   for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos2d, edge_labels=edge_labels,
                                 ax=ax, font_size=6, font_color="#555555")

    # ── draw nodes grouped by element ───────────────────────────────────────
    elements = set(nx.get_node_attributes(G, "symbol").values())
    for sym in sorted(elements):
        node_ids = [n for n, d in G.nodes(data=True) if d["symbol"] == sym]
        nx.draw_networkx_nodes(G, pos2d, nodelist=node_ids, ax=ax,
                               node_color=_node_color(sym), node_size=600,
                               edgecolors="black", linewidths=0.8,
                               label=sym)

    # ── node index labels ────────────────────────────────────────────────────
    nx.draw_networkx_labels(G, pos2d, ax=ax, font_size=8,
                            font_color="black", font_weight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(title="Elements", fontsize=8, loc="best")
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Demo  (run this file directly to see an example)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ase.build import molecule, bulk
    from ase.io import read   # noqa – available if you want to load a file

    # ── Example 1: small molecule ────────────────────────────────────────────
    # print("\n── Example 1: Ethanol (C2H5OH) ─────────────────────────────────")
    # atoms = molecule("CH3CH2OH")
    images = read(f'{path}/../proj_metals/dataset/total.traj', ':')
    structure_index = 100
    atoms = images[structure_index]
    G = atoms_to_graph(atoms, scale_factor=1.1)
    print_summary(G)

    # ── 2D: three layouts side-by-side ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ethanol (C2H5OH) — 2D Graph Views", fontsize=14, fontweight="bold")

    configs = [
        ("coords",  "xy",   "Coordinate projection (XY plane)"),
        ("spring",  None,   "Spring / force-directed layout"),
        ("kamada",  None,   "Kamada-Kawai layout (bond-distance weighted)"),
    ]

    for ax, (layout, proj, subtitle) in zip(axes, configs):
        if layout == "coords":
            plane = (0, 1)
            pos2d = {n: G.nodes[n]["pos"][list(plane)] for n in G.nodes()}
            ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        elif layout == "spring":
            pos2d = nx.spring_layout(G, seed=42, weight=None)
        else:  # kamada
            for u, v, d in G.edges(data=True):
                G[u][v]["inv_dist"] = 1.0 / max(d["distance"], 1e-6)
            pos2d = nx.kamada_kawai_layout(G, weight="inv_dist")

        nx.draw_networkx_edges(G, pos2d, ax=ax,
                               edge_color="#AAAAAA", width=1.8, alpha=0.8)
        edge_labels = {(u, v): f"{d['distance']:.2f}Å"
                       for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos2d, edge_labels=edge_labels,
                                     ax=ax, font_size=6, font_color="#555555")

        elements = set(nx.get_node_attributes(G, "symbol").values())
        for sym in sorted(elements):
            node_ids = [n for n, d in G.nodes(data=True) if d["symbol"] == sym]
            nx.draw_networkx_nodes(G, pos2d, nodelist=node_ids, ax=ax,
                                   node_color=_node_color(sym), node_size=500,
                                   edgecolors="black", linewidths=0.8, label=sym)
        nx.draw_networkx_labels(G, pos2d, ax=ax,
                                font_size=8, font_weight="bold")

        ax.set_title(subtitle, fontsize=10)
        ax.legend(title="Elements", fontsize=7)
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.tight_layout()
    plt.savefig(f"{path}/geometry_graph_2d.png", dpi=300)
    print("2D plot saved → geometry_graph_2d.png")
    plt.show()

    # ── Example 2: periodic crystal ─────────────────────────────────────────
    print("\n── Example 2: FCC Cu (2×2×2 supercell) ─────────────────────────")
    atoms2 = bulk("Cu", "fcc", a=3.6).repeat([2, 2, 2])
    G2 = atoms_to_graph(atoms2, scale_factor=1.1)
    print(f"  Nodes={G2.number_of_nodes()}  Edges={G2.number_of_edges()}")
    plot_graph_2d(G2, title="FCC Cu 2×2×2 — Spring Layout", layout="spring",
                  save_path=f"{path}/geometry_graph_2d_cu.png")

    # ── Example 3: load from file ─────────────────────────────────────────────
    # atoms3 = read("your_structure.cif")
    # G3 = atoms_to_graph(atoms3)
    # plot_graph_2d(G3, layout="kamada")