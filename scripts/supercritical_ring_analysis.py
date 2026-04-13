"""
supercritical_ring_analysis.py

Compares H-bond ring-size distributions between:
  Supercritical  — SCAN/130/800K/small.md  (130% density, 800 K, T >> Tc=647K)
  Subcritical    — SCAN/100/300K/small.md  (ambient density, 300 K)

Panels:
  1. Adjacency matrix — supercritical (last frame)
  2. Adjacency matrix — subcritical   (last frame)
  3. Ring-size distribution: NetworkX simple_cycles, both conditions overlaid

Physical note
-------------
k=2  mutual pairs       k=5  pentagons (liquid water)
k=3  triangles          k=6  hexagons  (ice Ih)
k=4  squares (high-P)   k>=7 large floppy rings
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ase.io import read
from collections import Counter

# -- Configuration -------------------------------------------------------------

CONDITIONS = {
    "Supercritical (130%, 800 K)": r"C:\Users\yupei\Desktop\SHP\SCAN\130\800K\small.md",
    "Subcritical (100%, 300 K)":   r"C:\Users\yupei\Desktop\SHP\SCAN\100\300K\small.md",
}
MAX_RING   = 8
FRAME_SKIP = 5

# -- Core functions ------------------------------------------------------------

def build_hbond_graph(atoms):
    o_idx = [a.index for a in atoms if a.symbol == 'O']
    h_idx = [a.index for a in atoms if a.symbol == 'H']
    G = nx.DiGraph()
    G.add_nodes_from(o_idx)
    for h in h_idx:
        dists  = atoms.get_distances(h, o_idx, mic=True)
        ranked = np.argsort(dists)
        donor, acceptor = o_idx[ranked[0]], o_idx[ranked[1]]
        if not G.has_edge(donor, acceptor):
            G.add_edge(donor, acceptor)
    return G

def ring_counts_exact(G, max_k=MAX_RING):
    try:
        cycles = list(nx.simple_cycles(G, length_bound=max_k))
    except TypeError:
        cycles = [c for c in nx.simple_cycles(G) if len(c) <= max_k]
    counts = Counter(len(c) for c in cycles)
    return {k: counts.get(k, 0) for k in range(2, max_k + 1)}

def analyse_trajectory(label, path):
    print(f"\n[{label}]  Loading {path} ...")
    traj     = read(path, format='castep-md', index=':')
    n_frames = len(traj)
    fids     = list(range(0, n_frames, FRAME_SKIP))
    print(f"  {n_frames} frames, analysing {len(fids)} (every {FRAME_SKIP}).")

    exact_tot   = {k: 0.0 for k in range(2, MAX_RING + 1)}
    n_mol       = None
    last_adj    = None
    last_nbonds = 0

    for i, fi in enumerate(fids):
        G = build_hbond_graph(traj[fi])
        if n_mol is None:
            n_mol = G.number_of_nodes()
        for k, v in ring_counts_exact(G).items():
            exact_tot[k] += v
        if i == len(fids) - 1:
            last_adj    = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
            last_nbonds = G.number_of_edges()
        print(f"  frame {fi+1}/{n_frames}", end='\r')

    print()
    ns = len(fids)
    ks = list(range(2, MAX_RING + 1))
    return {
        'label':     label,
        'adj':       last_adj,
        'n_mol':     n_mol,
        'n_bonds':   last_nbonds,
        'ks':        ks,
        'exact_avg': [exact_tot[k] / ns / n_mol for k in ks],
        'n_frames':  ns,
    }

# -- Run analysis --------------------------------------------------------------

results = [analyse_trajectory(lbl, path) for lbl, path in CONDITIONS.items()]
print("\nAll done.\n")

# -- Plotting ------------------------------------------------------------------

COLORS = ['#2166ac', '#d6604d']   # blue = supercritical, red = subcritical
ks = results[0]['ks']
w  = 0.35
x  = np.array(ks)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("H-Bond Network: Supercritical vs Subcritical Water", fontsize=14)

# Panels 0 & 1: Adjacency matrices --------------------------------------------
for col, res in enumerate(results):
    ax = axes[col]
    adj = res['adj']
    n   = adj.shape[0]
    rows_idx, cols_idx = np.where(adj == 1)
    ax.scatter(cols_idx, rows_idx, s=4, color='black', marker='.')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(f"Adjacency matrix\n{res['label']}\n"
                 f"({res['n_mol']} O atoms, {res['n_bonds']} H-bonds, last frame)")
    ax.set_xlabel("Oxygen index j  (acceptor)")
    ax.set_ylabel("Oxygen index i  (donor)")
    ax.set_aspect('equal')

# Panel 2: Ring-size distribution (exact) ------------------------------------
ax3 = axes[2]
for i, res in enumerate(results):
    offset = (i - 0.5) * w
    ax3.bar(x + offset, res['exact_avg'], width=w,
            label=res['label'],
            color=COLORS[i], alpha=0.85, edgecolor='k', linewidth=0.5)
ax3.set_title("Ring-size distribution\n(NetworkX simple_cycles, exact)")
ax3.set_xlabel("Ring size  k")
ax3.set_ylabel("Mean directed cycles per molecule")
ax3.set_xticks(ks)
ax3.legend()
ax3.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_path = r"C:\Users\yupei\Desktop\SHP\supercritical_ring_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()

# -- Summary table ------------------------------------------------------------
print(f"Saved: {out_path}\n")
notes = {2:"mutual pairs", 3:"triangles", 4:"squares",
         5:"pentagons", 6:"hexagons", 7:"", 8:""}
print(f"{'k':>4}  {'Supercrit/mol':>15}  {'Subcrit/mol':>13}  Note")
print("-" * 52)
for ki, k in enumerate(ks):
    print(f"{k:>4}  {results[0]['exact_avg'][ki]:>15.4f}  "
          f"{results[1]['exact_avg'][ki]:>13.4f}  {notes[k]}")
