"""
plot_all_isoT.py

Plots percolation order parameter (S_max/N) and mean H-bond coordination
number (n_HB) vs density for ALL available temperatures on the same axes,
revealing consistent sigmoidal patterns across isotherms.

Usage:  python plot_all_isoT.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

from H2O import SHPDataSet

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCAN_ROOT = "C:/Users/yupei/Desktop/SHP/SCAN"

TEMPERATURES = [300, 400, 500, 600, 700, 800, 900, 1000]

# Per-temperature H-bond cutoffs from O-H RDF 2nd peak (H...O distance, Angstrom)
CUTOFF_VALUES = [1.88, 1.88, 2.01, 2.01, 2.01, 2.01, 2.08, 2.08]
CUTOFF_MEAN   = float(np.mean(CUTOFF_VALUES))
CUTOFF_SE     = float(np.std(CUTOFF_VALUES, ddof=1) / np.sqrt(len(CUTOFF_VALUES)))

# Minimum equilibrated frames required to trust a data point
MIN_EQ_FRAMES = 50

BOHR_TO_ANG      = 0.529177210903
ANG3_TO_CM3      = 1e-24
AVOGADRO         = 6.02214076e23
WATER_MOLAR_MASS = 18.015

DENSITY_RE = re.compile(r"^\d+$")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def read_cell_volume(md_path):
    volumes, h_buf = [], []
    try:
        with open(md_path) as fh:
            for line in fh:
                if "<-- h" in line:
                    v = line.split()
                    h_buf.append([float(v[0]), float(v[1]), float(v[2])])
                    if len(h_buf) == 3:
                        V = abs(np.linalg.det(np.array(h_buf))) * BOHR_TO_ANG**3
                        volumes.append(V)
                        h_buf = []
    except Exception:
        return None
    if not volumes:
        return None
    half = len(volumes) // 2
    return float(np.mean(volumes[half:]))



def load_oh_rdf_cutoff(cache_npz, r_min=1.2, r_max=2.5):
    """First peak of g_OH(r) for r_min < r < r_max (the H-bond O...H peak)."""
    try:
        d = np.load(cache_npz, allow_pickle=True)
        r, g = d["oh_rdf_r"], d["oh_rdf_g"]
        mask = (r > r_min) & (r < r_max)
        r_m, g_m = r[mask], g[mask]
        if len(r_m) < 3:
            return None
        return float(r_m[int(np.argmax(g_m))])
    except Exception:
        return None


def compute_percolation_stats(df, cutoff):
    n_frames = int(df["frame"].max()) + 1
    start    = n_frames // 2
    df_eq    = df[df["frame"] >= start]
    valid    = df_eq[df_eq["dist"] <= cutoff]
    grouped  = valid.groupby("frame")
    sizes    = []
    for f in range(start, n_frames):
        if f in grouped.groups:
            fd      = grouped.get_group(f)
            n_total = int(fd.iloc[0]["n_mol"])
            G       = nx.Graph()
            G.add_edges_from(zip(fd["u"], fd["v"]))
            largest = (len(max(nx.connected_components(G), key=len))
                       if len(G) > 0 else 1)
            sizes.append(largest / n_total)
        else:
            sizes.append(1.0 / int(df.iloc[0]["n_mol"]))
    if not sizes:
        return float("nan"), float("nan")
    return float(np.mean(sizes)), float(np.std(sizes))


def compute_coordination(df, cutoff):
    n_frames = int(df["frame"].max()) + 1
    start    = n_frames // 2
    n_mol    = int(df["n_mol"].iloc[0])
    df_eq    = df[df["frame"] >= start]
    grouped  = df_eq[df_eq["dist"] <= cutoff].groupby("frame")
    per_frame = [
        2.0 * len(grouped.get_group(f)) / n_mol if f in grouped.groups else 0.0
        for f in range(start, n_frames)
    ]
    if not per_frame:
        return float("nan"), float("nan")
    return float(np.mean(per_frame)), float(np.std(per_frame))


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT ONE ISOTHERM
# ─────────────────────────────────────────────────────────────────────────────

def collect_isotherm(scan_root, temp_k):
    density_dirs = sorted(
        [d for d in os.listdir(scan_root)
         if os.path.isdir(os.path.join(scan_root, d)) and DENSITY_RE.match(d)],
        key=int,
    )
    rows = []
    for dens in density_dirs:
        folder  = os.path.join(scan_root, dens, f"{temp_k}K")
        md_path = os.path.join(folder, "small.md")
        if not os.path.isdir(folder) or not os.path.exists(md_path):
            continue

        dataset = SHPDataSet(os.path.join(scan_root, dens))
        df = dataset.get_temperature_data(temp_k)
        if df is None or df.empty:
            continue

        n_frames_total = int(df["frame"].max()) + 1
        n_eq = n_frames_total - n_frames_total // 2
        if n_eq < MIN_EQ_FRAMES:
            print(f"    [skip] {dens}/{temp_k}K: only {n_eq} eq frames")
            continue

        s, s_err = compute_percolation_stats(df, CUTOFF_MEAN)
        if np.isnan(s_err):
            continue

        n_hb, n_hb_err = compute_coordination(df, CUTOFF_MEAN)

        avg_V = read_cell_volume(md_path)
        if avg_V is None:
            continue

        n_mol = int(df["n_mol"].iloc[0])
        rho   = (n_mol * WATER_MOLAR_MASS / AVOGADRO) / (avg_V * ANG3_TO_CM3)
        rows.append((rho, s, s_err, n_hb, n_hb_err))

    return sorted(rows, key=lambda r: r[0])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Fixed cutoff : {CUTOFF_MEAN:.4f} +/- {CUTOFF_SE:.4f} A")
    print(f"Min eq frames: {MIN_EQ_FRAMES}\n")

    all_data = {}   # temp -> [(rho, S, chi, n_HB), ...]
    for temp_k in TEMPERATURES:
        print(f"T = {temp_k} K ...", end=" ", flush=True)
        rows = collect_isotherm(SCAN_ROOT, temp_k)
        if rows:
            all_data[temp_k] = rows
            print(f"{len(rows)} density points")
        else:
            print("no data")

    if not all_data:
        print("No data collected.")
        import sys; sys.exit(1)

    # Colormap: cold (blue) -> hot (red)
    cmap   = cm.get_cmap("coolwarm", len(TEMPERATURES))
    colors = {t: cmap(i) for i, t in enumerate(TEMPERATURES)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for temp_k, rows in all_data.items():
        rhos     = [r[0] for r in rows]
        Ss       = [r[1] for r in rows]
        S_errs   = [r[2] for r in rows]
        n_hbs    = [r[3] for r in rows]
        nhb_errs = [r[4] for r in rows]
        c        = colors[temp_k]
        ax1.errorbar(rhos, Ss,    yerr=S_errs,   fmt="o-", color=c,
                     linewidth=1.8, markersize=5, capsize=3, elinewidth=0.8,
                     label=f"{temp_k} K")
        ax2.errorbar(rhos, n_hbs, yerr=nhb_errs, fmt="o-", color=c,
                     linewidth=1.8, markersize=5, capsize=3, elinewidth=0.8,
                     label=f"{temp_k} K")

    # Panel 1: S vs density
    ax1.axhline(0.5, color="grey", linestyle="--", linewidth=1.0,
                label="S = 0.5 midpoint")
    ax1.set_xlabel("Density (g/cc)", fontsize=13)
    ax1.set_ylabel(r"Mean largest cluster  $\langle S_{max} / N \rangle$", fontsize=13)
    ax1.set_title("Percolation order parameter\n(all isotherms)", fontsize=12)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.4)

    # Panel 2: n_HB vs density
    ax2.set_xlabel("Density (g/cc)", fontsize=13)
    ax2.set_ylabel(r"Mean H-bond coord.  $\langle n_{HB} \rangle$", fontsize=13)
    ax2.set_title("H-bond coordination number\n(all isotherms)", fontsize=12)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.4)

    plt.suptitle(
        f"H-bond network vs density  -  all temperatures  "
        f"(cutoff = {CUTOFF_MEAN:.3f} +/- {CUTOFF_SE:.3f} A,  2nd-half frames only)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    fname = "all_isoT_overlay.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {fname}")
    plt.show()
