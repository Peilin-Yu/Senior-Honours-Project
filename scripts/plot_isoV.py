"""
plot_isoV.py

Plots percolation susceptibility chi and largest cluster size vs temperature
at a fixed density (volume), scanning across all temperature subfolders.

Usage:  python plot_isoV.py [density_label]
        default density_label = 140
        e.g.   python plot_isoV.py 130
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from H2O import SHPDataSet

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCAN_ROOT    = "C:/Users/yupei/Desktop/SHP/SCAN"
DENSITY_LABEL = sys.argv[1] if len(sys.argv) > 1 else "140"

# Fixed H-bond cutoff (mean +/- SE from percolation analysis)
CUTOFF_VALUES = [1.88, 1.88, 2.01, 2.01, 2.01, 2.01, 2.08, 2.08]
CUTOFF_MEAN   = float(np.mean(CUTOFF_VALUES))
CUTOFF_SE     = float(np.std(CUTOFF_VALUES, ddof=1) / np.sqrt(len(CUTOFF_VALUES)))

# Unit conversions
BOHR_TO_ANG      = 0.529177210903
ANG3_TO_CM3      = 1e-24
AVOGADRO         = 6.02214076e23
WATER_MOLAR_MASS = 18.015

TEMP_RE = re.compile(r"^(\d+)K$")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def read_cell_volume(md_path, max_frames=300):
    """Read average cell volume (Ang^3) from CASTEP .md lattice vectors.
    Only the second half of frames is used to skip the warm-up period."""
    volumes, h_buf = [], []
    try:
        with open(md_path) as fh:
            for line in fh:
                if len(volumes) >= max_frames:
                    break
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


def compute_percolation_stats(df, cutoff):
    """Return (mean_S, chi) over equilibrated frames (second half only).

    mean_S : mean of S_max/N  — order parameter, shows sigmoidal step
    chi    : std  of S_max/N  — susceptibility, shows peak at threshold
    """
    n_frames  = int(df["frame"].max()) + 1
    start     = n_frames // 2          # skip first half (warm-up)
    df_eq     = df[df["frame"] >= start]
    valid     = df_eq[df_eq["dist"] <= cutoff]
    grouped   = valid.groupby("frame")
    sizes     = []
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


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT DATA
# ─────────────────────────────────────────────────────────────────────────────

def collect(scan_root, density_label):
    folder_root = os.path.join(scan_root, density_label)
    if not os.path.isdir(folder_root):
        print(f"Density folder not found: {folder_root}")
        sys.exit(1)

    # Discover temperature subfolders
    temp_dirs = sorted(
        [int(m.group(1))
         for name in os.listdir(folder_root)
         if os.path.isdir(os.path.join(folder_root, name))
         for m in [TEMP_RE.match(name)] if m],
    )

    dataset = SHPDataSet(folder_root)
    rows = []

    for temp_k in temp_dirs:
        md_path = os.path.join(folder_root, f"{temp_k}K", "small.md")
        if not os.path.exists(md_path):
            continue

        print(f"  {density_label}/{temp_k}K ", end="", flush=True)

        df = dataset.get_temperature_data(temp_k)
        if df is None or df.empty:
            print("no data -- skipped.")
            continue

        s_mid, chi_mid = compute_percolation_stats(df, CUTOFF_MEAN)
        s_lo,  chi_lo  = compute_percolation_stats(df, CUTOFF_MEAN - CUTOFF_SE)
        s_hi,  chi_hi  = compute_percolation_stats(df, CUTOFF_MEAN + CUTOFF_SE)

        avg_V = read_cell_volume(md_path)
        if avg_V is None:
            print("no volume -- skipped.")
            continue

        n_mol = int(df["n_mol"].iloc[0])
        density_gcc = (n_mol * WATER_MOLAR_MASS / AVOGADRO) / (avg_V * ANG3_TO_CM3)

        print(f"chi={chi_mid:.4f}  S={s_mid:.4f}  rho={density_gcc:.4f} g/cc")

        rows.append({
            "temp_k":      temp_k,
            "density_gcc": density_gcc,
            "chi_mid":     chi_mid,
            "chi_lo":      chi_lo,
            "chi_hi":      chi_hi,
            "s_mid":       s_mid,
            "s_lo":        s_lo,
            "s_hi":        s_hi,
        })

    return sorted(rows, key=lambda r: r["temp_k"])


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows, density_label):
    temps  = [r["temp_k"]      for r in rows]
    chi    = [r["chi_mid"]     for r in rows]
    chi_lo = [r["chi_lo"]      for r in rows]
    chi_hi = [r["chi_hi"]      for r in rows]
    s      = [r["s_mid"]       for r in rows]
    s_lo   = [r["s_lo"]        for r in rows]
    s_hi   = [r["s_hi"]        for r in rows]

    # Use the mean density across temperatures for the title
    mean_rho = float(np.mean([r["density_gcc"] for r in rows]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    cutoff_label = f"Cutoff = {CUTOFF_MEAN:.3f} +/- {CUTOFF_SE:.3f} A"

    # ── Left panel: susceptibility (chi) — shows a peak ──────────────────────
    ax1.fill_between(temps, chi_lo, chi_hi, alpha=0.2, color="steelblue",
                     label="Cutoff uncertainty")
    ax1.plot(temps, chi, marker="o", color="steelblue", linewidth=2,
             label=cutoff_label)
    ax1.set_xlabel("Temperature (K)", fontsize=13)
    ax1.set_ylabel(r"Susceptibility $\chi$ = std($S_{max}$ / N)", fontsize=13)
    ax1.set_title(f"Susceptibility vs Temperature  (density folder {density_label})",
                  fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    # ── Right panel: mean largest cluster size — shows a sigmoidal step ──────
    ax2.fill_between(temps, s_lo, s_hi, alpha=0.2, color="crimson",
                     label="Cutoff uncertainty")
    ax2.plot(temps, s, marker="o", color="crimson", linewidth=2,
             label=cutoff_label)
    ax2.set_xlabel("Temperature (K)", fontsize=13)
    ax2.set_ylabel(r"Mean largest cluster  $\langle S_{max} / N \rangle$",
                   fontsize=13)
    ax2.set_title(f"Largest cluster size vs Temperature  (density folder {density_label})",
                  fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.suptitle(
        f"H-bond percolation  (density folder {density_label},  "
        f"~{mean_rho:.3f} g/cc)  --  equilibrated frames only (2nd half)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    fname = f"chi_vs_T_{density_label}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {fname}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Density folder : {DENSITY_LABEL}")
    print(f"Fixed cutoff   : {CUTOFF_MEAN:.4f} +/- {CUTOFF_SE:.4f} A\n")

    rows = collect(SCAN_ROOT, DENSITY_LABEL)

    if not rows:
        print("No data collected.")
        sys.exit(1)

    print(f"\nCollected {len(rows)} temperature points.\n")
    plot(rows, DENSITY_LABEL)
