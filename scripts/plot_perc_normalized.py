"""
plot_perc_normalized.py

Re-plots the percolation S-vs-cutoff curves from analyse_all.py caches,
with the x-axis rescaled to the dimensionless quantity

    r_norm = r / V^(1/3)

where V is the mean simulation-cell volume (Angstrom^3) taken from the
second half of each trajectory.  This removes the direct effect of
changing box size / density, so transitions at different densities can
be compared on equal footing.

Reads : SCAN/*/*/analysis_cache.npz  +  SCAN/*/*/small.md
Output: perc_normalized.png
"""

import os, glob, colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCAN_ROOT   = "C:/Users/yupei/Desktop/SHP/SCAN"
BOHR_TO_ANG = 0.529177210903


# ── helpers ────────────────────────────────────────────────────────────────

def read_mean_volume_ang3(md_path, max_frames=300):
    """Mean cell volume (Ang^3) from the equilibrated second half."""
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
    return float(np.mean(volumes[len(volumes) // 2:]))


# ── collect data ────────────────────────────────────────────────────────────

def collect():
    pattern  = os.path.join(SCAN_ROOT, "*", "*", "small.md")
    md_files = sorted(glob.glob(pattern))
    results  = []
    for md_file in md_files:
        cache = md_file.replace("small.md", "analysis_cache.npz")
        if not os.path.exists(cache):
            continue
        rel   = os.path.relpath(md_file, SCAN_ROOT)
        parts = rel.replace(os.sep, "/").split("/")
        density, temp = parts[0], parts[1]
        label = density + "/" + temp
        try:
            d = np.load(cache, allow_pickle=True)
            if "perc_r" not in d or "perc_S" not in d:
                continue
            perc_r = d["perc_r"]
            perc_S = d["perc_S"]
        except Exception as e:
            print(f"  [skip] {label}: cache error – {e}")
            continue
        V = read_mean_volume_ang3(md_file)
        if V is None or V <= 0:
            print(f"  [skip] {label}: cannot read volume")
            continue
        V_cbrt = V ** (1.0 / 3.0)          # Ang
        r_norm = perc_r / V_cbrt            # dimensionless
        dens_int = int(density) if density.isdigit() else 0
        temp_k   = int("".join(filter(str.isdigit, temp.split("_")[0])))
        results.append(dict(label=label, dens_int=dens_int, temp_k=temp_k,
                            r_norm=r_norm, perc_S=perc_S, V_cbrt=V_cbrt))
    results.sort(key=lambda x: (x["dens_int"], x["temp_k"]))
    return results


# ── plot ────────────────────────────────────────────────────────────────────

def main():
    results = collect()
    if not results:
        print("No data found.")
        return
    print(f"Loaded {len(results)} trajectories")

    # colour by temperature using coolwarm
    all_temps = sorted(set(r["temp_k"] for r in results))
    cmap   = cm.get_cmap("coolwarm", len(all_temps))
    t_idx  = {t: i for i, t in enumerate(all_temps)}

    fig, ax = plt.subplots(figsize=(13, 7))

    legend_added = set()
    for res in results:
        c     = cmap(t_idx[res["temp_k"]])
        lbl   = f"{res['temp_k']} K" if res["temp_k"] not in legend_added else "_nolegend_"
        legend_added.add(res["temp_k"])
        ax.plot(res["r_norm"], res["perc_S"],
                color=c, linewidth=1.4, alpha=0.85, label=lbl)

    ax.axhline(0.5, color="dimgrey", linestyle="--", linewidth=1.2,
               label="S = 0.5")
    ax.set_xlabel(r"Volume-normalised cutoff  $r\,/\,V^{1/3}$  (dimensionless)",
                  fontsize=16)
    ax.set_ylabel(r"Largest cluster fraction  $\langle S_\mathrm{max}/N\rangle$",
                  fontsize=16)
    ax.set_title(
        r"Percolation order parameter vs volume-normalised H-bond cutoff  $r/V^{1/3}$"
        "\n(all 84 conditions; colour = temperature)",
        fontsize=12,
    )
    ax.set_xlim(0, 0.25)
    ax.legend(fontsize=9, ncol=2, loc="upper left",
              title="Temperature", title_fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    out = "perc_normalized.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
