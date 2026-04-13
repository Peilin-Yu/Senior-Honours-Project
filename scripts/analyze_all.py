import os, glob, colorsys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ase.io import read
from ase.geometry.analysis import Analysis

SCAN_ROOT    = "C:/Users/yupei/Desktop/SHP/SCAN"
OUTPUT_HTML  = "C:/Users/yupei/Desktop/SHP/analysis_all.html"
RDF_RMAX, RDF_NBINS, RECOMPUTE = 5.0, 100, False
CACHE_VERSION = "v3"

PERC_R_START, PERC_R_END, PERC_STEPS = 1.5, 3.5, 40

def compute_msd(traj):
    box  = traj[0].get_cell().diagonal()
    p0   = traj[0].get_positions()
    n0   = len(p0)
    uw   = p0.copy(); prev = p0.copy(); msd = [0.0]
    for frame in traj[1:]:
        c = frame.get_positions()
        if len(c) != n0:
            msd.append(msd[-1]); continue
        d = c - prev
        d -= np.round(d / box) * box
        uw += d; prev = c.copy()
        msd.append(float(np.mean(np.sum((uw - p0)**2, axis=1))))
    return np.array(msd)

def compute_oo_rdf(traj):
    cell_min = float(np.min(np.abs(traj[0].get_cell().diagonal())))
    rmax = min(RDF_RMAX, 0.49 * cell_min)
    ana  = Analysis(traj)
    rdf  = ana.get_rdf(rmax=rmax, nbins=RDF_NBINS, elements=["O", "O"])
    return np.linspace(0, rmax, RDF_NBINS), np.mean(rdf, axis=0)

def compute_oh_rdf(traj):
    """True partial O-H RDF g_OH(r): only O-H pairs.

    ASE Analysis.get_rdf(elements=["O","H"]) computes the full RDF of the
    O+H subsystem (mixes O-O, H-H, O-H), giving a spurious peak at ~1.51 A
    from the intramolecular H-H distance (~1.52 A). We compute g_OH manually.
    """
    cell_min = float(np.min(np.abs(traj[0].get_cell().diagonal())))
    rmax  = min(RDF_RMAX, 0.49 * cell_min)
    dr    = rmax / RDF_NBINS
    bins  = np.arange(RDF_NBINS) * dr + dr / 2.0
    g_sum = np.zeros(RDF_NBINS)
    for atoms in traj:
        pos  = atoms.get_positions()
        cell = np.diag(atoms.get_cell())
        syms = np.array(atoms.get_chemical_symbols())
        o_idx = np.where(syms == "O")[0]
        h_idx = np.where(syms == "H")[0]
        N_O, N_H = len(o_idx), len(h_idx)
        if N_O == 0 or N_H == 0:
            continue
        V     = float(np.prod(cell))
        rho_H = N_H / V
        diff  = pos[h_idx][np.newaxis, :, :] - pos[o_idx][:, np.newaxis, :]
        diff -= np.round(diff / cell) * cell
        dists = np.sqrt(np.sum(diff**2, axis=2)).flatten()
        counts, _ = np.histogram(dists, bins=RDF_NBINS, range=(0.0, rmax))
        shell_vols = 4.0 * np.pi * bins**2 * dr
        expected   = N_O * rho_H * shell_vols
        with np.errstate(invalid="ignore", divide="ignore"):
            g_sum += np.where(expected > 0, counts / expected, 0.0)
    return bins, g_sum / len(traj)

def compute_adf(traj):
    all_angles = []
    for atoms in traj:
        pos   = atoms.get_positions()
        cell  = np.diag(atoms.get_cell())
        syms  = np.array(atoms.get_chemical_symbols())
        o_idx = np.where(syms == "O")[0]
        h_idx = np.where(syms == "H")[0]
        pos_O = pos[o_idx]; pos_H = pos[h_idx]
        diff  = pos_H[np.newaxis, :, :] - pos_O[:, np.newaxis, :]
        diff -= np.round(diff / cell) * cell
        dists = np.sqrt(np.sum(diff**2, axis=2))
        near2 = np.argsort(dists, axis=1)[:, :2]
        v1    = diff[np.arange(len(o_idx)), near2[:, 0], :]
        v2    = diff[np.arange(len(o_idx)), near2[:, 1], :]
        n1    = np.linalg.norm(v1, axis=1)
        n2    = np.linalg.norm(v2, axis=1)
        cos_a = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-12)
        all_angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    return np.concatenate(all_angles) if all_angles else np.array([])

def _find_root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def compute_percolation_sweep(traj):
    cutoffs = np.linspace(PERC_R_START, PERC_R_END, PERC_STEPS)
    frame_bonds = []
    frame_nO    = []
    for atoms in traj:
        syms  = np.array(atoms.get_chemical_symbols())
        o_idx = np.where(syms == "O")[0]
        h_idx = np.where(syms == "H")[0]
        n_O   = len(o_idx)
        frame_nO.append(n_O)
        if n_O < 2 or len(h_idx) == 0:
            frame_bonds.append(np.empty((0, 3))); continue
        pos   = atoms.get_positions()
        cell  = np.diag(atoms.get_cell())
        pos_O = pos[o_idx]; pos_H = pos[h_idx]
        diff = pos_O[np.newaxis, :, :] - pos_H[:, np.newaxis, :]
        diff -= np.round(diff / cell) * cell
        dists = np.sqrt(np.sum(diff**2, axis=2))
        sorted_i      = np.argsort(dists, axis=1)
        donors_loc    = sorted_i[:, 0]
        acceptors_loc = sorted_i[:, 1]
        bond_dists    = dists[np.arange(len(h_idx)), acceptors_loc]
        frame_bonds.append(np.column_stack([donors_loc, acceptors_loc, bond_dists]))
    n_frames = len(traj)
    S_mean = np.zeros(PERC_STEPS)
    S_std  = np.zeros(PERC_STEPS)
    for ri, r in enumerate(cutoffs):
        frame_S = []
        for f in range(n_frames):
            n_O   = frame_nO[f]
            bonds = frame_bonds[f]
            if n_O == 0:
                frame_S.append(0.0); continue
            active = bonds[bonds[:, 2] <= r] if len(bonds) else bonds
            parent = list(range(n_O))
            for row in active:
                d_node, a_node = int(row[0]), int(row[1])
                pd = _find_root(parent, d_node)
                pa = _find_root(parent, a_node)
                if pd != pa:
                    parent[pd] = pa
            roots  = [_find_root(parent, i) for i in range(n_O)]
            sizes  = np.bincount(roots, minlength=n_O)
            frame_S.append(sizes.max() / n_O)
        S_mean[ri] = np.mean(frame_S)
        S_std[ri]  = np.std(frame_S)
    return cutoffs, S_mean, S_std

def cache_path(f):
    return f.replace("small.md", "analysis_cache.npz")

_REQUIRED = ["msd", "rdf_r", "rdf_g", "oh_rdf_r", "oh_rdf_g",
             "angles", "perc_r", "perc_S", "perc_chi", "cache_version"]

def load_or_compute(md_file, label):
    cp = cache_path(md_file)
    if not RECOMPUTE and os.path.exists(cp):
        d = np.load(cp, allow_pickle=True)
        if (all(k in d for k in _REQUIRED) and str(d["cache_version"]) == CACHE_VERSION):
            print("  [cache]   " + label)
            return (d["msd"], d["rdf_r"], d["rdf_g"],
                    d["oh_rdf_r"], d["oh_rdf_g"], d["angles"],
                    d["perc_r"], d["perc_S"], d["perc_chi"])
    print("  [compute] " + label)
    traj = read(md_file, format="castep-md", index=":")
    print("            " + str(len(traj)) + " frames")
    msd                      = compute_msd(traj);               print("            MSD done")
    rdf_r,    rdf_g          = compute_oo_rdf(traj);            print("            O-O RDF done")
    oh_rdf_r, oh_rdf_g       = compute_oh_rdf(traj);            print("            O-H RDF done")
    angles                   = compute_adf(traj);                print("            ADF done (" + str(len(angles)) + " angles)")
    perc_r, perc_S, perc_chi = compute_percolation_sweep(traj); print("            Percolation sweep done")
    np.savez(cp, msd=msd, rdf_r=rdf_r, rdf_g=rdf_g,
             oh_rdf_r=oh_rdf_r, oh_rdf_g=oh_rdf_g, angles=angles,
             perc_r=perc_r, perc_S=perc_S, perc_chi=perc_chi,
             cache_version=np.array(CACHE_VERSION))
    return (msd, rdf_r, rdf_g, oh_rdf_r, oh_rdf_g, angles, perc_r, perc_S, perc_chi)

def make_colors(n):
    if n == 0:
        return []
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.78)
        colors.append("rgb(%d,%d,%d)" % (int(r * 255), int(g * 255), int(b * 255)))
    return colors

def main():
    pattern  = os.path.join(SCAN_ROOT, "*", "*", "small.md")
    md_files = sorted(glob.glob(pattern))
    if not md_files:
        print("No small.md files found under " + SCAN_ROOT); return
    print("Found " + str(len(md_files)) + " trajectories.")
    results = []
    for md_file in md_files:
        rel   = os.path.relpath(md_file, SCAN_ROOT)
        parts = rel.replace(os.sep, "/").split("/")
        density, temp = parts[0], parts[1]
        label = density + "/" + temp
        try:
            (msd, rdf_r, rdf_g, oh_rdf_r, oh_rdf_g,
             angles, perc_r, perc_S, perc_chi) = load_or_compute(md_file, label)
            temp_k = int("".join(filter(str.isdigit, temp.split("_")[0])))
        except Exception as e:
            print("  [ERROR] " + label + ": " + str(e)); continue
        results.append(dict(label=label, density=density, temp_k=temp_k,
                            msd=msd, rdf_r=rdf_r, rdf_g=rdf_g,
                            oh_rdf_r=oh_rdf_r, oh_rdf_g=oh_rdf_g,
                            angles=angles, perc_r=perc_r, perc_S=perc_S, perc_chi=perc_chi))
    if not results:
        print("No results to plot."); return
    results.sort(key=lambda x: (x["density"], x["temp_k"]))
    colors = make_colors(len(results))
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "O-O Radial Distribution Function g(r)",
            "O-H RDF g(r)  [intermolecular, r > 1.2 A]",
            "H-O-H Angle Distribution",
            "Mean Squared Displacement",
            "Percolation: Largest-Cluster Fraction S vs Cutoff",
            "",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    bins = np.linspace(80, 180, 101)
    cx   = 0.5 * (bins[:-1] + bins[1:])
    for i, res in enumerate(results):
        c_col = colors[i]
        lb = res["label"]
        kw = dict(color=c_col, width=1.5)
        fig.add_trace(go.Scatter(x=res["rdf_r"], y=res["rdf_g"], mode="lines",
            name=lb, legendgroup=lb, line=kw), row=1, col=1)
        fig.add_trace(go.Scatter(x=res["oh_rdf_r"], y=res["oh_rdf_g"], mode="lines",
            name=lb, legendgroup=lb, showlegend=False, line=kw), row=1, col=2)
        if len(res["angles"]) > 0:
            cnt, _ = np.histogram(res["angles"], bins=bins, density=True)
            fig.add_trace(go.Scatter(x=cx, y=cnt, mode="lines",
                name=lb, legendgroup=lb, showlegend=False, line=kw), row=1, col=3)
        fr = np.arange(len(res["msd"]))
        fig.add_trace(go.Scatter(x=fr, y=res["msd"], mode="lines",
            name=lb, legendgroup=lb, showlegend=False, line=kw), row=2, col=1)
        fig.add_trace(go.Scatter(x=res["perc_r"], y=res["perc_S"], mode="lines",
            name=lb, legendgroup=lb, showlegend=False, line=kw), row=2, col=2)
    fig.add_vline(x=104.5, line=dict(color="gray", dash="dot", width=1),
                  row=1, col=3, annotation_text="104.5 deg",
                  annotation_font_color="gray", annotation_position="top right")
    fig.add_hline(y=0.5, line=dict(color="gray", dash="dot", width=1),
                  row=2, col=2, annotation_text="S = 0.5",
                  annotation_font_color="gray", annotation_position="right")
    fig.update_xaxes(title_text="r (A)",                        row=1, col=1)
    fig.update_yaxes(title_text="g(r)",                         row=1, col=1)
    fig.update_xaxes(title_text="r (A)",  range=[1.2, 5.0],    row=1, col=2)
    fig.update_yaxes(title_text="g(r)",   range=[0, 3.0],      row=1, col=2)
    fig.update_xaxes(title_text="Angle (deg)",                  row=1, col=3)
    fig.update_yaxes(title_text="Probability density",          row=1, col=3)
    fig.update_xaxes(title_text="Frame",                        row=2, col=1)
    fig.update_yaxes(title_text="MSD (A^2)",                    row=2, col=1)
    fig.update_xaxes(title_text="H-bond cutoff r (A)",          row=2, col=2)
    fig.update_yaxes(title_text="S (largest cluster fraction)", row=2, col=2)
    fig.update_layout(
        title=dict(text="Water MD Analysis - " + str(len(results)) + " conditions",
                   font=dict(size=14, color="black")),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(title="density/temp", bgcolor="rgba(245,245,245,0.9)",
                    bordercolor="#aaaaaa", borderwidth=1, font=dict(size=9)),
        height=900, width=1700,
    )
    fig.update_xaxes(gridcolor="#e8e8e8", zeroline=False, showline=True, linecolor="#bbbbbb", mirror=True,
                     title_font=dict(size=16))
    fig.update_yaxes(gridcolor="#e8e8e8", zeroline=False, showline=True, linecolor="#bbbbbb", mirror=True,
                     title_font=dict(size=16))
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print("Saved -> " + OUTPUT_HTML)

if __name__ == "__main__":
    main()
