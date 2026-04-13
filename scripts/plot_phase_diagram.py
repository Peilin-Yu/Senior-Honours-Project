# plot_phase_diagram.py
# Saturation dome: IAPWS-IF97 via iapws 1.5.4 (Wagner & Kruse 2008)
# Critical point: Tc=647.096 K, rhoc=322 kg/m3 (IAPWS-IF97)

import os, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from iapws import IAPWS97, IAPWS95
from H2O import SHPDataSet

SCAN_ROOT = "C:/Users/yupei/Desktop/SHP/SCAN"
TEMPERATURES = [300, 400, 500, 600, 700, 800, 900, 1000]
CUTOFF_VALUES = [1.88, 1.88, 2.01, 2.01, 2.01, 2.01, 2.08, 2.08]
CUTOFF_MEAN   = float(np.mean(CUTOFF_VALUES))
CUTOFF_SE     = float(np.std(CUTOFF_VALUES, ddof=1) / np.sqrt(len(CUTOFF_VALUES)))
MIN_EQ_FRAMES = 50
BOHR_TO_ANG   = 0.529177210903
ANG3_TO_CM3   = 1e-24
AVOGADRO      = 6.02214076e23
WATER_MOLAR_MASS = 18.015
DENSITY_RE    = re.compile(r"^\d+$")
WATER_Tc      = 647.096
WATER_RHOc    = 0.322
WATER_Vc      = 1.0 / WATER_RHOc
WATER_Ttp     = 273.16          # K  (IAPWS-95 definition)
WATER_Ptp     = 611.657e-6      # MPa (= 611.657 Pa)

_T_SAT_K = sorted(set([273.16] + list(range(280, 640, 10)) + [630, 635, 640, 643, 645, 646, 647.096]))
_RHO_LIQ, _RHO_VAP = [], []
for _T in _T_SAT_K:
    _RHO_LIQ.append(IAPWS97(T=_T, x=0).rho / 1000.0)
    _RHO_VAP.append(IAPWS97(T=_T, x=1).rho / 1000.0)
_T_SAT_K  = np.array(_T_SAT_K)
_RHO_LIQ  = np.array(_RHO_LIQ)
_RHO_VAP  = np.array(_RHO_VAP)
_VLIQ_SAT = 1.0 / _RHO_LIQ
_VVAP_SAT = 1.0 / _RHO_VAP
_P_SAT_MPa = np.array([IAPWS97(T=_T, x=0).P for _T in _T_SAT_K])


def rho_to_pressure_MPa(rho_gcc, temp_k):
    try:
        state = IAPWS95(T=float(temp_k), rho=float(rho_gcc) * 1000.0)
        return float(state.P)
    except Exception:
        return None


def read_cell_volume(md_path, max_frames=300):
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
    return float(np.mean(volumes[len(volumes)//2:]))

def compute_coordination_stats(df, cutoff):
    n_frames = int(df["frame"].max()) + 1
    start    = n_frames // 2
    n_mol    = int(df["n_mol"].iloc[0])
    df_eq    = df[df["frame"] >= start]
    grouped  = df_eq[df_eq["dist"] <= cutoff].groupby("frame")
    nhb_per_frame = []
    for f in range(start, n_frames):
        n_bonds = len(grouped.get_group(f)) if f in grouped.groups else 0
        nhb_per_frame.append(2.0 * n_bonds / n_mol)
    if not nhb_per_frame:
        return float("nan"), float("nan")
    return float(np.mean(nhb_per_frame)), float(np.std(nhb_per_frame))


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
        s, chi_perc = compute_percolation_stats(df, CUTOFF_MEAN)
        if np.isnan(chi_perc):
            continue
        n_hb, chi_nhb = compute_coordination_stats(df, CUTOFF_MEAN)
        avg_V = read_cell_volume(md_path)
        if avg_V is None:
            continue
        n_mol = int(df["n_mol"].iloc[0])
        rho   = (n_mol * WATER_MOLAR_MASS / AVOGADRO) / (avg_V * ANG3_TO_CM3)
        rows.append((rho, chi_perc, s, n_hb, chi_nhb))
    return sorted(rows, key=lambda r: r[0])

def find_transition(rows):
    if len(rows) < 3:
        return None, None, None, None
    rhos     = np.array([r[0] for r in rows])
    chis     = np.array([r[1] for r in rows])
    Ss       = np.array([r[2] for r in rows])
    n_hbs    = np.array([r[3] for r in rows])
    chi_nhbs = np.array([r[4] for r in rows])
    peak_idx     = int(np.argmax(chis))
    rho_chi_perc = float(rhos[peak_idx]) if 0 < peak_idx < len(chis) - 1 else None
    rho_S_half = None
    for i in range(len(Ss) - 1):
        if (Ss[i] - 0.5) * (Ss[i + 1] - 0.5) <= 0:
            t = (0.5 - Ss[i]) / (Ss[i + 1] - Ss[i])
            rho_S_half = float(rhos[i] + t * (rhos[i + 1] - rhos[i]))
            break
    nhb_chi_idx = int(np.argmax(chi_nhbs))
    rho_nhb_chi = (float(rhos[nhb_chi_idx])
                   if 0 < nhb_chi_idx < len(rhos) - 1 else None)
    NHB_MID = 2.0
    rho_nhb_infl = None
    for i in range(len(n_hbs) - 1):
        if (n_hbs[i] - NHB_MID) * (n_hbs[i + 1] - NHB_MID) <= 0:
            t = (NHB_MID - n_hbs[i]) / (n_hbs[i + 1] - n_hbs[i])
            rho_nhb_infl = float(rhos[i] + t * (rhos[i + 1] - rhos[i]))
            break
    return rho_chi_perc, rho_S_half, rho_nhb_chi, rho_nhb_infl



def _compute_widom_line(P_min=23.0, P_max=200.0, n_P=25, n_T=100):
    """Locus of Cp maximum along supercritical isobars (IAPWS-95)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ps, Ts = [], []
        for P in np.logspace(np.log10(P_min), np.log10(P_max), n_P):
            T_arr = np.linspace(WATER_Tc + 2, WATER_Tc * 2.5, n_T)
            cp_arr = []
            for T in T_arr:
                try:
                    s = IAPWS95(T=T, P=P)
                    cp_arr.append(s.cp if (s.cp and np.isfinite(s.cp)) else np.nan)
                except Exception:
                    cp_arr.append(np.nan)
            cp_arr = np.array(cp_arr)
            idx = int(np.nanargmax(cp_arr))
            if 1 <= idx <= len(cp_arr)-2 and cp_arr[idx] > 1.5*np.nanmin(cp_arr):
                Ps.append(P); Ts.append(T_arr[idx])
        return np.array(Ps), np.array(Ts)


def _compute_frenkel_line_water():
    """Frenkel line for water via LJ corresponding-states scaling.

    Source: LJ cv = 2 kB per particle data from Brazhkin et al.,
    PRL 111, 145901 (2013), scaled to water using the principle of
    corresponding states:
        T_water   = (T* / T*_c,LJ)  * Tc_water
        rho_water = (rho* / rho*_c,LJ) * rhoc_water
    LJ critical point: T*_c = 1.321, rho*_c = 0.316
    (Kolafa & Nezbeda, Fluid Phase Equilib. 100, 1, 1994).

    Returns (T_arr_K, rho_arr_gcc) for the rho-T diagram.
    """
    # Digitised cv = 2 kB locus for LJ fluid (Brazhkin et al. PRL 2013, Fig. 1)
    LJ_T_star   = np.array([1.5,   2.0,   3.0,   4.0,   5.0,   7.0,  10.0])
    LJ_rho_star = np.array([0.685, 0.725, 0.800, 0.855, 0.905, 0.980,  1.07])
    LJ_Tc_star   = 1.321
    LJ_rhoc_star = 0.316
    T_water   = (LJ_T_star   / LJ_Tc_star)   * WATER_Tc
    rho_water = (LJ_rho_star / LJ_rhoc_star) * WATER_RHOc
    # Interpolate to a smooth curve
    T_fine   = np.linspace(T_water[0], T_water[-1], 300)
    rho_fine = np.interp(T_fine, T_water, rho_water)
    return T_fine, rho_fine


def _add_dome(ax, use_v=True):
    x_liq = _VLIQ_SAT if use_v else _RHO_LIQ
    x_vap = _VVAP_SAT if use_v else _RHO_VAP
    x_cp  = WATER_Vc  if use_v else WATER_RHOc
    ax.plot(x_liq, _T_SAT_K, color="black", linewidth=2.0)
    ax.plot(x_vap, _T_SAT_K, color="black", linewidth=2.0,
            label="Saturation dome (IAPWS-IF97)")
    ax.scatter([x_cp], [WATER_Tc], marker="*", s=250, color="gold",
               edgecolors="black", linewidths=0.7, zorder=6,
               label=f"Critical point ({WATER_Tc:.1f} K, {WATER_RHOc:.3f} g/cm$^3$)")
    x_tp = _VLIQ_SAT[0] if use_v else _RHO_LIQ[0]
    ax.scatter([x_tp], [WATER_Ttp], marker="v", s=120, color="royalblue",
               edgecolors="black", linewidths=0.7, zorder=6,
               label=f"Triple point ({WATER_Ttp:.2f} K)")


if __name__ == "__main__":
    print(f"Fixed cutoff : {CUTOFF_MEAN:.4f} +/- {CUTOFF_SE:.4f} A")
    print(f"Min eq frames: {MIN_EQ_FRAMES}")
    chi_perc_pts, s_half_pts, nhb_chi_pts, nhb_infl_pts = [], [], [], []
    for temp_k in TEMPERATURES:
        print(f"T = {temp_k} K ...", end=" ", flush=True)
        rows = collect_isotherm(SCAN_ROOT, temp_k)
        rho_chi, rho_s, rho_nhb_chi, rho_nhb_infl = find_transition(rows)
        print(f"perc_chi={rho_chi}  S=0.5={rho_s}  nhb_chi={rho_nhb_chi}  nhb_infl={rho_nhb_infl}")
        def _v(r): return 1.0/r if r is not None else None
        def _p(r): return rho_to_pressure_MPa(r, temp_k) if r is not None else None
        if rho_chi      is not None: chi_perc_pts.append((temp_k, rho_chi, _v(rho_chi), _p(rho_chi)))
        if rho_s        is not None: s_half_pts.append(  (temp_k, rho_s,   _v(rho_s),   _p(rho_s)))
        if rho_nhb_chi  is not None: nhb_chi_pts.append( (temp_k, rho_nhb_chi, _v(rho_nhb_chi), _p(rho_nhb_chi)))
        if rho_nhb_infl is not None: nhb_infl_pts.append((temp_k, rho_nhb_infl, _v(rho_nhb_infl), _p(rho_nhb_infl)))

    STYLES = [
        (chi_perc_pts, "o-",  "steelblue",  r"$\chi_\mathrm{perc}$ peak"),
        (s_half_pts,   "s--", "crimson",     r"$\langle S_\mathrm{max}/N\rangle = 0.5$"),
        (nhb_chi_pts,  "^-",  "darkorange",  r"$\chi_{n_\mathrm{HB}}$ peak"),
        (nhb_infl_pts, "D:",  "forestgreen", r"$\langle n_\mathrm{HB}\rangle = 2$ crossing"),
    ]

    print("Computing Widom line from IAPWS-95 ...")
    widom_P_arr, widom_T_arr = _compute_widom_line()
    print(f"  {len(widom_P_arr)} points, P = {widom_P_arr[0]:.1f}-{widom_P_arr[-1]:.1f} MPa")
    print("Computing Frenkel line (LJ corresponding states) ...")
    frenkel_T_arr, frenkel_rho_arr = _compute_frenkel_line_water()
    print(f"  {len(frenkel_T_arr)} points, T = {frenkel_T_arr[0]:.0f}-{frenkel_T_arr[-1]:.0f} K")

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    _add_dome(ax1, use_v=True)
    for pts, fmt, color, label in STYLES:
        if pts:
            ax1.plot([p[2] for p in pts], [p[0] for p in pts],
                     fmt, color=color, linewidth=2.0, markersize=8, label=label)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Specific volume $v = 1/\rho$ (cm$^3$/g)", fontsize=13)
    ax1.set_ylabel("Temperature (K)", fontsize=13)
    ax1.set_title("H-bond percolation transition on water phase diagram (v-T)", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, which="both", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("phase_diagram_v_T.png", dpi=150, bbox_inches="tight")
    print("Saved -> phase_diagram_v_T.png")

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    _add_dome(ax2, use_v=False)
    for pts, fmt, color, label in STYLES:
        if pts:
            ax2.plot([p[1] for p in pts], [p[0] for p in pts],
                     fmt, color=color, linewidth=2.0, markersize=8, label=label)
    if len(frenkel_rho_arr) > 1:
        ax2.plot(frenkel_rho_arr, frenkel_T_arr, color="darkorange", linewidth=2.0,
                 linestyle="-.", zorder=5, label="Frenkel line (LJ corr. states, Brazhkin 2013)")
    ax2.set_xlabel(r"Density $\rho$ (g/cm$^3$)", fontsize=13)
    ax2.set_ylabel("Temperature (K)", fontsize=13)
    ax2.set_title(r"H-bond percolation transition on water phase diagram ($\rho$-T)", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("phase_diagram_rho_T.png", dpi=150, bbox_inches="tight")
    print("Saved -> phase_diagram_rho_T.png")

    fig3, ax3 = plt.subplots(figsize=(9, 7))
    _add_dome(ax3, use_v=True)
    for pts, fmt, color, label in STYLES:
        if pts:
            ax3.plot([p[2] for p in pts], [p[0] for p in pts],
                     fmt, color=color, linewidth=2.0, markersize=9, label=label)
    ax3.set_xlim(0.5, 3.5)
    ax3.set_ylim(350, 1050)
    ax3.set_xlabel(r"Specific volume $v = 1/\rho$ (cm$^3$/g)", fontsize=13)
    ax3.set_ylabel("Temperature (K)", fontsize=13)
    ax3.set_title("H-bond percolation transition - zoomed near v = 1 cm$^3$/g", fontsize=12)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("phase_diagram_v_T_zoom.png", dpi=150, bbox_inches="tight")
    print("Saved -> phase_diagram_v_T_zoom.png")

    fig4, ax4 = plt.subplots(figsize=(10, 7))
    ax4.plot(_T_SAT_K, _P_SAT_MPa, color="black", linewidth=2.0,
             label="Saturation curve (IAPWS-97)")
    _cp_P = float(IAPWS97(T=WATER_Tc, x=0).P)
    ax4.scatter([WATER_Tc], [_cp_P],
                marker="*", s=250, color="gold", edgecolors="black",
                linewidths=0.7, zorder=6,
                label=f"Critical point ({WATER_Tc:.1f} K, {_cp_P:.2f} MPa)")
    ax4.scatter([WATER_Ttp], [WATER_Ptp], marker="v", s=120, color="royalblue",
                edgecolors="black", linewidths=0.7, zorder=6,
                label=f"Triple point ({WATER_Ttp:.2f} K, 611.7 Pa)")
    if len(widom_P_arr) > 1:
        ax4.plot(widom_T_arr, widom_P_arr, color="darkorchid", linewidth=2.0,
                 linestyle="--", label="Widom line (IAPWS-95, Cp max)")
    for pts, fmt, color, label in STYLES:
        valid = [(p[0], p[3]) for p in pts if p[3] is not None]
        if valid:
            Ts, Ps = zip(*valid)
            ax4.plot(list(Ts), list(Ps), fmt, color=color, linewidth=2.0,
                     markersize=8, label=label)
    ax4.set_xlabel("Temperature (K)", fontsize=13)
    ax4.set_ylabel("Pressure (MPa)", fontsize=13)
    ax4.set_title("H-bond percolation transition on water phase diagram (P-T)", fontsize=12)
    ax4.set_yscale("log")
    ax4.legend(fontsize=9, loc="upper left")
    ax4.grid(True, which="both", alpha=0.3)
    fig4.tight_layout()
    fig4.savefig("phase_diagram_P_T.png", dpi=150, bbox_inches="tight")
    print("Saved -> phase_diagram_P_T.png")

    plt.show()
