"""
H2O.py
======
Reusable module for hydrogen-bond network percolation analysis of water
from CASTEP MD trajectories.

Importable usage
----------------
    from H2O import SHPDataSet, PercolationAnalysis

    dataset     = SHPDataSet("path/to/SCAN/140")
    percolation = PercolationAnalysis(r_min=1.5, r_max=4.0, steps=40)

    df              = dataset.get_temperature_data(500)
    cutoffs, flucts = percolation.run(df)
    chi_max, r_star = percolation.peak(df)

Standalone usage
----------------
    python H2O.py [path/to/density/folder]
    e.g.  python H2O.py C:/Users/yupei/Desktop/SHP/SCAN/140
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ase.io import read


# =============================================================================
# CLASS 1: DATA MANAGER
# =============================================================================

class SHPDataSet:
    """
    Manages geometry data for one pressure/density folder.

    Discovers temperature sub-folders (e.g. 300K, 400K …), loads cached
    geometry DataFrames from CSV, and parses CASTEP .md trajectories when
    no cache is present.

    Parameters
    ----------
    root_dir : str
        Path to a density folder, e.g. ``C:/…/SCAN/140``.

    Attributes
    ----------
    root_dir        : str   Absolute path supplied at construction.
    available_temps : list  Sorted list of integer temperatures (K) found.
    """

    _TEMP_RE = re.compile(r"^(\d+)K$")

    def __init__(self, root_dir: str):
        self.root_dir        = root_dir
        self.available_temps = self._scan_temperatures()

    # ------------------------------------------------------------------ private

    def _scan_temperatures(self) -> list:
        """Return sorted list of integer temperatures found in root_dir."""
        temps = []
        for name in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, name)):
                m = self._TEMP_RE.match(name)
                if m:
                    temps.append(int(m.group(1)))
        return sorted(temps)

    # ------------------------------------------------------------------ public

    def get_temperature_data(
        self,
        temp_k:   int,
        md_name:  str = "small.md",
        csv_name: str = "geometry_cache.csv",
    ) -> "pd.DataFrame | None":
        """
        Return the geometry DataFrame for *temp_k*.

        1. Returns the cached CSV if it exists and is non-empty.
        2. Falls back to parsing the CASTEP .md file (and saves the cache).
        3. Returns ``None`` if neither source is available or parsing fails.

        Parameters
        ----------
        temp_k   : int   Temperature in Kelvin.
        md_name  : str   CASTEP MD filename inside the temperature folder.
        csv_name : str   Cache CSV filename.

        Returns
        -------
        pd.DataFrame or None
            Columns: frame, h_idx, u (donor O index), v (acceptor O index),
            dist (H → acceptor O distance in Å), n_mol (O-atom count).
        """
        folder_path = os.path.join(self.root_dir, f"{temp_k}K")
        csv_path    = os.path.join(folder_path, csv_name)
        md_path     = os.path.join(folder_path, md_name)

        # 1. Cache
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            if not df.empty:
                print(f"[{temp_k}K] Loading cached data from {csv_name}...")
                return df
            print(f"[{temp_k}K] Cache is empty — re-parsing {md_name}...")

        # 2. Parse MD
        if not os.path.exists(md_path):
            print(f"[{temp_k}K] Warning: {md_name} not found.")
            return None

        print(f"[{temp_k}K] Parsing {md_name} (this may take a while)...")
        try:
            traj = read(md_path, format="castep-md", index=":")
            data = SHPDataSet.calculate_geometry(traj)
            df   = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            print(f"[{temp_k}K] Cache saved to {csv_name}.")
            return df
        except Exception as exc:
            print(f"[{temp_k}K] Error processing MD: {exc}")
            return None

    @staticmethod
    def calculate_geometry(traj: list) -> list:
        """
        Compute H-bond donor/acceptor geometry for every frame.

        For each hydrogen atom finds the nearest oxygen (donor) and the
        second-nearest oxygen (acceptor).  Periodic boundary conditions
        are applied via ASE's minimum-image convention.

        Parameters
        ----------
        traj : list of ase.Atoms
            Trajectory as returned by ``ase.io.read(..., index=':')``.

        Returns
        -------
        list of dict
            One record per H atom per frame with keys:
            ``frame``, ``h_idx``, ``u`` (donor O index),
            ``v`` (acceptor O index), ``dist`` (H→acceptor Å), ``n_mol``.
        """
        records = []
        for frame_idx, atoms in enumerate(traj):
            o_indices = [a.index for a in atoms if a.symbol == "O"]
            h_indices = [a.index for a in atoms if a.symbol == "H"]
            n_mol     = len(o_indices)

            if len(o_indices) < 2:
                continue  # skip frames with fewer than 2 O atoms (corrupt/empty frames)

            for h_idx in h_indices:
                dists      = atoms.get_distances(h_idx, o_indices, mic=True)
                sorted_idx = np.argsort(dists)
                if len(sorted_idx) >= 2:
                    records.append({
                        "frame": frame_idx,
                        "h_idx": h_idx,
                        "u":     o_indices[sorted_idx[0]],  # donor O
                        "v":     o_indices[sorted_idx[1]],  # acceptor O
                        "dist":  dists[sorted_idx[1]],      # H → acceptor Å
                        "n_mol": n_mol,
                    })
        return records


# =============================================================================
# CLASS 2: PERCOLATION ANALYSER
# =============================================================================

class PercolationAnalysis:
    """
    Percolation sweep on a hydrogen-bond DataFrame.

    At each cutoff distance *r* in [r_min, r_max] the O–O bond network is
    built from all H-bonds with H → acceptor distance ≤ *r*.  The standard
    deviation of the normalised largest-cluster size across frames is the
    susceptibility χ.  The peak of χ locates the percolation threshold.

    Parameters
    ----------
    r_min  : float  Lower bound of the cutoff sweep (Å).
    r_max  : float  Upper bound (Å).
    steps  : int    Number of cutoff values (resolution of the sweep).
    """

    def __init__(self, r_min: float = 1.5, r_max: float = 4.0, steps: int = 40):
        self.r_min   = r_min
        self.r_max   = r_max
        self.steps   = steps
        self.cutoffs = np.linspace(r_min, r_max, steps)

    # ------------------------------------------------------------------ helpers

    def _sweep(self, df: pd.DataFrame) -> np.ndarray:
        """
        Internal sweep; returns the fluctuation array aligned with self.cutoffs.
        """
        n_frames     = int(df["frame"].max()) + 1
        fluctuations = []

        for r in self.cutoffs:
            valid   = df[df["dist"] <= r]
            grouped = valid.groupby("frame")
            sizes   = []

            for f in range(n_frames):
                if f in grouped.groups:
                    fd      = grouped.get_group(f)
                    n_total = int(fd.iloc[0]["n_mol"])
                    G       = nx.Graph()
                    G.add_edges_from(zip(fd["u"], fd["v"]))
                    largest = (
                        len(max(nx.connected_components(G), key=len))
                        if len(G) > 0 else 1
                    )
                    sizes.append(largest / n_total)
                else:
                    sizes.append(1.0 / int(df.iloc[0]["n_mol"]))

            fluctuations.append(np.std(sizes))

        return np.array(fluctuations)

    # ------------------------------------------------------------------ public

    def run(self, df: "pd.DataFrame | None") -> tuple:
        """
        Run the full percolation sweep and return the complete curve.

        Parameters
        ----------
        df : pd.DataFrame or None

        Returns
        -------
        (np.ndarray, np.ndarray) or (None, None)
            ``cutoffs``      — 1-D array of cutoff distances (Å)
            ``fluctuations`` — susceptibility χ at each cutoff
        """
        if df is None or df.empty:
            return None, None

        print("   -> Running percolation sweep...")
        flucts = self._sweep(df)
        return self.cutoffs, flucts

    def peak(self, df: "pd.DataFrame | None") -> tuple:
        """
        Return only the peak susceptibility and the corresponding cutoff.

        Convenience wrapper around :meth:`run` for scripts that only need
        the scalar summary rather than the full curve.

        Parameters
        ----------
        df : pd.DataFrame or None

        Returns
        -------
        (chi_max, r_critical) or (None, None)
            ``chi_max``    — maximum χ across the sweep
            ``r_critical`` — H-bond cutoff (Å) at that maximum
        """
        cutoffs, flucts = self.run(df)
        if flucts is None or flucts.max() == 0:
            return None, None
        idx = int(np.argmax(flucts))
        return float(flucts[idx]), float(cutoffs[idx])


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    work_dir = (sys.argv[1] if len(sys.argv) > 1
                else "C:/Users/yupei/Desktop/SHP/SCAN/140")

    dataset     = SHPDataSet(work_dir)
    percolation = PercolationAnalysis(r_min=1.5, r_max=4.0, steps=40)

    print(f"Density folder : {work_dir}")
    print(f"Found temperatures: {dataset.available_temps} K\n")

    results = {}

    for temp in dataset.available_temps:
        df = dataset.get_temperature_data(temp)
        if df is None or df.empty:
            continue

        x, y = percolation.run(df)
        if x is None:
            print(f"[{temp}K] No percolation result — skipping.\n")
            continue

        peak_idx        = int(np.argmax(y))
        results[temp]   = (x, y, x[peak_idx])
        print(f"[{temp}K] r* = {x[peak_idx]:.3f} A   chi_max = {y[peak_idx]:.4f}\n")

    if not results:
        print("No results to plot.")
        sys.exit(0)

    plt.figure(figsize=(10, 6))
    for temp, (x, y, r_crit) in results.items():
        plt.plot(x, y, label=f"{temp} K  (r* = {r_crit:.2f} Å)")

    plt.xlabel("H-bond cutoff (Å)", fontsize=14)
    plt.ylabel("Susceptibility  χ", fontsize=14)
    plt.title("Percolation threshold vs Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("percolationcutoff_300-1000K.png", dpi=150, bbox_inches="tight")
    print("Saved -> percolationcutoff_300-1000K.png")
    plt.show()
