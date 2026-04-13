"""
3D Interactive H-Bond Network Visualizer
=========================================
Generates a self-contained interactive HTML file (via Plotly) showing
the hydrogen-bond network for a single MD frame.

Usage:
    python visualize_network_3d.py

Output:
    hbond_network_3d.html  — open in any browser for full 3D rotation/zoom.

Requirements:
    pip install plotly pandas numpy
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Configuration ─────────────────────────────────────────────────────────────

XYZ_FILE    = r"C:\Users\yupei\Desktop\SHP\SCAN\140\1000K\X.xyz"
CACHE_FILE  = r"C:\Users\yupei\Desktop\SHP\SCAN\140\1000K\geometry_cache.csv"
CELL_FILE   = r"C:\Users\yupei\Desktop\SHP\SCAN\140\1000K\small.cell"  # CASTEP cell for box size
FRAME_IDX   = 4                              # which MD frame to show
CUTOFF      = 1.9                            # H→O distance cutoff in A
SHOW_H      = True                          # overlay H atoms as tiny grey dots
OUTPUT_HTML = "hbond_network_3d.html"

# ── XYZ parser ────────────────────────────────────────────────────────────────

def parse_xyz_frame(filepath: str, frame_idx: int) -> pd.DataFrame:
    """Return DataFrame [element, x, y, z] for the requested frame."""
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    i = 0
    current_frame = 0
    while i < len(lines):
        n_atoms = int(lines[i].strip())
        block_end = i + 2 + n_atoms
        if current_frame == frame_idx:
            rows = []
            for ln in lines[i + 2 : block_end]:
                parts = ln.split()
                rows.append({
                    "element": parts[0],
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                })
            return pd.DataFrame(rows)
        current_frame += 1
        i = block_end

    raise IndexError(
        f"Frame {frame_idx} not found in {filepath} "
        f"(file contains {current_frame} frames)."
    )


# ── Network builder ───────────────────────────────────────────────────────────

def parse_box_size(cell_file):
    """Parse [Lx, Ly, Lz] from a CASTEP .cell file. Returns None on failure."""
    try:
        with open(cell_file, "r") as fh:
            text = fh.read()
        import re
        block = re.search(
            r"%BLOCK\s+lattice_cart(.*?)%ENDBLOCK\s+lattice_cart",
            text, re.IGNORECASE | re.DOTALL)
        if not block:
            return None
        ls = [l.strip() for l in block.group(1).strip().splitlines()
              if l.strip() and l.strip().upper() not in ("ANG", "BOHR")]
        vecs = np.array([[float(x) for x in ln.split()] for ln in ls[:3]])
        return np.array([vecs[0, 0], vecs[1, 1], vecs[2, 2]])
    except Exception:
        return None


PBC_STUB_LEN = 1.5  # length (A) of outward stub for PBC-crossing bonds

def split_pbc_edge(p_u, p_v, h_pos, box):
    """
    For a bond that crosses the periodic boundary, return two short outward
    stubs pointing in the bond direction rather than one long line across the box.

    Non-PBC: returns [(p_u, h_pos, p_v)], False  -- 3-point segment
    PBC    : returns [(p_u, end_u), (p_v, end_v)], True  -- two 2-point stubs
      end_u = p_u + PBC_STUB_LEN * direction   (stub pointing away from u)
      end_v = p_v - PBC_STUB_LEN * direction   (stub pointing away from v)
    """
    delta = p_v - p_u
    shift = np.zeros(3)
    for i in range(3):
        if delta[i] > box[i] / 2:
            shift[i] = -box[i]
        elif delta[i] < -box[i] / 2:
            shift[i] = box[i]
    crosses = np.any(shift != 0)
    if not crosses:
        return [(p_u, h_pos, p_v)], False
    # Minimum image vector u->v and its unit direction
    min_vec = delta + shift
    dist = np.linalg.norm(min_vec)
    direction = min_vec / dist if dist > 0 else min_vec
    end_u = p_u - PBC_STUB_LEN * direction
    end_v = p_v + PBC_STUB_LEN * direction
    return [(p_u, end_u), (p_v, end_v)], True


def build_network(atoms_df, cache_df, frame_idx, cutoff):
    """
    Returns
    -------
    o_pos   : ndarray (n_O, 3)  XYZ positions of O atoms
    o_index : ndarray (n_O,)    original atom indices (200-299 for 100-mol sys)
    edges   : list of (u_local, v_local, h_pos, dist)
              h_pos is the 3D coordinate of the bridging H atom
    all_pos : ndarray (n_atoms, 3)  all atom positions (for free H lookup)
    """
    o_mask  = atoms_df["element"] == "O"
    o_df    = atoms_df[o_mask].reset_index()
    o_index = o_df["index"].values
    o_pos   = o_df[["x", "y", "z"]].values
    all_pos = atoms_df[["x", "y", "z"]].values

    idx_to_local = {orig: loc for loc, orig in enumerate(o_index)}

    frame_cache = cache_df[
        (cache_df["frame"] == frame_idx) & (cache_df["dist"] <= cutoff)
    ]

    edges = []
    bonded_h = set()
    for _, row in frame_cache.iterrows():
        u, v, h = int(row["u"]), int(row["v"]), int(row["h_idx"])
        if u in idx_to_local and v in idx_to_local:
            h_pos = all_pos[h]
            edges.append((idx_to_local[u], idx_to_local[v], h_pos, row["dist"]))
            bonded_h.add(h)

    return o_pos, o_index, edges, all_pos, bonded_h


def find_largest_cluster(n_nodes, edges):
    """BFS; returns set of node indices in the largest connected component."""
    adj = {i: [] for i in range(n_nodes)}
    for u, v, *_ in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()
    best = set()
    for start in range(n_nodes):
        if start in visited:
            continue
        comp = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            queue.extend(adj[node])
        if len(comp) > len(best):
            best = comp
    return best


# ── Plotly figure ─────────────────────────────────────────────────────────────

def make_figure(atoms_df, o_pos, o_index, edges, all_pos, bonded_h, frame_idx, cutoff, show_h, box):
    degree = np.zeros(len(o_pos), dtype=int)
    for u, v, h_pos, _ in edges:
        degree[u] += 1
        degree[v] += 1

    largest = find_largest_cluster(len(o_pos), edges)

    # ── Build edge coordinate lists ──────────────────────────────────────
    ex, ey, ez = [], [], []          # non-largest bonds (cluster view)
    lx, ly, lz = [], [], []          # largest-cluster bonds (cluster view)
    ax, ay, az = [], [], []          # all bonds combined (degree view)
    hx, hy, hz = [], [], []          # H midpoints on bonds
    pbc_ex, pbc_ey, pbc_ez = [], [], []

    for u, v, h_pos, _ in edges:
        in_largest = (u in largest and v in largest)
        if box is not None:
            segments, crosses = split_pbc_edge(o_pos[u], o_pos[v], h_pos, box)
        else:
            segments, crosses = [(o_pos[u], h_pos, o_pos[v])], False
        for seg in segments:
            if crosses:
                a, b = seg
                pbc_ex += [a[0], b[0], None]
                pbc_ey += [a[1], b[1], None]
                pbc_ez += [a[2], b[2], None]
            else:
                a, mid, b = seg
                pts = [a[0], mid[0], b[0], None]
                ax += pts; ay += [a[1], mid[1], b[1], None]; az += [a[2], mid[2], b[2], None]
                if in_largest:
                    lx += pts; ly += [a[1], mid[1], b[1], None]; lz += [a[2], mid[2], b[2], None]
                else:
                    ex += pts; ey += [a[1], mid[1], b[1], None]; ez += [a[2], mid[2], b[2], None]
                hx.append(mid[0]); hy.append(mid[1]); hz.append(mid[2])

    # ── O atom groupings ─────────────────────────────────────────────────
    rest_idx = [i for i in range(len(o_pos)) if i not in largest]
    lc_idx   = [i for i in range(len(o_pos)) if i in largest]

    def o_text(i):
        return (f"O atom {o_index[i]}<br>degree: {degree[i]}<br>"
                f"({o_pos[i,0]:.2f}, {o_pos[i,1]:.2f}, {o_pos[i,2]:.2f}) A")

    # ── H free atoms ─────────────────────────────────────────────────────
    h_df = atoms_df[atoms_df["element"] == "H"].reset_index()
    free_h = h_df[~h_df["index"].isin(bonded_h)] if show_h else h_df.iloc[0:0]
    has_free_h = len(free_h) > 0

    # ── Assemble traces (fixed order for button visibility toggles) ───────
    # Trace 0 : all bonds steelblue          -- degree view
    # Trace 1 : non-largest bonds steelblue  -- cluster view
    # Trace 2 : largest bonds tomato         -- cluster view
    # Trace 3 : PBC stubs orange             -- both views
    # Trace 4 : H markers white              -- both views
    # Trace 5 : O atoms Viridis by degree    -- degree view
    # Trace 6 : O other steelblue            -- cluster view
    # Trace 7 : O largest tomato             -- cluster view
    # Trace 8 : H free grey                  -- both views (optional)

    traces = []

    traces.append(go.Scatter3d(                              # 0
        x=ax, y=ay, z=az, mode="lines",
        line=dict(color="steelblue", width=5),
        hoverinfo="none", name="H-bonds", opacity=0.6, visible=False))

    traces.append(go.Scatter3d(                              # 1
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="steelblue", width=5),
        hoverinfo="none", name="H-bonds", opacity=0.6, visible=True))

    traces.append(go.Scatter3d(                              # 2
        x=lx, y=ly, z=lz, mode="lines",
        line=dict(color="tomato", width=7),
        hoverinfo="none", name="Largest cluster bonds", opacity=0.85, visible=True))

    traces.append(go.Scatter3d(                              # 3
        x=pbc_ex, y=pbc_ey, z=pbc_ez, mode="lines",
        line=dict(color="orange", width=5, dash="dash"),
        hoverinfo="none", name="H-bonds (PBC)", opacity=0.6, visible=True))

    traces.append(go.Scatter3d(                              # 4
        x=hx, y=hy, z=hz, mode="markers",
        marker=dict(size=3, color="white", opacity=0.85),
        hoverinfo="none", name="H (bonded)", visible=True))

    traces.append(go.Scatter3d(                              # 5
        x=o_pos[:, 0], y=o_pos[:, 1], z=o_pos[:, 2], mode="markers",
        marker=dict(size=6, color=degree, colorscale="Viridis",
                    colorbar=dict(title="Degree", thickness=15, x=1.02),
                    showscale=True, line=dict(width=0.5, color="black")),
        text=[o_text(i) for i in range(len(o_pos))],
        hoverinfo="text", name="O atoms (degree)", visible=False))

    rx = o_pos[rest_idx, 0] if rest_idx else []
    ry = o_pos[rest_idx, 1] if rest_idx else []
    rz = o_pos[rest_idx, 2] if rest_idx else []
    traces.append(go.Scatter3d(                              # 6
        x=rx, y=ry, z=rz, mode="markers",
        marker=dict(size=6, color="steelblue", line=dict(width=0.5, color="black")),
        text=[o_text(i) for i in rest_idx], hoverinfo="text",
        name="O atoms (other)", visible=True))

    lcx = o_pos[lc_idx, 0] if lc_idx else []
    lcy = o_pos[lc_idx, 1] if lc_idx else []
    lcz = o_pos[lc_idx, 2] if lc_idx else []
    traces.append(go.Scatter3d(                              # 7
        x=lcx, y=lcy, z=lcz, mode="markers",
        marker=dict(size=8, color="tomato", line=dict(width=1, color="black")),
        text=[o_text(i) for i in lc_idx], hoverinfo="text",
        name=f"Largest cluster ({len(lc_idx)} O)", visible=True))

    if has_free_h:                                           # 8 (optional)
        traces.append(go.Scatter3d(
            x=free_h["x"], y=free_h["y"], z=free_h["z"], mode="markers",
            marker=dict(size=2.5, color="lightgrey", opacity=0.4),
            hoverinfo="none", name="H (free)", visible=True))

    # ── Visibility masks for the two views ────────────────────────────────
    extra = [True] if has_free_h else []
    cluster_vis = [False, True,  True,  True, True, False, True,  True ] + extra
    degree_vis  = [True,  False, False, True, True, True,  False, False] + extra

    updatemenus = [dict(
        type="buttons",
        direction="left",
        active=0,
        buttons=[
            dict(label="Cluster view",
                 method="update",
                 args=[{"visible": cluster_vis}]),
            dict(label="Degree view",
                 method="update",
                 args=[{"visible": degree_vis}]),
        ],
        pad={"r": 10, "t": 10},
        showactive=True,
        bgcolor="#2e2e4e",
        bordercolor="#aaaacc",
        font=dict(color="white"),
        x=0.5, xanchor="center",
        y=1.10, yanchor="top",
    )]

    box_label = (
        f" | box {box[0]:.1f}x{box[1]:.1f}x{box[2]:.1f} A"
        if box is not None else ""
    )
    layout = go.Layout(
        title=dict(
            text=(f"H-Bond Network — Frame {frame_idx} | "
                  f"cutoff {cutoff} A | "
                  f"{len(o_pos)} O atoms | {len(edges)} bonds | "
                  f"largest cluster: {len(largest)} O{box_label}"),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="x (A)", showgrid=True, zeroline=False),
            yaxis=dict(title="y (A)", showgrid=True, zeroline=False),
            zaxis=dict(title="z (A)", showgrid=True, zeroline=False),
            aspectmode="data",
        ),
        updatemenus=updatemenus,
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="white"),
    )

    return go.Figure(data=traces, layout=layout)


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize(
    xyz_file   = XYZ_FILE,
    cache_file = CACHE_FILE,
    cell_file  = CELL_FILE,
    frame_idx  = FRAME_IDX,
    cutoff     = CUTOFF,
    show_h     = SHOW_H,
    output_html= OUTPUT_HTML,
):
    print(f"Loading frame {frame_idx} from {xyz_file} …")
    atoms_df = parse_xyz_frame(xyz_file, frame_idx)

    print(f"Loading geometry cache from {cache_file} …")
    cache_df = pd.read_csv(cache_file)

    n_frames = cache_df["frame"].nunique()
    print(f"  Cache has {n_frames} frames, {len(cache_df)} total H-bond entries.")

    box = parse_box_size(cell_file)
    if box is not None:
        print(f"  Box dimensions: {box[0]:.3f} x {box[1]:.3f} x {box[2]:.3f} A")
    else:
        print("  Warning: could not parse box size — PBC bond splitting disabled.")

    print(f"Building network (cutoff = {cutoff} A) …")
    o_pos, o_index, edges, all_pos, bonded_h = build_network(atoms_df, cache_df, frame_idx, cutoff)

    deg = np.zeros(len(o_pos), dtype=int)
    for u, v, h_pos, _ in edges:
        deg[u] += 1
        deg[v] += 1
    print(f"  {len(o_pos)} O atoms | {len(edges)} H-bonds")
    print(f"  Mean degree: {np.mean(deg):.2f}  Max: {deg.max() if len(deg) else 0}")

    print("Rendering Plotly figure …")
    fig = make_figure(atoms_df, o_pos, o_index, edges, all_pos, bonded_h, frame_idx, cutoff, show_h, box)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_html)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\nSaved → {out_path}")
    print("Open it in any browser for 3D rotation / zoom / pan.")


if __name__ == "__main__":
    visualize()
