"""Static 3D visualizer for a SuperMap output folder (self-contained HTML, opens in any browser).

Usage:
    .venv/bin/python scripts/visualize.py <out_dir> [--subset PATH] [--config PATH] [--out PATH]
        [--max-dense-points 80000] [--max-instance-points 4000] [--no-trajectory]

Shows the dense map, per-instance points / boxes / labels, spatial and temporal edges,
VLM waypoints and the camera trajectory. Each trace is toggleable in the legend.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from supermap.config import dataset_config, load_config  # noqa: E402
from supermap.datasets import build_sequence  # noqa: E402
from supermap.geometry.extrinsics import camera_pose_at  # noqa: E402

REL_COLORS = {"on": "orange", "beside": "steelblue", "under": "purple"}
# 12 edges of a box given corner indices (bit0=x, bit1=y, bit2=z)
BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _rgb(c: np.ndarray) -> list[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def _sample(n: int, k: int) -> np.ndarray:
    idx = np.arange(n)
    if n > k:
        idx = np.sort(np.random.default_rng(0).choice(n, k, replace=False))
    return idx


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_outputs(out_dir: Path) -> tuple[dict, dict, dict]:
    """Return (map_points npz as dict, graph.json, run_summary.json or {})."""
    pts = dict(np.load(out_dir / "map_points.npz"))
    graph = json.loads((out_dir / "graph.json").read_text())
    summ_p = out_dir / "run_summary.json"
    summary = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    return pts, graph, summary


def camera_positions(config: Path, subset: Path):
    """Return (sequence, (n, 3) camera positions in world frame)."""
    cfg = load_config(config)
    sub = yaml.safe_load(subset.read_text())
    ds_cfg = dataset_config(cfg, sub.get("dataset", "replica"), root=sub.get("root"),
                            scene=sub.get("scene"), nominal_fps=sub.get("nominal_fps"))
    seq = build_sequence(ds_cfg, sub, base_dir=REPO_ROOT)
    pos = np.stack([camera_pose_at(seq.pose_provider, fid, seq.T_BC)[:3, 3] for fid in seq.frame_ids])
    return seq, pos


def instance_traces(pts: dict, graph: dict, max_pts: int) -> tuple[list[go.Scatter3d], int]:
    """Points, bbox and label trace per instance; returns (traces, drawn point count)."""
    traces: list[go.Scatter3d] = []
    ids = pts["object_ids"]
    disappeared = {int(k): v for k, v in graph.get("disappeared", {}).items()}
    palette = plotly.colors.qualitative.Dark24
    drawn = 0
    for k, node in enumerate(sorted(graph["nodes"], key=lambda n: n["id"])):
        nid, name = node["id"], f"Instance_{node['id']} {node['label']}"
        color, group = palette[k % len(palette)], f"inst{nid}"
        text = name
        if nid in disappeared:
            color, text = "grey", f"{name} (disappeared t={disappeared[nid]})"
        sel = np.flatnonzero(ids == nid)
        sel = sel[_sample(len(sel), max_pts)]
        drawn += len(sel)
        p = pts["object_points"][sel]
        traces.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                                   legendgroup=group, marker=dict(size=2.5, color=color),
                                   hovertext=name, hoverinfo="text"))
        lo, hi = np.asarray(node["bbox"], dtype=float)
        corners = np.array([[(hi if i >> a & 1 else lo)[a] for a in range(3)] for i in range(8)])
        seg = np.full((len(BOX_EDGES) * 3, 3), np.nan)
        for j, (a, b) in enumerate(BOX_EDGES):
            seg[3 * j], seg[3 * j + 1] = corners[a], corners[b]
        traces.append(go.Scatter3d(x=[None if np.isnan(v) else v for v in seg[:, 0]],
                                   y=[None if np.isnan(v) else v for v in seg[:, 1]],
                                   z=[None if np.isnan(v) else v for v in seg[:, 2]],
                                   mode="lines", line=dict(color=color, width=3),
                                   legendgroup=group, showlegend=False, hoverinfo="skip"))
        c = node["centroid"]
        traces.append(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="text", text=[text],
                                   textfont=dict(color=color), legendgroup=group,
                                   showlegend=False, hoverinfo="skip"))
    known = {n["id"] for n in graph["nodes"]}
    rest = np.flatnonzero(~np.isin(ids, list(known)))
    if len(rest):
        rest = rest[_sample(len(rest), max_pts)]
        p = pts["object_points"][rest]
        traces.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   name="Unassigned points", visible="legendonly",
                                   marker=dict(size=2.5, color="grey"), hoverinfo="skip"))
    return traces, drawn


def edge_traces(graph: dict) -> list[go.Scatter3d]:
    """Spatial edges (one line + midpoint hover trace per relation) and temporal edges."""
    cen = {n["id"]: n["centroid"] for n in graph["nodes"]}
    traces: list[go.Scatter3d] = []
    for rel, color in REL_COLORS.items():
        es = [e for e in graph["spatial_edges"] if e[1] == rel and e[0] in cen and e[2] in cen]
        if not es:
            continue
        xyz = [[], [], []]
        mids, hover = [], []
        for a, _, b in es:
            for d in range(3):
                xyz[d] += [cen[a][d], cen[b][d], None]
            mids.append([(cen[a][d] + cen[b][d]) / 2 for d in range(3)])
            hover.append(f"Instance_{a} {rel} Instance_{b}")
        m = np.array(mids)
        traces.append(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode="lines", name=f"Spatial: {rel}",
                                   legendgroup=f"rel_{rel}", line=dict(color=color, width=3),
                                   hoverinfo="skip"))
        traces.append(go.Scatter3d(x=m[:, 0], y=m[:, 1], z=m[:, 2], mode="markers",
                                   legendgroup=f"rel_{rel}", showlegend=False, hovertext=hover,
                                   hoverinfo="text", marker=dict(size=3, color=color)))
    te = graph.get("temporal_edges", [])
    if te:
        xyz = [[], [], []]
        for e in te:
            for d in range(3):
                xyz[d] += [e["centroid_prev"][d], e["centroid_next"][d], None]
        nxt = np.array([e["centroid_next"] for e in te], dtype=float)
        hover = [f"Instance_{e['id']} t={e['t_prev']} -> t={e['t_next']}" for e in te]
        traces.append(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode="lines", name="Temporal edges",
                                   legendgroup="temporal", line=dict(color="darkred", width=3, dash="dash"),
                                   hoverinfo="skip"))
        traces.append(go.Scatter3d(x=nxt[:, 0], y=nxt[:, 1], z=nxt[:, 2], mode="markers",
                                   legendgroup="temporal", showlegend=False, hovertext=hover,
                                   hoverinfo="text", marker=dict(size=4, color="darkred")))
    return traces


def trajectory_traces(seq, pos: np.ndarray) -> list[go.Scatter3d]:
    """Camera path colored by stream index, plus start/end markers."""
    hover = [f"t={i + 1} frame={fid} stamp={seq.stamp(fid):.2f}s" for i, fid in enumerate(seq.frame_ids)]
    path = go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines+markers",
                        name="Camera trajectory", legendgroup="traj", line=dict(color="grey", width=2),
                        marker=dict(size=3, color=np.arange(len(pos)), colorscale="Viridis",
                                    colorbar=dict(title="frame", len=0.5)),
                        hovertext=hover, hoverinfo="text")
    ends = [go.Scatter3d(x=[pos[i, 0]], y=[pos[i, 1]], z=[pos[i, 2]], mode="markers", name=n,
                         legendgroup="traj", marker=dict(size=7, color=c), hoverinfo="name")
            for i, n, c in ((0, "Start", "green"), (-1, "End", "red"))]
    return [path, *ends]


def build_figure(out_dir: Path, args: argparse.Namespace) -> tuple[go.Figure, dict]:
    pts, graph, summary = load_outputs(out_dir)
    fig, stats = go.Figure(), {}
    d = _sample(len(pts["dense_points"]), args.max_dense_points)
    p = pts["dense_points"][d]
    fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name="Dense map",
                               marker=dict(size=1.5, color=_rgb(pts["dense_colors"][d])), hoverinfo="skip"))
    inst, stats["inst_pts"] = instance_traces(pts, graph, args.max_instance_points)
    stats["dense_pts"] = len(d)
    for t in inst + edge_traces(graph):
        fig.add_trace(t)
    vlm = summary.get("vlm") or {}
    if vlm.get("status") == "ok" and vlm.get("waypoints"):
        w = np.asarray(vlm["waypoints"], dtype=float).reshape(-1, 3)
        hover = [f"waypoint {k}: Instance_{i}" for k, i in enumerate(vlm.get("ids", []))][:len(w)]
        hover += [f"waypoint {k}" for k in range(len(hover), len(w))]
        fig.add_trace(go.Scatter3d(x=w[:, 0], y=w[:, 1], z=w[:, 2], mode="markers",
                                   name=f'VLM waypoints: "{vlm.get("query", "")}"', hovertext=hover,
                                   hoverinfo="text", marker=dict(size=9, color="gold", symbol="diamond",
                                                                 line=dict(color="black", width=2))))
    stats["frames"] = 0
    if not args.no_trajectory:
        try:
            subset = _resolve(args.subset) if args.subset else out_dir / "subset.yaml"
            if not subset.exists():
                subset = REPO_ROOT / "configs/subset_replica_office0.yaml"
                print(f"warning: no subset.yaml in {out_dir}; trajectory uses {subset.name} (pass --subset)")
            seq, pos = camera_positions(_resolve(args.config), subset)
            for t in trajectory_traces(seq, pos):
                fig.add_trace(t)
            stats["frames"] = len(pos)
        except Exception as e:  # trajectory is optional
            print(f"warning: trajectory skipped ({type(e).__name__}: {e})")
    scene = yaml.safe_load(subset.read_text()).get("scene", out_dir.name) if stats["frames"] else out_dir.name
    fig.update_layout(title=f"SuperMap: {scene} ({len(graph['nodes'])} instances, t={graph.get('t')})",
                      scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]",
                                 zaxis_title="z [m]"),
                      showlegend=True, margin=dict(l=0, r=0, t=40, b=0), height=900)
    stats["nodes"] = len(graph["nodes"])
    stats["spatial"] = len(graph["spatial_edges"])
    stats["temporal"] = len(graph.get("temporal_edges", []))
    return fig, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-dense-points", type=int, default=80000)
    ap.add_argument("--max-instance-points", type=int, default=4000)
    ap.add_argument("--no-trajectory", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out = Path(args.out) if args.out else out_dir / "viz.html"
    fig, s = build_figure(out_dir, args)
    fig.write_html(out, include_plotlyjs=True)
    print(f"points drawn: dense={s['dense_pts']} instance={s['inst_pts']}")
    print(f"nodes={s['nodes']} spatial_edges={s['spatial']} temporal_edges={s['temporal']} trajectory_frames={s['frames']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
