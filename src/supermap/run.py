"""E2E run on a dataset subset (REQ-062; Fig. 2 full pipeline, D-002, D-003).

Usage:
    python -m supermap.run --config configs/default.yaml --subset configs/subset_replica_office0.yaml --out <dir>

Outputs in <dir> (world frame W, meters; t = stream index, D-008):
    map_objects.json  per instance: id, label, label_probs, centroid, bbox, num_points, status,
                      first_seen, last_seen (REMOVED instances keep their last known geometry, D-094)
    map_points.npz    instance points/colors/ids/log-odds and the dense map points/colors
    graph.json        nodes, spatial edges E_s, temporal edges E_t, disappeared ids
    serialized.txt    schema + local subgraph + spatio-temporal cues (REQ-052..REQ-056)
    run_summary.json  frame counts, N_T, module run counts, VLM status
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from supermap.config import REPO_ROOT, dataset_config, load_config
from supermap.datasets import build_sequence
from supermap.graph import select_local_subgraph, serialize_graph
from supermap.pipeline import FrameResult, SuperMapPipeline
from supermap.vlm import find_api_key, make_backend, run_grounding
from supermap.vlm.backend import PROJECT_DOTENV

log = logging.getLogger("supermap.run")


# REQ-062: JSON-friendly list of floats
def _vec(x) -> list[float] | None:
    return None if x is None else [float(v) for v in np.asarray(x).reshape(-1)]


# REQ-062, REQ-005: per-instance record of M_T
def object_records(pipe: SuperMapPipeline) -> list[dict]:
    out = []
    for o in pipe.map.all_objects():  # ACTIVE objects plus REMOVED status records (REQ-037)
        c, bb = pipe.object_geometry(o)
        out.append({
            "id": o.id,
            "label": pipe.map.class_names[o.label] if o.num_classes else "unknown",
            "label_probs": _vec(o.label_probs),
            "centroid": _vec(c),
            "bbox": None if bb is None else [_vec(bb[0]), _vec(bb[1])],
            "num_points": o.num_points,
            "status": o.status.value,
            "first_seen": o.first_seen,
            "last_seen": o.last_seen,
            "exist_log_odds": float(o.exist_log_odds),  # REQ-035: D-113
        })
    return out


# REQ-062, REQ-046, REQ-050: graph as JSON
def graph_record(graph) -> dict:
    return {
        "t": graph.t,
        "nodes": [{"id": n.id, "label": n.label, "centroid": _vec(n.centroid),
                   "bbox": [_vec(n.bbox[0]), _vec(n.bbox[1])], "first_seen": n.first_seen,
                   "last_seen": n.last_seen} for _, n in sorted(graph.nodes.items())],
        "spatial_edges": [[a, r, b] for a, r, b in graph.E_s],
        "temporal_edges": [{"id": e.id, "t_prev": e.t_prev, "t_next": e.t_next,
                            "centroid_prev": _vec(e.centroid_prev), "centroid_next": _vec(e.centroid_next)}
                           for e in graph.E_t],
        "disappeared": {str(k): v for k, v in sorted(graph.disappeared.items())},
    }


# REQ-062: instance points and dense map arrays
def save_points(pipe: SuperMapPipeline, path: Path) -> None:
    objs = [o for o in pipe.map if o.num_points > 0]
    cat = (lambda xs, shape, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dt))
    np.savez(path,
             object_points=cat([o.points for o in objs], (0, 3), np.float32),
             object_colors=cat([o.colors for o in objs], (0, 3), np.uint8),
             object_log_odds=cat([o.log_odds for o in objs], (0,), np.float32),
             object_ids=cat([np.full(o.num_points, o.id) for o in objs], (0,), np.int64),
             dense_points=pipe.dense_map.points.astype(np.float32),
             dense_colors=pipe.dense_map.colors)


# REQ-062: frame log line
def _frame_line(r: FrameResult, pipe: SuperMapPipeline, dt: float) -> str:
    mods = "".join(k[0].upper() if v else "-" for k, v in r.ran.items())
    n_act = len(pipe.map.active_objects())
    s = f"t={r.t:3d} stamp={r.stamp:5.2f} [{mods}] N_t={n_act} removed={len(pipe.map.removed_objects())}"
    if r.ran.get("segmentation"):
        s += (f" dets={r.num_detections} T_trk={r.t_trk_ids} O_nbr={r.o_nbr_ids} O_on={r.o_on_ids}"
              f" O_front={r.o_front_ids} O_neg={r.o_neg_ids} new={r.new_ids}")
    if r.removed_ids:
        s += f" removed={r.removed_ids}"
    return s + f" ({dt:.2f}s)"


# REQ-062: run the pipeline over a subset and write all outputs
def run(config: str | Path, subset: str | Path, out: str | Path) -> dict:
    t_start = time.time()
    cfg = load_config(config)
    sub = yaml.safe_load(Path(subset).read_text())
    name = sub.get("dataset", "replica")
    ds_cfg = dataset_config(cfg, name, root=sub.get("root"), scene=sub.get("scene"),
                            nominal_fps=sub.get("nominal_fps"))
    seq = build_sequence(ds_cfg, sub, base_dir=REPO_ROOT)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = SuperMapPipeline(cfg, pose_provider=seq.pose_provider, T_BC=seq.T_BC)
    print(f"dataset={name} scene={ds_cfg['scene']} frames={len(seq)} rates={pipe.scheduler.rates}", flush=True)

    last = [time.time()]

    # REQ-062: per-frame progress
    def on_frame(r: FrameResult) -> None:
        now = time.time()
        print(_frame_line(r, pipe, now - last[0]), flush=True)
        last[0] = now

    pipe.run(seq, on_frame)
    graph = pipe.final_graph()

    (out / "map_objects.json").write_text(json.dumps(object_records(pipe), indent=1))
    save_points(pipe, out / "map_points.npz")
    (out / "graph.json").write_text(json.dumps(graph_record(graph), indent=1))
    center = pipe.last_obs.pose[:3, 3] if (pipe.last_obs is not None and
                                          cfg.get("run", {}).get("subgraph_center") == "camera") else None
    sub_graph = select_local_subgraph(graph, center=center)
    text = serialize_graph(sub_graph)
    (out / "serialized.txt").write_text(text + "\n")

    vlm = {"status": "skipped", "reason": ""}
    vcfg = cfg["vlm"]
    # REQ-057: key from environment, then the project-root .env (D-098); the value is never logged
    key = find_api_key(vcfg.get("api_key_env", ()), dotenv_path=PROJECT_DOTENV)
    if key is None:
        vlm["reason"] = "no GEMINI_API_KEY/GOOGLE_API_KEY in environment or .env; stopped after serialization (D-003)"
    else:
        try:
            backend = make_backend(vcfg, api_key=key)
            q = cfg.get("run", {}).get("vlm_query", "")
            g = run_grounding(q, graph, pipe.map, backend, center=center)
            vlm = {"status": "ok", "model": getattr(backend, "model", vcfg.get("model")), "query": q,
                   "response": g.response, "ids": g.ids, "waypoints": g.waypoints.tolist()}
        except RuntimeError as e:  # SDK missing or call failure
            vlm = {"status": "skipped", "reason": f"VLM backend unavailable: {e}"}

    active = pipe.map.active_objects()
    summary = {
        "frames_in_subset": len(seq),
        "frames_processed": len(pipe.frames),
        "frames_skipped": pipe.skipped_frames,
        "N_T": len(active),  # REQ-005: N_t counts only ACTIVE objects of M_T (D-096)
        "absorbed": int(sum(len(r.absorbed) for r in pipe.frames)),  # REQ-024: D-111
        "component_pruned_points": int(sum(sum(r.component_pruned.values()) for r in pipe.frames)),  # REQ-024: iter 15
        "map_matched": int(sum(r.map_matched for r in pipe.frames)),  # REQ-022
        "gate_rejected": int(sum(r.gate_rejected for r in pipe.frames)),  # REQ-022: iter-15 3D gate
        "map_dropped": int(sum(r.map_dropped for r in pipe.frames)),  # REQ-026
        "absorbed_pairs": [[int(o), int(n)] for r in pipe.frames for o, n in r.absorbed.items()][:200],
        "N_T_active": len(active),
        "N_T_removed": len(pipe.map.removed_objects()),
        "removed_by_existence": int(pipe.removed_by_existence),  # REQ-035: D-113
        "module_runs": dict(pipe.scheduler.counts),
        "rates_hz": dict(pipe.scheduler.rates),
        "dense_map_points": len(pipe.dense_map),
        "instance_points": int(sum(o.num_points for o in pipe.map.all_objects())),
        "graph": {"nodes": len(graph.nodes), "spatial_edges": len(graph.E_s),
                  "temporal_edges": len(graph.E_t), "disappeared": len(graph.disappeared)},
        "serialized_nodes": len(sub_graph.nodes),
        "vlm": vlm,
        "runtime_s": round(time.time() - t_start, 2),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# REQ-062: command-line entry point
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SuperMap end-to-end run")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--subset", default=str(REPO_ROOT / "configs" / "subset_replica_office0.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    run(args.config, args.subset, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
