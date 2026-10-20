#!/usr/bin/env python
"""Offline study tool for segment-based instances with multi-view consensus (D-124).

extract: GroundingDINO + SAM2 front end of the paper pipeline -> SegmentStore cache (GPU)
cluster: consensus clustering of the cached segments -> map_objects.json / map_points.npz (CPU)
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supermap.config import REPO_ROOT, dataset_config, load_config  # noqa: E402
from supermap.perception import GroundingDINODetector, PromptList, SAM2Segmenter  # noqa: E402
from supermap.segments.store import Keyframe, SegmentStore  # noqa: E402

VOXEL = 0.02  # per-segment voxel deduplication (m)


def prompt_list(cfg):
    pc = cfg["perception"]
    return PromptList.from_config(pc["prompts"], pc.get("prompt_aliases"))


# background class indices from tracking.merge.contact_exclude, resolved like the pipeline does
def background_indices(cfg, prompts):
    names = cfg["tracking"]["merge"].get("contact_exclude", [])
    return [prompts.index(p) for p in names if " ".join(str(p).strip().lower().split()) in prompts.phrases]


# fallback same-frame mask dedup: union-find over pairs with IoU >= iou
def _local_dedup_groups(masks, iou=0.8):
    n = len(masks)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    flat = [np.asarray(m, bool) for m in masks]
    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(flat[i], flat[j]).sum()
            union = np.logical_or(flat[i], flat[j]).sum()
            if union > 0 and inter / union >= iou:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for _, g in sorted(groups.items())]


def get_dedup():
    try:
        from supermap.segments.dedup import dedup_groups
        return dedup_groups, "supermap.segments.dedup"
    except ImportError:
        return _local_dedup_groups, "local union-find"


def extract(a):
    from supermap.geometry.dense_map import voxel_keys
    from supermap.pipeline import SuperMapPipeline
    from supermap.scheduler import ModuleScheduler
    from supermap.tracking.depth_edges import depth_edge_mask
    from supermap.tracking.instance_ops import unproject_masks
    from supermap.tracking.mask_ops import resolve_mask_overlaps
    from supermap.datasets import build_sequence

    t_start = time.time()
    cfg = load_config(a.config)
    sub = yaml.safe_load(Path(a.subset).read_text())
    name = sub.get("dataset", "replica")
    ds_cfg = dataset_config(cfg, name, root=sub.get("root"), scene=sub.get("scene"),
                            nominal_fps=sub.get("nominal_fps"))
    seq = build_sequence(ds_cfg, sub, base_dir=REPO_ROOT)
    prompts = prompt_list(cfg)
    detector = GroundingDINODetector.from_config(cfg["perception"])
    segmenter = SAM2Segmenter.from_config(cfg["perception"])
    scheduler = ModuleScheduler.from_config(cfg.get("rates"))
    pipe = SuperMapPipeline(cfg, detector=detector, segmenter=segmenter, pose_provider=seq.pose_provider,
                            T_BC=seq.T_BC, scheduler=scheduler)
    dedup_groups, dedup_src = get_dedup()
    bg_idx = background_indices(cfg, prompts)
    bg_set = set(bg_idx)
    ucfg = pipe.unproject_cfg
    s = int(a.scale)
    store = SegmentStore()
    n_before = n_after = 0
    print(f"scene={ds_cfg['scene']} frames={len(seq)} dedup={dedup_src} background={bg_idx}", flush=True)

    for obs in seq:
        if not scheduler.due("pose", obs.stamp):
            continue
        seg_due = scheduler.due("segmentation", obs.stamp)
        scheduler.due("mapping", obs.stamp)
        scheduler.due("scene_graph", obs.stamp)
        if not seg_due:
            continue
        obs = pipe.pose(obs)
        H, W = obs.depth.shape[:2]
        det = detector.detect(obs.rgb, t=obs.t)
        keep = np.flatnonzero(det.scores >= a.mask_score_min)
        masks = segmenter.segment(obs.rgb, det.boxes[keep]) if keep.size else np.zeros((0, H, W), bool)
        n_before += len(masks)
        groups = dedup_groups(list(masks), a.dedup_iou) if len(masks) else []
        n_after += len(groups)
        # one union mask per group; label observations of all members
        merged, labs = [], []
        for k, g in enumerate(groups):
            merged.append((k, np.logical_or.reduce([masks[i] for i in g])))
            labs.append([(int(det.label_ids[keep[i]]), float(det.scores[keep[i]])) for i in g])
        merged = resolve_mask_overlaps(merged, pipe.resolve_overlaps)
        lifted = unproject_masks(obs, merged, labels=None, scores=None, depth_min=pipe.depth_min,
                                 depth_max=pipe.depth_max, min_points=pipe.min_points, **ucfg)
        seg_of = {}  # mask index k -> segment
        for o in lifted:
            _, first = np.unique(voxel_keys(o.points, VOXEL), return_index=True)
            first = np.sort(first)
            seg_of[int(o.id)] = store.add_segment(obs.t, o.points[first], o.colors[first], labs[int(o.id)])
        # keyframe record
        depth_full = np.asarray(obs.depth, np.float32)
        ok_full = depth_edge_mask(depth_full, np.ones(depth_full.shape, bool), ucfg["edge_jump"],
                                  ucfg["edge_rel"], ucfg["edge_radius"])
        depth = depth_full[::s, ::s].copy()
        fin = np.isfinite(depth) & (depth > 0)
        valid = fin & ok_full[::s, ::s]
        depth[~fin] = 0.0
        mask_ids = np.full(depth.shape, -1, np.int32)
        seg_ids, bg = [], []
        for k, m in merged:
            if k not in seg_of:
                continue
            mask_ids[np.asarray(m, bool)[::s, ::s]] = len(seg_ids)
            sg = seg_of[k]
            seg_ids.append(sg.id)
            vote = sg.label_scores(prompts.C)
            bg.append(bool(vote.sum() > 0 and int(vote.argmax()) in bg_set))
        store.add_keyframe(Keyframe(t=obs.t, pose=obs.pose, K=obs.K, scale=s, depth=depth, valid=valid,
                                    mask_ids=mask_ids, segment_ids=np.asarray(seg_ids, np.int64),
                                    background=np.asarray(bg, bool)))
        print(f"t={obs.t:4d} boxes={len(keep):3d} masks={len(groups):3d} segments={len(seg_ids):3d}", flush=True)

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    store.save(cache)
    npts = [sg.num_points for sg in store.segments.values()]
    summary = {"n_keyframes": len(store.keyframes), "n_segments": len(store),
               "n_masks_before_dedup": n_before, "n_masks_after_dedup": n_after,
               "mean_points_per_segment": float(np.mean(npts)) if npts else 0.0,
               "wall_time_s": round(time.time() - t_start, 2), "config": str(a.config), "subset": str(a.subset),
               "mask_score_min": a.mask_score_min, "scale": s, "dedup_iou": a.dedup_iou,
               "dedup_impl": dedup_src, "background_classes": bg_idx, "num_classes": prompts.C}
    (cache / "extract_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


# parse --set key=value with yaml.safe_load
def parse_sets(sets):
    out = {}
    for item in sets:
        if "=" not in item:
            sys.exit(f"bad --set '{item}', expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def _vec(x):
    return None if x is None else [float(v) for v in np.asarray(x).reshape(-1)]


# same per-instance record as supermap.run.object_records
def object_record(o, names):
    lo, hi = o.bbox if o.num_points > 0 else (None, None)
    return {"id": o.id, "label": names[o.label] if o.num_classes else "unknown",
            "label_probs": _vec(o.label_probs), "centroid": _vec(o.centroid) if o.num_points else None,
            "bbox": None if lo is None else [_vec(lo), _vec(hi)], "num_points": o.num_points,
            "status": "active", "first_seen": o.first_seen, "last_seen": o.last_seen,
            "exist_log_odds": float(o.exist_log_odds)}


def cluster(a):
    from supermap.segments.build import clusters_to_instances
    from supermap.segments.cluster import consensus_cluster
    from supermap.segments.params import ConsensusParams
    from supermap.segments.visibility import segment_histograms

    cfg = load_config(a.config)
    prompts = prompt_list(cfg)
    C = len(prompts.phrases)
    cache = Path(a.cache)
    store = SegmentStore.load(cache)
    ext = json.loads((cache / "extract_summary.json").read_text()) if (cache / "extract_summary.json").exists() else {}
    bg_idx = ext.get("background_classes", background_indices(cfg, prompts))
    params = ConsensusParams.from_dict({"background": bg_idx, **parse_sets(a.sets)})
    t0 = time.time()
    hists = segment_histograms(store, params)
    t1 = time.time()
    res = consensus_cluster(store, hists, params, num_classes=C)
    t2 = time.time()
    objs = clusters_to_instances(store, res, C, params.voxel, params)
    t3 = time.time()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "map_objects.json").write_text(json.dumps([object_record(o, prompts.phrases) for o in objs], indent=1))
    live = [o for o in objs if o.num_points > 0]
    cat = (lambda xs, shape, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dt))
    np.savez(out / "map_points.npz",
             object_points=cat([o.points for o in live], (0, 3), np.float32),
             object_colors=cat([o.colors for o in live], (0, 3), np.uint8),
             object_log_odds=cat([o.log_odds for o in live], (0,), np.float32),
             object_ids=cat([np.full(o.num_points, o.id) for o in live], (0,), np.int64),
             dense_points=np.zeros((0, 3), np.float32), dense_colors=np.zeros((0, 3), np.uint8))
    (out / "cluster_log.json").write_text(json.dumps(res.pair_log, indent=1, default=str))
    summary = {"backend": "segments_offline", "N_T": len(objs), "n_segments": len(store),
               "n_clusters_fg": len(res.clusters) - int(sum(res.background)),
               "n_clusters_bg": int(sum(res.background)),
               "merges": res.merges, "rounds": res.rounds, "params": dataclasses.asdict(params),
               "timings_s": {"histograms": round(t1 - t0, 2), "clustering": round(t2 - t1, 2),
                             "build": round(t3 - t2, 2)}}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=1, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["extract", "cluster", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mask-score-min", type=float, default=0.35)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--dedup-iou", type=float, default=0.8)
    ap.add_argument("--set", action="append", default=[], dest="sets", metavar="KEY=VALUE")
    a = ap.parse_args()
    if a.stage in ("extract", "all") and a.subset is None:
        ap.error("--subset is required for extract")
    if a.stage in ("cluster", "all") and a.out is None:
        ap.error("--out is required for cluster")
    if a.stage in ("extract", "all"):
        extract(a)
    if a.stage in ("cluster", "all"):
        cluster(a)


if __name__ == "__main__":
    main()
