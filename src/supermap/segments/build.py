"""Turn segment clusters into map instances (D-124)."""
from __future__ import annotations

import numpy as np

from supermap.geometry.dense_map import voxel_keys
from supermap.map import ObjectInstance
from supermap.segments.cluster import ClusterResult
from supermap.segments.store import SegmentStore


# REQ-023 (extension): one ObjectInstance per cluster; points voxel-deduplicated, label vote summed
def clusters_to_instances(store: SegmentStore, result: ClusterResult, num_classes: int,
                          voxel: float = 0.02) -> list[ObjectInstance]:
    out: list[ObjectInstance] = []
    for idx, members in enumerate(result.clusters):
        segs = [store.segments[i] for i in members]
        pts = np.concatenate([s.points for s in segs]).astype(np.float64)
        cols = np.concatenate([s.colors for s in segs])
        if pts.shape[0]:
            _, first = np.unique(voxel_keys(pts, voxel), return_index=True)
            first.sort()
            pts, cols = pts[first], cols[first]
        probs = np.sum([s.label_scores(num_classes) for s in segs], axis=0)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(num_classes, 1.0 / num_classes)
        out.append(ObjectInstance(id=idx, points=pts, colors=cols, log_odds=np.zeros(pts.shape[0]),
                                  label_probs=probs, first_seen=min(s.t for s in segs),
                                  last_seen=max(s.t for s in segs), exist_log_odds=0.0))
    return out
