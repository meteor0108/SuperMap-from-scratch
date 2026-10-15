"""3D distance gate for the ByteTrack association: extends the Eq. (5) prior into 3D (iter-15 experiment).

A (track, detection) pair is rejected when the 3D centroid of the detection is farther than the gate
from the 3D centroid of the track's mapped instance.
"""
from __future__ import annotations

import numpy as np

from supermap.geometry.backproject import unproject_pixels


# REQ-022: 3D centroid of a detection box from the depth inside its central region (iter-15 gate, paper-silent)
def detection_centroids(depth, K, P_t, boxes, shrink=0.5, depth_min=0.0, depth_max=None) -> np.ndarray:
    """Per-axis median of the world points inside the central sub-box; (M, 3), NaN rows when no valid depth."""
    depth = np.asarray(depth)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    H, W = depth.shape
    out = np.full((boxes.shape[0], 3), np.nan, dtype=np.float64)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        hw, hh = max(x2 - x1, 0.0) * shrink / 2.0, max(y2 - y1, 0.0) * shrink / 2.0
        u0 = int(np.clip(np.floor(cx - hw), 0, W - 1))
        u1 = int(np.clip(np.ceil(cx + hw), u0 + 1, W))
        v0 = int(np.clip(np.floor(cy - hh), 0, H - 1))
        v1 = int(np.clip(np.ceil(cy + hh), v0 + 1, H))
        d = depth[v0:v1, u0:u1]
        valid = np.isfinite(d) & (d > depth_min)
        if depth_max is not None:
            valid &= d < depth_max
        v, u = np.nonzero(valid)
        if v.size == 0:
            continue
        uv = np.stack([u + u0, v + v0], axis=1).astype(np.float64)
        out[i] = np.median(unproject_pixels(uv, d[v, u], K, P_t), axis=0)
    return out


# REQ-022: pairwise 3D gate matrix, True where the pair must be rejected
def gate_matrix(track_centroids, det_centroids, gate) -> np.ndarray:
    t = np.asarray(track_centroids, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(det_centroids, dtype=np.float64).reshape(-1, 3)
    if t.shape[0] == 0 or d.shape[0] == 0:
        return np.zeros((t.shape[0], d.shape[0]), dtype=bool)
    diff = t[:, None, :] - d[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    both = np.isfinite(t).all(axis=1)[:, None] & np.isfinite(d).all(axis=1)[None, :]
    with np.errstate(invalid="ignore"):
        return both & (dist > gate)
