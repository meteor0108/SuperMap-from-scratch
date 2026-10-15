"""Pixel footprints of map objects for re-activation boxes and map-guided association (REQ-026)."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from supermap.camera import project


# REQ-026: pixel footprint (robust percentile box) of world points; None when < 3 points project in the image
def footprint_box(points: np.ndarray, P_t: np.ndarray, K: np.ndarray, image_size: tuple[int, int],
                  lo: float = 2.0, hi: float = 98.0) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return None
    H, W = image_size
    uv, _, valid = project(pts, P_t, K)
    ok = valid & np.all(np.isfinite(uv), axis=1)
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if int(ok.sum()) < 3:
        return None
    u, v = uv[ok, 0], uv[ok, 1]
    return np.array([np.percentile(u, lo), np.percentile(v, lo),
                     np.percentile(u, hi), np.percentile(v, hi)], dtype=np.float64)


# REQ-026: footprint of every O_on object over the points selected by point_mask (as _object_geometry)
def map_footprints(O_on: Iterable, P_t: np.ndarray, K: np.ndarray, image_size: tuple[int, int],
                   point_mask: Mapping[int, np.ndarray] | None = None) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for obj in O_on:
        oid = int(obj.id)
        pts = obj.points
        m = None if point_mask is None else point_mask.get(oid)
        if m is not None:
            m = np.asarray(m, dtype=bool).reshape(-1)
            if m.shape[0] == obj.num_points and m.any():
                pts = pts[m]
        box = footprint_box(pts, P_t, K, image_size)
        if box is not None:
            out[oid] = box
    return out
