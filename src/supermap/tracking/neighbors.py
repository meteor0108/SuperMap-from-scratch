"""Local map context O_nbr = GetNeighborObjects(M, Odom) (Supp. Alg. 1 lines 5-6).

Neighbor criterion, default mode "points" (D-110, supersedes D-032 as default): an ACTIVE object is a
neighbor if any of up to `sample_points` evenly strided points projects inside the image (camera z > 0)
and lies within max_range of the camera center p_t of P_t. Mode "centroid" keeps D-032: the centroid
X_j projects inside the image (camera z > 0) and ||X_j - p_t|| <= max_range.
"""
from __future__ import annotations

import numpy as np

from supermap.camera import PinholeCamera, project
from supermap.map import GlobalMap, ObjectInstance


# REQ-021: O_nbr: active objects in the view frustum and within max_range (m); see module docstring
def get_neighbor_objects(global_map: GlobalMap, P_t: np.ndarray, K: np.ndarray,
                         image_size: tuple[int, int], max_range: float | None,
                         mode: str = "points", sample_points: int = 256) -> list[ObjectInstance]:
    """image_size = (H, W) pixels. Returns objects sorted by id (references into the map)."""
    objs = [o for o in global_map.active_objects() if o.num_points > 0]
    if not objs:
        return []
    H, W = int(image_size[0]), int(image_size[1])
    cam = PinholeCamera.from_K(K, W, H)
    center = np.asarray(P_t, np.float64)[:3, 3]
    if mode == "centroid":  # REQ-021: D-032 centroid criterion
        X = np.stack([o.centroid for o in objs])
        uv, _, valid = project(X, P_t, K)
        keep = valid & cam.in_image(uv)
        if max_range is not None:
            keep &= np.linalg.norm(X - center, axis=1) <= float(max_range)
        return [o for o, k in zip(objs, keep) if k]
    if mode != "points":
        raise ValueError(f"unknown neighbors mode: {mode}")
    # REQ-021: D-110 points criterion, one batched projection over all sampled points
    samples, owner = [], []
    for i, o in enumerate(objs):
        pts = o.points
        # REQ-021: even stride ceil(n / sample_points) over the whole array: indices 0, c, 2c, ... up to the
        # last point, at most sample_points of them (D-110, I-8-6)
        s = pts[::-(-pts.shape[0] // int(sample_points))]
        samples.append(s)
        owner.append(np.full(s.shape[0], i, dtype=np.int64))
    X = np.concatenate(samples)
    owner = np.concatenate(owner)
    uv, _, valid = project(X, P_t, K)
    ok = valid & cam.in_image(uv)
    if max_range is not None:
        ok &= np.linalg.norm(X - center, axis=1) <= float(max_range)
    keep = np.zeros(len(objs), dtype=bool)
    keep[owner[ok]] = True
    return [o for o, k in zip(objs, keep) if k]
