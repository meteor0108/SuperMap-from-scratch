"""3D contact criterion for ObjMerge absorb (REQ-024, paper-silent detail)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# REQ-024: fraction of new_points whose nearest obj_points neighbor is within dist
def contact_fraction(new_points: np.ndarray, obj_points: np.ndarray, dist: float,
                     tree: cKDTree | None = None) -> float:
    if new_points.shape[0] == 0 or obj_points.shape[0] == 0:
        return 0.0
    if tree is None:
        tree = cKDTree(obj_points)
    d, _ = tree.query(new_points, distance_upper_bound=dist)
    return float(np.isfinite(d).mean())


# REQ-024: lazily built per-candidate KD-trees for one obj_merge call
class TreeCache:
    def __init__(self) -> None:
        self._trees: dict[int, cKDTree] = {}

    # REQ-024: contact fraction of new_points against candidate obj, tree cached by object id
    def fraction(self, obj_id: int, obj_points: np.ndarray, new_points: np.ndarray, dist: float) -> float:
        if obj_points.shape[0] == 0:
            return 0.0
        if obj_id not in self._trees:
            self._trees[obj_id] = cKDTree(obj_points)
        return contact_fraction(new_points, obj_points, dist, self._trees[obj_id])
