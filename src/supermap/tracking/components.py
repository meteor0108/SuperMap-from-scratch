"""Largest connected component of a point set on a voxel grid (iter-15 outlier removal)."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_OFFSETS = np.array([o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)],
                    dtype=np.int64)


# REQ-023: mask of the points in the largest 26-connected voxel component at grid `voxel` (iter-15 outlier removal, paper-silent)
def largest_component_mask(points: np.ndarray, voxel: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool)
    ijk = np.floor(points / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0)  # non-negative, so a mixed-radix key is unique
    dims = ijk.max(axis=0) + 3
    ijk += 1
    stride = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    keys = ijk @ stride
    ukeys, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    m = ukeys.shape[0]
    rows, cols = [], []
    for off in _OFFSETS:
        nk = ukeys + int(off @ stride)
        pos = np.searchsorted(ukeys, nk)
        pos[pos >= m] = m - 1
        hit = ukeys[pos] == nk
        rows.append(np.nonzero(hit)[0])
        cols.append(pos[hit])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    graph = coo_matrix((np.ones(r.shape[0], dtype=np.int8), (r, c)), shape=(m, m))
    _, comp = connected_components(graph, directed=False)
    point_comp = comp[inverse]
    best = np.argmax(np.bincount(point_comp))
    return point_comp == best
