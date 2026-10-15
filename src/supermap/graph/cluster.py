"""Centroid-distance clustering of object instances (REQ-047, p.5 Sec. IV-C).

Symbol map (paper -> code):
    X_j (centroid, R^3)  -> centroids[j]
    cluster threshold    -> d_clu (meters, D-070)
Only pairs inside the same cluster are tested with spatial predicates.
"""
from __future__ import annotations

import numpy as np


# REQ-047: single-linkage clustering: i, j linked when ||X_i - X_j|| <= d_clu (D-070)
def cluster_centroids(centroids: np.ndarray, d_clu: float) -> np.ndarray:
    """centroids (N,3) world meters -> cluster labels (N,) int, numbered by first member index."""
    X = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    n = X.shape[0]
    if d_clu < 0:
        raise ValueError("d_clu must be >= 0")
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        ii, jj = np.nonzero(np.triu(dist <= d_clu, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    labels = np.empty(n, dtype=np.int64)
    remap: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        if r not in remap:
            remap[r] = len(remap)
        labels[i] = remap[r]
    return labels


# REQ-047: candidate index pairs (i < j) that share a cluster
def candidate_pairs(labels: np.ndarray) -> list[tuple[int, int]]:
    lab = np.asarray(labels).reshape(-1)
    iu, ju = np.triu_indices(lab.shape[0], k=1)
    keep = lab[iu] == lab[ju]
    return list(zip(iu[keep].tolist(), ju[keep].tolist()))
