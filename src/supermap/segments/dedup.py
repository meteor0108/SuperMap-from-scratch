"""Same-frame duplicate mask grouping by IoU (D-124)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

_MAX_PIXELS = 4_000_000  # above M*H*W the stack is stride-downsampled before the matmul


# REQ-024 (extension): pairwise mask IoU of a (M, H, W) bool stack
def mask_iou_matrix(masks: np.ndarray) -> np.ndarray:
    m = np.asarray(masks, dtype=bool)
    if m.ndim != 3:
        raise ValueError("masks must be (M, H, W)")
    n = m.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    step = 1
    while n * (m.shape[1] // step + 1) * (m.shape[2] // step + 1) > _MAX_PIXELS:
        step += 1
    if step > 1:
        m = m[:, ::step, ::step]
    flat = m.reshape(n, -1).astype(np.float32)
    inter = (flat @ flat.T).astype(np.float64)
    area = np.diag(inter).copy()
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)


# REQ-024 (extension): union-find grouping of masks with pairwise IoU >= iou
def dedup_groups(masks: Sequence[np.ndarray], iou: float = 0.8) -> list[list[int]]:
    n = len(masks)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mat = mask_iou_matrix(np.stack([np.asarray(m, dtype=bool) for m in masks]))
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] >= iou:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]
