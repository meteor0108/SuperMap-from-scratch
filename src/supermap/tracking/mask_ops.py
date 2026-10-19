"""Mask clean-up before lifting (Supp. Alg. 1 line 20 Unproject; D-106, D-107).

Symbol map (paper -> code):
    M_mask                -> list of (track_id, (H, W) bool) masks of one frame
    erode_mask            -> REQ-023 binary erosion of a SAM2 mask (D-106)
    depth_inlier_mask     -> REQ-023 MAD-based depth outlier removal inside a mask (D-106)
    resolve_mask_overlaps -> REQ-014/REQ-023 one owner per pixel across the masks of a frame (D-107);
                              background classes win shared pixels first (D-122)
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_erosion


# REQ-023: erode a mask by erode_px pixels; keep the original if too large a fraction would be lost (D-106)
def erode_mask(mask: np.ndarray, erode_px: int, min_keep: float = 0.5) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if erode_px <= 0 or not mask.any():
        return mask
    k = 2 * int(erode_px) + 1
    m2 = binary_erosion(mask, structure=np.ones((k, k), dtype=bool))
    if m2.sum() < min_keep * mask.sum():
        return mask
    return m2


# REQ-023: drop mask pixels whose depth deviates from the mask median by more than k * sigma_MAD (D-106)
def depth_inlier_mask(depth: np.ndarray, mask: np.ndarray, k: float, floor: float = 0.05,
                      depth_min: float = 0.0, depth_max: float | None = None) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > depth_min)
    if depth_max is not None:
        valid &= depth <= depth_max
    sel = mask & valid
    if int(sel.sum()) < 3:
        return mask
    d = depth[sel].astype(np.float64)
    med = np.median(d)
    sigma = max(1.4826 * np.median(np.abs(d - med)), floor)
    keep = np.abs(depth.astype(np.float64) - med) <= k * sigma
    return mask & ~(valid & ~keep)


# REQ-014, REQ-023: smaller-wins owner map over the masks of one group (full-mask areas), writing only `allowed` pixels (D-107)
def _owner_map(group: list[np.ndarray], allowed: np.ndarray) -> np.ndarray:
    areas = [int(m.sum()) for m in group]
    owner = np.full(allowed.shape, -1, dtype=np.int64)
    # largest first, stable on ties so the earlier mask is written last and wins
    for k in sorted(range(len(group)), key=lambda i: (-areas[i], -i)):
        owner[group[k] & allowed] = k
    return owner


# REQ-014, REQ-023: a pixel covered by several masks is kept only in the smallest one (D-107);
# masks of background classes claim shared pixels before the foreground masks (D-122)
def resolve_mask_overlaps(masks: Sequence[tuple[int, np.ndarray]], rule: str | None = "smaller",
                          labels: dict[int, int] | None = None,
                          background: Iterable[int] = ()) -> list[tuple[int, np.ndarray]]:
    if rule is None or len(masks) == 0:
        return list(masks)
    if rule != "smaller":
        raise ValueError(f"unknown overlap rule: {rule}")
    stack = [np.asarray(m, dtype=bool) for _, m in masks]
    bg = {int(b) for b in background}
    is_bg = [labels is not None and int(tid) in labels and int(labels[tid]) in bg for tid, _ in masks]
    free = np.ones(stack[0].shape, dtype=bool)
    out: list[np.ndarray | None] = [None] * len(masks)
    for flag in (True, False):
        idx = [i for i in range(len(masks)) if is_bg[i] == flag]
        if not idx:
            continue
        owner = _owner_map([stack[i] for i in idx], free)
        for j, i in enumerate(idx):
            out[i] = owner == j
        if flag:
            free = free & (owner < 0)
    return [(tid, out[k]) for k, (tid, _) in enumerate(masks)]
