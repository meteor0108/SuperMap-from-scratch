"""Spatial predicates on world axis-aligned bboxes (REQ-048, REQ-049, p.5 Sec. IV-C).

Symbol map (paper -> code):
    B_A, B_B                -> (min_xyz, max_xyz) tuples, world frame, meters (D-011)
    z^A_min, z^B_max        -> bbox[0][axis], bbox[1][axis]; vertical axis = world z (D-072)
    IoU_xy(B_A, B_B)        -> iou_xy
    gamma                   -> gamma (D-071)
    "approximately equal"   -> |z^A_min - z^B_max| <= delta (D-071)
Relation strings are exactly "on", "beside", "under".
"""
from __future__ import annotations

import numpy as np

ON = "on"
BESIDE = "beside"
UNDER = "under"
RELATIONS = (ON, BESIDE, UNDER)

BBox = tuple[np.ndarray, np.ndarray]


# REQ-048: horizontal axes are the two axes other than the vertical one
def _horizontal_axes(axis: int) -> list[int]:
    if axis not in (0, 1, 2):
        raise ValueError("vertical axis must be 0, 1 or 2")
    return [a for a in range(3) if a != axis]


# REQ-048: bbox as two float arrays (min_xyz, max_xyz)
def _as_bbox(b: BBox) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(b[0], dtype=np.float64).reshape(3)
    hi = np.asarray(b[1], dtype=np.float64).reshape(3)
    if np.any(hi < lo):
        raise ValueError("bbox max must be >= min")
    return lo, hi


# REQ-048: IoU of the two bboxes projected onto the horizontal (xy) plane
def iou_xy(b_a: BBox, b_b: BBox, axis: int = 2) -> float:
    (alo, ahi), (blo, bhi) = _as_bbox(b_a), _as_bbox(b_b)
    h = _horizontal_axes(axis)
    inter_sides = np.clip(np.minimum(ahi[h], bhi[h]) - np.maximum(alo[h], blo[h]), 0.0, None)
    inter = float(np.prod(inter_sides))
    area_a = float(np.prod(ahi[h] - alo[h]))
    area_b = float(np.prod(bhi[h] - blo[h]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# REQ-048: On(A,B) <=> (z^A_min ~= z^B_max) and (IoU_xy(B_A, B_B) > gamma)
def is_on(b_a: BBox, b_b: BBox, gamma: float, delta: float, axis: int = 2) -> bool:
    (alo, _), (_, bhi) = _as_bbox(b_a), _as_bbox(b_b)
    return bool(abs(alo[axis] - bhi[axis]) <= delta and iou_xy(b_a, b_b, axis) > gamma)


# REQ-049: Under(A,B) <=> On(B,A) (D-073)
def is_under(b_a: BBox, b_b: BBox, gamma: float, delta: float, axis: int = 2) -> bool:
    return is_on(b_b, b_a, gamma, delta, axis)


# REQ-049: horizontal gap between the xy rectangles (0 when they overlap)
def horizontal_gap(b_a: BBox, b_b: BBox, axis: int = 2) -> float:
    (alo, ahi), (blo, bhi) = _as_bbox(b_a), _as_bbox(b_b)
    h = _horizontal_axes(axis)
    d = np.maximum(0.0, np.maximum(blo[h] - ahi[h], alo[h] - bhi[h]))
    return float(np.linalg.norm(d))


# REQ-049: Beside(A,B) <=> vertical intervals overlap, IoU_xy <= gamma, horizontal gap < d_bes (D-073)
def is_beside(b_a: BBox, b_b: BBox, gamma: float, d_bes: float, axis: int = 2) -> bool:
    (alo, ahi), (blo, bhi) = _as_bbox(b_a), _as_bbox(b_b)
    vertical_overlap = alo[axis] <= bhi[axis] and blo[axis] <= ahi[axis]
    return bool(vertical_overlap and iou_xy(b_a, b_b, axis) <= gamma
                and horizontal_gap(b_a, b_b, axis) < d_bes)


# REQ-048, REQ-049: array form of is_on / is_under / is_beside for candidate pairs (D-103)
def pair_relations(lo: np.ndarray, hi: np.ndarray, ii: np.ndarray, jj: np.ndarray,
                   gamma_i: np.ndarray, gamma_j: np.ndarray, delta: float, d_bes: float,
                   axis: int = 2) -> dict[str, np.ndarray]:
    """lo, hi (N,3) bbox corners; ii, jj (P,) pair indices; gamma_i/gamma_j (P,) per-pair gamma.

    Returns (P,) bool arrays reproducing the per-pair scalar calls of the graph builder:
    on_ij = is_on(i,j,gamma_j), under_ij = is_under(i,j,gamma_i), on_ji = is_on(j,i,gamma_i),
    under_ji = is_under(j,i,gamma_j), beside_ij = is_beside(i,j,gamma_j).
    """
    h = _horizontal_axes(axis)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1, 3)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1, 3)
    if np.any(hi < lo):
        raise ValueError("bbox max must be >= min")
    ii, jj = np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64)
    gi, gj = np.asarray(gamma_i, dtype=np.float64), np.asarray(gamma_j, dtype=np.float64)
    li, hi_i, lj, hj = lo[ii], hi[ii], lo[jj], hi[jj]
    sides = np.clip(np.minimum(hi_i[:, h], hj[:, h]) - np.maximum(li[:, h], lj[:, h]), 0.0, None)
    inter = np.prod(sides, axis=1)
    union = np.prod(hi_i[:, h] - li[:, h], axis=1) + np.prod(hj[:, h] - lj[:, h], axis=1) - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    gap = np.linalg.norm(np.maximum(0.0, np.maximum(lj[:, h] - hi_i[:, h], li[:, h] - hj[:, h])), axis=1)
    z_ij = np.abs(li[:, axis] - hj[:, axis]) <= delta   # z^i_min ~= z^j_max
    z_ji = np.abs(lj[:, axis] - hi_i[:, axis]) <= delta  # z^j_min ~= z^i_max
    vert = (li[:, axis] <= hj[:, axis]) & (lj[:, axis] <= hi_i[:, axis])
    return {
        "on_ij": z_ij & (iou > gj),
        "under_ij": z_ji & (iou > gi),
        "on_ji": z_ji & (iou > gi),
        "under_ji": z_ij & (iou > gj),
        "beside_ij": vert & (iou <= gj) & (gap < d_bes),
    }
