"""Depth-edge pixel exclusion before back-projection (REQ-023, D-106 revision candidate)."""
from __future__ import annotations

import cv2
import numpy as np


# REQ-023: drop mask pixels on depth discontinuities (or next to invalid depth); thresholds 0 = off
def depth_edge_mask(depth: np.ndarray, mask: np.ndarray, jump_abs: float, jump_rel: float,
                    radius: int = 1) -> np.ndarray:
    if jump_abs <= 0 and jump_rel <= 0:
        return mask
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    k = np.ones((2 * int(radius) + 1, 2 * int(radius) + 1), np.uint8)
    lo = cv2.erode(np.where(valid, d, np.inf).astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)
    hi = cv2.dilate(np.where(valid, d, -np.inf).astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)
    near_invalid = cv2.dilate((~valid).astype(np.uint8), k, borderType=cv2.BORDER_REPLICATE) > 0
    with np.errstate(invalid="ignore"):
        jump = hi - lo
        thr = np.maximum(float(jump_abs), float(jump_rel) * np.where(valid, d, 0.0))
        edge = near_invalid | (jump > thr)
    return np.asarray(mask, dtype=bool) & ~edge
