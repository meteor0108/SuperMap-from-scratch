"""Depth back-projection to world points, p.4 Sec. IV-A (i); Supp. Alg. 1 `Unproject`.

Symbol map (paper -> code):
    u = (u, v)  -> uv (n, 2) pixel coordinates, u = column, v = row (D-010)
    d = D_t(u)  -> d (n,) z-depth in meters along camera z (D-015)
    K           -> K (3, 3)
    P_t         -> P_t (4, 4) camera-to-world (Eq. 3)
    X_c         -> X_c = d K^{-1} [u, v, 1]^T, camera frame (OpenCV axes)
    X_w         -> X_w = [P_t [X_c; 1]]_{1:3}, world frame, meters
"""
from __future__ import annotations

import numpy as np

from supermap.types import Observation


# REQ-010: X_w = [P_t [d K^{-1} [u, v, 1]^T ; 1]]_{1:3}
def unproject_pixels(uv: np.ndarray, d: np.ndarray, K: np.ndarray, P_t: np.ndarray) -> np.ndarray:
    """(n, 2) pixels + (n,) depths -> (n, 3) float64 world points. No validity filtering."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if uv.shape[1] != 2 or uv.shape[0] != d.shape[0]:
        raise ValueError(f"uv must be (n, 2) matching d (n,), got {uv.shape} and {d.shape}")
    K = np.asarray(K, dtype=np.float64)
    P_t = np.asarray(P_t, dtype=np.float64)
    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    X_c = (uv1 @ np.linalg.inv(K).T) * d[:, None]
    return X_c @ P_t[:3, :3].T + P_t[:3, 3]


# REQ-010: back-project all valid depth pixels (optionally inside a mask) with their colors
def backproject(depth: np.ndarray, K: np.ndarray, P_t: np.ndarray, rgb: np.ndarray | None = None,
                mask: np.ndarray | None = None, depth_min: float = 0.0,
                depth_max: float | None = None):
    """Back-project a depth image.

    Args:
        depth: (H, W) meters. Pixels that are non-finite, <= depth_min, or > depth_max are dropped.
        K: (3, 3) intrinsics. P_t: (4, 4) camera-to-world.
        rgb: optional (H, W, 3) uint8 colors.
        mask: optional (H, W) bool; only pixels inside the mask are used.
    Returns:
        X_w: (n, 3) float64 world points
        colors: (n, 3) uint8 (zeros if rgb is None)
        pix: (n, 2) int64 (u, v) pixel indices of each point, row-major order
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    valid = np.isfinite(depth) & (depth > depth_min)
    if depth_max is not None:
        valid &= depth <= depth_max
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"mask {mask.shape} and depth {depth.shape} sizes differ")
        valid &= mask
    v, u = np.nonzero(valid)
    pix = np.stack([u, v], axis=1).astype(np.int64)
    X_w = unproject_pixels(pix, depth[v, u], K, P_t)
    if rgb is None:
        colors = np.zeros((pix.shape[0], 3), dtype=np.uint8)
    else:
        rgb = np.asarray(rgb)
        if rgb.shape[:2] != depth.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb {rgb.shape} must be (H, W, 3) matching depth {depth.shape}")
        colors = rgb[v, u].astype(np.uint8, copy=False)
    return X_w, colors, pix


# REQ-010: back-project an Observation Q_t with P_t = obs.pose
def backproject_observation(obs: Observation, mask: np.ndarray | None = None,
                            depth_min: float = 0.0, depth_max: float | None = None):
    return backproject(obs.depth, obs.K, obs.pose, rgb=obs.rgb, mask=mask,
                       depth_min=depth_min, depth_max=depth_max)
