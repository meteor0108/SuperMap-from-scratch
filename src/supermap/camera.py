"""Pinhole camera model and perspective projection pi (Eq. 5, Eq. 7).

Symbol map (paper -> code):
    K          -> K (3, 3) [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    pi(.)      -> pi(): [a, b, c] -> [a/c, b/c]
    P_t        -> P_t (4, 4) T_WC camera-to-world
    P_t^{-1}   -> T_CW, world-to-camera
    X          -> X_w (N, 3) world points, meters

Conventions (D-010): OpenCV camera axes (x right, y down, z forward), no lens
distortion, continuous pixel coordinates with integer values at pixel centers.
Points with camera z <= 0 are not projectable (pixel = NaN, valid = False).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# REQ-004: pinhole intrinsics K
@dataclass(frozen=True)
class PinholeCamera:
    """Pinhole intrinsics with image size (pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return make_K(self.fx, self.fy, self.cx, self.cy)

    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int) -> "PinholeCamera":
        K = np.asarray(K, dtype=np.float64)
        return cls(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
                   int(width), int(height))

    # REQ-004: pixel-in-image test for projected coordinates
    def in_image(self, uv: np.ndarray) -> np.ndarray:
        """(N,) bool: rounded pixel index lies inside [0, W) x [0, H)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        ok = np.all(np.isfinite(uv), axis=1)
        idx = np.full(uv.shape, -1, dtype=np.int64)
        idx[ok] = pixel_index(uv[ok])
        return ok & (idx[:, 0] >= 0) & (idx[:, 0] < self.width) & (idx[:, 1] >= 0) & (idx[:, 1] < self.height)


# REQ-004: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
def make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# REQ-004: perspective division pi([a, b, c]) = [a/c, b/c]
def pi(x: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 2). Rows with c <= 0 map to NaN."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.full((x.shape[0], 2), np.nan, dtype=np.float64)
    ok = x[:, 2] > 0
    out[ok] = x[ok, :2] / x[ok, 2:3]
    return out


# REQ-004: [P_t^{-1} X]_{1:3}, world points into the camera frame
def world_to_camera(X_w: np.ndarray, P_t: np.ndarray) -> np.ndarray:
    """(N, 3) world -> (N, 3) camera coordinates, using the rigid inverse of T_WC."""
    X_w = np.atleast_2d(np.asarray(X_w, dtype=np.float64))
    P_t = np.asarray(P_t, dtype=np.float64)
    R, p = P_t[:3, :3], P_t[:3, 3]
    return (X_w - p) @ R  # R^T (X - p), row-vector form


# REQ-004: pixel = pi(K [P_t^{-1} X]_{1:3}) (Eq. 5, Eq. 7)
def project(X_w: np.ndarray, P_t: np.ndarray, K: np.ndarray):
    """Project world points.

    Returns:
        uv: (N, 2) float64 pixels (NaN where not projectable)
        z: (N,) float64 camera-frame depth
        valid: (N,) bool, z > 0
    """
    X_c = world_to_camera(X_w, P_t)
    x = X_c @ np.asarray(K, dtype=np.float64).T
    z = X_c[:, 2].copy()
    return pi(x), z, z > 0


# REQ-004: pixel index from continuous coordinates, round to nearest (D-010)
def pixel_index(uv: np.ndarray) -> np.ndarray:
    """(N, 2) float pixels -> (N, 2) int64 (u, v) indices."""
    return np.floor(np.asarray(uv, dtype=np.float64) + 0.5).astype(np.int64)
