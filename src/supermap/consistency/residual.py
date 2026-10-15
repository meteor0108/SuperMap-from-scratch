"""Depth residual (Eq. 7) and Gaussian measurement likelihood (Sec. IV-B3).

Symbol map (paper -> code):
    X_k          -> X_k (n, 3) world points of M_{t-1}, meters
    T_CW         -> inverse of P_t (P_t = T_WC, 4x4)
    d_proj       -> d_proj = [T_CW X_k]_z, camera z (not Euclidean norm, D-045)
    u = pi(X_k)  -> u = pi(K [T_CW X_k]_{1:3}), rounded to nearest pixel (D-010)
    D(u)         -> depth[v, u] raw sensor depth, meters
    Delta d      -> delta_d = d_proj - D(u)
    sigma        -> sigma, p(Delta d) ~ N(0, sigma^2)
"""
from __future__ import annotations

import math

import numpy as np

from supermap.camera import pixel_index, project


# REQ-029: Eq. (7) Delta d = d_proj - D(u) with u = pi(X_k)
def depth_residual(X_k: np.ndarray, P_t: np.ndarray, K: np.ndarray, depth: np.ndarray):
    """Per-point depth residual of world points against depth image D_t.

    Args:
        X_k: (n, 3) world points. P_t: (4, 4) camera-to-world. K: (3, 3). depth: (H, W) meters.
    Returns:
        delta_d: (n,) float64, NaN where mask is False
        mask: (n,) bool; False if behind the camera, outside the image, or D(u) invalid (<= 0 / non-finite)
        pix: (n, 2) int64 (u, v) pixel index, -1 where not projectable
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    X_k = np.asarray(X_k, dtype=np.float64).reshape(-1, 3)
    n = X_k.shape[0]
    delta_d = np.full(n, np.nan, dtype=np.float64)
    pix = np.full((n, 2), -1, dtype=np.int64)
    if n == 0:
        return delta_d, np.zeros(0, dtype=bool), pix
    uv, d_proj, in_front = project(X_k, P_t, K)
    ok = in_front & np.all(np.isfinite(uv), axis=1)
    pix[ok] = pixel_index(uv[ok])
    H, W = depth.shape
    ok &= (pix[:, 0] >= 0) & (pix[:, 0] < W) & (pix[:, 1] >= 0) & (pix[:, 1] < H)
    D_u = np.full(n, np.nan, dtype=np.float64)
    D_u[ok] = depth[pix[ok, 1], pix[ok, 0]]
    ok &= np.isfinite(D_u) & (D_u > 0)
    delta_d[ok] = d_proj[ok] - D_u[ok]
    pix[~ok] = -1
    return delta_d, ok, pix


# REQ-030: Gaussian likelihood p(Delta d) = N(Delta d; 0, sigma^2)
def gaussian_likelihood(delta_d, sigma: float):
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    x = np.asarray(delta_d, dtype=np.float64)
    out = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    return float(out) if out.ndim == 0 else out
