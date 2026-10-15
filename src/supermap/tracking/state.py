"""Hybrid tracking state and 3D-to-2D motion-compensated Kalman prediction (Sec. IV-B1/B2, Eq. 4-6).

Symbol map (paper -> code):
    S_i(t)            -> S (6,) = [x, y, w, h, x_dot, y_dot]; 8-dim adds [w_dot, h_dot] (Eq. 4)
    c_i(t) = [x, y]   -> S[0:2], 2D image centroid, pixels
    s_i(t) = [w, h]   -> S[2:4], box width/height, pixels
    c_dot_i(t)        -> S[4:6], image-plane velocity, pixels per dt
    c_hat_i(t)        -> project_centroid(X_i, P_t, K) = pi(K P_t^{-1} X_i) (Eq. 5)
    F, Q              -> make_F(dt, dim), make_Q(...) (Eq. 6, REQ-019)
    w_t ~ N(0, Q)     -> covariance propagation Sigma_hat = F Sigma F^T + Q
    Sigma             -> Sigma (dim, dim) state covariance
    R, H              -> measurement noise (4, 4) and H = [I_4, 0] for z = [x, y, w, h] (D-031)
Boxes are xyxy in pixels. Poses P_t are T_WC (4, 4); K is (3, 3).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from supermap.camera import project

STATE_DIMS = (6, 8)


# REQ-016: xyxy box -> S_i = [c, s, c_dot] (Eq. 4), velocities initialized to 0
def box_to_state(box: Sequence[float], dim: int = 6) -> np.ndarray:
    if dim not in STATE_DIMS:
        raise ValueError(f"state dim must be 6 or 8, got {dim}")
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.shape[0] != 4:
        raise ValueError(f"box must have 4 values (xyxy), got {b.shape}")
    S = np.zeros(dim, dtype=np.float64)
    S[0:2] = (b[0:2] + b[2:4]) / 2.0
    S[2:4] = b[2:4] - b[0:2]
    return S


# REQ-016: S_i -> xyxy box from centroid and size
def state_to_box(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    if S.shape[0] not in STATE_DIMS:
        raise ValueError(f"state must have length 6 or 8, got {S.shape[0]}")
    half = S[2:4] / 2.0
    return np.concatenate([S[0:2] - half, S[0:2] + half])


# REQ-019: constant-velocity F (ByteTrack [33] motion model): c += dt c_dot (and s += dt s_dot if 8-dim)
def make_F(dt: float = 1.0, dim: int = 6) -> np.ndarray:
    if dim not in STATE_DIMS:
        raise ValueError(f"state dim must be 6 or 8, got {dim}")
    F = np.eye(dim, dtype=np.float64)
    F[0, 4] = F[1, 5] = dt
    if dim == 8:
        F[2, 6] = F[3, 7] = dt
    return F


# REQ-019: diagonal process noise Q; variances for [c, s, c_dot(, s_dot)] (D-031)
def make_Q(var_centroid: float, var_size: float, var_velocity: float,
           dim: int = 6, var_size_velocity: float | None = None) -> np.ndarray:
    if dim not in STATE_DIMS:
        raise ValueError(f"state dim must be 6 or 8, got {dim}")
    d = [var_centroid] * 2 + [var_size] * 2 + [var_velocity] * 2
    if dim == 8:
        d += [var_velocity if var_size_velocity is None else var_size_velocity] * 2
    q = np.asarray(d, dtype=np.float64)
    if np.any(q < 0):
        raise ValueError("process noise variances must be non-negative")
    return np.diag(q)


# REQ-017: c_hat_i(t) = pi(K P_t^{-1} X_i) (Eq. 5); None if X_i is behind the camera
def project_centroid(X_i: np.ndarray, P_t: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    uv, _, valid = project(np.asarray(X_i, dtype=np.float64).reshape(1, 3), P_t, K)
    if not bool(valid[0]):
        return None
    return uv[0].copy()


# REQ-026: image-plane size [w, h] of a projected world axis-aligned 3D bbox (8 corners)
def project_bbox_size(bbox: tuple[np.ndarray, np.ndarray], P_t: np.ndarray,
                      K: np.ndarray) -> np.ndarray | None:
    lo, hi = (np.asarray(v, dtype=np.float64).reshape(3) for v in bbox)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    uv, _, valid = project(corners, P_t, K)
    if not np.all(valid):
        return None
    return uv.max(axis=0) - uv.min(axis=0)


# REQ-018: Eq. (6) prediction S_hat = F S + w_t, Sigma_hat = F Sigma F^T + Q,
# with the centroid replaced by the Eq. (5) projection prior when given (D-030)
def kf_predict(S: np.ndarray, Sigma: np.ndarray, F: np.ndarray, Q: np.ndarray,
               c_hat: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    S_hat = F @ S
    Sigma_hat = F @ Sigma @ F.T + Q
    if c_hat is not None:
        S_hat = compensate_centroid(S_hat, c_hat)
    return S_hat, Sigma_hat


# REQ-018, REQ-022: 3D-to-2D motion compensation: predicted centroid := c_hat, size/velocity kept (D-030)
def compensate_centroid(S_hat: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
    out = np.asarray(S_hat, dtype=np.float64).reshape(-1).copy()
    out[0:2] = np.asarray(c_hat, dtype=np.float64).reshape(2)
    return out


# REQ-018: standard KF measurement update with z = [x, y, w, h], H = [I_4, 0] (D-031)
def kf_update(S_hat: np.ndarray, Sigma_hat: np.ndarray, z: np.ndarray,
              R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dim = S_hat.shape[0]
    H = np.zeros((4, dim), dtype=np.float64)
    H[:, :4] = np.eye(4)
    y = np.asarray(z, dtype=np.float64).reshape(4) - H @ S_hat
    Sxx = H @ Sigma_hat @ H.T + R
    G = Sigma_hat @ H.T @ np.linalg.inv(Sxx)
    S = S_hat + G @ y
    Sigma = (np.eye(dim) - G @ H) @ Sigma_hat
    return S, 0.5 * (Sigma + Sigma.T)
