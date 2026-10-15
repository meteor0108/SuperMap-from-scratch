"""Observation datatype Q_t (Sec. III, Sec. IV-A).

Symbol map (paper -> code):
    C_t   -> Observation.rgb    (H, W, 3) uint8, RGB order
    D_t   -> Observation.depth  (H, W) float32, meters; <= 0 or non-finite = invalid
    K     -> Observation.K      (3, 3) float64 pinhole intrinsics
    P_t   -> Observation.pose   (4, 4) float64, T_WC camera-to-world (Eq. 3)
    t     -> Observation.t      int, 1-based time index t in {1..T} (Sec. III)
    stamp -> Observation.stamp  float, seconds
    frame_id -> dataset file index (not a paper symbol, D-008)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SE3_ATOL = 1e-6


# REQ-002: SE(3) check used to validate P_t (last row [0,0,0,1], R^T R = I, det R = 1)
def is_se3(T: np.ndarray, atol: float = SE3_ATOL) -> bool:
    """Return True if T is a (4, 4) finite rigid transform."""
    T = np.asarray(T)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        return False
    return bool(abs(np.linalg.det(R) - 1.0) <= atol)


# REQ-002: observation Q_t = {C_t, D_t} bundled with P_t, K, t, stamp (Sec. III, Sec. IV-A)
@dataclass(frozen=True)
class Observation:
    """One synchronized RGB-D observation Q_t.

    Every downstream module consumes this type only. Arrays are validated and
    normalized to the dtypes documented in the module header at construction.
    """

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    pose: np.ndarray
    t: int
    stamp: float
    frame_id: int = -1

    # REQ-002: shape/dtype validation, raises ValueError on malformed input
    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth)
        K = np.asarray(self.K, dtype=np.float64)
        pose = np.asarray(self.pose, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
        if depth.ndim != 2:
            raise ValueError(f"depth must be (H, W), got {depth.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth {depth.shape} and rgb {rgb.shape[:2]} sizes differ")
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"depth must be float meters, got {depth.dtype}")
        if K.shape != (3, 3):
            raise ValueError(f"K must be (3, 3), got {K.shape}")
        if not is_se3(pose):
            raise ValueError("pose must be a valid SE(3) (4, 4) matrix")
        if int(self.t) != self.t:
            raise ValueError(f"t must be an integer, got {self.t}")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth", depth.astype(np.float32, copy=False))
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "t", int(self.t))
        object.__setattr__(self, "stamp", float(self.stamp))
        object.__setattr__(self, "frame_id", int(self.frame_id))

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    # REQ-006: depth <= 0 or non-finite is invalid (D-012)
    def valid_depth_mask(self) -> np.ndarray:
        """(H, W) bool mask of pixels with a usable depth reading."""
        d = self.depth
        return np.isfinite(d) & (d > 0)
