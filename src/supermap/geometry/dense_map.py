"""Colorized dense 3D point map, p.4 Sec. IV-A; Fig. 2 "Dense 3D Model".

Each mapping step back-projects the full depth frame (REQ-010) and appends the
colored world points. Optional voxel downsampling (D-016): at most one point per
voxel of edge voxel_size, the first inserted point is kept. voxel_size None
disables downsampling. The dense map carries no log-odds (D-017).

Data: points (N, 3) float64 world meters, colors (N, 3) uint8 RGB.
"""
from __future__ import annotations

import numpy as np

from supermap.geometry.backproject import backproject_observation
from supermap.types import Observation

_KEY_BITS = 21
_KEY_OFFSET = 1 << (_KEY_BITS - 1)


# REQ-011: integer voxel coordinates floor(X / v) packed into one int64 key (D-016)
def voxel_keys(X_w: np.ndarray, voxel_size: float) -> np.ndarray:
    ijk = np.floor(np.asarray(X_w, dtype=np.float64) / voxel_size).astype(np.int64) + _KEY_OFFSET
    if ijk.size and (ijk.min() < 0 or ijk.max() >= (1 << _KEY_BITS)):
        raise ValueError("point outside the representable voxel range")
    return (ijk[:, 0] << (2 * _KEY_BITS)) | (ijk[:, 1] << _KEY_BITS) | ijk[:, 2]


# REQ-011: world-frame colorized dense point map accumulated over mapping steps
class DenseMap:
    """Accumulated colored point cloud in the world frame."""

    # REQ-011: voxel_size None = no downsampling; depth range applied at back-projection
    def __init__(self, voxel_size: float | None = None, depth_min: float = 0.0,
                 depth_max: float | None = None) -> None:
        if voxel_size is not None and not voxel_size > 0:
            raise ValueError("voxel_size must be > 0 or None")
        self.voxel_size = None if voxel_size is None else float(voxel_size)
        self.depth_min = float(depth_min)
        self.depth_max = None if depth_max is None else float(depth_max)
        self._points = np.zeros((0, 3), dtype=np.float64)
        self._colors = np.zeros((0, 3), dtype=np.uint8)
        self._keys = np.zeros((0,), dtype=np.int64)  # sorted voxel keys of stored points
        self.num_integrated = 0

    # REQ-011: build from the `geometry` config section
    @classmethod
    def from_config(cls, geometry_cfg: dict) -> "DenseMap":
        dm = geometry_cfg.get("dense_map", {})
        return cls(voxel_size=dm.get("voxel_size"), depth_min=geometry_cfg.get("depth_min", 0.0),
                   depth_max=geometry_cfg.get("depth_max"))

    # REQ-011: add colored world points, deduplicated per voxel when enabled
    def add_points(self, X_w: np.ndarray, colors: np.ndarray) -> int:
        """Append points; returns the number of points actually added."""
        X_w = np.asarray(X_w, dtype=np.float64).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if X_w.shape[0] != colors.shape[0]:
            raise ValueError("points and colors lengths differ")
        if self.voxel_size is not None and X_w.shape[0]:
            keys = voxel_keys(X_w, self.voxel_size)
            keys, first = np.unique(keys, return_index=True)  # first occurrence per voxel
            pos = np.searchsorted(self._keys, keys)
            hit = pos < self._keys.shape[0]
            hit[hit] = self._keys[pos[hit]] == keys[hit]
            new = ~hit
            keys, sel = keys[new], np.sort(first[new])
            X_w, colors = X_w[sel], colors[sel]
            self._keys = np.sort(np.concatenate([self._keys, keys]))
        self._points = np.concatenate([self._points, X_w])
        self._colors = np.concatenate([self._colors, colors])
        return int(X_w.shape[0])

    # REQ-011: one mapping step, back-project the full depth frame of Q_t (REQ-010)
    def integrate(self, obs: Observation) -> int:
        X_w, colors, _ = backproject_observation(obs, depth_min=self.depth_min,
                                                 depth_max=self.depth_max)
        self.num_integrated += 1
        return self.add_points(X_w, colors)

    # REQ-011: (N, 3) float64 world points
    @property
    def points(self) -> np.ndarray:
        return self._points

    # REQ-011: (N, 3) uint8 RGB colors aligned with points
    @property
    def colors(self) -> np.ndarray:
        return self._colors

    # REQ-011: number of stored points N
    def __len__(self) -> int:
        return int(self._points.shape[0])

    # REQ-011: world-frame axis-aligned bounds of the map, (min_xyz, max_xyz)
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            raise ValueError("dense map is empty")
        return self._points.min(axis=0), self._points.max(axis=0)
