"""RGB-D video stream abstraction (Sec. III).

A sequence yields Observation Q_t for t = 1..T in order, loading each frame
lazily so processing frame t never touches frames after t.
P_t is formed as T_WB^(t) @ T_BC with T_BC = I_4 by default (D-009); the
Eq. (3) composition lives in supermap.geometry.extrinsics (REQ-009).
"""
from __future__ import annotations

import abc
from typing import Iterator, Sequence

import numpy as np

from supermap.geometry.extrinsics import camera_pose_at
from supermap.pose import DatasetPoseProvider, PoseProvider
from supermap.types import Observation, is_se3


# REQ-001: incremental RGB-D video sequence C_{1:T}, D_{1:T}
class RGBDSequence(abc.ABC):
    """Ordered, lazily loaded stream of Observations."""

    # REQ-001, REQ-003: pose source and fixed body-to-camera extrinsic
    def __init__(self, pose_provider: PoseProvider, T_BC: np.ndarray | None = None) -> None:
        self.pose_provider = pose_provider
        T_BC = np.eye(4) if T_BC is None else np.asarray(T_BC, dtype=np.float64)
        if not is_se3(T_BC):
            raise ValueError("T_BC must be a valid SE(3) matrix")
        self.T_BC = T_BC

    @property
    @abc.abstractmethod
    def frame_ids(self) -> list[int]:
        """Dataset frame indices in stream order."""
        raise RuntimeError("abstract")

    @property
    @abc.abstractmethod
    def K(self) -> np.ndarray:
        """(3, 3) float64 intrinsics of the depth/RGB pair as delivered."""
        raise RuntimeError("abstract")

    @abc.abstractmethod
    def load_frame(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb (H, W, 3) uint8, depth (H, W) float32 meters)."""
        raise RuntimeError("abstract")

    @abc.abstractmethod
    def stamp(self, frame_id: int) -> float:
        """Timestamp in seconds for a dataset frame index."""
        raise RuntimeError("abstract")

    def __len__(self) -> int:
        return len(self.frame_ids)

    # REQ-001, REQ-002: build Q_t for the i-th stream item (t = i + 1)
    def observation(self, i: int) -> Observation:
        fid = self.frame_ids[i]
        rgb, depth = self.load_frame(fid)
        P_t = camera_pose_at(self.pose_provider, fid, self.T_BC)  # REQ-009: Eq. (3)
        return Observation(rgb=rgb, depth=depth, K=self.K, pose=P_t,
                           t=i + 1, stamp=self.stamp(fid), frame_id=fid)

    # REQ-001: yield Q_1..Q_T in order, one frame at a time
    def __iter__(self) -> Iterator[Observation]:
        for i in range(len(self)):
            yield self.observation(i)


# REQ-001: in-memory sequence for synthetic streams
class ArraySequence(RGBDSequence):
    """Sequence over pre-loaded arrays; frame ids are 0..T-1."""

    def __init__(self, rgbs: Sequence[np.ndarray], depths: Sequence[np.ndarray], K: np.ndarray,
                 poses: np.ndarray, stamps: Sequence[float], T_BC: np.ndarray | None = None) -> None:
        n = len(rgbs)
        if not (len(depths) == n == len(poses) == len(stamps)):
            raise ValueError("rgbs, depths, poses and stamps must have equal length")
        super().__init__(DatasetPoseProvider(np.asarray(poses, dtype=np.float64)), T_BC)
        self._rgbs = list(rgbs)
        self._depths = list(depths)
        self._K = np.asarray(K, dtype=np.float64)
        self._stamps = [float(s) for s in stamps]

    @property
    def frame_ids(self) -> list[int]:
        return list(range(len(self._rgbs)))

    @property
    def K(self) -> np.ndarray:
        return self._K.copy()

    def load_frame(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self._rgbs[frame_id]),
                np.asarray(self._depths[frame_id]).astype(np.float32, copy=False))

    def stamp(self, frame_id: int) -> float:
        return self._stamps[frame_id]


# REQ-006, REQ-007: frame index selection start, start+stride, ... (at most max_frames)
def select_frame_ids(available: Sequence[int], start: int, stride: int,
                     max_frames: int | None) -> list[int]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    ids = sorted(int(a) for a in available)
    ids = [i for i in ids if i >= start][::stride]
    return ids if max_frames is None else ids[:max_frames]
