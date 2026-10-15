"""Pose source for the geometric layer (Sec. IV-A).

The paper obtains T_WB^(t) from SuperOdometry [35]; per D-001 this project
uses a swappable PoseProvider interface whose default implementation returns
benchmark-dataset poses. Dataset poses are already camera poses, so
T_WB := dataset pose and T_BC := I_4 by default (D-009).

Symbol map: T_WB^(t) -> PoseProvider.get(frame_id), (4, 4) float64 body-to-world.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from supermap.types import is_se3


# REQ-003: swappable pose source interface returning T_WB^(t) in SE(3)
class PoseProvider(abc.ABC):
    """Abstract pose source. Implementations must return (4, 4) float64 T_WB."""

    @abc.abstractmethod
    def get(self, frame_id: int) -> np.ndarray:
        """Return T_WB for the given dataset frame index; raise KeyError if unknown."""
        raise RuntimeError("abstract")

    @abc.abstractmethod
    def frame_ids(self) -> list[int]:
        """Sorted frame indices this provider can serve."""
        raise RuntimeError("abstract")

    def __contains__(self, frame_id: int) -> bool:
        return int(frame_id) in set(self.frame_ids())


# REQ-003: default provider backed by dataset-provided poses (D-001, D-009)
class DatasetPoseProvider(PoseProvider):
    """Poses indexed by dataset frame index."""

    # REQ-003: accepts a (N, 4, 4) array (ids 0..N-1) or a {frame_id: 4x4} mapping
    def __init__(self, poses: np.ndarray | Mapping[int, np.ndarray],
                 frame_ids: Sequence[int] | None = None) -> None:
        if isinstance(poses, Mapping):
            items = {int(k): np.asarray(v, dtype=np.float64) for k, v in poses.items()}
        else:
            arr = np.asarray(poses, dtype=np.float64)
            if arr.ndim != 3 or arr.shape[1:] != (4, 4):
                raise ValueError(f"poses must be (N, 4, 4), got {arr.shape}")
            ids = list(range(arr.shape[0])) if frame_ids is None else [int(i) for i in frame_ids]
            if len(ids) != arr.shape[0]:
                raise ValueError("frame_ids length does not match poses")
            items = dict(zip(ids, arr))
        for k, T in items.items():
            if not is_se3(T):
                raise ValueError(f"pose for frame {k} is not a valid SE(3) matrix")
        self._poses = items

    def get(self, frame_id: int) -> np.ndarray:
        key = int(frame_id)
        if key not in self._poses:
            raise KeyError(f"no pose for frame {frame_id}")
        return self._poses[key].copy()

    def frame_ids(self) -> list[int]:
        return sorted(self._poses)

    def __len__(self) -> int:
        return len(self._poses)

    # REQ-003, REQ-006: trajectory file with one row-major 4x4 per line (16 numbers)
    @classmethod
    def from_trajectory_file(cls, path: str | Path) -> "DatasetPoseProvider":
        rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
        for i, r in enumerate(rows):
            if len(r) != 16:
                raise ValueError(f"{path}:{i + 1}: expected 16 numbers, got {len(r)}")
        arr = np.array(rows, dtype=np.float64).reshape(-1, 4, 4)
        return cls(arr)
