"""ScanNet RGB-D loader for SensReader exports (Sec. V-A, V-B benchmark data).

Layout (official ScanNet SensReader export):
    <root>/scans/<scene>/color/<i>.jpg
    <root>/scans/<scene>/depth/<i>.png          uint16 millimeters
    <root>/scans/<scene>/pose/<i>.txt           4x4 camera-to-world
    <root>/scans/<scene>/intrinsic/intrinsic_depth.txt  4x4, K = [:3, :3]
Resolution alignment (D-013): color is resized to the depth resolution and
K is taken from intrinsic_depth.txt. Frames with non-finite poses are skipped.
Exports carry no timestamps: stamp = frame_id / nominal_fps.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from supermap.datasets.base import RGBDSequence, select_frame_ids
from supermap.datasets.replica import depth_from_raw
from supermap.pose import DatasetPoseProvider
from supermap.types import is_se3


# REQ-007: ScanNet scene -> Observation stream (REQ-001/REQ-002)
class ScanNetSequence(RGBDSequence):

    def __init__(self, root: str | Path, scene: str, depth_scale: float, nominal_fps: float,
                 start: int = 0, stride: int = 1, max_frames: int | None = None,
                 T_BC: np.ndarray | None = None) -> None:
        self.scene_dir = Path(root) / "scans" / scene
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"ScanNet scene not found: {self.scene_dir}")
        if nominal_fps <= 0 or depth_scale <= 0:
            raise ValueError("nominal_fps and depth_scale must be > 0")
        self.depth_scale = float(depth_scale)
        self.nominal_fps = float(nominal_fps)
        intr = np.loadtxt(self.scene_dir / "intrinsic" / "intrinsic_depth.txt", dtype=np.float64)
        self._K = intr[:3, :3].copy()
        poses = {}
        for p in (self.scene_dir / "pose").glob("*.txt"):
            T = np.loadtxt(p, dtype=np.float64).reshape(4, 4)
            if is_se3(T):  # D-013: skip invalid (-inf / NaN) poses
                poses[int(p.stem)] = T
        super().__init__(DatasetPoseProvider(poses), T_BC)
        have = {int(p.stem) for p in (self.scene_dir / "color").glob("*.jpg")}
        have &= {int(p.stem) for p in (self.scene_dir / "depth").glob("*.png")}
        available = [i for i in self.pose_provider.frame_ids() if i in have]
        self._frame_ids = select_frame_ids(available, start, stride, max_frames)

    @property
    def frame_ids(self) -> list[int]:
        return list(self._frame_ids)

    @property
    def K(self) -> np.ndarray:
        return self._K.copy()

    # REQ-007: depth mm -> m; color resized to depth size (D-013)
    def load_frame(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        bgr = cv2.imread(str(self.scene_dir / "color" / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(self.scene_dir / "depth" / f"{frame_id}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            raise FileNotFoundError(f"missing ScanNet frame {frame_id} in {self.scene_dir}")
        depth = depth_from_raw(raw, self.depth_scale)
        h, w = depth.shape
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth

    def stamp(self, frame_id: int) -> float:
        return frame_id / self.nominal_fps
