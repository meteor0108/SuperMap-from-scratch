"""Replica RGB-D loader, NICE-SLAM layout (development substitute, D-002).

Layout:
    <root>/cam_params.json (or <root>/<scene>/cam_params.json)
    <root>/<scene>/results/frameXXXXXX.jpg   RGB
    <root>/<scene>/results/depthXXXXXX.png   uint16, meters = raw / scale
    <root>/<scene>/traj.txt                  one row-major 4x4 T_WC per line
Replica has no timestamps: stamp = frame_id / nominal_fps (D-006).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supermap.camera import make_K
from supermap.datasets.base import RGBDSequence, select_frame_ids
from supermap.pose import DatasetPoseProvider


# REQ-006: Replica scene -> Observation stream (REQ-001/REQ-002)
class ReplicaSequence(RGBDSequence):

    def __init__(self, root: str | Path, scene: str, nominal_fps: float, start: int = 0,
                 stride: int = 1, max_frames: int | None = None, T_BC: np.ndarray | None = None) -> None:
        self.scene_dir = Path(root) / scene
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"Replica scene not found: {self.scene_dir}")
        cam_path = self.scene_dir / "cam_params.json"
        if not cam_path.exists():
            cam_path = Path(root) / "cam_params.json"
        cam = json.loads(cam_path.read_text())["camera"]
        self.width, self.height = int(cam["w"]), int(cam["h"])
        self.depth_scale = float(cam["scale"])
        self._K = make_K(cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        if nominal_fps <= 0:
            raise ValueError("nominal_fps must be > 0")
        self.nominal_fps = float(nominal_fps)
        super().__init__(DatasetPoseProvider.from_trajectory_file(self.scene_dir / "traj.txt"), T_BC)
        n_rgb = len(list((self.scene_dir / "results").glob("frame*.jpg")))
        available = [i for i in self.pose_provider.frame_ids() if i < n_rgb]
        self._frame_ids = select_frame_ids(available, start, stride, max_frames)

    @property
    def frame_ids(self) -> list[int]:
        return list(self._frame_ids)

    @property
    def K(self) -> np.ndarray:
        return self._K.copy()

    # REQ-006: read RGB (BGR->RGB) and depth raw / scale; zeros stay 0 (invalid, D-012)
    def load_frame(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        res = self.scene_dir / "results"
        bgr = cv2.imread(str(res / f"frame{frame_id:06d}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(res / f"depth{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            raise FileNotFoundError(f"missing Replica frame {frame_id} in {res}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth_from_raw(raw, self.depth_scale)

    # REQ-006: nominal time axis (D-006)
    def stamp(self, frame_id: int) -> float:
        return frame_id / self.nominal_fps


# REQ-006, REQ-007: uint16 depth image -> float32 meters
def depth_from_raw(raw: np.ndarray, scale: float) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"depth image must be single channel, got {raw.shape}")
    return (raw.astype(np.float64) / float(scale)).astype(np.float32)
