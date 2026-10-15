"""Geometric layer (Sec. IV-A): camera pose (Eq. 3), back-projection, dense map."""
# REQ-009, REQ-010, REQ-011: public API of the geometric layer
from supermap.geometry.backproject import backproject, backproject_observation, unproject_pixels
from supermap.geometry.dense_map import DenseMap, voxel_keys
from supermap.geometry.extrinsics import camera_pose_at, compose_camera_pose

__all__ = [
    "backproject", "backproject_observation", "unproject_pixels",
    "DenseMap", "voxel_keys",
    "camera_pose_at", "compose_camera_pose",
]
