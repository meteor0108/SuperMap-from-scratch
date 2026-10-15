"""Camera pose from body pose and fixed extrinsic, Eq. (3), p.4 Sec. IV-A.

Symbol map (paper -> code):
    T_WB^(t)       -> T_WB (4, 4) float64, body-to-world at time t (PoseProvider)
    T_BC           -> T_BC (4, 4) float64, fixed camera-to-body extrinsic (configs extrinsics.T_BC)
    P_t = T_WC^(t) -> P_t  (4, 4) float64, camera-to-world
"""
from __future__ import annotations

import numpy as np

from supermap.pose import PoseProvider
from supermap.types import is_se3


# REQ-009: Eq. (3) P_t = T_WC^(t) = T_WB^(t) . T_BC
def compose_camera_pose(T_WB: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
    """Return P_t = T_WB @ T_BC. Both inputs must be SE(3); raises ValueError otherwise."""
    T_WB = np.asarray(T_WB, dtype=np.float64)
    T_BC = np.asarray(T_BC, dtype=np.float64)
    if not is_se3(T_WB):
        raise ValueError("T_WB must be a valid SE(3) (4, 4) matrix")
    if not is_se3(T_BC):
        raise ValueError("T_BC must be a valid SE(3) (4, 4) matrix")
    return T_WB @ T_BC


# REQ-009: P_t for a dataset frame, T_WB^(t) from the PoseProvider (REQ-003)
def camera_pose_at(provider: PoseProvider, frame_id: int, T_BC: np.ndarray) -> np.ndarray:
    return compose_camera_pose(provider.get(frame_id), T_BC)
