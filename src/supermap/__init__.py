"""SuperMap: spatio-temporal SLAM for visual-language navigation (paper-faithful implementation).

Frames (SPEC section 0): world W, body B, camera C. Camera axes follow OpenCV
(x right, y down, z forward). P_t = T_WC (4x4 camera-to-world). Depth in meters.
"""
# REQ-001, REQ-002: package entry; see types.py for the observation datatype.
from supermap.types import Observation, is_se3
from supermap.camera import PinholeCamera, pi, project, world_to_camera
from supermap.pose import PoseProvider, DatasetPoseProvider
from supermap.map import GlobalMap, ObjectInstance, ObjectStatus

__all__ = [
    "Observation", "is_se3",
    "PinholeCamera", "pi", "project", "world_to_camera",
    "PoseProvider", "DatasetPoseProvider",
    "GlobalMap", "ObjectInstance", "ObjectStatus",
]
