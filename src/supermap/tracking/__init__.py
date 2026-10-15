"""Instance layer tracking and association (Sec. IV-B1/B2, Supp. Alg. 1 lines 5-6, 11-20, 28-29).

Data flow per frame (SPEC 0): O_nbr = get_neighbor_objects(M, P_t) -> [CompareDepth, M4 -> O_on]
-> Tracker.step(B_det, O_on, P_t, K) = predict (Eq. 6) + update_from_map (Eq. 5) + re-activation
+ ByteTrack associate -> T_trk -> SAM2 masks -> unproject_masks -> O_add
-> obj_merge(O_nbr, O_add) -> map_merge(M, O_nbr).
"""
# REQ-016..REQ-024: public tracking API
from supermap.tracking.bytetrack import (Track, TrackedDetection, Tracker, TrackState, as_prompts,
                                         iou_matrix, linear_assignment)
from supermap.tracking.instance_ops import ObservedObject, map_merge, obj_merge, unproject_masks
from supermap.tracking.neighbors import get_neighbor_objects
from supermap.tracking.state import (box_to_state, compensate_centroid, kf_predict, kf_update, make_F,
                                     make_Q, project_bbox_size, project_centroid, state_to_box)

__all__ = [
    "Track", "TrackedDetection", "Tracker", "TrackState", "as_prompts", "iou_matrix",
    "linear_assignment", "ObservedObject", "map_merge", "obj_merge", "unproject_masks",
    "get_neighbor_objects", "box_to_state", "compensate_centroid", "kf_predict", "kf_update",
    "make_F", "make_Q", "project_bbox_size", "project_centroid", "state_to_box",
]
