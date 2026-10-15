"""Detection observation passed to the instance layer (REQ-015; Sec. IV-B4, Supp. Alg. 1 line 1).

Stream element S = (B_det, I, D_I, Odom):
    B_det -> DetectionObservation.detections (boxes (M, 4), labels, scores)
    I     -> .I     = Observation.rgb   C_t (H, W, 3) uint8
    D_I   -> .D_I   = Observation.depth D_t (H, W) float32 meters
    Odom  -> .Odom  = Observation.pose  P_t (4, 4) T_WC
    z_t   -> .z     (M,) int64 class indices, 0-based into the prompt list (D-022)
Detection scores are carried but not used as fusion weights (D-024).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from supermap.perception.detector import Detections
from supermap.types import Observation


# REQ-015: (B_det, I, D_I, Odom) tuple of one frame; all parts come from the same t
@dataclass(frozen=True)
class DetectionObservation:
    detections: Detections
    observation: Observation

    # REQ-015: consistency of image size and time index between B_det and Q_t
    def __post_init__(self) -> None:
        d, o = self.detections, self.observation
        if d.image_size is not None and d.image_size != (o.height, o.width):
            raise ValueError(f"detections image size {d.image_size} != observation {(o.height, o.width)}")
        if d.t is not None and d.t != o.t:
            raise ValueError(f"detections t={d.t} != observation t={o.t}")

    @property
    def B_det(self) -> np.ndarray:
        return self.detections.boxes

    @property
    def I(self) -> np.ndarray:
        return self.observation.rgb

    @property
    def D_I(self) -> np.ndarray:
        return self.observation.depth

    @property
    def Odom(self) -> np.ndarray:
        return self.observation.pose

    @property
    def t(self) -> int:
        return self.observation.t

    # REQ-015: detector label observations z_t for Eq. (10)
    @property
    def z(self) -> np.ndarray:
        return self.detections.label_ids

    @property
    def scores(self) -> np.ndarray:
        return self.detections.scores

    # REQ-015: Supp. Alg. 1 stream element (B_det, I, D_I, Odom)
    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.B_det, self.I, self.D_I, self.Odom


# REQ-015: build the per-frame stream element from Q_t and its detections
def make_detection_observation(obs: Observation, detections: Detections) -> DetectionObservation:
    if detections.t is None or detections.image_size is None:
        detections = Detections(detections.boxes, detections.labels, detections.label_ids,
                                detections.scores, detections.prompts,
                                detections.image_size or (obs.height, obs.width),
                                obs.t if detections.t is None else detections.t)
    return DetectionObservation(detections, obs)
