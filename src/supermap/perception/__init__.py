"""2D perception for the instance layer (Sec. IV-B, Sec. V-B b), Supp. Alg. 1).

Data flow (SPEC 0): C_t -> GroundingDINO(C_t, prompt list) -> B_det (Detections)
-> [tracker, M3] -> T_trk -> SAM2(C_t, T_trk boxes) -> per-track masks.
"""
# REQ-012, REQ-013, REQ-014, REQ-015: public perception API
from supermap.perception.detector import Detections, GroundingDINODetector, resolve_device
from supermap.perception.observation import DetectionObservation, make_detection_observation
from supermap.perception.prompts import PromptList, labels_to_indices
from supermap.perception.segmenter import SAM2Segmenter

__all__ = ["Detections", "GroundingDINODetector", "resolve_device", "DetectionObservation",
           "make_detection_observation", "PromptList", "labels_to_indices", "SAM2Segmenter"]
