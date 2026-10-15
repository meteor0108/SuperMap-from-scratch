"""Class-agnostic non-maximum suppression on xyxy boxes (REQ-012, D-020 option)."""
from __future__ import annotations

import numpy as np


# REQ-012: IoU of one xyxy box against many, 0 where the union area is 0
def _iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(box[2], others[:, 2]) - np.maximum(box[0], others[:, 0]), 0.0, None)
    ih = np.clip(np.minimum(box[3], others[:, 3]) - np.maximum(box[1], others[:, 1]), 0.0, None)
    inter = iw * ih
    area = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    areas = np.clip(others[:, 2] - others[:, 0], 0.0, None) * np.clip(others[:, 3] - others[:, 1], 0.0, None)
    union = area + areas - inter
    return np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)


# REQ-012: greedy NMS, keeps a box unless IoU >= iou_thresh with a kept one; returns int64 indices by descending score
def nms_keep(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    for i in order:
        if kept and np.any(_iou_one_to_many(boxes[i], boxes[kept]) >= iou_thresh):
            continue
        kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)
