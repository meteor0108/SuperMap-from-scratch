"""GroundingDINO 2D detector (REQ-012; Sec. IV-B, Sec. V-B b), ref. [12]).

Interface:
    input : C_t rgb (H, W, 3) uint8 RGB, PromptList
    output: Detections with boxes (M, 4) float64 xyxy pixels clipped to [0, W] x [0, H],
            labels (M,) str (elements of the prompt list), label_ids (M,) int64 in 0..C-1,
            scores (M,) float64 in [0, 1]
Symbol map (paper -> code): B_det -> Detections.boxes, z_t -> Detections.label_ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from supermap.perception.nms import nms_keep
from supermap.perception.prompts import PromptList, labels_to_indices


# REQ-012: select the torch device ("auto" -> cuda if available)
def resolve_device(device: str | None = "auto") -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# REQ-012, REQ-015: per-frame detection set B_det with labels and scores
@dataclass(frozen=True)
class Detections:
    """Detections of one frame. Arrays are validated at construction."""

    boxes: np.ndarray
    labels: tuple[str, ...]
    label_ids: np.ndarray
    scores: np.ndarray
    prompts: PromptList
    image_size: tuple[int, int] | None = None  # (H, W) of the source image
    t: int | None = None  # time index of the source frame, if known

    # REQ-012: shape/range validation, raises ValueError on malformed input
    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float64)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        label_ids = np.asarray(self.label_ids, dtype=np.int64).reshape(-1)
        labels = tuple(str(l) for l in self.labels)
        M = boxes.shape[0]
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes must be (M, 4), got {boxes.shape}")
        if not (len(labels) == M and scores.shape[0] == M and label_ids.shape[0] == M):
            raise ValueError("boxes, labels, label_ids and scores must have the same length")
        if not np.all(np.isfinite(boxes)):
            raise ValueError("boxes must be finite")
        if M and (np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1])):
            raise ValueError("boxes must be xyxy with x2 >= x1 and y2 >= y1")
        if M and (np.any(scores < 0.0) or np.any(scores > 1.0)):
            raise ValueError("scores must be in [0, 1]")
        if M and (np.any(label_ids < 0) or np.any(label_ids >= self.prompts.C)):
            raise ValueError("label_ids must index the prompt list")
        if [self.prompts.phrases[i] for i in label_ids] != list(labels):
            raise ValueError("labels must equal prompts[label_ids]")
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "label_ids", label_ids)
        object.__setattr__(self, "labels", labels)
        if self.image_size is not None:
            object.__setattr__(self, "image_size", (int(self.image_size[0]), int(self.image_size[1])))
        if self.t is not None:
            object.__setattr__(self, "t", int(self.t))

    # REQ-013, REQ-015: build from label strings; z indices are derived from the prompt list
    @classmethod
    def from_labels(cls, boxes, labels: Sequence[str], scores, prompts: PromptList | Sequence[str],
                    image_size: tuple[int, int] | None = None, t: int | None = None) -> "Detections":
        pl = prompts if isinstance(prompts, PromptList) else PromptList.from_config(prompts)
        ids = labels_to_indices(labels, pl)
        return cls(boxes, tuple(pl.phrases[i] for i in ids), ids, scores, pl, image_size, t)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


# REQ-012: GroundingDINO wrapper, weights loaded lazily on first call
class GroundingDINODetector:
    """Language-grounded detector [12] (D-004 checkpoint, D-020/D-021 thresholds and labels)."""

    # REQ-012: store settings only; no model is loaded here
    def __init__(self, prompts: PromptList | Sequence[str], model_id: str,
                 box_threshold: float, device: str | None = "auto", local_files_only: bool = True,
                 nms_iou: float = 0.0):
        self.prompts = prompts if isinstance(prompts, PromptList) else PromptList.from_config(prompts)
        if not 0.0 <= float(box_threshold) <= 1.0:
            raise ValueError("box_threshold must be in [0, 1]")
        if not 0.0 <= float(nms_iou) <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")
        self.nms_iou = float(nms_iou)
        self.model_id = model_id
        self.box_threshold = float(box_threshold)
        self.device = resolve_device(device)
        self.local_files_only = bool(local_files_only)
        self.processor = None
        self.model = None

    # REQ-012: build from configs/default.yaml `perception` section
    @classmethod
    def from_config(cls, pcfg: dict) -> "GroundingDINODetector":
        d = pcfg["detector"]
        return cls(PromptList.from_config(pcfg["prompts"], pcfg.get("prompt_aliases")), d["model_id"], d["box_threshold"], d.get("device", "auto"),
                   d.get("local_files_only", True), d.get("nms_iou", 0.0))

    # REQ-012: lazy load of the real GroundingDINO weights (HF transformers)
    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            self.model_id, local_files_only=self.local_files_only).to(self.device).eval()

    # REQ-012: run GroundingDINO on C_t with the prompt list text
    @torch.no_grad()
    def detect(self, rgb: np.ndarray, t: int | None = None) -> Detections:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
        self.load()
        H, W = rgb.shape[:2]
        text = self.prompts.text()
        enc = self.processor(images=rgb, text=text, return_tensors="pt")
        offsets = self.processor.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        if len(offsets) != enc["input_ids"].shape[1]:
            raise RuntimeError("tokenizer offsets do not match the model input ids")
        out = self.model(**enc.to(self.device))
        # (Q, L) token probabilities; logits beyond the text length are unused
        prob = out.logits[0, :, : len(offsets)].sigmoid().float().cpu().numpy()
        boxes_n = out.pred_boxes[0].float().cpu().numpy()  # (Q, 4) normalized cx, cy, w, h
        return self._decode(prob, boxes_n, offsets, H, W, t)

    # REQ-012: per-query class = prompt phrase holding the max token probability (D-021)
    def _decode(self, prob: np.ndarray, boxes_n: np.ndarray, offsets, H: int, W: int,
                t: int | None) -> Detections:
        tok_cls = self.prompts.token_class_ids(offsets)
        C = self.prompts.C
        cls_prob = np.zeros((prob.shape[0], C), dtype=np.float64)
        for c in range(C):
            cols = np.flatnonzero(tok_cls == c)
            if cols.size:
                cls_prob[:, c] = prob[:, cols].max(axis=1)
        label_ids = cls_prob.argmax(axis=1)
        scores = cls_prob[np.arange(prob.shape[0]), label_ids]
        keep = scores > self.box_threshold
        cx, cy, bw, bh = (boxes_n[keep, k].astype(np.float64) for k in range(4))
        boxes = np.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], axis=1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, W)  # D-020: clip to the image
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, H)
        ids = label_ids[keep]
        sc = scores[keep]
        order = np.argsort(-sc, kind="stable")
        ids, sc, boxes = ids[order], sc[order], boxes[order]
        if self.nms_iou > 0.0 and len(sc):  # class-agnostic NMS, order stays descending score
            k = nms_keep(boxes, sc, self.nms_iou)
            ids, sc, boxes = ids[k], sc[k], boxes[k]
        return Detections(boxes.reshape(-1, 4), tuple(self.prompts.phrases[i] for i in ids), ids,
                          sc, self.prompts, (H, W), t)

    __call__ = detect
