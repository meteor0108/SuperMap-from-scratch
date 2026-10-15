"""SAM2 instance masks from tracked boxes (REQ-014; Sec. V-B b), ref. [21]; Supp. Alg. 1 line 19).

Interface:
    segment(rgb, boxes)        : C_t (H, W, 3) uint8, boxes (K, 4) xyxy pixels -> (K, H, W) bool
    segment_tracks(rgb, T_trk) : T_trk = [(track_id, box), ...] -> [(track_id, mask (H, W) bool), ...]
Output order equals input order; one mask per box (multimask_output=False, D-023).
Symbol map (paper -> code): M_mask -> returned masks, T_trk -> tracks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from supermap.perception.detector import resolve_device


# REQ-014: SAM2 wrapper with box prompts, weights loaded lazily on first call
class SAM2Segmenter:
    """GenerateMasks(I, T_trk) of Supp. Alg. 1 (D-005 checkpoint)."""

    # REQ-014: store settings only; no model is loaded here
    def __init__(self, model_id: str, mask_threshold: float = 0.0, multimask_output: bool = False,
                 device: str | None = "auto", local_files_only: bool = True):
        self.model_id = model_id
        self.mask_threshold = float(mask_threshold)
        self.multimask_output = bool(multimask_output)
        self.device = resolve_device(device)
        self.local_files_only = bool(local_files_only)
        self.processor = None
        self.model = None

    # REQ-014: build from configs/default.yaml `perception` section
    @classmethod
    def from_config(cls, pcfg: dict) -> "SAM2Segmenter":
        s = pcfg["segmenter"]
        return cls(s["model_id"], s.get("mask_threshold", 0.0), s.get("multimask_output", False),
                   s.get("device", "auto"), s.get("local_files_only", True))

    # REQ-014: lazy load of the real SAM2 weights (HF transformers)
    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoProcessor, Sam2Model
        self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        self.model = Sam2Model.from_pretrained(
            self.model_id, local_files_only=self.local_files_only).to(self.device).eval()

    # REQ-014: one SAM2 mask per box prompt, in input order
    @torch.no_grad()
    def segment(self, rgb: np.ndarray, boxes) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
        H, W = rgb.shape[:2]
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.size == 0:
            return np.zeros((0, H, W), dtype=bool)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.all(np.isfinite(boxes)):
            raise ValueError(f"boxes must be finite (K, 4) xyxy, got {boxes.shape}")
        self.load()
        inputs = self.processor(images=rgb, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        out = self.model(**inputs, multimask_output=self.multimask_output)
        logits = out.pred_masks  # (1, K, n_masks, h, w)
        if self.multimask_output:
            # D-023: keep the candidate with the highest predicted IoU
            best = out.iou_scores[0].argmax(dim=-1)  # (K,)
            logits = logits[:, torch.arange(logits.shape[1], device=logits.device), best][:, :, None]
        masks = self.processor.post_process_masks(
            logits.float().cpu(), inputs["original_sizes"].cpu(), mask_threshold=self.mask_threshold,
            binarize=True)[0]  # (K, 1, H, W)
        masks = masks[:, 0].numpy().astype(bool)
        if masks.shape != (boxes.shape[0], H, W):
            raise RuntimeError(f"unexpected SAM2 mask shape {masks.shape}")
        return masks

    # REQ-014: masks keep the track id of their box, order preserved
    def segment_tracks(self, rgb: np.ndarray, tracks: Sequence[tuple[int, Sequence[float]]]
                       ) -> list[tuple[int, np.ndarray]]:
        ids = [int(tid) for tid, _ in tracks]
        boxes = np.asarray([list(b) for _, b in tracks], dtype=np.float64).reshape(-1, 4)
        masks = self.segment(rgb, boxes)
        return list(zip(ids, masks))

    __call__ = segment
