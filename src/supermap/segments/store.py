"""Segment store: per-frame lifted masks and the keyframe records needed to re-observe them (D-124).

Symbol map (design -> code):
    segment s, P_s, z_s   -> Segment (points, label_obs)
    keyframe f            -> Keyframe (pose, K, downsampled depth / validity / mask-id images)
    M(s, f) lookup data   -> Keyframe.mask_ids (index into Keyframe.segment_ids)
Downsampling convention: a full-resolution pixel (u, v) maps to (u // scale, v // scale); the small
images are stride samples, i.e. depth[v_s, u_s] = depth_full[v_s * scale, u_s * scale].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# REQ-023 (extension): one lifted mask of one keyframe
@dataclass
class Segment:
    id: int
    t: int                                  # stream index of the keyframe it came from
    points: np.ndarray                      # (n, 3) float32 world, voxel-deduplicated
    colors: np.ndarray                      # (n, 3) uint8
    label_obs: list[tuple[int, float]] = field(default_factory=list)  # (class index z, detector score)

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.t = int(self.t)
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 3)
        self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
        self.label_obs = [(int(z), float(s)) for z, s in self.label_obs]

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    # score-summed label vote of this segment, (C,) float64
    def label_scores(self, num_classes: int) -> np.ndarray:
        v = np.zeros(num_classes, dtype=np.float64)
        for z, s in self.label_obs:
            if 0 <= z < num_classes:
                v[z] += s
        return v


# REQ-023 (extension): what is needed to test "visible" and "contained" for any segment later
@dataclass
class Keyframe:
    t: int
    pose: np.ndarray            # (4, 4) T_WC
    K: np.ndarray               # (3, 3) intrinsics of the FULL-resolution image
    scale: int                  # downsample factor of the small images
    depth: np.ndarray           # (H/s, W/s) float32 meters, 0 = invalid
    valid: np.ndarray           # (H/s, W/s) bool: depth valid and not on a depth edge
    mask_ids: np.ndarray        # (H/s, W/s) int32 index into segment_ids, -1 = no mask
    segment_ids: np.ndarray     # (M,) int64 segment id of each mask index
    background: np.ndarray      # (M,) bool, mask belongs to a background-class segment

    def __post_init__(self) -> None:
        self.t = int(self.t)
        self.pose = np.asarray(self.pose, dtype=np.float64).reshape(4, 4)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.scale = int(self.scale)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.mask_ids = np.asarray(self.mask_ids, dtype=np.int32)
        self.segment_ids = np.asarray(self.segment_ids, dtype=np.int64).reshape(-1)
        self.background = np.asarray(self.background, dtype=bool).reshape(-1)
        if self.depth.shape != self.valid.shape or self.depth.shape != self.mask_ids.shape:
            raise ValueError("depth, valid and mask_ids must have the same shape")
        if self.segment_ids.shape != self.background.shape:
            raise ValueError("segment_ids and background must have the same length")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])


# REQ-023 (extension): all segments and keyframes of a run; saved as one npz per keyframe + a segments npz
class SegmentStore:
    def __init__(self) -> None:
        self.segments: dict[int, Segment] = {}
        self.keyframes: list[Keyframe] = []
        self._next_id = 0

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add_segment(self, t: int, points: np.ndarray, colors: np.ndarray,
                    label_obs: list[tuple[int, float]]) -> Segment:
        seg = Segment(self.new_id(), t, points, colors, label_obs)
        self.segments[seg.id] = seg
        return seg

    def add_keyframe(self, kf: Keyframe) -> Keyframe:
        self.keyframes.append(kf)
        return kf

    def __len__(self) -> int:
        return len(self.segments)

    # persist to a directory: segments.npz (concatenated points) + keyframes/<i>.npz + meta.json
    def save(self, path: str | Path) -> None:
        path = Path(path)
        (path / "keyframes").mkdir(parents=True, exist_ok=True)
        segs = [self.segments[k] for k in sorted(self.segments)]
        counts = np.array([s.num_points for s in segs], dtype=np.int64)
        np.savez(path / "segments.npz",
                 ids=np.array([s.id for s in segs], dtype=np.int64),
                 t=np.array([s.t for s in segs], dtype=np.int64),
                 counts=counts,
                 points=(np.concatenate([s.points for s in segs]) if segs else np.zeros((0, 3), np.float32)),
                 colors=(np.concatenate([s.colors for s in segs]) if segs else np.zeros((0, 3), np.uint8)))
        meta = {"next_id": self._next_id, "num_keyframes": len(self.keyframes),
                "label_obs": {str(s.id): s.label_obs for s in segs}}
        (path / "meta.json").write_text(json.dumps(meta))
        for i, kf in enumerate(self.keyframes):
            np.savez(path / "keyframes" / f"{i:05d}.npz", t=kf.t, pose=kf.pose, K=kf.K, scale=kf.scale,
                     depth=kf.depth, valid=kf.valid, mask_ids=kf.mask_ids,
                     segment_ids=kf.segment_ids, background=kf.background)

    @classmethod
    def load(cls, path: str | Path) -> "SegmentStore":
        path = Path(path)
        st = cls()
        meta = json.loads((path / "meta.json").read_text())
        d = np.load(path / "segments.npz")
        # read each array once: every NpzFile[] access decompresses the whole array again
        points, colors = d["points"], d["colors"]
        off = 0
        for sid, t, n in zip(d["ids"], d["t"], d["counts"]):
            lab = [(int(z), float(s)) for z, s in meta["label_obs"][str(int(sid))]]
            st.segments[int(sid)] = Segment(int(sid), int(t), points[off:off + n].copy(),
                                            colors[off:off + n].copy(), lab)
            off += int(n)
        st._next_id = int(meta["next_id"])
        for i in range(int(meta["num_keyframes"])):
            k = np.load(path / "keyframes" / f"{i:05d}.npz")
            st.keyframes.append(Keyframe(int(k["t"]), k["pose"], k["K"], int(k["scale"]), k["depth"],
                                         k["valid"], k["mask_ids"], k["segment_ids"], k["background"]))
        return st
