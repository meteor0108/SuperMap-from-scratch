"""Hybrid 2D-3D tracker: ByteTrack association with 3D-to-2D motion compensation
(Sec. IV-B2, Supp. Alg. 1 lines 11-18, Supp. VIII-B; ByteTrack [33] Algorithm 1).

Symbol map (paper -> code):
    T (tracker)       -> Tracker
    T_alive           -> Tracker.alive_tracks() (tracked + lost, not yet removed)
    T_trk             -> list[TrackedDetection] returned by Tracker.associate()
    B_det             -> detection boxes (M, 4) xyxy pixels with scores and labels z
    UpdateFromMap(O)  -> Tracker.update_from_map(O_on, P_t, K) (Eq. 5 prior on alive tracks)
    D_high / D_low    -> score > high_thresh / low_thresh < score <= high_thresh
    epsilon           -> new_track_thresh (track initialization score)
    track id          -> instance id (same identifier, REQ-022)
    gate_3d           -> 3D extension of the Eq. (5) prior (iter 15 experiment)
Association is IoU matching plus label_cost * [label mismatch] in the first pass (D-033 class-agnostic at 0;
D-117 default 1.0 is a hard class gate because the cost then exceeds match_thresh). Track ids come from an allocator
shared with the global map so ids are never reused (REQ-024).
"""
from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from supermap.tracking.gate3d import gate_matrix
from supermap.tracking.footprint import footprint_box, map_footprints
from supermap.tracking.map_match import match_leftovers
from supermap.tracking.state import (box_to_state, compensate_centroid, kf_predict, kf_update,
                                     make_F, make_Q, project_bbox_size, project_centroid,
                                     state_to_box)

_INFEASIBLE = 1e6


# REQ-020: (N, M) matrix, 1.0 where both labels are known (>= 0) and differ (D-108)
def label_mismatch_matrix(track_labels, det_labels) -> np.ndarray:
    tl = np.asarray(track_labels, dtype=np.int64).reshape(-1, 1)
    dl = np.asarray(det_labels, dtype=np.int64).reshape(1, -1)
    return ((tl >= 0) & (dl >= 0) & (tl != dl)).astype(np.float64)


# REQ-022, REQ-026, REQ-028: X_O over the points selected by point_mask when one is given (D-105)
def _object_geometry(obj, point_mask):
    """Returns (centroid, bbox) of the masked points; all points when no usable mask (None, wrong length, or empty)."""
    if point_mask is not None:
        m = np.asarray(point_mask, dtype=bool).reshape(-1)
        if m.shape[0] == obj.num_points and m.any():
            pts = obj.points[m]
            return pts.mean(axis=0), (pts.min(axis=0), pts.max(axis=0))
    return obj.centroid, obj.bbox


# REQ-020: pairwise IoU of xyxy boxes, (N, 4) x (M, 4) -> (N, M)
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


# REQ-020: Hungarian assignment on cost = 1 - IoU, pairs with cost > thresh rejected
def linear_assignment(cost: np.ndarray, thresh: float):
    """Returns (matches [(row, col)], unmatched rows, unmatched cols)."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    c = np.where(cost <= thresh, cost, _INFEASIBLE)
    rows, cols = linear_sum_assignment(c)
    matches = [(int(r), int(k)) for r, k in zip(rows, cols) if cost[r, k] <= thresh]
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return matches, [r for r in range(n) if r not in mr], [k for k in range(m) if k not in mc]


# REQ-020: ByteTrack track states
class TrackState(str, enum.Enum):
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


# REQ-016, REQ-018: one tracklet with Kalman state S_i (Eq. 4) and covariance
@dataclass
class Track:
    track_id: int
    S: np.ndarray
    Sigma: np.ndarray
    label: int
    score: float
    start_frame: int
    last_matched_frame: int
    state: TrackState = TrackState.TRACKED
    compensated: bool = False   # centroid replaced by Eq. (5) in the current frame
    reactivated: bool = False   # re-created from the map in the current frame (D-035)

    # REQ-016: current xyxy box from S_i
    @property
    def box(self) -> np.ndarray:
        return state_to_box(self.S)


# REQ-020: one element of T_trk: (track id == instance id, box, label z, score)
class TrackedDetection(NamedTuple):
    track_id: int
    box: np.ndarray
    label: int
    score: float


# REQ-014, REQ-020: (track_id, box) prompts for SAM2Segmenter.segment_tracks
def as_prompts(t_trk: Sequence[TrackedDetection]) -> list[tuple[int, np.ndarray]]:
    return [(t.track_id, t.box) for t in t_trk]


# REQ-015, REQ-020: B_det boxes, scores and labels z from the detection stream element
def _detection_arrays(detections):
    """Accept Detections, DetectionObservation, or (boxes, scores, labels)."""
    if hasattr(detections, "detections"):
        detections = detections.detections
    if hasattr(detections, "boxes"):
        return (np.asarray(detections.boxes, np.float64).reshape(-1, 4),
                np.asarray(detections.scores, np.float64).reshape(-1),
                np.asarray(detections.label_ids, np.int64).reshape(-1))
    boxes, scores, labels = detections
    return (np.asarray(boxes, np.float64).reshape(-1, 4), np.asarray(scores, np.float64).reshape(-1),
            np.asarray(labels, np.int64).reshape(-1))


# REQ-020, REQ-022, REQ-026: hybrid tracker (Supp. Alg. 1 T), ByteTrack [33] Algorithm 1 association
class Tracker:
    """Per-frame usage (Supp. Alg. 1 order): predict() -> update_from_map(O_on) ->
    reactivate_from_map(O_on) -> associate(B_det). step() runs all four."""

    # REQ-019, REQ-020: F, Q, R, initial covariance and ByteTrack thresholds (configs/default.yaml `tracking`)
    def __init__(self, F: np.ndarray, Q: np.ndarray, R: np.ndarray, Sigma0: np.ndarray,
                 high_thresh: float = 0.5, low_thresh: float = 0.1, new_track_thresh: float = 0.6,
                 match_thresh: float = 0.8, second_match_thresh: float = 0.5, track_buffer: int = 30,
                 motion_compensation: bool = True, reactivation: bool = True,
                 id_allocator: Callable[[], int] | None = None, label_cost: float = 0.0,
                 footprint_boxes: bool = False, map_match_iou: float = 0.0,
                 gate_3d: float = 0.0) -> None:
        self.F = np.asarray(F, np.float64)
        self.dim = self.F.shape[0]
        self.Q = np.asarray(Q, np.float64)
        self.R = np.asarray(R, np.float64)
        self.Sigma0 = np.asarray(Sigma0, np.float64)
        if self.Q.shape != (self.dim, self.dim) or self.Sigma0.shape != (self.dim, self.dim):
            raise ValueError("Q and Sigma0 must match the state dimension of F")
        if self.R.shape != (4, 4):
            raise ValueError("R must be (4, 4)")
        if not 0.0 <= low_thresh <= high_thresh:
            raise ValueError("need 0 <= low_thresh <= high_thresh")
        if track_buffer < 0:
            raise ValueError("track_buffer must be >= 0")
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.second_match_thresh = float(second_match_thresh)
        self.track_buffer = int(track_buffer)
        if not 0.0 <= map_match_iou <= 1.0:
            raise ValueError("map_match_iou must be in [0, 1]")
        # REQ-026: candidate box from the pixel footprint of the projected points (I-11-3 candidate)
        self.footprint_boxes = bool(footprint_boxes)
        # REQ-022/026: leftover D_high detections take the id of the O_on object with footprint IoU >= this; 0 = off
        self.map_match_iou = float(map_match_iou)
        if gate_3d < 0.0:
            raise ValueError("gate_3d must be >= 0")
        # REQ-022: reject pairs whose 3D centroids are farther apart than this (m); 0 = off (iter 15)
        self.gate_3d = float(gate_3d)
        self.gate_rejected = self.gate_rejected_total = 0
        self._map_boxes: dict[int, np.ndarray] = {}
        self.map_matched = self.map_dropped = 0
        self.map_matched_total = self.map_dropped_total = 0
        # REQ-020: extra first-pass cost for differing labels (D-108); 0 = class-agnostic ByteTrack
        self.label_cost = float(label_cost)
        # REQ-025: 3D-compensated prior keeps IDs under rapid ego-motion (Sec. IV-B2); false = 2D-only control
        self.motion_compensation = bool(motion_compensation)
        self.reactivation = bool(reactivation)
        if id_allocator is None:
            counter = itertools.count()
            id_allocator = lambda: next(counter)  # noqa: E731
        self._new_id = id_allocator
        self.frame = 0
        self.tracks: dict[int, Track] = {}      # alive: tracked + lost
        self.removed: dict[int, Track] = {}     # expired tracks, kept by id
        self._candidates: dict[int, Track] = {}  # map re-activation hypotheses (this frame)
        self._predicted = False

    # REQ-019, REQ-020: build from the `tracking` config section
    @classmethod
    def from_config(cls, tcfg: dict, id_allocator: Callable[[], int] | None = None) -> "Tracker":
        kf, bt = tcfg["kalman"], tcfg["bytetrack"]
        dim = int(kf["state_dim"])
        F = make_F(float(kf["dt"]), dim)
        Q = make_Q(kf["q_centroid"], kf["q_size"], kf["q_velocity"], dim, kf.get("q_size_velocity"))
        R = np.diag([kf["r_centroid"]] * 2 + [kf["r_size"]] * 2).astype(np.float64)
        Sigma0 = make_Q(kf["p0_centroid"], kf["p0_size"], kf["p0_velocity"], dim,
                        kf.get("p0_size_velocity"))
        return cls(F, Q, R, Sigma0, bt["high_thresh"], bt["low_thresh"], bt["new_track_thresh"],
                   bt["match_thresh"], bt["second_match_thresh"], bt["track_buffer"],
                   tcfg.get("motion_compensation", True), tcfg.get("reactivation", True), id_allocator,
                   float(bt.get("label_cost", 0.0)), bool(bt.get("footprint_boxes", False)),
                   float(bt.get("map_match_iou", 0.0)), float(bt.get("gate_3d", 0.0)))

    # REQ-020: T_alive = tracked + lost tracks (Supp. Alg. 1 line 11)
    def alive_tracks(self) -> list[Track]:
        return [self.tracks[k] for k in sorted(self.tracks)]

    # REQ-020: ids of T_alive
    @property
    def alive_ids(self) -> set[int]:
        return set(self.tracks)

    # REQ-016, REQ-020: start a track from a box; id from the shared allocator unless given
    def init_track(self, box: Sequence[float], score: float = 1.0, label: int = -1,
                   track_id: int | None = None) -> Track:
        tid = self._new_id() if track_id is None else int(track_id)
        if tid in self.tracks:
            raise ValueError(f"track id {tid} is already alive")
        self.removed.pop(tid, None)
        trk = Track(tid, box_to_state(box, self.dim), self.Sigma0.copy(), int(label), float(score),
                    self.frame, self.frame)
        self.tracks[tid] = trk
        return trk

    # REQ-024, REQ-037: drop a track (e.g. its instance was pruned) so its id is never re-associated
    def remove_track(self, track_id: int) -> None:
        trk = self.tracks.pop(int(track_id), None)
        if trk is not None:
            trk.state = TrackState.REMOVED
        self.removed.pop(int(track_id), None)

    # REQ-020, REQ-024: D-111 the track of an absorbed id continues under the absorbing id; never allocates
    def remap_id(self, old: int, new: int) -> None:
        old, new = int(old), int(new)
        if old == new:
            return
        trk = self.tracks.pop(old, None)
        self.removed.pop(old, None)
        self._candidates.pop(old, None)
        if trk is not None and new not in self.tracks:
            trk.track_id = new
            self.removed.pop(new, None)
            self.tracks[new] = trk

    # REQ-018: Eq. (6) prediction for every alive track; starts a new frame
    def predict(self) -> None:
        self.frame += 1
        self._candidates = {}
        self._map_boxes = {}
        self.map_matched = self.map_dropped = 0
        self.gate_rejected = 0
        for trk in self.tracks.values():
            # REQ-028: plain 2D KF prediction for objects not in O_on (e.g. a person crossing the view) keeps one ID
            trk.S, trk.Sigma = kf_predict(trk.S, trk.Sigma, self.F, self.Q)
            trk.compensated = False
            trk.reactivated = False
        self._predicted = True

    # REQ-022: UpdateFromMap (Supp. Alg. 1 lines 12-16): for O in O_on with O.id in T_alive,
    # predicted centroid := c_hat = pi(K P_t^{-1} X_O) (Eq. 5). Returns compensated ids.
    # REQ-025, REQ-028: point_mask (pipeline: points not Disappeared in this frame) selects X_O (D-105)
    def update_from_map(self, O_on: Iterable, P_t: np.ndarray, K: np.ndarray,
                        point_mask: Mapping[int, np.ndarray] | None = None) -> list[int]:
        if not self.motion_compensation:
            return []
        done = []
        for obj in O_on:
            trk = self.tracks.get(int(obj.id))
            if trk is None:
                continue
            c, _ = _object_geometry(obj, None if point_mask is None else point_mask.get(int(obj.id)))
            c_hat = project_centroid(c, P_t, K)
            if c_hat is None:
                continue
            # REQ-025: projected centroid replaces the 2D constant-velocity prediction
            trk.S = compensate_centroid(trk.S, c_hat)
            trk.compensated = True
            done.append(trk.track_id)
        return done

    # REQ-026: re-activation (D-035): O in O_on whose id is not alive becomes a candidate track with
    # the same id, centroid from Eq. (5), size from the projected 3D bbox (else the last known size)
    # centroid and size use the points selected by point_mask when one is given (D-105)
    def reactivate_from_map(self, O_on: Iterable, P_t: np.ndarray, K: np.ndarray,
                            point_mask: Mapping[int, np.ndarray] | None = None,
                            image_size: tuple[int, int] | None = None) -> list[int]:
        O_on = list(O_on)
        if image_size is not None:
            # REQ-026: footprints of all O_on objects for map-guided association
            self._map_boxes = map_footprints(O_on, P_t, K, image_size, point_mask)
        if not (self.motion_compensation and self.reactivation):
            return []
        out = []
        for obj in O_on:
            oid = int(obj.id)
            if oid in self.tracks or oid in self._candidates:
                continue
            c, bb = _object_geometry(obj, None if point_mask is None else point_mask.get(oid))
            fb = self._map_boxes.get(oid) if self.footprint_boxes and image_size is not None else None
            if fb is not None:
                c_hat, size = (fb[0:2] + fb[2:4]) / 2.0, fb[2:4] - fb[0:2]
            else:
                c_hat = project_centroid(c, P_t, K)
                if c_hat is None:
                    continue
                size = project_bbox_size(bb, P_t, K)
            old = self.removed.get(oid)
            if (size is None or np.any(size <= 0)) and old is not None:
                size = old.S[2:4].copy()
            if size is None or np.any(size <= 0):
                continue
            S = np.zeros(self.dim, np.float64)
            S[0:2], S[2:4] = c_hat, size
            label = old.label if old is not None else -1
            start = old.start_frame if old is not None else self.frame
            last = old.last_matched_frame if old is not None else self.frame
            self._candidates[oid] = Track(oid, S, self.Sigma0.copy(), label, 0.0, start, last,
                                          TrackState.LOST, compensated=True, reactivated=True)
            out.append(oid)
        return out

    # REQ-020: ByteTrack Algorithm 1: high-score IoU matching, low-score second matching,
    # new tracks from unmatched high-score detections, removal after track_buffer frames
    # REQ-022: optional 3D gate from det_centroids (M, 3) and track_centroids {track id: (3,)} (iter 15)
    def associate(self, detections, det_centroids=None,
                  track_centroids: Mapping[int, np.ndarray] | None = None) -> list[TrackedDetection]:
        if not self._predicted:
            raise RuntimeError("call predict() before associate()")
        self._predicted = False
        boxes, scores, labels = _detection_arrays(detections)
        if not (boxes.shape[0] == scores.shape[0] == labels.shape[0]):
            raise ValueError("boxes, scores and labels must have the same length")
        high = np.nonzero(scores > self.high_thresh)[0]
        low = np.nonzero((scores > self.low_thresh) & (scores <= self.high_thresh))[0]

        pool = self.alive_tracks() + [self._candidates[k] for k in sorted(self._candidates)]
        pool_boxes = np.array([t.box for t in pool]).reshape(-1, 4)
        out: list[tuple[int, TrackedDetection]] = []

        # first association: all tracks vs D_high; cost = 1 - IoU + label_cost * mismatch (D-108).
        # label_cost > match_thresh (or >= with any IoU < 1) acts as a hard class gate
        cost = 1.0 - iou_matrix(pool_boxes, boxes[high])
        if self.label_cost > 0.0:
            cost = cost + self.label_cost * label_mismatch_matrix([t.label for t in pool], labels[high])
        gate = self.gate_3d > 0.0 and det_centroids is not None and track_centroids is not None
        if gate:
            det_c = np.asarray(det_centroids, np.float64).reshape(-1, 3)
            pool_c = np.array([track_centroids.get(t.track_id, np.full(3, np.nan)) for t in pool],
                              dtype=np.float64).reshape(-1, 3)
            rej = gate_matrix(pool_c, det_c[high], self.gate_3d)
            self.gate_rejected += int((rej & (cost <= self.match_thresh)).sum())
            cost[rej] = _INFEASIBLE
        m1, rem_trk, rem_high = linear_assignment(cost, self.match_thresh)
        for r, k in m1:
            out.append(self._apply_match(pool[r], high[k], boxes, scores, labels))

        # second association: T_remain vs D_low
        rem_pool = [pool[r] for r in rem_trk]
        cost2 = 1.0 - iou_matrix(np.array([t.box for t in rem_pool]).reshape(-1, 4), boxes[low])
        if gate:
            rej2 = gate_matrix(pool_c[np.asarray(rem_trk, dtype=int)].reshape(-1, 3), det_c[low], self.gate_3d)
            self.gate_rejected += int((rej2 & (cost2 <= self.second_match_thresh)).sum())
            cost2[rej2] = _INFEASIBLE
        m2, rem_trk2, _ = linear_assignment(cost2, self.second_match_thresh)
        for r, k in m2:
            out.append(self._apply_match(rem_pool[r], low[k], boxes, scores, labels))

        # unmatched tracks: lost, removed after track_buffer frames; unmatched candidates dropped
        just_removed: set[int] = set()
        for r in rem_trk2:
            trk = rem_pool[r]
            if trk.reactivated:
                continue
            trk.state = TrackState.LOST
            if self.frame - trk.last_matched_frame > self.track_buffer:
                trk.state = TrackState.REMOVED
                self.removed[trk.track_id] = self.tracks.pop(trk.track_id)
                just_removed.add(trk.track_id)
        self._candidates = {}

        # REQ-022, REQ-026: map-guided association of the leftover D_high detections
        if self.map_match_iou > 0.0 and self._map_boxes:
            rem_high = match_leftovers(self, high, rem_high, boxes, scores, labels, just_removed, out)
            self.map_matched_total += self.map_matched
            self.map_dropped_total += self.map_dropped
        self.gate_rejected_total += self.gate_rejected

        # new tracks from unmatched D_high with score >= epsilon
        # REQ-027: a newly appearing object gets a fresh, never-used ID; matched neighbors keep theirs
        for k in rem_high:
            d = high[k]
            if scores[d] < self.new_track_thresh:
                continue
            trk = self.init_track(boxes[d], scores[d], labels[d])
            out.append((int(d), TrackedDetection(trk.track_id, boxes[d].copy(), int(labels[d]),
                                                 float(scores[d]))))
        out.sort(key=lambda x: x[0])
        return [t for _, t in out]

    # REQ-018, REQ-020: KF measurement update of a matched track; candidate becomes alive
    def _apply_match(self, trk: Track, d: int, boxes, scores, labels) -> tuple[int, TrackedDetection]:
        z = box_to_state(boxes[d], 6)[:4]
        trk.S, trk.Sigma = kf_update(trk.S, trk.Sigma, z, self.R)
        trk.state = TrackState.TRACKED
        trk.last_matched_frame = self.frame
        trk.score = float(scores[d])
        trk.label = int(labels[d])
        if trk.reactivated:
            self.removed.pop(trk.track_id, None)
            self.tracks[trk.track_id] = trk
        return int(d), TrackedDetection(trk.track_id, boxes[d].copy(), int(labels[d]), float(scores[d]))

    # REQ-020, REQ-022, REQ-026: one frame in Supp. Alg. 1 order (lines 11-18)
    def step(self, detections, O_on: Iterable, P_t: np.ndarray, K: np.ndarray,
             point_mask: Mapping[int, np.ndarray] | None = None, det_centroids=None,
             track_centroids: Mapping[int, np.ndarray] | None = None) -> list[TrackedDetection]:
        O_on = list(O_on)
        self.predict()
        self.update_from_map(O_on, P_t, K, point_mask=point_mask)
        self.reactivate_from_map(O_on, P_t, K, point_mask=point_mask)
        return self.associate(detections, det_centroids, track_centroids)
