"""Per-keyframe visibility and containment histograms of segments (D-124).

For segment s and keyframe f: V(s, f) = points of s that project into the small image on a valid pixel
with depth agreement; the mask-id image is then read at those pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from supermap.camera import pixel_index, project
from supermap.segments.params import ConsensusParams
from supermap.segments.store import SegmentStore


# REQ-024 (extension): what keyframe f sees of a segment (or of a cluster, when summed)
@dataclass
class FrameHist:
    n_points: int            # |P_s|
    n_vis: int               # |V(s,f)|
    counts: dict[int, int] = field(default_factory=dict)   # mask index -> visible points inside it


# REQ-024 (extension): visible iff |V| / |P| >= alpha and |V| >= n_min
def is_visible(n_points: int, n_vis: int, p: ConsensusParams) -> bool:
    return n_points > 0 and n_vis >= p.n_min and n_vis >= p.alpha * n_points


# REQ-024 (extension): mask holding >= beta of the visible points, else -1
def contained_mask(n_vis: int, counts: dict[int, int], p: ConsensusParams) -> int:
    if n_vis <= 0 or not counts:
        return -1
    best_m, best_c = -1, 0
    for m in sorted(counts):
        if counts[m] > best_c:
            best_m, best_c = m, counts[m]
    return best_m if best_c >= p.beta * n_vis else -1


# REQ-024 (extension): elementwise sum of two histograms (cluster = sum of members)
def merge_hists(a: FrameHist | None, b: FrameHist | None) -> FrameHist | None:
    if a is None:
        return None if b is None else FrameHist(b.n_points, b.n_vis, dict(b.counts))
    if b is None:
        return FrameHist(a.n_points, a.n_vis, dict(a.counts))
    counts = dict(a.counts)
    for m, c in b.counts.items():
        counts[m] = counts.get(m, 0) + c
    return FrameHist(a.n_points + b.n_points, a.n_vis + b.n_vis, counts)


_CORNER_SIGN = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)


# REQ-024 (extension): stacked points of the segments with their frustum-check corners
@dataclass
class _Stack:
    ids: list[int]
    npts: np.ndarray
    offs: np.ndarray
    pts: np.ndarray
    seg_of: np.ndarray
    corners: np.ndarray


def _stack(store: SegmentStore, ids: list[int]) -> _Stack:
    segs = [store.segments[i] for i in ids]
    npts = np.array([s.num_points for s in segs], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(npts)[:-1]])
    pts = np.concatenate([s.points for s in segs]).astype(np.float64)
    lo = np.minimum.reduceat(pts, offs, axis=0)
    hi = np.maximum.reduceat(pts, offs, axis=0)
    corners = (lo[:, None, :] + _CORNER_SIGN[None] * (hi - lo)[:, None, :]).reshape(-1, 3)
    return _Stack(ids, npts, offs, pts, np.repeat(np.arange(len(ids)), npts), corners)


# REQ-024 (extension): project the candidate points into one keyframe; returns
# (point index into st.pts, u, v, z, depth at the pixel) for points on a valid in-image pixel, or None
def _project_keyframe(st: _Stack, kf):
    n_seg = len(st.ids)
    h, w = kf.shape
    W, H = w * kf.scale, h * kf.scale
    # frustum pre-check on the 8 bbox corners
    uvc, zc, _ = project(st.corners, kf.pose, kf.K)
    front = (zc > 0).reshape(n_seg, 8)
    uvc = uvc.reshape(n_seg, 8, 2)
    all_front = front.all(axis=1)
    any_front = front.any(axis=1)
    uu, vv = uvc[..., 0], uvc[..., 1]
    umin = np.where(front, uu, np.inf).min(axis=1)
    umax = np.where(front, uu, -np.inf).max(axis=1)
    vmin = np.where(front, vv, np.inf).min(axis=1)
    vmax = np.where(front, vv, -np.inf).max(axis=1)
    outside = (umax < -0.5) | (umin > W - 0.5) | (vmax < -0.5) | (vmin > H - 0.5)
    cand = any_front & ~(all_front & outside)
    if not cand.any():
        return None
    sel = np.flatnonzero(cand[st.seg_of])
    uv, z, ok_z = project(st.pts[sel], kf.pose, kf.K)
    vi = np.flatnonzero(ok_z)
    if vi.size == 0:
        return None
    pix = pixel_index(uv[vi])
    u, v = pix[:, 0] // kf.scale, pix[:, 1] // kf.scale
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    vi, u, v = vi[inb], u[inb], v[inb]
    okp = kf.valid[v, u]
    vi, u, v = vi[okp], u[okp], v[okp]
    if vi.size == 0:
        return None
    return sel[vi], u, v, z[vi], kf.depth[v, u].astype(np.float64)


# REQ-024 (extension): sparse seg_id -> {keyframe index -> FrameHist} for all keyframes where n_vis > 0
def segment_histograms(store: SegmentStore, params: ConsensusParams,
                       segment_ids: Sequence[int] | None = None) -> dict[int, dict[int, FrameHist]]:
    ids = sorted(store.segments) if segment_ids is None else [int(i) for i in segment_ids]
    out: dict[int, dict[int, FrameHist]] = {i: {} for i in ids}
    ids = [i for i in ids if store.segments[i].num_points > 0]
    if not ids or not store.keyframes:
        return out
    st = _stack(store, ids)
    n_seg = len(ids)
    for fi, kf in enumerate(store.keyframes):
        r = _project_keyframe(st, kf)
        if r is None:
            continue
        pi_, u, v, z, d = r
        good = np.abs(z - d) < params.tau_d
        pi_, u, v = pi_[good], u[good], v[good]
        if pi_.size == 0:
            continue
        sl = st.seg_of[pi_]
        n_vis = np.bincount(sl, minlength=n_seg)
        n_masks = int(kf.segment_ids.shape[0])
        cdict: dict[int, dict[int, int]] = {}
        if n_masks:
            mid = kf.mask_ids[v, u].astype(np.int64)
            has = mid >= 0
            if has.any():
                uniq, cnt = np.unique(sl[has] * n_masks + mid[has], return_counts=True)
                for key, c in zip(uniq.tolist(), cnt.tolist()):
                    cdict.setdefault(key // n_masks, {})[key % n_masks] = c
        for s in np.flatnonzero(n_vis).tolist():
            out[ids[s]][fi] = FrameHist(int(st.npts[s]), int(n_vis[s]), cdict.get(s, {}))
    return out


# REQ-024 (extension): per-point free-space evidence: (confirmed, free) int32 arrays aligned with points
def point_evidence(store: SegmentStore, params: ConsensusParams,
                   segment_ids: Sequence[int] | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ids = sorted(store.segments) if segment_ids is None else [int(i) for i in segment_ids]
    n_all = {i: store.segments[i].num_points for i in ids}
    ids = [i for i in ids if n_all[i] > 0]
    conf = np.zeros(0, np.int32)
    free = np.zeros(0, np.int32)
    st = None
    if ids and store.keyframes:
        st = _stack(store, ids)
        conf = np.zeros(st.pts.shape[0], np.int32)
        free = np.zeros(st.pts.shape[0], np.int32)
        for kf in store.keyframes:
            r = _project_keyframe(st, kf)
            if r is None:
                continue
            pi_, _, _, z, d = r
            conf[pi_[np.abs(z - d) < params.tau_d]] += 1
            free[pi_[z < d - params.ghost_tau]] += 1
    out = {i: (np.zeros(n_all[i], np.int32), np.zeros(n_all[i], np.int32)) for i in n_all}
    if st is not None:
        for k, i in enumerate(st.ids):
            o, n = int(st.offs[k]), int(st.npts[k])
            out[i] = (conf[o:o + n].copy(), free[o:o + n].copy())
    return out
