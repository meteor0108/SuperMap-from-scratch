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


# REQ-024 (extension): sparse seg_id -> {keyframe index -> FrameHist} for all keyframes where n_vis > 0
def segment_histograms(store: SegmentStore, params: ConsensusParams,
                       segment_ids: Sequence[int] | None = None) -> dict[int, dict[int, FrameHist]]:
    ids = sorted(store.segments) if segment_ids is None else [int(i) for i in segment_ids]
    out: dict[int, dict[int, FrameHist]] = {i: {} for i in ids}
    ids = [i for i in ids if store.segments[i].num_points > 0]
    if not ids or not store.keyframes:
        return out
    segs = [store.segments[i] for i in ids]
    n_seg = len(ids)
    npts = np.array([s.num_points for s in segs], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(npts)[:-1]])
    all_pts = np.concatenate([s.points for s in segs]).astype(np.float64)
    seg_of = np.repeat(np.arange(n_seg), npts)
    lo = np.minimum.reduceat(all_pts, offs, axis=0)
    hi = np.maximum.reduceat(all_pts, offs, axis=0)
    corners = (lo[:, None, :] + _CORNER_SIGN[None] * (hi - lo)[:, None, :]).reshape(-1, 3)

    for fi, kf in enumerate(store.keyframes):
        h, w = kf.shape
        W, H = w * kf.scale, h * kf.scale
        # frustum pre-check on the 8 bbox corners
        uvc, zc, _ = project(corners, kf.pose, kf.K)
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
            continue
        sel = np.flatnonzero(cand[seg_of])
        uv, z, ok_z = project(all_pts[sel], kf.pose, kf.K)
        vi = np.flatnonzero(ok_z)
        if vi.size == 0:
            continue
        pix = pixel_index(uv[vi])
        u, v = pix[:, 0] // kf.scale, pix[:, 1] // kf.scale
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        vi, u, v = vi[inb], u[inb], v[inb]
        good = kf.valid[v, u] & (np.abs(z[vi] - kf.depth[v, u]) < params.tau_d)
        vi, u, v = vi[good], u[good], v[good]
        if vi.size == 0:
            continue
        sl = seg_of[sel[vi]]
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
            out[ids[s]][fi] = FrameHist(int(npts[s]), int(n_vis[s]), cdict.get(s, {}))
    return out
