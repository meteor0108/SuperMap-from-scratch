"""Pairwise view-consensus evidence between segments or clusters (D-124)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from supermap.segments.params import ConsensusParams
from supermap.segments.store import SegmentStore
from supermap.segments.visibility import FrameHist, contained_mask, is_visible


# REQ-024 (extension): support / conflict counters of a pair
@dataclass
class PairEvidence:
    support: int = 0
    conflict: int = 0

    @property
    def evidence(self) -> int:
        return self.support + self.conflict

    @property
    def consensus(self) -> float:
        return self.support / self.evidence if self.evidence > 0 else 0.0


# REQ-024 (extension): pairs (a < b) whose bboxes, expanded by near_dist, overlap
def candidate_pairs(store: SegmentStore, segment_ids: Sequence[int], near_dist: float) -> list[tuple[int, int]]:
    ids = np.array(sorted(int(i) for i in segment_ids if store.segments[int(i)].num_points > 0), dtype=np.int64)
    n = ids.shape[0]
    if n < 2:
        return []
    lo = np.empty((n, 3))
    hi = np.empty((n, 3))
    for k, i in enumerate(ids.tolist()):
        lo[k], hi[k] = store.segments[i].bbox
    pairs: list[tuple[int, int]] = []
    for k in range(n - 1):
        near = np.all((lo[k + 1:] <= hi[k] + near_dist) & (lo[k] <= hi[k + 1:] + near_dist), axis=1)
        for j in np.flatnonzero(near).tolist():
            pairs.append((int(ids[k]), int(ids[k + 1 + j])))
    return pairs


# REQ-024 (extension): support if both are contained in the same mask, conflict if in different masks
def pair_evidence(hist_a: dict[int, FrameHist], hist_b: dict[int, FrameHist], p: ConsensusParams,
                  distrust: set[tuple[int, int]] | None = None) -> PairEvidence:
    ev = PairEvidence()
    small, big = (hist_a, hist_b) if len(hist_a) <= len(hist_b) else (hist_b, hist_a)
    for f in small:
        if f not in big:
            continue
        ha, hb = hist_a[f], hist_b[f]
        if not (is_visible(ha.n_points, ha.n_vis, p) and is_visible(hb.n_points, hb.n_vis, p)):
            continue
        ma = contained_mask(ha.n_vis, ha.counts, p)
        mb = contained_mask(hb.n_vis, hb.counts, p)
        if ma < 0 or mb < 0:
            continue
        if ma == mb:
            if distrust is None or (f, ma) not in distrust:
                ev.support += 1
        elif p.partial_share > 0 and (ha.counts.get(mb, 0) >= p.partial_share * hb.n_vis
                                      or hb.counts.get(ma, 0) >= p.partial_share * ha.n_vis):
            continue    # one is a part of the other: neither support nor conflict
        else:
            ev.conflict += 1
    return ev
