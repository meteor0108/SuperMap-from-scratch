"""Greedy consensus clustering of segments into instances (D-124).

Foreground segments are merged by view consensus evaluated at cluster level (histograms of a cluster are
the sum of its members'); background segments are merged by voxel overlap.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from scipy.spatial import cKDTree

from supermap.geometry.dense_map import voxel_keys
from supermap.segments.consensus import PairEvidence, candidate_pairs, pair_evidence
from supermap.segments.params import ConsensusParams
from supermap.segments.store import Segment, SegmentStore
from supermap.segments.visibility import FrameHist, contained_mask, is_visible, merge_hists

_EPS = 1e-9


# REQ-024 (extension): clusters of segment ids plus diagnostics
@dataclass
class ClusterResult:
    clusters: list[list[int]]          # segment ids per instance, foreground first then background
    background: list[bool]             # per cluster
    merges: int = 0
    rounds: int = 0
    pair_log: list[tuple[int, int, int, int, float]] = field(default_factory=list)  # (a, b, support, conflict, theta)
    dropped: list[list[int]] = field(default_factory=list)  # clusters below min_cluster_points
    fallback_merges: int = 0   # merges made by the contact fallback (theta -1.0 in pair_log)


# REQ-024 (extension): background iff the argmax of the segment label vote is a background class
def is_background_segment(seg: Segment, num_classes: int, background: tuple[int, ...]) -> bool:
    if not background:
        return False
    v = seg.label_scores(num_classes)
    return bool(v.sum() > 0 and int(np.argmax(v)) in set(background))


# REQ-024 (extension): hist of a cluster = elementwise sum over members and keyframes
def _merge_cluster_hists(ha: dict[int, FrameHist], hb: dict[int, FrameHist]) -> dict[int, FrameHist]:
    out = {f: merge_hists(ha.get(f), hb.get(f)) for f in set(ha) | set(hb)}
    return {f: h for f, h in out.items() if h is not None}


# REQ-024 (extension): masks that contain >= 2 clusters already known to be distinct
def _distrust_set(hist: dict[int, dict[int, FrameHist]], p: ConsensusParams) -> set[tuple[int, int]]:
    by_slot: dict[tuple[int, int], list[int]] = {}
    for c, hc in hist.items():
        for f, h in hc.items():
            if is_visible(h.n_points, h.n_vis, p):
                m = contained_mask(h.n_vis, h.counts, p)
                if m >= 0:
                    by_slot.setdefault((f, m), []).append(c)
    known: dict[tuple[int, int], bool] = {}
    out: set[tuple[int, int]] = set()
    for slot, cs in by_slot.items():
        if len(cs) < 2:
            continue
        cs = sorted(cs)
        for i, a in enumerate(cs):
            if slot in out:
                break
            for b in cs[i + 1:]:
                if (a, b) not in known:
                    ev = pair_evidence(hist[a], hist[b], p)
                    known[(a, b)] = ev.evidence >= p.k_min and ev.consensus < 0.5
                if known[(a, b)]:
                    out.add(slot)
                    break
    return out


# REQ-024 (extension): merge cluster b into a (a < b): members, histograms, adjacency
def _merge_pair(a: int, b: int, members: dict, hist: dict, nbrs: dict) -> None:
    members[a].extend(members.pop(b))
    hist[a] = _merge_cluster_hists(hist[a], hist.pop(b))
    nb = (nbrs[a] | nbrs[b]) - {a, b}
    for n in nbrs.pop(b):
        nbrs[n].discard(b)
    nbrs[a] = nb
    for n in nb:
        nbrs[n].add(a)


# REQ-024 (extension): theta-scheduled greedy merging with cluster-level evidence
def _cluster_foreground(store: SegmentStore, hists: dict[int, dict[int, FrameHist]], ids: list[int],
                        p: ConsensusParams, distrust: set[tuple[int, int]] | None):
    hist = {i: dict(hists.get(i, {})) for i in ids}
    members = {i: [i] for i in ids}
    counter = itertools.count(1)
    version = {i: 0 for i in ids}
    nbrs: dict[int, set[int]] = {i: set() for i in ids}
    for a, b in candidate_pairs(store, ids, p.near_dist):
        nbrs[a].add(b)
        nbrs[b].add(a)
    cache: dict[tuple[int, int], tuple[int, int, PairEvidence]] = {}
    merges = rounds = 0
    log: list[tuple[int, int, int, int, float]] = []

    def evidence(a: int, b: int) -> PairEvidence:
        hit = cache.get((a, b))
        if hit is not None and hit[0] == version[a] and hit[1] == version[b]:
            return hit[2]
        ev = pair_evidence(hist[a], hist[b], p, distrust)
        cache[(a, b)] = (version[a], version[b], ev)
        return ev

    for theta in p.thetas:
        while True:
            rounds += 1
            cands = []
            for a in sorted(nbrs):
                for b in nbrs[a]:
                    if b > a:
                        ev = evidence(a, b)
                        if ev.evidence >= p.k_min and ev.consensus >= theta - _EPS:
                            cands.append((-ev.consensus, -ev.evidence, a, b))
            if not cands:
                break
            cands.sort()
            touched: set[int] = set()
            n_round = 0
            for _, _, a, b in cands:
                if a in touched or b in touched:
                    continue
                ev = evidence(a, b)
                touched.update((a, b))
                _merge_pair(a, b, members, hist, nbrs)
                version[a] = next(counter)
                log.append((a, b, ev.support, ev.conflict, float(theta)))
                n_round += 1
            merges += n_round
            if n_round == 0:
                break
    return members, hist, nbrs, merges, rounds, log


# REQ-024 (extension): geometric fallback for pairs the consensus could not decide (evidence < k_min)
def _contact_fallback(store: SegmentStore, members: dict, hist: dict, nbrs: dict, p: ConsensusParams,
                      num_classes: int, distrust: set[tuple[int, int]] | None,
                      log: list[tuple[int, int, int, int, float]]) -> int:
    counter = itertools.count(1)
    version = {c: 0 for c in members}
    pts: dict[int, np.ndarray] = {}
    votes: dict[int, np.ndarray] = {}

    def cluster_pts(c: int) -> np.ndarray:
        if c not in pts:
            pts[c] = np.concatenate([store.segments[i].points for i in members[c]]).astype(np.float64)
        return pts[c]

    def cluster_vote(c: int) -> np.ndarray:
        if c not in votes:
            votes[c] = np.sum([store.segments[i].label_scores(num_classes) for i in members[c]], axis=0)
        return votes[c]

    def labels_agree(a: int, b: int) -> bool:
        va, vb = cluster_vote(a), cluster_vote(b)
        if va.sum() <= 0 or vb.sum() <= 0:
            return True
        return int(np.argmax(va)) == int(np.argmax(vb))

    def contact(a: int, b: int) -> float:
        pa, pb = cluster_pts(a), cluster_pts(b)
        small, big = (pa, pb) if (len(pa), a) <= (len(pb), b) else (pb, pa)
        if len(big) > 5000:
            big = big[:: -(-len(big) // 5000)]
        d, _ = cKDTree(big).query(small, k=1, distance_upper_bound=p.near_dist)
        return float(np.mean(np.isfinite(d)))

    ev_cache: dict[tuple[int, int], tuple[int, int, PairEvidence]] = {}
    ct_cache: dict[tuple[int, int], tuple[int, int, float]] = {}
    total = 0
    while True:
        cands = []
        for a in sorted(nbrs):
            for b in nbrs[a]:
                if b <= a:
                    continue
                hit = ev_cache.get((a, b))
                if hit is None or hit[0] != version[a] or hit[1] != version[b]:
                    hit = (version[a], version[b], pair_evidence(hist[a], hist[b], p, distrust))
                    ev_cache[(a, b)] = hit
                if hit[2].evidence >= p.k_min or not labels_agree(a, b):
                    continue
                ch = ct_cache.get((a, b))
                if ch is None or ch[0] != version[a] or ch[1] != version[b]:
                    ch = (version[a], version[b], contact(a, b))
                    ct_cache[(a, b)] = ch
                if ch[2] >= p.fallback_contact - _EPS:
                    cands.append((-ch[2], a, b, hit[2]))
        if not cands:
            return total
        cands.sort(key=lambda x: x[:3])
        touched: set[int] = set()
        for _, a, b, ev in cands:
            if a in touched or b in touched:
                continue
            touched.update((a, b))
            _merge_pair(a, b, members, hist, nbrs)
            version[a] = next(counter)
            pts.pop(a, None), pts.pop(b, None), votes.pop(a, None), votes.pop(b, None)
            log.append((a, b, ev.support, ev.conflict, -1.0))
            total += 1


# REQ-024 (extension): background segments linked by voxel overlap of the smaller one
def _cluster_background(store: SegmentStore, ids: list[int], p: ConsensusParams) -> list[list[int]]:
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = {i: np.unique(voxel_keys(store.segments[i].points, p.voxel)) for i in ids}
    for a, b in candidate_pairs(store, ids, p.near_dist):
        small = min(keys[a].size, keys[b].size)
        if small == 0:
            continue
        inter = np.intersect1d(keys[a], keys[b], assume_unique=True).size
        if inter / small >= p.bg_overlap:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


# REQ-024 (extension): full clustering of a segment store
def consensus_cluster(store: SegmentStore, hists: dict[int, dict[int, FrameHist]], params: ConsensusParams,
                      num_classes: int) -> ClusterResult:
    fg_ids, bg_ids = [], []
    for sid in sorted(store.segments):
        seg = store.segments[sid]
        if seg.num_points == 0:
            continue
        (bg_ids if is_background_segment(seg, num_classes, params.background) else fg_ids).append(sid)

    members, hist, nbrs, merges, rounds, log = _cluster_foreground(store, hists, fg_ids, params, None)
    distrust = None
    if params.distrust_split_masks and fg_ids:
        distrust = _distrust_set(hist, params) or None
        if distrust:
            members, hist, nbrs, merges, rounds, log = _cluster_foreground(store, hists, fg_ids, params,
                                                                           distrust)
    n_fallback = 0
    if params.fallback == "contact" and fg_ids:
        n_fallback = _contact_fallback(store, members, hist, nbrs, params, num_classes, distrust, log)
    fg = [sorted(m) for m in members.values()]
    bg = _cluster_background(store, bg_ids, params) if bg_ids else []

    def order(cs: list[list[int]]) -> list[list[int]]:
        return sorted(cs, key=lambda c: (min(store.segments[i].t for i in c), min(c)))

    kept_fg, kept_bg, dropped = [], [], []
    for cs, kept in ((order(fg), kept_fg), (order(bg), kept_bg)):
        for cl in cs:
            if sum(store.segments[i].num_points for i in cl) < params.min_cluster_points:
                dropped.append(cl)
            else:
                kept.append(cl)
    return ClusterResult(clusters=kept_fg + kept_bg, background=[False] * len(kept_fg) + [True] * len(kept_bg),
                         merges=merges, rounds=rounds, pair_log=log, dropped=dropped,
                         fallback_merges=n_fallback)
