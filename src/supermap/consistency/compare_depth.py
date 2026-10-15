"""Object-level CompareDepth (Supp. Alg. 1 lines 7-9).

Symbol map (paper -> code):
    O_nbr     -> o_nbr, list of ObjectInstance (neighbor objects of M_{t-1})
    D_I, Odom -> obs.depth, obs.pose (P_t = T_WC), obs.K
    O_front   -> CompareDepthResult.o_front  (majority Disappeared: point in front of surface)
    O_on      -> CompareDepthResult.o_on     (majority Observable: on the surface)
    O_behind  -> CompareDepthResult.o_behind (majority Unobservable: occluded)
    O_neg     -> CompareDepthResult.o_neg    initialized as a copy of O_front
Per-point states (Eq. 9) are kept in `states` so Eq. (8) and ObjNegate reuse them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import cv2
import numpy as np

from supermap.consistency.classify import PointState, classify
from supermap.consistency.params import ConsistencyParams, as_params
from supermap.consistency.residual import depth_residual
from supermap.map import ObjectInstance, ObjectStatus
from supermap.types import Observation

# Tie-break order for the object-level majority rule (D-047): conservative states first.
_TIE_ORDER = (PointState.OBSERVABLE, PointState.UNOBSERVABLE, PointState.DISAPPEARED)


# REQ-032: result of CompareDepth; unpacks as (o_front, o_on, o_behind)
@dataclass
class CompareDepthResult:
    o_front: list[ObjectInstance] = field(default_factory=list)
    o_on: list[ObjectInstance] = field(default_factory=list)
    o_behind: list[ObjectInstance] = field(default_factory=list)
    o_neg: list[ObjectInstance] = field(default_factory=list)
    states: dict[int, np.ndarray] = field(default_factory=dict)       # id -> (n,) PointState
    delta_d: dict[int, np.ndarray] = field(default_factory=dict)      # id -> (n,) Eq. 7 residual
    object_state: dict[int, PointState] = field(default_factory=dict)  # id -> object-level state
    verdict_states: dict[int, np.ndarray] = field(default_factory=dict)  # id -> (n,) window-tolerant states (I-9-4)

    def __iter__(self) -> Iterator[list[ObjectInstance]]:
        return iter((self.o_front, self.o_on, self.o_behind))


# REQ-032: object-level state = majority of in-view point states (D-047)
def object_state(states: np.ndarray) -> PointState:
    s = np.asarray(states).reshape(-1)
    s = s[s != PointState.NOT_IN_VIEW]
    if s.size == 0:
        return PointState.NOT_IN_VIEW
    c = np.bincount(s.astype(np.int64), minlength=int(max(PointState)) + 1)
    best = max(int(c[st]) for st in _TIE_ORDER)
    if best == 0:
        return PointState.NOT_IN_VIEW
    return next(st for st in _TIE_ORDER if int(c[st]) == best)


# REQ-029, REQ-031: per-point Eq. (7) residuals and Eq. (9) states of one object
def point_states(obj: ObjectInstance, obs: Observation, tau_eps: float):
    delta_d, mask, _ = depth_residual(obj.points, obs.pose, obs.K, obs.depth)
    return delta_d, classify(delta_d, mask, tau_eps)


# REQ-031, REQ-032: I-9-4 mitigation, no decision recorded. DISAPPEARED points whose depth is consistent with some valid depth in the
# (2r+1)^2 window around their pixel count as OBSERVABLE for the object-level verdict only
def tolerant_states(st: np.ndarray, dd: np.ndarray, pix: np.ndarray, depth: np.ndarray,
                    tau_eps: float, r: int) -> np.ndarray:
    out = st.copy()
    if r <= 0:
        return out
    cand = np.flatnonzero(st == PointState.DISAPPEARED)
    if cand.size == 0:
        return out
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    lo = cv2.erode(np.where(valid, d, np.float32(np.inf)), k, borderType=cv2.BORDER_REPLICATE)
    hi = cv2.dilate(np.where(valid, d, np.float32(-np.inf)), k, borderType=cv2.BORDER_REPLICATE)
    u, v = pix[cand, 0], pix[cand, 1]
    z = dd[cand] + d[v, u].astype(np.float64)
    ok = (z >= lo[v, u] - tau_eps) & (z <= hi[v, u] + tau_eps)
    out[cand[ok]] = PointState.OBSERVABLE
    return out


# REQ-032: (O_front, O_on, O_behind) <- CompareDepth(D_I, O_nbr, Odom); O_neg <- O_front
def compare_depth(obs: Observation, o_nbr: Iterable[ObjectInstance],
                  params: ConsistencyParams | dict | None = None) -> CompareDepthResult:
    """Split neighbor objects by depth consistency against the current depth image.

    Only ACTIVE objects are considered. Objects with no in-view point are in no set.
    """
    p = as_params(params)
    res = CompareDepthResult()
    objs = [o for o in o_nbr if o.status == ObjectStatus.ACTIVE]
    # REQ-032: one projection for all O_nbr points (D-103)
    sizes = [o.num_points for o in objs]
    if sum(sizes) > 0:
        X = np.concatenate([o.points for o in objs])
        dd, mask, pix = depth_residual(X, obs.pose, obs.K, obs.depth)
        st_all = classify(dd, mask, p.tau_eps)
        vs_all = tolerant_states(st_all, dd, pix, obs.depth, p.tau_eps, p.verdict_window_px)  # I-9-4
        cuts = np.cumsum(sizes)[:-1]
        dd_parts, st_parts, vs_parts = np.split(dd, cuts), np.split(st_all, cuts), np.split(vs_all, cuts)
    else:
        dd_parts = [np.zeros(0, np.float64) for _ in objs]
        st_parts = [np.zeros(0, np.int8) for _ in objs]
        vs_parts = st_parts
    for obj, delta_d, s, vs in zip(objs, dd_parts, st_parts, vs_parts):
        res.states[obj.id] = s
        res.delta_d[obj.id] = delta_d
        res.verdict_states[obj.id] = vs
        st = object_state(vs)
        res.object_state[obj.id] = st
        if st == PointState.DISAPPEARED:
            res.o_front.append(obj)
        elif st == PointState.OBSERVABLE:
            res.o_on.append(obj)
        elif st == PointState.UNOBSERVABLE:
            res.o_behind.append(obj)
    res.o_neg = list(res.o_front)  # Alg. 1 line 9: inconsistent depth
    return res
