"""Log-odds occupancy update (Eq. 8), inverse sensor model, dynamic penalty and pruning.

Symbol map (paper -> code):
    L(o_k | Q_1:t) -> ObjectInstance.log_odds (n,)
    L(n)           -> logit(n) = log(n / (1 - n))
    P(o_k | Q_t)   -> inverse_sensor_prob(states): p_obs / p_dis / p_unobs by s_k^(t)
    L_prune        -> points with L < L_prune are removed (REQ-035)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from supermap.consistency.classify import PointState
from supermap.consistency.params import ConsistencyParams, as_params
from supermap.map import ObjectInstance, ObjectStatus


# REQ-033: L(n) = log(n / (1 - n))
def logit(p):
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("probabilities must be in (0, 1)")
    out = np.log(p / (1.0 - p))
    return float(out) if out.ndim == 0 else out


# REQ-033: probability recovered from log-odds, p = 1 / (1 + exp(-L))
def prob_from_log_odds(L):
    out = 1.0 / (1.0 + np.exp(-np.asarray(L, dtype=np.float64)))
    return float(out) if out.ndim == 0 else out


# REQ-034: inverse sensor model P(o_k | Q_t) from thresholded states (Eq. 9)
def inverse_sensor_prob(states: np.ndarray, params: ConsistencyParams | dict | None = None) -> np.ndarray:
    """(n,) PointState -> (n,) probability; NOT_IN_VIEW -> 0.5 (logit 0, no update)."""
    p = as_params(params)
    s = np.asarray(states).reshape(-1)
    out = np.full(s.shape, 0.5, dtype=np.float64)
    out[s == PointState.OBSERVABLE] = p.p_obs
    out[s == PointState.DISAPPEARED] = p.p_dis
    out[s == PointState.UNOBSERVABLE] = p.p_unobs
    return out


# REQ-033: Eq. (8) L_t = L_{t-1} + logit(P(o_k | Q_t)); optional symmetric clamp (D-048)
def logodds_update(L_prev, p, clamp: float | None = None):
    L = np.asarray(L_prev, dtype=np.float64) + logit(p)
    if clamp is not None:
        L = np.clip(L, -clamp, clamp)
    return float(L) if np.ndim(L) == 0 else L


# REQ-035: record kept when an object loses all points (history for REQ-037)
@dataclass(frozen=True)
class RemovalEvent:
    id: int
    t: int
    centroid: np.ndarray
    bbox: tuple
    last_seen: int
    reason: str = "points"  # REQ-035: "points" (all points pruned) or "existence" (D-113)


# REQ-035: remove points with L < L_prune; an object left without points becomes REMOVED
def prune(obj: ObjectInstance, L_prune: float, t: int = -1) -> tuple[int, RemovalEvent | None]:
    """Returns (number of pruned points, RemovalEvent if the object was removed). ID is kept."""
    if obj.status != ObjectStatus.ACTIVE or obj.num_points == 0:
        return 0, None
    drop = obj.log_odds < L_prune
    n_drop = int(np.count_nonzero(drop))
    if n_drop == 0:
        return 0, None
    event = None
    if n_drop == obj.num_points:
        event = RemovalEvent(obj.id, int(t), obj.centroid.copy(), obj.bbox, obj.last_seen)
    obj.remove_points(drop)
    if obj.num_points == 0:
        obj.status = ObjectStatus.REMOVED
    return n_drop, event


# REQ-033, REQ-034: apply Eq. (8) to one object's points given per-point probabilities
def apply_point_update(obj: ObjectInstance, p_k: np.ndarray, params: ConsistencyParams) -> None:
    obj.log_odds = np.asarray(logodds_update(obj.log_odds, p_k, params.L_clamp), dtype=np.float64).reshape(-1)


# REQ-035: summary of one geometric update
@dataclass
class UpdateResult:
    pruned: dict[int, int] = field(default_factory=dict)       # id -> pruned point count
    removed: list[RemovalEvent] = field(default_factory=list)  # objects that became REMOVED

    @property
    def num_pruned(self) -> int:
        return int(sum(self.pruned.values()))


# REQ-035: prune every object in a collection, collecting counts and removal events
def prune_objects(objects: Iterable[ObjectInstance], params: ConsistencyParams | dict | None = None,
                  t: int = -1, result: UpdateResult | None = None) -> UpdateResult:
    p = as_params(params)
    res = result if result is not None else UpdateResult()
    for obj in objects:
        n, ev = prune(obj, p.L_prune, t)
        if n:
            res.pruned[obj.id] = res.pruned.get(obj.id, 0) + n
        if ev is not None:
            res.removed.append(ev)
    return res
