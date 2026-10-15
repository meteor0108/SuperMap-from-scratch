"""Semantic consistency check and ObjNegate (Supp. Alg. 1 lines 21-27, Supp. VIII-A).

Symbol map (paper -> code):
    O_on   -> o_on, geometrically visible objects from CompareDepth
    T_trk  -> t_trk_ids, set of int track IDs associated in this frame (track ID == instance ID)
    O_neg  -> o_neg, objects receiving negative evidence
    ObjNegate(O_nbr, O_neg) -> obj_negate: Eq. (8) with P(o_k|Q_t) = p_dis on the negated
        objects' in-view, non-occluded points (D-050)
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from supermap.consistency.classify import PointState
from supermap.consistency.logodds import apply_point_update
from supermap.consistency.params import ConsistencyParams, as_params
from supermap.map import ObjectInstance, ObjectStatus


# REQ-036: identifier of an object given as ObjectInstance or int ID
def _oid(o) -> int:
    return int(o.id) if hasattr(o, "id") else int(o)


# REQ-036: for O in O_on, if O.id not in T_trk then O_neg <- O_neg U {O}
def semantic_consistency(o_on: Iterable, t_trk_ids: Iterable[int], o_neg: Iterable) -> list:
    """Returns the extended O_neg (O_front members first, then semantic ones in O_on order)."""
    trk = {int(i) for i in t_trk_ids}
    out = list(o_neg)
    seen = {_oid(o) for o in out}
    for o in o_on:
        i = _oid(o)
        if i not in trk and i not in seen:
            out.append(o)
            seen.add(i)
    return out


# REQ-036: points of a negated object that receive negative evidence (D-050)
def negated_point_mask(obj: ObjectInstance, states: np.ndarray | None) -> np.ndarray:
    """In-view points that are not occluded (Observable or Disappeared); all points if states is None."""
    if states is None:
        return np.ones(obj.num_points, dtype=bool)
    s = np.asarray(states).reshape(-1)
    if s.shape[0] != obj.num_points:
        raise ValueError(f"states length {s.shape[0]} != points of object {obj.id}")
    return (s == PointState.OBSERVABLE) | (s == PointState.DISAPPEARED)


# REQ-036: O_nbr <- ObjNegate(O_nbr, O_neg); one logit(p_dis) decrement per selected point
def obj_negate(o_nbr: Iterable[ObjectInstance], o_neg: Iterable,
               params: ConsistencyParams | Mapping | None = None,
               states: Mapping[int, np.ndarray] | None = None) -> list[ObjectInstance]:
    """Apply negative evidence to members of O_nbr whose ID is in O_neg; returns O_nbr (modified in place).

    states: id -> (n,) PointState from CompareDepth; restricts the update to in-view,
    non-occluded points. Without it every point of a negated object is updated.
    """
    p = as_params(params)
    neg_ids = {_oid(o) for o in o_neg}
    objs = list(o_nbr)
    for obj in objs:
        if obj.id not in neg_ids or obj.status != ObjectStatus.ACTIVE:
            continue
        sel = negated_point_mask(obj, None if states is None else states.get(obj.id))
        p_k = np.where(sel, p.p_dis, 0.5)  # logit(0.5) = 0: unselected points unchanged
        apply_point_update(obj, p_k, p)
    return objs
