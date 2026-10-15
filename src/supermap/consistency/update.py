"""Per-frame probabilistic geometric consistency update (Sec. IV-B3, Supp. Alg. 1).

Frame order (Alg. 1): CompareDepth (lines 7-9) -> tracker gives T_trk (M3) ->
semantic check + ObjNegate (lines 21-27) -> Eq. (8) on the remaining points -> pruning.
Each map point receives exactly one Eq. (8) update per frame (D-051):
    - objects in O_neg: ObjNegate, p_dis on in-view non-occluded points;
    - all other objects: inverse sensor model of their per-point state (Eq. 9).
Without T_trk (no segmentation this frame) only the per-point geometric update runs (D-052).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from supermap.consistency.compare_depth import CompareDepthResult, compare_depth
from supermap.consistency.existence import existence_prune, existence_update
from supermap.consistency.logodds import UpdateResult, apply_point_update, inverse_sensor_prob, prune_objects
from supermap.consistency.negate import obj_negate, semantic_consistency
from supermap.consistency.params import ConsistencyParams, as_params
from supermap.map import ObjectInstance, ObjectStatus
from supermap.types import Observation


# REQ-032, REQ-035, REQ-036: outcome of one consistency update
@dataclass
class ConsistencyResult:
    cd: CompareDepthResult
    o_neg: list = field(default_factory=list)
    update: UpdateResult = field(default_factory=UpdateResult)

    @property
    def removed_ids(self) -> list[int]:
        return [e.id for e in self.update.removed]


# REQ-033, REQ-034, REQ-035: Eq. (8) per point with P(o_k|Q_t) from Eq. (9) states, then pruning
def geometric_update(o_nbr: Iterable[ObjectInstance], obs: Observation,
                     params: ConsistencyParams | Mapping | None = None,
                     cd: CompareDepthResult | None = None, skip_ids: Iterable[int] = (),
                     do_prune: bool = True) -> ConsistencyResult:
    """o_nbr: objects (list or GlobalMap). Objects in skip_ids get no update here."""
    p = as_params(params)
    objs = list(o_nbr)
    if cd is None:
        cd = compare_depth(obs, objs, p)
    skip = {int(i) for i in skip_ids}
    for obj in objs:
        if obj.status != ObjectStatus.ACTIVE or obj.id in skip or obj.id not in cd.states:
            continue
        apply_point_update(obj, inverse_sensor_prob(cd.states[obj.id], p), p)
    # REQ-035: D-113 one existence update per object; O_neg = skip set (semantic) plus geometric O_neg
    # REQ-035, REQ-036: existence evidence = geometric O_front (cd.o_neg) plus, if exist_semantic, the semantic O_neg (skip)
    existence_update(objs, cd, {o.id for o in cd.o_neg} | (skip if p.exist_semantic else set()), p)
    res = ConsistencyResult(cd=cd, o_neg=list(cd.o_neg))
    if do_prune:
        prune_objects(objs, p, obs.t, res.update)
        existence_prune(objs, p, obs.t, res.update)  # REQ-035: after point pruning, no double report
    return res


# REQ-036, REQ-035: Alg. 1 lines 21-27 plus Eq. (8) and pruning for one frame
# REQ-037: disappeared objects are pruned to REMOVED with ID kept; REQ-038: occluded/out-of-view points get no update
# REQ-039: stale points of moved objects are pruned (geometric), untracked visible points too (semantic)
def consistency_update(o_nbr: Iterable[ObjectInstance], obs: Observation,
                       params: ConsistencyParams | Mapping | None = None,
                       t_trk_ids: Iterable[int] | None = None,
                       cd: CompareDepthResult | None = None) -> ConsistencyResult:
    """t_trk_ids: IDs associated by the tracker this frame; None = no segmentation this frame."""
    p = as_params(params)
    objs = list(o_nbr)
    if cd is None:
        cd = compare_depth(obs, objs, p)
    if t_trk_ids is None:
        return geometric_update(objs, obs, p, cd=cd)
    o_neg = semantic_consistency(cd.o_on, t_trk_ids, cd.o_neg)
    obj_negate(objs, o_neg, p, cd.states)
    res = geometric_update(objs, obs, p, cd=cd, skip_ids=[o.id for o in o_neg], do_prune=False)
    res.o_neg = o_neg
    prune_objects(objs, p, obs.t, res.update)
    existence_prune(objs, p, obs.t, res.update)  # REQ-035: D-113
    return res
