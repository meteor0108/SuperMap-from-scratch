"""Object-level existence log-odds L_exist (D-113; abstract: existence-and-label confidence update).

One update per ACTIVE instance per frame in which it has an object-level state (D-047):
    DISAPPEARED or member of O_neg -> += logit(p_dis);  OBSERVABLE -> += logit(p_obs);
    UNOBSERVABLE / no in-view point -> unchanged. L_exist < exist_prune -> REMOVED like an
    instance that lost all points (id kept, RemovalEvent for the temporal graph, D-053).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from supermap.consistency.classify import PointState
from supermap.consistency.compare_depth import CompareDepthResult
from supermap.consistency.logodds import RemovalEvent, UpdateResult, logodds_update
from supermap.consistency.params import ConsistencyParams
from supermap.map import ObjectInstance, ObjectStatus


# REQ-035: I-9-4 mitigation, no decision recorded; enough in-view points (count and share) for the object-level verdict to count
def _covered(obj: ObjectInstance, cd: CompareDepthResult, params: ConsistencyParams) -> bool:
    if params.exist_min_points <= 0 and params.exist_min_share <= 0.0:
        return True
    s = cd.states.get(obj.id)
    if s is None:
        return True
    n_in = int(np.count_nonzero(np.asarray(s) != PointState.NOT_IN_VIEW))
    return n_in >= params.exist_min_points and n_in >= params.exist_min_share * max(len(s), 1)


# REQ-033, REQ-035: Eq. (8)-style scalar update of the instance existence log-odds (D-113)
def existence_update(objs: Iterable[ObjectInstance], cd: CompareDepthResult, o_neg_ids: Iterable[int],
                     params: ConsistencyParams) -> None:
    if params.exist_prune is None:  # REQ-035: rule disabled, previous behaviour
        return
    neg = {int(i) for i in o_neg_ids}
    for obj in objs:
        if obj.status != ObjectStatus.ACTIVE or obj.id not in cd.object_state:
            continue
        if not _covered(obj, cd, params):  # REQ-035: I-9-4 mitigation, no decision recorded coverage gate
            continue
        state = cd.object_state[obj.id]
        if obj.id in neg or state == PointState.DISAPPEARED:  # REQ-036: negative evidence
            p = params.p_dis
        elif state == PointState.OBSERVABLE:  # REQ-034: positive evidence
            p = params.p_obs
        else:  # REQ-038: unobservable / out of view -> no update
            continue
        obj.exist_log_odds = float(logodds_update(obj.exist_log_odds, p, params.exist_clamp))


# REQ-035, REQ-037: remove instances whose existence log-odds fell below exist_prune (id kept)
def existence_prune(objs: Iterable[ObjectInstance], params: ConsistencyParams, t: int,
                    result: UpdateResult) -> None:
    if params.exist_prune is None:  # REQ-035: rule disabled
        return
    for obj in objs:
        if obj.status != ObjectStatus.ACTIVE or not obj.exist_log_odds < params.exist_prune:
            continue
        if obj.num_points > 0:  # REQ-039: event carries geometry before the drop
            ev = RemovalEvent(obj.id, int(t), obj.centroid.copy(), obj.bbox, obj.last_seen, "existence")
            result.pruned[obj.id] = result.pruned.get(obj.id, 0) + obj.num_points
            obj.remove_points(np.ones(obj.num_points, dtype=bool))
        else:  # REQ-037: pointless ACTIVE record; no geometry to report
            ev = RemovalEvent(obj.id, int(t), np.zeros(3), (np.zeros(3), np.zeros(3)), obj.last_seen, "existence")
        obj.status = ObjectStatus.REMOVED
        result.removed.append(ev)
