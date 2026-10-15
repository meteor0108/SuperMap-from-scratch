"""Probabilistic geometric consistency update (Sec. IV-B3, Eq. 7-9; Supp. Alg. 1 lines 7-9, 21-27)."""
# REQ-029..REQ-036: public API of the consistency layer
from supermap.consistency.params import ConsistencyParams, as_params
from supermap.consistency.residual import depth_residual, gaussian_likelihood
from supermap.consistency.classify import PointState, classify
from supermap.consistency.compare_depth import CompareDepthResult, compare_depth, object_state, point_states
from supermap.consistency.logodds import (RemovalEvent, UpdateResult, inverse_sensor_prob, logit,
                                          logodds_update, prob_from_log_odds, prune, prune_objects)
from supermap.consistency.negate import negated_point_mask, obj_negate, semantic_consistency
from supermap.consistency.update import ConsistencyResult, consistency_update, geometric_update

__all__ = [
    "ConsistencyParams", "as_params", "depth_residual", "gaussian_likelihood", "PointState", "classify",
    "CompareDepthResult", "compare_depth", "object_state", "point_states", "RemovalEvent", "UpdateResult",
    "inverse_sensor_prob", "logit", "logodds_update", "prob_from_log_odds", "prune", "prune_objects",
    "negated_point_mask", "obj_negate", "semantic_consistency", "ConsistencyResult",
    "consistency_update", "geometric_update",
]
