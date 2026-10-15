"""Per-instance semantic update, gating, and low-confidence removal (Sec. IV-B4, p.5).

Symbol map (paper -> code):
    P(L_j = c)            -> ObjectInstance.label_probs (C,)
    s_k^(t) = Observable  -> `observable` input (bool, or per-point bool mask from the Eq. 9 check)
    posterior belief      -> posterior_belief(probs) = max_c P(L_j = c) (D-063)
    p_min                 -> removal threshold (configs/default.yaml semantics.p_min)
This module takes the Eq. (9) result as input and does not compute point states itself.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from supermap.map import GlobalMap, ObjectInstance, ObjectStatus
from supermap.semantics.fusion import bayes_update, uniform_prior


# REQ-043: instance-level gate: True iff the instance has at least one Observable point (D-062)
def _has_observable(observable: bool | np.ndarray) -> bool:
    if isinstance(observable, (bool, np.bool_)):
        return bool(observable)
    arr = np.asarray(observable)
    if arr.dtype != bool:
        raise ValueError("observable must be a bool or a boolean per-point mask")
    return bool(arr.any())


# REQ-040, REQ-043: new instance: uniform prior + one Eq. (10) step with its first z (D-060)
def init_from_first_observation(instance: ObjectInstance, z: int, confusion: np.ndarray) -> np.ndarray:
    C = np.asarray(confusion).shape[0]
    posterior = bayes_update(uniform_prior(C), z, confusion)
    instance.set_label_probs(posterior)
    return instance.label_probs


# REQ-041, REQ-043: apply Eq. (10) only if the instance has Observable points this frame
def semantic_update(instance: ObjectInstance, z: int, confusion: np.ndarray,
                    observable: bool | np.ndarray) -> bool:
    """Returns True if the distribution was updated; otherwise label_probs is left untouched."""
    if not _has_observable(observable):
        return False
    posterior = bayes_update(instance.label_probs, z, confusion)
    instance.set_label_probs(posterior)
    return True


# REQ-044: posterior belief of the representative label, max_c P(L_j = c) (D-063)
def posterior_belief(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("empty label distribution")
    return float(p.max())


# REQ-044: belief strictly below p_min means the instance is removed
def is_low_confidence(probs: np.ndarray, p_min: float) -> bool:
    return posterior_belief(probs) < float(p_min)


# REQ-044: remove all points of low-belief instances; entry stays as REMOVED so its ID is never reused
def remove_low_confidence(target: ObjectInstance | GlobalMap | Iterable[ObjectInstance],
                          p_min: float) -> list[int]:
    """Returns the IDs removed in this call. Only ACTIVE instances with a distribution are checked."""
    if isinstance(target, ObjectInstance):
        objs = [target]
    elif isinstance(target, GlobalMap):
        objs = target.active_objects()
    else:
        objs = list(target)
    removed = []
    for obj in objs:
        if obj.status != ObjectStatus.ACTIVE or obj.num_classes == 0:
            continue
        if is_low_confidence(obj.label_probs, p_min):
            obj.remove_points(np.ones(obj.num_points, dtype=bool))
            obj.status = ObjectStatus.REMOVED
            removed.append(obj.id)
    return removed
