"""Instance IDs -> 3D centroid waypoints (REQ-059, p.5 Sec. IV-D b), Sec. V-G a)).

Output: (k, 3) float64 array of centroids X_j in the world frame, meters, in the order of the IDs.
IDs missing from M_t, REMOVED, or without points are skipped with a logged warning (D-083).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from supermap.map import GlobalMap, ObjectStatus

log = logging.getLogger(__name__)


# REQ-059: look up X_j in M_t for each parsed ID, order preserved
def ids_to_waypoints(ids: Iterable[int], gmap: GlobalMap) -> np.ndarray:
    out = []
    for i in ids:
        i = int(i)
        if i not in gmap:
            log.warning("instance %d not in map; skipped", i)
            continue
        obj = gmap[i]
        if obj.status != ObjectStatus.ACTIVE:
            log.warning("instance %d is %s; skipped", i, obj.status.value)
            continue
        if obj.num_points == 0:
            log.warning("instance %d has no points; skipped", i)
            continue
        out.append(obj.centroid)
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)
