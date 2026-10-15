"""Three-state geometric consistency classification s_k^(t) (Eq. 9).

Symbol map (paper -> code):
    s_k^(t)     -> PointState per point
    tau_epsilon -> tau_eps (meters)
Observable if |Dd| <= tau; Unobservable if Dd > tau (point behind surface);
Disappeared if Dd < -tau (point in front of surface). Points without a residual
(mask False) are NOT_IN_VIEW, a state the paper does not name (D-046); they get no update.
"""
from __future__ import annotations

import enum

import numpy as np


# REQ-031: point states of Eq. (9) plus NOT_IN_VIEW for points without a residual
class PointState(enum.IntEnum):
    OBSERVABLE = 0
    UNOBSERVABLE = 1
    DISAPPEARED = 2
    NOT_IN_VIEW = 3


# REQ-031: Eq. (9) thresholded classification of Delta d
def classify(delta_d: np.ndarray, mask: np.ndarray, tau_eps: float) -> np.ndarray:
    """(n,) residuals + (n,) mask -> (n,) int8 PointState values."""
    if tau_eps < 0:
        raise ValueError("tau_eps must be >= 0")
    dd = np.asarray(delta_d, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if dd.shape != m.shape:
        raise ValueError("delta_d and mask lengths differ")
    s = np.full(dd.shape, PointState.NOT_IN_VIEW, dtype=np.int8)
    s[m & (np.abs(dd) <= tau_eps)] = PointState.OBSERVABLE
    s[m & (dd > tau_eps)] = PointState.UNOBSERVABLE
    s[m & (dd < -tau_eps)] = PointState.DISAPPEARED
    return s
