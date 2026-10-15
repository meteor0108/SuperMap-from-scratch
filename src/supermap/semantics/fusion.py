"""Eq. (10) Bayesian label fusion (Sec. IV-B4, p.5).

Symbol map (paper -> code):
    P(L_j = c | z_{1:t-1})  -> prior (C,)
    P(L_j = c | z_{1:t})    -> posterior (C,)
    z_t                     -> z, 0-based class index (D-022)
    P(z_t | L_j = c)        -> confusion[c, z] (rows = true class c, cols = observed z)
    eta                     -> eta = 1 / sum_c P(z_t | c) P(c | z_{1:t-1})
"""
from __future__ import annotations

import numpy as np


# REQ-040: uniform categorical distribution over C prompt classes (D-011, D-060)
def uniform_prior(C: int) -> np.ndarray:
    if int(C) < 1:
        raise ValueError("C must be >= 1")
    return np.full(int(C), 1.0 / int(C), dtype=np.float64)


# REQ-040: representative label = argmax_c P(L_j = c); ties go to the lowest index (D-063)
def representative_label(probs: np.ndarray) -> int:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("empty label distribution")
    return int(np.argmax(p))


# REQ-041: Eq. (10) P(L=c|z_1:t) = eta * P(z_t|L=c) * P(L=c|z_1:t-1)
def bayes_update(prior: np.ndarray, z: int, confusion: np.ndarray) -> np.ndarray:
    """One Bayesian step. prior (C,), z int in [0, C), confusion (C, C) -> posterior (C,)."""
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    confusion = np.asarray(confusion, dtype=np.float64)
    C = prior.shape[0]
    if C == 0:
        raise ValueError("empty prior")
    if confusion.shape != (C, C):
        raise ValueError(f"confusion must be ({C}, {C}), got {confusion.shape}")
    if np.any(prior < 0) or not np.all(np.isfinite(prior)):
        raise ValueError("prior must be finite and non-negative")
    z = int(z)
    if not 0 <= z < C:
        raise ValueError(f"observed label z={z} outside [0, {C})")
    likelihood = confusion[:, z]            # P(z_t | L_j = c) for every c
    unnorm = likelihood * prior
    s = unnorm.sum()
    if not s > 0:
        raise ValueError("posterior has zero mass; confusion column and prior do not overlap")
    eta = 1.0 / s
    return eta * unnorm
