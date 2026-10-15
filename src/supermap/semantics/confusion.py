"""Detector confusion matrix P(z_t | L_j = c) (Sec. IV-B4, p.5; REQ-042).

Layout: confusion[c, z] = P(z_t = z | L_j = c). Rows = true class c, cols = observed z; rows sum to 1.
The paper gives no values; the default is a constant-diagonal parameterization (D-061).
"""
from __future__ import annotations

import numpy as np

ROW_SUM_ATOL = 1e-6


# REQ-042: row-stochastic check with strictly positive entries (no posterior collapse)
def validate_confusion(confusion: np.ndarray, C: int | None = None) -> np.ndarray:
    M = np.asarray(confusion, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
        raise ValueError(f"confusion must be a non-empty square matrix, got shape {M.shape}")
    if C is not None and M.shape[0] != int(C):
        raise ValueError(f"confusion must be ({C}, {C}), got {M.shape}")
    if not np.all(np.isfinite(M)) or np.any(M <= 0):
        raise ValueError("confusion entries must be finite and > 0")
    if not np.allclose(M.sum(axis=1), 1.0, atol=ROW_SUM_ATOL):
        raise ValueError(f"confusion rows must sum to 1, got {M.sum(axis=1)}")
    return M


# REQ-042: diagonal p_c, off-diagonal (1 - p_c) / (C - 1) (D-061)
def confusion_matrix_from_config(C: int, diag_p: float) -> np.ndarray:
    C = int(C)
    p = float(diag_p)
    if C < 1:
        raise ValueError("C must be >= 1")
    if C == 1:
        return np.ones((1, 1), dtype=np.float64)
    if not 0.0 < p < 1.0:
        raise ValueError(f"diag_p must be in (0, 1) so every entry is > 0, got {p}")
    M = np.full((C, C), (1.0 - p) / (C - 1), dtype=np.float64)
    np.fill_diagonal(M, p)
    return validate_confusion(M, C)


# REQ-042: build confusion from configs/default.yaml `semantics.confusion` (explicit matrix wins)
def confusion_from_config(section: dict, C: int) -> np.ndarray:
    """section = cfg['semantics']['confusion'] with keys diag_prob and optional matrix."""
    matrix = section.get("matrix")
    if matrix is not None:
        return validate_confusion(np.asarray(matrix, dtype=np.float64), C)
    return confusion_matrix_from_config(C, section["diag_prob"])
