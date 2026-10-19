"""Hyperparameters of the view-consensus segment clustering (D-124)."""
from __future__ import annotations

from dataclasses import dataclass, fields


# REQ-024 (extension): thresholds of visible / contained / merge decisions
@dataclass(frozen=True)
class ConsensusParams:
    tau_d: float = 0.08      # depth agreement for "visible" (m)
    alpha: float = 0.3       # visible fraction |V| / |P_s|
    n_min: int = 30          # minimum visible points
    beta: float = 0.8        # contained fraction
    k_min: int = 3           # minimum evidence (support + conflict)
    thetas: tuple[float, ...] = (0.95, 0.9, 0.85)   # merge consensus schedule, high to low
    near_dist: float = 0.05  # candidate pairs: bboxes expanded by this overlap (m)
    distrust_split_masks: bool = False  # a mask holding two clusters already known distinct gives no support
    background: tuple[int, ...] = ()    # class indices clustered by overlap instead of consensus
    bg_overlap: float = 0.3  # voxel-overlap fraction (of the smaller) linking two background segments
    voxel: float = 0.02      # voxel size for point de-duplication and overlap tests

    # tolerant constructor: unknown keys are ignored, lists become tuples
    @classmethod
    def from_dict(cls, d: dict) -> "ConsensusParams":
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in (d or {}).items():
            if k not in names:
                continue
            kw[k] = tuple(v) if isinstance(v, (list, tuple)) else v
        return cls(**kw)
