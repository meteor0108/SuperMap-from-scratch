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
    fallback: str = "contact"      # tier after the theta schedule for undecided pairs: "none" | "contact"
    fallback_contact: float = 0.5  # share of the smaller cluster's points within near_dist of the larger
    min_cluster_points: int = 50   # clusters with fewer points are dropped from the result
    partial_share: float = 0.0     # >0: a frame where one cluster fills the other's mask is no conflict (0 = off)
    ghost_free_min: int = 0        # >0: drop points seen in free space by >= this many keyframes (0 = off)
    ghost_confirmed_max: int = 1   # ... and confirmed by at most this many keyframes
    ghost_tau: float = 0.2         # measured surface behind the point by more than this = free space (m)
    min_point_support: int = 2     # keep points whose voxel is seen by >= k distinct member segments (1 = off)

    def __post_init__(self) -> None:
        if self.fallback not in ("none", "contact"):
            raise ValueError("fallback must be 'none' or 'contact'")

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
