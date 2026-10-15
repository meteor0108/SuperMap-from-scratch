"""Hyperparameters of the probabilistic geometric consistency update (Sec. IV-B3).

Symbol map (paper -> code):
    sigma        -> ConsistencyParams.sigma     Gaussian depth noise std, meters (REQ-030)
    tau_epsilon  -> ConsistencyParams.tau_eps   residual threshold of Eq. (9), meters (REQ-031)
    P(o_k|Q_t)   -> p_obs / p_dis / p_unobs     inverse sensor model per state (REQ-034)
    L_prune      -> ConsistencyParams.L_prune   log-odds pruning threshold (REQ-035)
Values live in configs/default.yaml `consistency:`; the paper gives no numbers (D-045..D-049).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


# REQ-030, REQ-031, REQ-033, REQ-034, REQ-035: container for the consistency config section
@dataclass(frozen=True)
class ConsistencyParams:
    sigma: float = 0.05
    tau_eps: float = 0.1
    p_obs: float = 0.7
    p_dis: float = 0.3
    p_unobs: float = 0.5
    L_prune: float = -1.0
    L_init: float | None = None     # None -> logit(p_obs) (D-048)
    L_clamp: float | None = None    # None -> no clamping (D-048)
    exist_clamp: float | None = 3.0   # REQ-035: symmetric clamp of the object existence log-odds (D-113)
    exist_prune: float | None = -1.0  # REQ-035: L_exist < this removes the object; None disables (D-113)
    exist_semantic: bool = True       # REQ-036: semantic O_neg (visible, untracked) also counts for L_exist (D-113; False = geometric only)
    verdict_window_px: int = 0        # REQ-032: window radius of the tolerant object-level verdict, 0 = off (I-9-4 mitigation, no decision recorded)
    exist_min_points: int = 0         # REQ-035: min in-view points for an L_exist update, 0 = off (I-9-4 mitigation, no decision recorded)
    exist_min_share: float = 0.0      # REQ-035: min in-view share of points for an L_exist update, 0 = off (I-9-4 mitigation, no decision recorded)

    # REQ-034: validate probability ranges and ordering p_dis < 0.5 < p_obs
    def __post_init__(self) -> None:
        for name in ("p_obs", "p_dis", "p_unobs"):
            p = getattr(self, name)
            if not 0.0 < p < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {p}")
        if not self.p_obs > 0.5:
            raise ValueError("p_obs must be > 0.5 (Observable raises occupancy)")
        if not self.p_dis < 0.5:
            raise ValueError("p_dis must be < 0.5 (Disappeared lowers occupancy)")
        if self.sigma <= 0 or self.tau_eps < 0:
            raise ValueError("sigma must be > 0 and tau_eps >= 0")
        if self.L_clamp is not None and self.L_clamp <= 0:
            raise ValueError("L_clamp must be positive or None")
        if self.exist_clamp is not None and self.exist_clamp <= 0:  # REQ-035: D-113
            raise ValueError("exist_clamp must be positive or None")
        if self.verdict_window_px < 0 or self.exist_min_points < 0 or not 0.0 <= self.exist_min_share <= 1.0:
            raise ValueError("verdict_window_px/exist_min_points must be >= 0, exist_min_share in [0, 1]")

    # REQ-033: initial log-odds of a new map point = logit of its observation probability
    @property
    def initial_log_odds(self) -> float:
        if self.L_init is not None:
            return float(self.L_init)
        return math.log(self.p_obs / (1.0 - self.p_obs))

    # REQ-030, REQ-031, REQ-034, REQ-035: build from the `consistency:` config section
    @classmethod
    def from_config(cls, cfg: Mapping | None = None) -> "ConsistencyParams":
        """cfg: full config dict (with a `consistency` key) or the section itself; None loads defaults."""
        if cfg is None:
            from supermap.config import load_config
            cfg = load_config()
        sec = cfg.get("consistency", cfg)
        keys = ("sigma", "tau_eps", "p_obs", "p_dis", "p_unobs", "L_prune", "L_init", "L_clamp",
                "exist_clamp", "exist_prune", "exist_semantic", "verdict_window_px", "exist_min_points",
                "exist_min_share")
        # REQ-036: exist_semantic is a flag (D-113); I-9-4 keys (verdict_window_px, exist_min_points) are ints; the other keys are floats or null
        ints = ("verdict_window_px", "exist_min_points")
        cast = lambda k, v: None if v is None else (bool(v) if k == "exist_semantic"
                                                    else int(v) if k in ints else float(v))
        return cls(**{k: cast(k, sec[k]) for k in keys if k in sec})


# REQ-030, REQ-031, REQ-034, REQ-035: accept params, a config dict, or None (defaults from config)
def as_params(params: "ConsistencyParams | Mapping | None") -> ConsistencyParams:
    if isinstance(params, ConsistencyParams):
        return params
    return ConsistencyParams.from_config(params)
