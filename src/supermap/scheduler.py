"""Asynchronous module rates on dataset replay (REQ-061; p.8 Sec. V-H, Supp. VII-D).

Paper rates (configs/default.yaml `rates:`): pose 10 Hz, 2D instance segmentation 1 Hz,
dense 3D mapping + 3D bbox update 3 Hz, 4D scene graph 5 Hz.

Replay is a simulated schedule driven by frame timestamps (D-087), not threads.
Each module m with period T_m = 1 / rate_m owns the time slots t0 + k T_m (t0 = first stamp).
A module is due at `stamp` when a slot index floor((stamp - t0 + tol) / T_m) not yet served
has been reached (D-086), so every module runs at the first stamp and a 3 Hz module runs
3 times per second on a 10 Hz stream (0.0, 0.4, 0.7, 1.0, ...).
Units: stamps and periods in seconds, rates in Hz.
"""
from __future__ import annotations

import math
from typing import Mapping

# REQ-061: module names used by the pipeline
MODULES = ("pose", "segmentation", "mapping", "scene_graph")
# REQ-061: paper rates in Hz (p.8 Sec. V-H, Supp. VII-D)
PAPER_RATES = {"pose": 10.0, "segmentation": 1.0, "mapping": 3.0, "scene_graph": 5.0}
DEFAULT_TOL = 1e-6


# REQ-061: timestamp-based scheduler with one independent period per module
class ModuleScheduler:
    """due(module, stamp) answers and records whether the module runs at this stamp."""

    # REQ-061: rates in Hz (> 0); missing modules fall back to the paper rates
    def __init__(self, rates: Mapping[str, float] | None = None, tol: float = DEFAULT_TOL) -> None:
        merged = dict(PAPER_RATES)
        for k, v in (rates or {}).items():
            merged[str(k)] = float(v)
        for k, v in merged.items():
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(f"rate of {k!r} must be a positive finite Hz value, got {v}")
        self.rates = merged
        self.periods = {k: 1.0 / v for k, v in merged.items()}
        self.tol = float(tol)
        self.t0: float | None = None
        self.last_stamp: float | None = None
        self._last_slot: dict[str, int] = {}
        self.counts: dict[str, int] = {k: 0 for k in merged}

    # REQ-061: build from the `rates:` config section (tolerance key optional)
    @classmethod
    def from_config(cls, sec: Mapping | None) -> "ModuleScheduler":
        sec = dict(sec or {})
        tol = float(sec.pop("tolerance", DEFAULT_TOL))
        return cls(sec, tol)

    # REQ-061: slot index of a stamp for one module
    def slot(self, module: str, stamp: float) -> int:
        if module not in self.periods:
            raise KeyError(f"unknown module {module!r}")
        t0 = float(stamp) if self.t0 is None else self.t0
        return int(math.floor((float(stamp) - t0 + self.tol) / self.periods[module]))

    # REQ-061: True (and recorded as a run) when a new slot of `module` is reached at `stamp`
    def due(self, module: str, stamp: float) -> bool:
        stamp = float(stamp)
        if not math.isfinite(stamp):
            raise ValueError("stamp must be finite")
        if self.last_stamp is not None and stamp < self.last_stamp - self.tol:
            raise ValueError(f"stamps must not decrease: {stamp} < {self.last_stamp}")
        if self.t0 is None:
            self.t0 = stamp
        self.last_stamp = max(stamp, self.last_stamp if self.last_stamp is not None else stamp)
        k = self.slot(module, stamp)
        last = self._last_slot.get(module)
        if last is not None and k <= last:
            return False
        self._last_slot[module] = k
        self.counts[module] += 1
        return True
