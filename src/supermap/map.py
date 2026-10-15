"""Global map M_t and object instances O_t^j (Sec. III, IV-B4, IV-C, IV-D).

Symbol map (paper -> code):
    M_t          -> GlobalMap
    O_t^j        -> ObjectInstance
    N_t          -> len(GlobalMap) == len(GlobalMap.active_objects()) (ACTIVE objects only, D-096)
    removed rec. -> GlobalMap.removed_objects(); ids/iter/lookup cover all records (REQ-037)
    X_j in R^3   -> ObjectInstance.centroid (mean of points, D-011)
    B_j          -> ObjectInstance.bbox, world axis-aligned (min_xyz, max_xyz), D-011
    P(L_j = c)   -> ObjectInstance.label_probs (C,), sums to 1
    L(o_k)       -> ObjectInstance.log_odds (n,), per-point occupancy log-odds (Eq. 8)
    z^A_min      -> ObjectInstance.bbox[0][2]
All geometry is in the world frame W, meters.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

LABEL_SUM_ATOL = 1e-9


# REQ-005: lifecycle status; removed objects keep their ID (see REQ-037)
class ObjectStatus(str, enum.Enum):
    ACTIVE = "active"
    REMOVED = "removed"


# REQ-005: object instance O^j with points, colors, log-odds, centroid, bbox, label distribution
@dataclass
class ObjectInstance:
    """One semantic instance in M_t. centroid/bbox are derived from points on access and cached
    per point-array version (D-102): change points only through add_points/remove_points or by
    assigning a new array, never by writing into the array in place."""

    id: int
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    log_odds: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float64))
    label_probs: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float64))
    status: ObjectStatus = ObjectStatus.ACTIVE
    first_seen: int = -1
    last_seen: int = -1
    exist_log_odds: float = 0.0  # REQ-035: object-level existence log-odds (D-113); set at creation by ObjMerge
    # REQ-005: X_j / B_j derived from points, cached per point-array version (D-102)
    _cache_key: tuple | None = field(default=None, init=False, repr=False, compare=False)
    _centroid: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _bbox: tuple | None = field(default=None, init=False, repr=False, compare=False)

    # REQ-005: normalize dtypes and check per-point array consistency
    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
        self.log_odds = np.asarray(self.log_odds, dtype=np.float64).reshape(-1)
        self._check_point_arrays()
        self.set_label_probs(self.label_probs)
        self.status = ObjectStatus(self.status)
        self._invalidate()

    # REQ-005: new instance with uniform label prior over C classes (D-011)
    @classmethod
    def create(cls, obj_id: int, num_classes: int, t: int,
               points: np.ndarray | None = None, colors: np.ndarray | None = None,
               log_odds: np.ndarray | float = 0.0) -> "ObjectInstance":
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        obj = cls(id=obj_id, label_probs=np.full(num_classes, 1.0 / num_classes),
                  first_seen=t, last_seen=t)
        if points is not None:
            obj.add_points(points, colors, log_odds)
        return obj

    def _check_point_arrays(self) -> None:
        n = self.points.shape[0]
        if self.colors.shape[0] != n or self.log_odds.shape[0] != n:
            raise ValueError(
                f"points/colors/log_odds lengths differ: {n}, {self.colors.shape[0]}, {self.log_odds.shape[0]}")

    # REQ-005: drop cached X_j / B_j after the point array changed (D-102)
    def _invalidate(self) -> None:
        self._cache_key = None

    # REQ-005: X_j / B_j derived from points, cached per point-array version (D-102)
    def _derived(self) -> None:
        key = (id(self.points), self.points.shape[0])
        if key != self._cache_key:
            self._centroid = self.points.mean(axis=0)
            self._bbox = (self.points.min(axis=0), self.points.max(axis=0))
            for arr in (self._centroid, *self._bbox):
                arr.flags.writeable = False
            self._cache_key = key

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.label_probs.shape[0])

    # REQ-005: X_j = mean of object points (D-011)
    @property
    def centroid(self) -> np.ndarray:
        if self.num_points == 0:
            raise ValueError(f"object {self.id} has no points")
        self._derived()
        return self._centroid

    # REQ-005: world axis-aligned bbox B_j = (min_xyz, max_xyz) (D-011)
    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        if self.num_points == 0:
            raise ValueError(f"object {self.id} has no points")
        self._derived()
        return self._bbox

    # REQ-005: most probable class index argmax_c P(L_j = c)
    @property
    def label(self) -> int:
        if self.num_classes == 0:
            raise ValueError(f"object {self.id} has no label distribution")
        return int(np.argmax(self.label_probs))

    # REQ-005: set P(L_j = c); must be non-negative and sum to 1
    def set_label_probs(self, probs: np.ndarray) -> None:
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        if p.size > 0:
            if np.any(p < 0) or not np.all(np.isfinite(p)):
                raise ValueError("label_probs must be finite and non-negative")
            if abs(p.sum() - 1.0) > LABEL_SUM_ATOL:
                raise ValueError(f"label_probs must sum to 1, got {p.sum()}")
        self.label_probs = p

    # REQ-005: append points; centroid and bbox follow automatically
    def add_points(self, points: np.ndarray, colors: np.ndarray | None = None,
                   log_odds: np.ndarray | float = 0.0) -> None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = pts.shape[0]
        cols = (np.zeros((n, 3), np.uint8) if colors is None
                else np.asarray(colors, dtype=np.uint8).reshape(-1, 3))
        lo = np.broadcast_to(np.asarray(log_odds, dtype=np.float64), (n,)).copy()
        if cols.shape[0] != n:
            raise ValueError("colors length must match points")
        self.points = np.concatenate([self.points, pts], axis=0)
        self.colors = np.concatenate([self.colors, cols], axis=0)
        self.log_odds = np.concatenate([self.log_odds, lo], axis=0)
        self._invalidate()

    # REQ-005: drop points where remove_mask is True
    def remove_points(self, remove_mask: np.ndarray) -> None:
        m = np.asarray(remove_mask, dtype=bool).reshape(-1)
        if m.shape[0] != self.num_points:
            raise ValueError("remove_mask length must equal num_points")
        keep = ~m
        self.points = self.points[keep]
        self.colors = self.colors[keep]
        self.log_odds = self.log_odds[keep]
        self._invalidate()


# REQ-005: global map M_t = {O_t^j}_{j=1..N_t}
class GlobalMap:
    """Container of object instances keyed by unique instance ID.

    len() is N_t, the number of ACTIVE objects in M_t. REMOVED objects are kept only as
    status records (id + history, REQ-037); they are reachable via iteration, ids, `in`,
    [] and removed_objects(), but are not counted in N_t (D-096).
    """

    # REQ-005: class_names is the label space c (prompt list, REQ-013)
    def __init__(self, class_names: Sequence[str] = ()) -> None:
        self.class_names: list[str] = list(class_names)
        self._objects: dict[int, ObjectInstance] = {}
        self._next_id = 0

    # REQ-005: N_t = number of objects currently in M_t (ACTIVE only, D-096)
    def __len__(self) -> int:
        return sum(1 for o in self._objects.values() if o.status == ObjectStatus.ACTIVE)

    # REQ-005, REQ-037: all stored records (ACTIVE and REMOVED) in id order
    def __iter__(self) -> Iterator[ObjectInstance]:
        return iter(self._objects[k] for k in sorted(self._objects))

    def __contains__(self, obj_id: int) -> bool:
        return int(obj_id) in self._objects

    def __getitem__(self, obj_id: int) -> ObjectInstance:
        return self._objects[int(obj_id)]

    @property
    def ids(self) -> list[int]:
        return sorted(self._objects)

    # REQ-005: monotonically increasing unique ID; never reused
    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # REQ-005: insert an object; its ID must be unique in the map
    def add(self, obj: ObjectInstance) -> ObjectInstance:
        if obj.id in self._objects:
            raise ValueError(f"object id {obj.id} already in map")
        self._objects[obj.id] = obj
        self._next_id = max(self._next_id, obj.id + 1)
        return obj

    # REQ-005: delete an object entry from the map
    def remove(self, obj_id: int) -> ObjectInstance:
        return self._objects.pop(int(obj_id))

    # REQ-005: objects whose status is ACTIVE
    def active_objects(self) -> list[ObjectInstance]:
        return [o for o in self if o.status == ObjectStatus.ACTIVE]

    # REQ-037: status records of pruned objects (not part of M_t, not counted in N_t)
    def removed_objects(self) -> list[ObjectInstance]:
        return [o for o in self if o.status == ObjectStatus.REMOVED]

    # REQ-005, REQ-037: every stored record, ACTIVE and REMOVED, in id order
    def all_objects(self) -> list[ObjectInstance]:
        return list(self)
