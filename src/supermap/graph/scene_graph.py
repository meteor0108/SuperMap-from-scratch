"""Spatio-temporal scene graph G = (V, E_s, E_t) built from M_t (REQ-046..REQ-051, p.5 Sec. IV-C).

Symbol map (paper -> code):
    M_t                -> supermap.map.GlobalMap
    V                  -> SceneGraph.nodes: dict[id -> Node] (ACTIVE instances only)
    E_s                -> SceneGraph.E_s: list[(src_id, rel, dst_id)], rel in {on, beside, under}
    E_t                -> SceneGraph.E_t: list[TemporalEdge] linking one ID across time
    I_t (association)  -> instance IDs of M_t (same ID = same object across frames)
    X_v                -> Node.centroid (world, meters)
Time t is the stream index of Sec. III (D-008). REMOVED instances leave V and E_s but keep
their temporal history and are listed in SceneGraph.disappeared (REQ-037).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from supermap.config import load_config
from supermap.graph.cluster import candidate_pairs, cluster_centroids
from supermap.graph.predicates import BESIDE, ON, UNDER, pair_relations
from supermap.map import GlobalMap, ObjectInstance, ObjectStatus

log = logging.getLogger(__name__)

UNKNOWN_LABEL = "unknown"


# REQ-047..REQ-050, REQ-056: scene graph hyperparameters (configs/default.yaml scene_graph)
@dataclass(frozen=True)
class SceneGraphConfig:
    d_clu: float
    gamma: float
    delta: float
    d_bes: float
    d_move: float
    subgraph_radius: float
    vertical_axis: int = 2
    gamma_per_class: dict = field(default_factory=dict)

    # REQ-047..REQ-050: build from the `scene_graph` config section
    @classmethod
    def from_config(cls, sec: dict) -> "SceneGraphConfig":
        return cls(d_clu=float(sec["d_clu"]), gamma=float(sec["gamma"]), delta=float(sec["delta"]),
                   d_bes=float(sec["d_bes"]), d_move=float(sec["d_move"]),
                   subgraph_radius=float(sec["subgraph_radius"]),
                   vertical_axis=int(sec.get("vertical_axis", 2)),
                   gamma_per_class=dict(sec.get("gamma_per_class") or {}))

    # REQ-047..REQ-050: values from configs/default.yaml
    @classmethod
    def default(cls) -> "SceneGraphConfig":
        return cls.from_config(load_config()["scene_graph"])

    # REQ-048: class-dependent gamma, keyed by the reference object's label (D-071)
    def gamma_for(self, label_b: str) -> float:
        return float(self.gamma_per_class.get(label_b, self.gamma))


# REQ-046: node v in V: instance ID, representative label, centroid, bbox, first/last seen
@dataclass(frozen=True)
class Node:
    id: int
    label: str
    centroid: np.ndarray
    bbox: tuple[np.ndarray, np.ndarray]
    first_seen: int
    last_seen: int


# REQ-050: temporal edge linking the same ID at t_prev and t_next
@dataclass(frozen=True)
class TemporalEdge:
    id: int
    t_prev: float
    t_next: float
    centroid_prev: np.ndarray
    centroid_next: np.ndarray


# REQ-050: temporal node: position of one ID from t_start, confirmed until t_end (D-074)
@dataclass
class TemporalState:
    t_start: float
    t_end: float
    centroid: np.ndarray
    status: ObjectStatus


# REQ-046: representative label name argmax_c P(L_j = c) via the map's prompt list (REQ-013)
def node_label(obj: ObjectInstance, class_names: Sequence[str]) -> str:
    if obj.num_classes == 0:
        return UNKNOWN_LABEL
    c = obj.label
    return str(class_names[c]) if c < len(class_names) else str(c)


# REQ-046: G = (V, E_s, E_t) with per-ID temporal history
class SceneGraph:
    """Scene graph over M_t. Call update(map, t) at each graph step to extend E_t."""

    # REQ-046: empty graph
    def __init__(self, config: SceneGraphConfig | None = None) -> None:
        self.config = config if config is not None else SceneGraphConfig.default()
        self.t: float | None = None
        self.nodes: dict[int, Node] = {}
        self.E_s: list[tuple[int, str, int]] = []
        self.E_t: list[TemporalEdge] = []
        self.history: dict[int, list[TemporalState]] = {}
        self.labels: dict[int, str] = {}
        self.seen: dict[int, tuple[int, int]] = {}
        self.disappeared: dict[int, float] = {}

    # REQ-046: build a fresh graph from M_t at time t
    @classmethod
    def build_from_map(cls, gmap: GlobalMap, t: float,
                       config: SceneGraphConfig | None = None) -> "SceneGraph":
        g = cls(config)
        g.update(gmap, t)
        return g

    # REQ-046, REQ-050: refresh V and E_s from M_t; append temporal nodes/edges per ID
    def update(self, gmap: GlobalMap, t: float) -> "SceneGraph":
        if self.t is not None and t < self.t:
            raise ValueError(f"graph time must not decrease: {t} < {self.t}")
        self.t = t
        present: set[int] = set()
        nodes: dict[int, Node] = {}
        for obj in gmap.all_objects():  # REMOVED records feed the temporal history (REQ-037)
            present.add(obj.id)
            self.labels[obj.id] = node_label(obj, gmap.class_names)
            self.seen[obj.id] = (obj.first_seen, obj.last_seen)
            if obj.status == ObjectStatus.ACTIVE:
                if obj.num_points == 0:
                    log.warning("instance %d has no points; not a graph node", obj.id)
                    continue
                nodes[obj.id] = Node(obj.id, self.labels[obj.id], obj.centroid.copy(),
                                     tuple(b.copy() for b in obj.bbox),
                                     obj.first_seen, obj.last_seen)
                self._observe_active(obj.id, obj.centroid, t)
            else:
                c = obj.centroid if obj.num_points > 0 else None
                self._observe_removed(obj.id, c, t, obj.last_seen)
        for obj_id in list(self.history):
            if obj_id not in present:
                self._observe_removed(obj_id, None, t, -1)
        self.nodes = nodes
        self.E_s = self._spatial_edges()
        return self

    # REQ-050: new temporal node (and edge) when X moves >= d_move or the ID reappears
    def _observe_active(self, obj_id: int, c: np.ndarray, t: float) -> None:
        c = np.asarray(c, dtype=np.float64).copy()
        hist = self.history.setdefault(obj_id, [])
        self.disappeared.pop(obj_id, None)
        if not hist:
            hist.append(TemporalState(t, t, c, ObjectStatus.ACTIVE))
            return
        last = hist[-1]
        prev_active = next((s for s in reversed(hist) if s.status == ObjectStatus.ACTIVE), None)
        moved = float(np.linalg.norm(c - last.centroid)) >= self.config.d_move
        if last.status != ObjectStatus.ACTIVE or moved:
            if prev_active is not None:
                p = prev_active
                self.E_t.append(TemporalEdge(obj_id, p.t_start, t, p.centroid.copy(), c))
            hist.append(TemporalState(t, t, c, ObjectStatus.ACTIVE))
        else:
            last.t_end = t

    # REQ-050, REQ-037: record disappearance once; history is kept (D-075)
    def _observe_removed(self, obj_id: int, c: np.ndarray | None, t: float, last_seen: int) -> None:
        if obj_id in self.disappeared:
            return
        hist = self.history.setdefault(obj_id, [])
        if hist:
            t_dis = t
            pos = hist[-1].centroid.copy()
        else:
            t_dis = last_seen if last_seen >= 0 else t
            if c is None:
                self.disappeared[obj_id] = t_dis
                return
            pos = np.asarray(c, dtype=np.float64).copy()
        self.disappeared[obj_id] = t_dis
        hist.append(TemporalState(t_dis, t_dis, pos, ObjectStatus.REMOVED))

    # REQ-047..REQ-049: cluster by centroid, then test predicates on in-cluster pairs
    def _spatial_edges(self) -> list[tuple[int, str, int]]:
        ids = sorted(self.nodes)
        if len(ids) < 2:
            return []
        cfg = self.config
        X = np.stack([self.nodes[i].centroid for i in ids])
        labels = cluster_centroids(X, cfg.d_clu)
        pairs = candidate_pairs(labels)
        if not pairs:
            return []
        nodes = [self.nodes[i] for i in ids]
        lo = np.stack([n.bbox[0] for n in nodes])
        hi = np.stack([n.bbox[1] for n in nodes])
        gam = np.array([cfg.gamma_for(n.label) for n in nodes])
        ii = np.array([p[0] for p in pairs], dtype=np.int64)
        jj = np.array([p[1] for p in pairs], dtype=np.int64)
        # REQ-048, REQ-049: array predicates over all in-cluster pairs (D-103)
        rel = pair_relations(lo, hi, ii, jj, gam[ii], gam[jj], cfg.delta, cfg.d_bes,
                             cfg.vertical_axis)
        edges: list[tuple[int, str, int]] = []
        for k, (i, j) in enumerate(pairs):
            a_id, b_id = ids[i], ids[j]
            if rel["on_ij"][k]:
                edges.append((a_id, ON, b_id))
            if rel["under_ij"][k]:
                edges.append((a_id, UNDER, b_id))
            if rel["on_ji"][k]:
                edges.append((b_id, ON, a_id))
            if rel["under_ji"][k]:
                edges.append((b_id, UNDER, a_id))
            if rel["beside_ij"][k]:
                edges.append((a_id, BESIDE, b_id))
        return sorted(edges)

    # REQ-050, REQ-051: trajectory of one ID: [(t, centroid)] in ascending time
    def trajectory(self, obj_id: int) -> list[tuple[float, np.ndarray]]:
        hist = self.history.get(int(obj_id), [])
        return [(s.t_start, s.centroid.copy()) for s in hist if s.status == ObjectStatus.ACTIVE]

    # REQ-050: temporal edges of one ID in time order
    def temporal_edges(self, obj_id: int) -> list[TemporalEdge]:
        return sorted((e for e in self.E_t if e.id == int(obj_id)), key=lambda e: e.t_prev)

    # REQ-046: all IDs known to the graph (active and disappeared)
    @property
    def known_ids(self) -> list[int]:
        return sorted(set(self.history) | set(self.nodes) | set(self.disappeared))

    # REQ-056: copy restricted to the given IDs; edges only between kept nodes
    def restricted(self, ids: Iterable[int]) -> "SceneGraph":
        keep = {int(i) for i in ids}
        g = SceneGraph(self.config)
        g.t = self.t
        g.nodes = {i: n for i, n in self.nodes.items() if i in keep}
        g.E_s = [e for e in self.E_s if e[0] in g.nodes and e[2] in g.nodes]
        g.E_t = [e for e in self.E_t if e.id in keep]
        g.history = {i: list(h) for i, h in self.history.items() if i in keep}
        g.labels = {i: s for i, s in self.labels.items() if i in keep}
        g.seen = {i: s for i, s in self.seen.items() if i in keep}
        g.disappeared = {i: s for i, s in self.disappeared.items() if i in keep}
        return g
