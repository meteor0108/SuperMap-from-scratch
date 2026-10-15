"""Spatio-temporal serialization of a local subgraph (REQ-052..REQ-056, p.5 Sec. IV-D a)).

Output text = schema block + nodes + spatial edges + temporal edges + spatio-temporal cues.
Formats (D-076..D-079):
    node:          "Instance_<id>: label=<label>, centroid=(x, y, z)"  (2-decimal fixed, meters)
    spatial edge:  "Instance_<a> <rel> Instance_<b>"                   sorted by (src, rel, dst)
    temporal edge: "Instance_<a> at t=<t1> -> t=<t2>: (x1, y1, z1) -> (x2, y2, z2)"
All ordering is by ascending instance ID, so output does not depend on insertion order.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from supermap.graph.predicates import RELATIONS
from supermap.graph.scene_graph import SceneGraph
from supermap.map import ObjectStatus

# REQ-054: fixed schema defining graph structure and coordinate frame (D-077)
SCHEMA = "\n".join([
    "[SCHEMA]",
    "The scene is a 4D scene graph G = (V, E_s, E_t) of object instances.",
    "Coordinate frame: world frame, units in meters, vertical (up) axis is world z.",
    "Node fields: ID (Instance_<id>), label (semantic class), centroid (x, y, z) in meters.",
    "Spatial edges E_s: 'Instance_a <relation> Instance_b', relation in {" + ", ".join(RELATIONS) + "}.",
    "  on: a rests on top of b; under: a is below b; beside: a is next to b at a similar height.",
    "Temporal edges E_t (temporal): 'Instance_a at t=<t1> -> t=<t2>: (x1, y1, z1) -> (x2, y2, z2)'"
    " links the same instance ID across time, tracing its trajectory.",
    "Cues: current time, first/last seen time per instance, disappeared instances with"
    " disappearance time, and movement history.",
    "[END SCHEMA]",
])


# REQ-053: time stamp text, e.g. t=3 or t=0.2
def fmt_t(t: float) -> str:
    return f"t={float(t):g}"


# REQ-052: centroid text with 2-decimal fixed format
def fmt_xyz(x: Sequence[float]) -> str:
    v = np.asarray(x, dtype=np.float64).reshape(3)
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


# REQ-052, REQ-053: instance token "Instance_<id>"
def inst(obj_id: int) -> str:
    return f"Instance_{int(obj_id)}"


# REQ-052: one line per node: ID, label, centroid
def serialize_node(obj_id: int, label: str, centroid: Sequence[float]) -> str:
    return f"{inst(obj_id)}: label={label}, centroid={fmt_xyz(centroid)}"


# REQ-052: node block, ascending ID
def serialize_nodes(graph: SceneGraph) -> list[str]:
    return [serialize_node(i, graph.nodes[i].label, graph.nodes[i].centroid)
            for i in sorted(graph.nodes)]


# REQ-053: spatial edge "Instance_a <rel> Instance_b"
def serialize_spatial_edge(src: int, rel: str, dst: int) -> str:
    return f"{inst(src)} {rel} {inst(dst)}"


# REQ-053: spatial edges sorted by (src, rel, dst)
def serialize_spatial_edges(edges: Iterable[tuple[int, str, int]]) -> list[str]:
    return [serialize_spatial_edge(s, r, d) for s, r, d in sorted((int(s), str(r), int(d)) for s, r, d in edges)]


# REQ-053: temporal edges sorted by (id, t_prev)
def serialize_temporal_edges(graph: SceneGraph) -> list[str]:
    out = []
    for e in sorted(graph.E_t, key=lambda e: (e.id, e.t_prev, e.t_next)):
        out.append(f"{inst(e.id)} at {fmt_t(e.t_prev)} -> {fmt_t(e.t_next)}: "
                   f"{fmt_xyz(e.centroid_prev)} -> {fmt_xyz(e.centroid_next)}")
    return out


# REQ-055: explicit spatio-temporal cues (D-078)
def serialize_cues(graph: SceneGraph) -> list[str]:
    out = [f"Current time: {fmt_t(graph.t if graph.t is not None else 0)}"]
    for i in sorted(graph.nodes):
        n = graph.nodes[i]
        out.append(f"{inst(i)} ({n.label}): first seen {fmt_t(n.first_seen)}, last seen {fmt_t(n.last_seen)}")
    for i in sorted(graph.disappeared):
        label = graph.labels.get(i, "unknown")
        hist = graph.history.get(i, [])
        where = f", last position {fmt_xyz(hist[-1].centroid)}" if hist else ""
        out.append(f"{inst(i)} ({label}): disappeared at {fmt_t(graph.disappeared[i])}{where}")
    for i in sorted(graph.history):
        states = [s for s in graph.history[i] if s.status == ObjectStatus.ACTIVE]
        if len(states) < 2:
            continue
        label = graph.labels.get(i, "unknown")
        steps = " -> ".join(f"{fmt_t(s.t_start)} {fmt_xyz(s.centroid)}" for s in states)
        line = f"{inst(i)} ({label}) movement history: {steps}"
        if states[-1].t_end > states[-1].t_start:
            line += f", stationary until {fmt_t(states[-1].t_end)}"
        out.append(line)
    return out


# REQ-056: local subgraph: nodes within radius of center (or given IDs); only edges among them
def select_local_subgraph(graph: SceneGraph, center: Sequence[float] | None = None,
                          radius: float | None = None,
                          node_ids: Iterable[int] | None = None) -> SceneGraph:
    """center: (3,) world meters (e.g. robot position). radius defaults to config subgraph_radius.
    Disappeared instances are kept when their last known position is within the radius."""
    if node_ids is not None:
        return graph.restricted(node_ids)
    if center is None:
        return graph.restricted(graph.known_ids)
    r = graph.config.subgraph_radius if radius is None else float(radius)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    keep = [i for i, n in graph.nodes.items() if np.linalg.norm(n.centroid - c) <= r]
    for i in graph.disappeared:
        hist = graph.history.get(i, [])
        if hist and np.linalg.norm(hist[-1].centroid - c) <= r:
            keep.append(i)
    return graph.restricted(keep)


# REQ-052..REQ-055: full serialized text of a (sub)graph, starting with the schema
def serialize_graph(graph: SceneGraph) -> str:
    blocks = [SCHEMA,
              "[NODES]", *serialize_nodes(graph),
              "[SPATIAL EDGES]", *serialize_spatial_edges(graph.E_s),
              "[TEMPORAL EDGES]", *serialize_temporal_edges(graph),
              "[SPATIO-TEMPORAL CUES]", *serialize_cues(graph)]
    return "\n".join(blocks)
