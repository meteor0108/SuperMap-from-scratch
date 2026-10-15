"""Grounded query: scene graph -> VLM -> IDs -> waypoints (REQ-056..REQ-059, Sec. IV-D).

Prompt = schema + serialized local subgraph + spatio-temporal cues (serialize_graph)
         + instruction (configs/default.yaml vlm.instruction) + query. One backend call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from supermap.config import load_config
from supermap.graph.scene_graph import SceneGraph
from supermap.graph.serialize import select_local_subgraph, serialize_graph
from supermap.map import GlobalMap
from supermap.vlm.backend import VLMBackend
from supermap.vlm.parser import parse_answer
from supermap.vlm.waypoints import ids_to_waypoints


# REQ-057: result of one grounded query
@dataclass
class GroundingResult:
    prompt: str
    response: str
    ids: list[int]
    waypoints: np.ndarray


# REQ-057: default instruction text from config
def default_instruction() -> str:
    return str(load_config()["vlm"]["instruction"])


# REQ-057: prompt = serialized (schema + subgraph + cues) + instruction + query
def build_prompt(query: str, graph: SceneGraph, instruction: str | None = None) -> str:
    instr = default_instruction() if instruction is None else instruction
    return f"{serialize_graph(graph)}\n[INSTRUCTION]\n{instr}\n[QUERY]\n{query}"


# REQ-056..REQ-059: select subgraph, call backend once, parse, look up centroids
def run_grounding(query: str, graph: SceneGraph, gmap: GlobalMap, backend: VLMBackend,
                  center: Sequence[float] | None = None, radius: float | None = None,
                  instruction: str | None = None) -> GroundingResult:
    sub = select_local_subgraph(graph, center=center, radius=radius)
    prompt = build_prompt(query, sub, instruction)
    response = backend.complete(prompt)
    ids = parse_answer(response)
    return GroundingResult(prompt, response, ids, ids_to_waypoints(ids, gmap))


# REQ-059: grounded query -> (k, 3) waypoints in world meters
def ground(query: str, graph: SceneGraph, gmap: GlobalMap, backend: VLMBackend,
           center: Sequence[float] | None = None, radius: float | None = None,
           instruction: str | None = None) -> np.ndarray:
    return run_grounding(query, graph, gmap, backend, center, radius, instruction).waypoints
