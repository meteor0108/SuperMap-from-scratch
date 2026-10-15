"""Topological layer: spatio-temporal scene graph (REQ-046..REQ-056, Sec. IV-C/D)."""
# REQ-046..REQ-056: public API of the topological layer
from supermap.graph.cluster import candidate_pairs, cluster_centroids
from supermap.graph.predicates import BESIDE, ON, RELATIONS, UNDER, iou_xy, is_beside, is_on, is_under
from supermap.graph.scene_graph import Node, SceneGraph, SceneGraphConfig, TemporalEdge, TemporalState
from supermap.graph.serialize import (SCHEMA, select_local_subgraph, serialize_cues, serialize_graph,
                                      serialize_node, serialize_nodes, serialize_spatial_edge,
                                      serialize_spatial_edges, serialize_temporal_edges)
