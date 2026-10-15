"""Lifting 2D masks to 3D and merging into the map (Supp. Alg. 1 lines 20, 28-29; Fig. 2).

Symbol map (paper -> code):
    M_mask             -> masks: list of (track_id, (H, W) bool)
    O_add              -> list[ObservedObject] = unproject_masks(obs, masks, labels)
    Unproject(D_I, M_mask, Odom) -> REQ-010 back-projection of each mask with P_t = obs.pose
    ObjMerge(O_nbr, O_add)       -> obj_merge(): same id -> append points, else new instance
    MapMerge(M, O_nbr)           -> map_merge(): write O_nbr back into the global map
Instance id == track id (REQ-022). New ids are allocated by the tracker from GlobalMap.new_id,
which never reuses an id (REQ-024).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from supermap.geometry.backproject import backproject
from supermap.geometry.dense_map import voxel_keys
from supermap.map import GlobalMap, ObjectInstance, ObjectStatus
from supermap.tracking.components import largest_component_mask
from supermap.tracking.contact import TreeCache
from supermap.tracking.depth_edges import depth_edge_mask
from supermap.tracking.mask_ops import depth_inlier_mask, erode_mask
from supermap.types import Observation


# REQ-023: observed object of O_add: track id, label observation z, world points and colors
@dataclass
class ObservedObject:
    id: int
    points: np.ndarray   # (n, 3) float64 world, meters
    colors: np.ndarray   # (n, 3) uint8
    label: int           # detector class index z_t (0-based, D-022), -1 if unknown
    score: float
    t: int
    pixels: np.ndarray | None = None  # (n, 2) int64 (u, v) source pixels

    # REQ-023: number of lifted points
    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


# REQ-023: O_add = Unproject(D_I, M_mask, Odom); empty masks (or no valid depth) give no object
def unproject_masks(obs: Observation, masks: Sequence[tuple[int, np.ndarray]],
                    labels: Sequence[int] | Mapping[int, int] | None = None,
                    scores: Sequence[float] | Mapping[int, float] | None = None,
                    depth_min: float = 0.0, depth_max: float | None = None,
                    min_points: int = 1, erode_px: int = 0, erode_min_keep: float = 0.5,
                    depth_outlier_k: float | None = None,
                    depth_outlier_floor: float = 0.05, edge_jump: float = 0.0,
                    edge_rel: float = 0.0, edge_radius: int = 1,
                    component_voxel: float | None = None) -> list[ObservedObject]:
    """labels/scores are per mask (same order) or keyed by track id.
    component_voxel (not None) keeps only the largest 26-connected voxel component of each lift.
    Defaults reproduce plain back-projection; erosion/outlier removal follow D-106."""
    # REQ-023: per-mask value from a sequence (mask order) or a mapping (track id)
    def _get(src, i, tid, default):
        if src is None:
            return default
        return src[tid] if isinstance(src, Mapping) else src[i]

    out = []
    for i, (tid, mask) in enumerate(masks):
        # REQ-023: mask erosion and depth outlier removal before lifting (D-106)
        if erode_px > 0:
            mask = erode_mask(mask, erode_px, erode_min_keep)
        if edge_jump > 0 or edge_rel > 0:
            mask = depth_edge_mask(obs.depth, mask, edge_jump, edge_rel, edge_radius)
        if depth_outlier_k is not None:
            mask = depth_inlier_mask(obs.depth, mask, depth_outlier_k, depth_outlier_floor,
                                     depth_min, depth_max)
        X_w, colors, pix = backproject(obs.depth, obs.K, obs.pose, rgb=obs.rgb, mask=mask,
                                       depth_min=depth_min, depth_max=depth_max)
        if component_voxel is not None and X_w.shape[0] > 0:
            # REQ-023: keep the largest connected component of the lifted points (iter 15, revises D-036)
            keep = largest_component_mask(X_w, float(component_voxel))
            X_w, colors = X_w[keep], colors[keep]
            pix = pix[keep] if pix is not None else None
        if X_w.shape[0] < max(1, int(min_points)):
            continue
        out.append(ObservedObject(int(tid), X_w, colors, int(_get(labels, i, tid, -1)),
                                  float(_get(scores, i, tid, 0.0)), int(obs.t), pix))
    return out


# REQ-024: voxel de-duplication of merged points, first point per voxel kept (D-037; default 0.02 m, D-101)
def _new_points_mask(existing: np.ndarray, new: np.ndarray, voxel_size: float | None) -> np.ndarray:
    if voxel_size is None or existing.shape[0] == 0 or new.shape[0] == 0:
        keep = np.ones(new.shape[0], dtype=bool)
    else:
        keep = ~np.isin(voxel_keys(new, voxel_size), voxel_keys(existing, voxel_size))
    if voxel_size is not None and keep.any():
        keys = voxel_keys(new[keep], voxel_size)
        _, first = np.unique(keys, return_index=True)
        sub = np.zeros(keys.shape[0], dtype=bool)
        sub[first] = True
        keep[np.nonzero(keep)[0]] = sub
    return keep


# REQ-024: append observed points to an instance and refresh last_seen
def _append(obj: ObjectInstance, points: np.ndarray, colors: np.ndarray, t: int,
            init_log_odds: float, voxel_size: float | None) -> None:
    keep = _new_points_mask(obj.points, points, voxel_size)
    obj.add_points(points[keep], colors[keep], init_log_odds)
    obj.last_seen = max(obj.last_seen, int(t))


# REQ-024: D-111 absorb decision of one new-id observation over the candidate objects (None = no absorb)
def _absorb_target(add: ObservedObject, cands: Sequence[ObjectInstance], key_sets: dict[int, np.ndarray],
                   trees: TreeCache, vsize: float, absorb_overlap: float | None, contact_dist: float,
                   contact_frac: float, contact_same_label: bool, excl: set[int],
                   absorb_same_label: bool, absorb_overlap_rev: float | None) -> ObjectInstance | None:
    best, best_f = None, -1.0
    if absorb_overlap is not None:
        ak = voxel_keys(add.points, vsize)
        ak_u = np.unique(ak) if absorb_overlap_rev is not None else None
        for c in cands:
            if c.id not in key_sets:
                key_sets[c.id] = voxel_keys(c.points, vsize)
            f = float(np.isin(ak, key_sets[c.id]).mean())
            if f < absorb_overlap:
                continue
            # REQ-024: label-aware / reverse-overlap absorb variants (iter 15 experiment)
            if absorb_same_label and not (add.label >= 0 and c.num_classes > 0
                                          and c.label == add.label):
                continue
            if absorb_overlap_rev is not None:
                ck = np.unique(key_sets[c.id])
                if np.isin(ck, ak_u).sum() / len(ck) < absorb_overlap_rev:
                    continue
            if f > best_f:
                best, best_f = c, f
    if not (best is not None and best_f >= absorb_overlap):
        best = None
    if best is None and contact_dist > 0:
        # REQ-024: contact absorb into the neighbor with the highest contact fraction
        best, best_f = None, -1.0
        for c in cands:
            if c.label in excl or (contact_same_label and c.label != add.label):
                continue
            f = trees.fraction(c.id, c.points, add.points, contact_dist)
            if f > best_f:
                best, best_f = c, f
        if best_f < contact_frac:
            best = None
    return best


# REQ-024: D-111 candidates are the ACTIVE objects of O_nbr with points at entry
def _absorb_cands(o_nbr: Sequence[ObjectInstance], absorb_overlap: float | None,
                  contact_dist: float) -> list[ObjectInstance]:
    if absorb_overlap is None and not contact_dist > 0:
        return []
    return [o for o in o_nbr if o.status == ObjectStatus.ACTIVE and o.num_points > 0]


# REQ-024, REQ-036: D-120 absorb plan: new-id observations -> absorbing O_nbr instance, decided before ObjMerge
def absorb_targets(o_nbr: Sequence[ObjectInstance], o_add: Iterable[ObservedObject],
                   voxel_size: float | None = None, absorb_overlap: float | None = None,
                   global_map: GlobalMap | None = None, contact_dist: float = 0.0,
                   contact_frac: float = 0.5, contact_same_label: bool = False,
                   contact_exclude: Sequence[int] = (), absorb_same_label: bool = False,
                   absorb_overlap_rev: float | None = None) -> dict[int, int]:
    """Returns {new id: absorbing object id} exactly as obj_merge would absorb them; mutates nothing."""
    known = {o.id for o in o_nbr}
    cands = _absorb_cands(o_nbr, absorb_overlap, contact_dist)
    plan: dict[int, int] = {}
    if not cands:
        return plan
    trees = TreeCache()
    excl = {int(x) for x in contact_exclude}
    key_sets: dict[int, np.ndarray] = {}
    vsize = voxel_size if voxel_size is not None else 0.02
    for add in o_add:
        if add.id in known or (global_map is not None and add.id in global_map) or add.num_points == 0:
            continue
        best = _absorb_target(add, cands, key_sets, trees, vsize, absorb_overlap, contact_dist,
                              contact_frac, contact_same_label, excl, absorb_same_label,
                              absorb_overlap_rev)
        if best is not None:
            plan[add.id] = best.id
    return plan


# REQ-024: ObjMerge(O_nbr, O_add): same id -> add points to that neighbor, else new instance.
# semantic_update(obj, z) is the hook for Eq. (10) fusion (REQ-041, M5); None = no label update.
def obj_merge(o_nbr: Sequence[ObjectInstance], o_add: Iterable[ObservedObject], num_classes: int,
              init_log_odds: float = 0.0, voxel_size: float | None = None,
              semantic_update: Callable[[ObjectInstance, int], None] | None = None,
              global_map: GlobalMap | None = None, absorb_overlap: float | None = None,
              absorbed: dict[int, int] | None = None, contact_dist: float = 0.0,
              contact_frac: float = 0.5, contact_same_label: bool = False,
              contact_exclude: Sequence[int] = (), absorb_same_label: bool = False,
              absorb_overlap_rev: float | None = None,
              absorb_plan: Mapping[int, int] | None = None) -> list[ObjectInstance]:
    """Returns the updated O_nbr (existing references mutated, new instances appended).
    If global_map is given, an id stored in the map but outside O_nbr reuses that map object.
    absorb_overlap (D-111): a new id whose points fall >= this fraction into the voxels of an ACTIVE
    original O_nbr object is appended to that object; absorbed[new_id] = object id.
    contact_dist > 0: if D-111 did not fire, a new id with >= contact_frac of its points within
    contact_dist of an ACTIVE O_nbr object (label not in contact_exclude, equal to add.label when
    contact_same_label) is absorbed into the object with the highest contact fraction.
    absorb_same_label / absorb_overlap_rev (iter 15 experiments): a D-111 candidate must also have
    label == add.label, and/or hold |A & C| / |C| >= absorb_overlap_rev of its own voxels.
    absorb_plan (D-120): decision from absorb_targets() taken earlier; a new id is absorbed into
    absorb_plan[id] when present, otherwise it creates an instance (nothing is re-decided)."""
    merged = {o.id: o for o in o_nbr}
    order = [o.id for o in o_nbr]
    # REQ-024: D-111 candidates are the objects of O_nbr at entry; voxel key sets built lazily
    cands = _absorb_cands(o_nbr, absorb_overlap, contact_dist)
    trees = TreeCache()
    excl = {int(x) for x in contact_exclude}
    key_sets: dict[int, np.ndarray] = {}
    vsize = voxel_size if voxel_size is not None else 0.02  # D-101 default grid when merge is voxel-free
    for add in o_add:
        # REQ-027: existing ids receive their points; only unseen ids create new instances
        obj = merged.get(add.id)
        if obj is None and global_map is not None and add.id in global_map:
            obj = global_map[add.id]
            merged[add.id] = obj
            order.append(add.id)
        if obj is None:
            best = None
            if absorb_plan is not None:
                best = merged.get(absorb_plan.get(add.id, -1))
            elif cands and add.num_points > 0:
                best = _absorb_target(add, cands, key_sets, trees, vsize, absorb_overlap, contact_dist,
                                      contact_frac, contact_same_label, excl, absorb_same_label,
                                      absorb_overlap_rev)
            if best is not None:
                # REQ-024: D-111 absorb into the neighbor holding most of the new points' voxels
                _append(best, add.points, add.colors, add.t, init_log_odds, voxel_size)
                if semantic_update is not None and add.label >= 0:
                    semantic_update(best, add.label)
                if absorbed is not None:
                    absorbed[add.id] = best.id
                continue
            obj = ObjectInstance.create(add.id, num_classes, add.t)
            obj.exist_log_odds = float(init_log_odds)  # REQ-035: D-113 initial existence log-odds
            merged[add.id] = obj
            order.append(add.id)
        _append(obj, add.points, add.colors, add.t, init_log_odds, voxel_size)
        if semantic_update is not None and add.label >= 0:
            semantic_update(obj, add.label)
    return [merged[k] for k in order]


# REQ-024: MapMerge(M, O_nbr): write O_nbr back; an id already stored as a different object
# (outside the neighborhood) receives the points instead of being replaced
def map_merge(global_map: GlobalMap, o_nbr: Iterable[ObjectInstance],
              voxel_size: float | None = None) -> GlobalMap:
    for obj in o_nbr:
        if obj.id not in global_map:
            global_map.add(obj)
            continue
        cur = global_map[obj.id]
        if cur is obj:
            continue
        keep = _new_points_mask(cur.points, obj.points, voxel_size)
        cur.add_points(obj.points[keep], obj.colors[keep], obj.log_odds[keep])
        cur.last_seen = max(cur.last_seen, obj.last_seen)
    return global_map


# REQ-024: drop the points of `objs` outside their largest connected component (iter-15 experiment)
def prune_components(objs: Iterable[ObjectInstance], voxel: float, min_points: int = 1) -> dict[int, int]:
    removed: dict[int, int] = {}
    for obj in objs:
        if obj.num_points < max(1, int(min_points)):
            continue
        keep = largest_component_mask(obj.points, voxel)
        if not keep.all():
            obj.remove_points(~keep)
            removed[obj.id] = int((~keep).sum())
    return removed
