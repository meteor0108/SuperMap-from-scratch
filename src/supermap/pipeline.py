"""End-to-end SuperMap frame processing (REQ-060, REQ-061; Eq. 2 p.4, Supp. Alg. 1 lines 3-30, Fig. 2).

Eq. (2): P(I_t, M_t, P_t | M_{t-1}, Q_t) = P(P_t | .) x P(I_t | ., P_t) x P(M_t | ., P_t, I_t),
so each frame runs pose -> instance association -> map update.

Symbol map (paper -> code):
    Q_t = {C_t, D_t}      -> Observation (rgb, depth, K, pose, t, stamp)
    P_t = T_WB T_BC        -> SuperMapPipeline.pose(obs) (Eq. 3, REQ-009)
    S = (B_det, I, D_I, Odom) -> DetectionObservation (REQ-015)
    M_t                    -> self.map (GlobalMap), dense geometry in self.dense_map
    T (tracker)            -> self.tracker (Tracker, ids from M.new_id)
    O_nbr                  -> GetNeighborObjects (REQ-021)
    O_front/O_on/O_behind  -> CompareDepth (REQ-032); O_neg <- O_front
    T_trk                  -> Tracker.associate (REQ-020)
    M_mask, O_add          -> SAM2 masks (REQ-014), Unproject (REQ-023)
    ObjNegate/ObjMerge/MapMerge -> REQ-036 / REQ-024
    G = (V, E_s, E_t)      -> self.graph (SceneGraph, REQ-046)

Stage hooks: callbacks fn(stage, ctx) receive the REQ-060 stages in this order on a
segmentation frame: pose, neighbors, compare_depth, update_from_map, associate, masks,
unproject, semantic_check, negate, merge, map_merge. Hooks registered with extra=True also
receive: prune, low_confidence, remove_tracks, predict_only, geometric_update, mapping,
scene_graph (all after map_merge on a segmentation frame).
Eq. (8) runs inside the negate stage and Eq. (10) inside merge (SPEC REQ-060 ambiguity, D-089).
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from supermap.consistency.existence import existence_prune
from supermap.consistency import (ConsistencyParams, PointState, UpdateResult, compare_depth,
                                  geometric_update, obj_negate, point_states, prune_objects,
                                  semantic_consistency)
from supermap.geometry import DenseMap, compose_camera_pose
from supermap.graph import SceneGraph, SceneGraphConfig
from supermap.map import GlobalMap, ObjectInstance, ObjectStatus
from supermap.perception import GroundingDINODetector, PromptList, SAM2Segmenter, make_detection_observation
from supermap.pose import PoseProvider
from supermap.scheduler import ModuleScheduler
from supermap.semantics import (confusion_from_config, init_from_first_observation, remove_low_confidence,
                                semantic_update)
from supermap.tracking.bytetrack import _object_geometry
from supermap.tracking.gate3d import detection_centroids
from supermap.tracking.mask_ops import resolve_mask_overlaps
from supermap.tracking import Tracker, as_prompts, get_neighbor_objects, map_merge, obj_merge, unproject_masks
from supermap.tracking.instance_ops import absorb_targets, prune_components
from supermap.types import Observation

log = logging.getLogger(__name__)

# REQ-060: canonical per-frame stage order (Alg. 1 lines 3-30)
STAGES = ("pose", "neighbors", "compare_depth", "update_from_map", "associate", "masks",
          "unproject", "semantic_check", "negate", "merge", "map_merge")
# REQ-060, REQ-061: stages reported only to hooks registered with extra=True
EXTRA_STAGES = ("prune", "low_confidence", "remove_tracks", "predict_only", "geometric_update",
                "mapping", "scene_graph")

Hook = Callable[[str, dict], None]


# REQ-060, REQ-061: what happened in one processed frame
@dataclass
class FrameResult:
    t: int
    stamp: float
    ran: dict[str, bool] = field(default_factory=dict)  # module -> ran this frame
    num_detections: int = 0
    t_trk_ids: list[int] = field(default_factory=list)
    o_nbr_ids: list[int] = field(default_factory=list)
    o_front_ids: list[int] = field(default_factory=list)
    o_on_ids: list[int] = field(default_factory=list)
    o_behind_ids: list[int] = field(default_factory=list)
    o_neg_ids: list[int] = field(default_factory=list)
    o_add_ids: list[int] = field(default_factory=list)
    new_ids: list[int] = field(default_factory=list)
    removed_ids: list[int] = field(default_factory=list)
    absorbed: dict[int, int] = field(default_factory=dict)  # D-111: absorbed track id -> instance id
    absorb_plan: dict[int, int] = field(default_factory=dict)  # REQ-036: D-120 absorb decision taken before the semantic check
    component_pruned: dict[int, int] = field(default_factory=dict)  # REQ-024: iter-15 points dropped outside the largest component
    map_matched: int = 0  # REQ-022: leftover detections matched to a map object id
    map_dropped: int = 0  # REQ-026: duplicate detection boxes dropped
    gate_rejected: int = 0  # REQ-022: pairs rejected by the 3D gate (iter 15)


# REQ-060: ids of a list of objects
def _ids(objs: Iterable) -> list[int]:
    return [int(o.id) for o in objs]


# REQ-060, REQ-061: SuperMap system: geometric, instance and topological layers (Fig. 2)
class SuperMapPipeline:
    """Online processing of an Observation stream. detector/segmenter are injectable:
    detector.detect(rgb, t) -> Detections; segmenter.segment_tracks(rgb, [(id, box)]) -> [(id, mask)]."""

    # REQ-060: build all modules from the config; InitMap() and InitTracker() (Alg. 1 line 2)
    def __init__(self, cfg: dict, detector=None, segmenter=None,
                 pose_provider: PoseProvider | None = None, T_BC: np.ndarray | None = None,
                 scheduler: ModuleScheduler | None = None) -> None:
        self.cfg = cfg
        self.prompts = PromptList.from_config(cfg["perception"]["prompts"], cfg["perception"].get("prompt_aliases"))
        self.C = self.prompts.C
        self.detector = detector if detector is not None else GroundingDINODetector.from_config(cfg["perception"])
        self.segmenter = segmenter if segmenter is not None else SAM2Segmenter.from_config(cfg["perception"])
        self.pose_provider = pose_provider
        self.T_BC = np.asarray(cfg["extrinsics"]["T_BC"] if T_BC is None else T_BC, dtype=np.float64)
        self.params = ConsistencyParams.from_config(cfg)
        self.confusion = confusion_from_config(cfg["semantics"]["confusion"], self.C)
        self.p_min = float(cfg["semantics"]["p_min"])
        tcfg = cfg["tracking"]
        self.max_range = tcfg["neighbors"]["max_range"]
        # REQ-021: neighbor criterion (D-110); REQ-024: absorb rule (D-111)
        self.nbr_mode = tcfg["neighbors"].get("mode", "points")
        self.nbr_samples = int(tcfg["neighbors"].get("sample_points", 256))
        self.absorb_overlap = tcfg["merge"].get("absorb_overlap", 0.5)
        mcfg = tcfg["merge"]
        self.contact = dict(contact_dist=float(mcfg.get("contact_dist", 0.0)),
                            contact_frac=float(mcfg.get("contact_frac", 0.5)),
                            contact_same_label=bool(mcfg.get("contact_same_label", False)),
                            contact_exclude=[self.prompts.index(p) for p in mcfg.get("contact_exclude", [])
                                             if " ".join(str(p).strip().lower().split()) in self.prompts.phrases])
        # REQ-024: label-aware / reverse-overlap absorb variants (iter 15 experiment, default off)
        self.absorb_variants = dict(
            absorb_same_label=bool(mcfg.get("absorb_same_label", False)),
            absorb_overlap_rev=(None if mcfg.get("absorb_overlap_rev") is None
                                else float(mcfg["absorb_overlap_rev"])))
        self.min_points = int(tcfg["unproject"]["min_points"])
        # REQ-023: mask erosion / depth outlier settings of Unproject (D-106), overlap rule (D-107)
        ucfg = tcfg["unproject"]
        self.unproject_cfg = {"erode_px": int(ucfg.get("erode_px", 0)),
                              "erode_min_keep": float(ucfg.get("erode_min_keep", 0.5)),
                              "depth_outlier_k": ucfg.get("depth_outlier_k"),
                              "depth_outlier_floor": float(ucfg.get("depth_outlier_floor", 0.05)),
                              "edge_jump": float(ucfg.get("edge_jump", 0.0)),
                              "edge_rel": float(ucfg.get("edge_rel", 0.0)),
                              "edge_radius": int(ucfg.get("edge_radius", 1))}
        if ucfg.get("component_voxel") is not None:  # REQ-023: iter-15 switch, key present only when on
            self.unproject_cfg["component_voxel"] = float(ucfg["component_voxel"])
        self.resolve_overlaps = ucfg.get("resolve_overlaps")
        # REQ-023: classes whose masks win shared pixels before smaller-wins (D-122); unknown names skipped
        self.overlap_background = [self.prompts.index(p) for p in ucfg.get("overlap_background", [])
                                   if " ".join(str(p).strip().lower().split()) in self.prompts.phrases]
        self.merge_voxel = tcfg["merge"]["voxel_size"]
        cv = tcfg["merge"].get("component_voxel")  # REQ-024: iter-15 instance-level largest-component pruning; None = off
        self.merge_component_voxel = None if cv is None else float(cv)
        self.depth_min = float(cfg["geometry"].get("depth_min", 0.0))
        self.depth_max = cfg["geometry"].get("depth_max")
        self.scheduler = scheduler if scheduler is not None else ModuleScheduler.from_config(cfg.get("rates"))
        self.graph_config = SceneGraphConfig.from_config(cfg["scene_graph"])
        self._hooks: list[tuple[Hook, bool]] = []
        self.last_known: dict[int, dict] = {}  # id -> centroid/bbox before removal
        self.frames: list[FrameResult] = []
        self.skipped_frames = 0
        self.removed_by_existence = 0  # REQ-035: D-113 count of existence removals
        self.last_obs: Observation | None = None
        self.init_map()
        self.init_tracker()

    # REQ-060: InitMap(): empty instance map M_0, dense map and scene graph
    def init_map(self) -> GlobalMap:
        self.map = GlobalMap(self.prompts.phrases)
        self.dense_map = DenseMap.from_config(self.cfg["geometry"])
        self.graph = SceneGraph(self.graph_config)
        return self.map

    # REQ-060: InitTracker(): hybrid tracker whose ids come from the map allocator (REQ-024)
    def init_tracker(self) -> Tracker:
        self.tracker = Tracker.from_config(self.cfg["tracking"], id_allocator=self.map.new_id)
        return self.tracker

    InitMap = init_map
    InitTracker = init_tracker

    # REQ-060: register a stage callback; extra=True also reports EXTRA_STAGES
    def add_hook(self, fn: Hook, extra: bool = False) -> None:
        self._hooks.append((fn, bool(extra)))

    # REQ-060: notify hooks of a finished stage
    def _emit(self, stage: str, **ctx) -> None:
        is_extra = stage in EXTRA_STAGES
        for fn, extra in self._hooks:
            if extra or not is_extra:
                fn(stage, ctx)

    # REQ-060 step (0), REQ-009: P_t = T_WB^(t) T_BC (Eq. 3) from the PoseProvider
    def pose(self, obs: Observation) -> Observation:
        """Returns Q_t with pose P_t. Without a PoseProvider the stream's P_t is used as given."""
        if self.pose_provider is None:
            return obs
        P_t = compose_camera_pose(self.pose_provider.get(obs.frame_id), self.T_BC)
        return dataclasses.replace(obs, pose=P_t)

    # REQ-060, REQ-061: process one frame; modules run according to the scheduler
    def process_frame(self, obs: Observation) -> FrameResult | None:
        """Returns None when the frame falls between pose outputs (D-088)."""
        sch = self.scheduler
        if not sch.due("pose", obs.stamp):
            self.skipped_frames += 1
            return None
        res = FrameResult(t=obs.t, stamp=obs.stamp)
        obs = self.pose(obs)
        self.last_obs = obs
        res.ran["pose"] = True
        self._emit("pose", obs=obs)
        seg = sch.due("segmentation", obs.stamp)
        mapping = sch.due("mapping", obs.stamp)
        graph = sch.due("scene_graph", obs.stamp)
        res.ran.update(segmentation=seg, mapping=mapping, scene_graph=graph)
        if seg:
            self._instance_step(obs, res)
        else:
            # D-085: between segmentation frames the tracker only predicts (Eq. 6)
            self.tracker.predict()
            self._emit("predict_only", obs=obs)
            if mapping:
                self._geometric_step(obs, res)
        if mapping:
            # REQ-011, REQ-061: dense 3D mapping; instance bboxes follow their points (D-011, D-090)
            self.dense_map.integrate(obs)
            self._emit("mapping", obs=obs, num_points=len(self.dense_map))
        if graph:
            # REQ-046, REQ-061: 4D scene graph update
            self.graph.update(self.map, obs.t)
            self._emit("scene_graph", graph=self.graph)
        self.frames.append(res)
        return res

    # REQ-060: Supp. Alg. 1 lines 5-30 for one segmentation frame
    def _instance_step(self, obs: Observation, res: FrameResult) -> None:
        # S = (B_det, I, D_I, Odom) (Alg. 1 line 3, REQ-015)
        S = make_detection_observation(obs, self.detector.detect(obs.rgb, t=obs.t))
        res.num_detections = len(S.detections)

        # (1) O_nbr = GetNeighborObjects(M, Odom)
        o_nbr = get_neighbor_objects(self.map, obs.pose, obs.K, (obs.height, obs.width), self.max_range,
                                    self.nbr_mode, self.nbr_samples)
        res.o_nbr_ids = _ids(o_nbr)
        self._emit("neighbors", o_nbr=o_nbr)

        # (2) (O_front, O_on, O_behind) <- CompareDepth(D_I, O_nbr, Odom); O_neg <- O_front
        cd = compare_depth(obs, o_nbr, self.params)
        res.o_front_ids, res.o_on_ids, res.o_behind_ids = _ids(cd.o_front), _ids(cd.o_on), _ids(cd.o_behind)
        self._emit("compare_depth", cd=cd)

        # (3) UpdateFromMap(O_on) on alive tracks (Eq. 5, 6) + re-activation, then ByteTrack Associate
        self.tracker.predict()
        # REQ-022, REQ-028: Eq. (5) X_O excludes only the points Eq. (9) marks Disappeared in this
        # frame (stale, about to be pruned); occluded and out-of-view points stay in X_O (D-105)
        current = {i: np.asarray(s) != PointState.DISAPPEARED for i, s in cd.states.items()}
        self.tracker.update_from_map(cd.o_on, obs.pose, obs.K, point_mask=current)
        self.tracker.reactivate_from_map(cd.o_on, obs.pose, obs.K, point_mask=current,
                                         image_size=(obs.height, obs.width))
        self._emit("update_from_map", o_on=cd.o_on)
        if self.tracker.gate_3d > 0.0:
            # REQ-022: 3D gate (iter 15): detection centroids from depth vs masked centroids of mapped instances
            det_c = detection_centroids(obs.depth, obs.K, obs.pose, S.detections.boxes,
                                        depth_min=self.depth_min, depth_max=self.depth_max)
            trk_c = {int(o.id): _object_geometry(o, current.get(int(o.id)))[0]
                     for o in o_nbr if o.num_points > 0}
            t_trk = self.tracker.associate(S, det_centroids=det_c, track_centroids=trk_c)
        else:
            t_trk = self.tracker.associate(S)
        res.gate_rejected = self.tracker.gate_rejected
        res.t_trk_ids = [td.track_id for td in t_trk]
        res.map_matched, res.map_dropped = self.tracker.map_matched, self.tracker.map_dropped
        self._emit("associate", t_trk=t_trk)

        # (4) M_mask <- SAM2(I, T_trk boxes); O_add <- Unproject(D_I, M_mask, Odom)
        masks = self.segmenter.segment_tracks(obs.rgb, as_prompts(t_trk))
        self._emit("masks", masks=masks)
        labels = {td.track_id: td.label for td in t_trk}
        scores = {td.track_id: td.score for td in t_trk}
        # REQ-023: Unproject with mask erosion / depth outlier removal (D-106) after overlap resolution (D-107)
        # REQ-023: background masks win shared pixels (D-122), then smaller-wins among the rest (D-107)
        masks = resolve_mask_overlaps(masks, self.resolve_overlaps, labels=labels,
                                      background=self.overlap_background)
        o_add = unproject_masks(obs, masks, labels, scores, self.depth_min, self.depth_max, self.min_points,
                                **self.unproject_cfg)
        res.o_add_ids = _ids(o_add)
        self._emit("unproject", o_add=o_add)

        # (5) semantic consistency check: O_on objects not in T_trk join O_neg (Alg. 1 lines 21-25)
        # REQ-036, REQ-024: D-120 the D-111 absorb decision is taken here; an O_on object that absorbs one of this
        # frame's detections is supported by the current segmentation and is not semantically inconsistent
        plan = absorb_targets(o_nbr, o_add, self.merge_voxel, self.absorb_overlap, global_map=self.map,
                              **self.contact, **self.absorb_variants)
        res.absorb_plan = dict(plan)
        supported = sorted(set(res.t_trk_ids) | set(plan.values()))
        o_neg = semantic_consistency(cd.o_on, supported, cd.o_neg)
        res.o_neg_ids = _ids(o_neg)
        self._emit("semantic_check", o_neg=o_neg)

        # ObjNegate (line 26) and Eq. (8) on the other neighbor points, one update per point (D-051)
        obj_negate(o_nbr, o_neg, self.params, cd.states)
        geometric_update(o_nbr, obs, self.params, cd=cd, skip_ids=res.o_neg_ids, do_prune=False)
        self._emit("negate", o_neg=o_neg)

        # ObjMerge(O_nbr, O_add) (line 28) with Eq. (10) fusion / first-observation init (REQ-040..043)
        known = set(self.map.ids)
        sem_hook = self._semantic_hook(obs, o_nbr, o_add, cd.states, known)
        absorbed: dict[int, int] = {}
        merged = obj_merge(o_nbr, o_add, self.C, init_log_odds=self.params.initial_log_odds,
                           voxel_size=self.merge_voxel, semantic_update=sem_hook, global_map=self.map,
                           absorb_overlap=self.absorb_overlap, absorbed=absorbed, absorb_plan=plan, **self.contact,
                           **self.absorb_variants)
        # REQ-024: D-111 tracks of absorbed ids continue under the absorbing instance id
        for old, new in absorbed.items():
            self.tracker.remap_id(old, new)
        res.absorbed = dict(absorbed)
        if self.merge_component_voxel is not None:
            # REQ-024: iter-15 keep the largest connected component of instances that received points
            touched = {a.id for a in o_add} | set(absorbed.values())
            res.component_pruned = prune_components([o for o in merged if o.id in touched],
                                                    self.merge_component_voxel)
        res.new_ids = [o.id for o in merged if o.id not in known]
        self._emit("merge", o_nbr=merged)

        # MapMerge(M, O_nbr) (line 29)
        map_merge(self.map, merged, self.merge_voxel)
        self._emit("map_merge", map=self.map)

        self._finish_updates(obs, merged, res, semantic=True)

    # REQ-043, REQ-040: Eq. (10) gate from pre-merge Eq. (9) states; new instances use D-060
    def _semantic_hook(self, obs: Observation, o_nbr: Sequence[ObjectInstance], o_add,
                       states: dict[int, np.ndarray], known: set[int]) -> Callable[[ObjectInstance, int], None]:
        observable: dict[int, np.ndarray] = {}

        # REQ-043: lazily computed observable mask of a pre-existing instance (D-091 outside O_nbr)
        def mask_of(obj: ObjectInstance) -> np.ndarray:
            if obj.id not in observable:
                s = states.get(obj.id)
                if s is None:
                    _, s = point_states(obj, obs, self.params.tau_eps)
                observable[obj.id] = np.asarray(s) == PointState.OBSERVABLE
            return observable[obj.id]

        # REQ-043: same-id instances get their pre-merge mask up front (as before D-111)
        for add in o_add:
            if add.id in known and add.id in self.map:
                mask_of(self.map[add.id])

        # REQ-041, REQ-043: per-instance Eq. (10) update used as the obj_merge hook
        def hook(obj: ObjectInstance, z: int) -> None:
            if obj.id in known:  # REQ-041: pre-existing instance (also an absorbing one, D-111)
                semantic_update(obj, z, self.confusion, mask_of(obj))
            else:
                init_from_first_observation(obj, z, self.confusion)
        return hook

    # REQ-039, REQ-052: remember centroid/bbox of objects before they lose their points
    def _remember(self, objs: Iterable[ObjectInstance]) -> None:
        for o in objs:
            if o.num_points > 0:
                lo, hi = o.bbox
                self.last_known[o.id] = {"centroid": o.centroid.copy(), "bbox": (lo.copy(), hi.copy())}

    # REQ-035, REQ-044, REQ-024: pruning, low-confidence removal, tracks of REMOVED objects dropped
    def _finish_updates(self, obs: Observation, objs: Sequence[ObjectInstance], res: FrameResult,
                        semantic: bool, update: UpdateResult | None = None) -> None:
        self._remember(objs)
        if update is None:
            update = prune_objects(objs, self.params, obs.t)
            existence_prune(objs, self.params, obs.t, update)  # REQ-035: D-113
            self._emit("prune", update=update)
        removed = [e.id for e in update.removed]
        self.removed_by_existence += sum(1 for e in update.removed if e.reason == "existence")  # REQ-035
        if semantic:
            self._remember(self.map.active_objects())
            low = remove_low_confidence(self.map, self.p_min)
            removed += low
            self._emit("low_confidence", removed=low)
        dropped = []
        for o in self.map.removed_objects():
            if o.id in self.tracker.alive_ids:
                self.tracker.remove_track(o.id)
                dropped.append(o.id)
        res.removed_ids = sorted(set(removed))
        self._emit("remove_tracks", ids=dropped)

    # REQ-061, D-052: mapping frame without segmentation: Eq. (7)-(9) + Eq. (8) + pruning on O_nbr
    def _geometric_step(self, obs: Observation, res: FrameResult) -> None:
        o_nbr = get_neighbor_objects(self.map, obs.pose, obs.K, (obs.height, obs.width), self.max_range,
                                    self.nbr_mode, self.nbr_samples)
        res.o_nbr_ids = _ids(o_nbr)
        self._emit("neighbors", o_nbr=o_nbr)
        cd = compare_depth(obs, o_nbr, self.params)
        res.o_front_ids, res.o_on_ids, res.o_behind_ids = _ids(cd.o_front), _ids(cd.o_on), _ids(cd.o_behind)
        self._emit("compare_depth", cd=cd)
        self._remember(o_nbr)
        out = geometric_update(o_nbr, obs, self.params, cd=cd, do_prune=True)
        self._emit("geometric_update", result=out)
        self._finish_updates(obs, o_nbr, res, semantic=False, update=out.update)

    # REQ-001, REQ-060: consume a stream once, frame by frame
    def run(self, sequence: Iterable[Observation], on_frame: Callable[[FrameResult], None] | None = None
            ) -> GlobalMap:
        for obs in sequence:
            r = self.process_frame(obs)
            if r is not None and on_frame is not None:
                on_frame(r)
        return self.map

    # REQ-046, REQ-061: graph at the last processed time (updated if the last frame was not a graph step)
    def final_graph(self) -> SceneGraph:
        if self.last_obs is not None and (self.graph.t is None or self.graph.t < self.last_obs.t):
            self.graph.update(self.map, self.last_obs.t)
        return self.graph

    # REQ-005, REQ-037: centroid/bbox of an object, last known values when it has no points
    def object_geometry(self, obj: ObjectInstance):
        if obj.num_points > 0:
            lo, hi = obj.bbox
            return obj.centroid, (lo, hi)
        k = self.last_known.get(obj.id)
        return (None, None) if k is None else (k["centroid"], k["bbox"])
