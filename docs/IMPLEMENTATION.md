# SuperMap reimplementation: implementation log and the defaults the experiments chose

*한국어: [IMPLEMENTATION.ko.md](IMPLEMENTATION.ko.md)*

This document is the **engineering log** of the reimplementation of "SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation"
(RSS 2026). The headline results are in [README.md](../README.md). This file collects the algorithm description, how the system was built, what
was chosen for every detail the paper leaves open, which experiment drove each default, what was tried and rejected, the evaluation protocol,
the per-iteration results, and the limitations. The order of authority is the paper > the requirement specification extracted from it > the decision records; the official code was not used.
Decision IDs (D-xxx), issue IDs (I-xx-x), requirement IDs (REQ-xxx) and iteration numbers point to the development records (section 10) and are kept as they are.

## Contents

1. [Algorithm](#1-algorithm)
2. [Scope decisions](#2-scope-decisions)
3. [Code layout](#3-code-layout)
4. [Parameters](#4-parameters)
5. [Details the paper leaves open, and what was chosen](#5-details-the-paper-leaves-open-and-what-was-chosen)
6. [Tried and not adopted](#6-tried-and-not-adopted)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [Results history](#8-results-history)
9. [Known limitations and open issues](#9-known-limitations-and-open-issues)
10. [Development harness](#10-development-harness)

Unless stated otherwise, every number is a mean over the 40 ScanNet v2 validation scenes, mIoU excludes the background (wall, floor), and the label
transfer radius is r = 0.05 m. The pipeline is deterministic, so the same configuration gives the same result.

---

## 1. Algorithm

Every frame is processed in the order of the paper's supplementary Algorithm 1. \
Four modules run asynchronously at the rates of Sec. V-H: pose 10 Hz, 2D segmentation 1 Hz, dense mapping 3 Hz and scene graph 5 Hz. \
A scheduler driven by the data timestamps reproduces these periods, and the heavy modules (detection and segmentation) only run on their own ticks.

```
RGB C_t, depth D_t, intrinsics K, pose
 |- Geometric layer    pose provider -> Eq.(3) P_t = T_WB * T_BC -> back-projection -> colorized dense map (2 cm voxels)
 |- Instance layer     GroundingDINO(C_t, prompts) -> detections
 |                     neighbor objects O_nbr of the map around P_t
 |                     depth comparison (Eq. 7, 9) -> O_front / O_on / O_behind, O_neg <- O_front
 |                     Kalman prediction, 3D-centroid prior (Eq. 5, 6), ByteTrack association -> tracks
 |                     SAM2(C_t, track boxes) -> masks -> Unproject -> O_add
 |                     semantic consistency -> O_neg;  ObjNegate, ObjMerge, MapMerge
 |                     Eq.(8) point log-odds, object existence log-odds, pruning, Eq.(10) label fusion
 `- Topological layer  clustering -> On / Beside edges, temporal edges -> serialization -> VLM -> <answer> ids -> waypoints
```

### 1.1 Geometric layer (Sec. IV-A, Eq. 3)

An odometry source gives the body pose T_WB, and a fixed extrinsic T_BC gives the camera pose P_t = T_WB T_BC (Eq. 3). \
Each depth image is back-projected with the intrinsics (z-depth, valid depth > 0) and transformed into the world frame. \
The points are integrated into a colorized dense map that keeps one point per 2 cm voxel. \
The dense map is the geometric layer and the source of the points that the instance layer later compares against new depth.

### 1.2 Open-vocabulary detection (Sec. IV-B, V-B)

GroundingDINO detects objects from a text prompt list; the prompt list defines the label set. \
For each query phrase the class score is the maximum token probability of that phrase, and the box label is the class with the highest score. \
A low box threshold (0.1) lets weak detections reach the low-score pass of the tracker. \
Extra phrases per class (aliases) can be added; they are sent to the detector only and the box is labelled with the class they belong to.

### 1.3 Neighbor objects and depth comparison (Sec. IV-B3, Eq. 7-9)

The neighbor set O_nbr contains the map objects that can be seen from the current pose: a sample of up to 256 points per object is projected, and the object is a neighbor if one of them lands in the image within 10 m. \
The points of these objects are projected into the current depth image. \
The residual between the projected depth and the measured depth classifies each point as Observable, Disappeared (the measured surface is behind the point) or Unobservable (the measured surface is in front of it), with the threshold tau_eps = 0.1 m (Eq. 9). \
Points outside the view or on invalid depth receive no update. \
The object-level state is the majority of the point states, and objects are split into O_front, O_on and O_behind; O_front objects are put into O_neg, the set that is negated instead of reinforced.

### 1.4 Hybrid 2D-3D tracking (Sec. IV-B2, Eq. 4-6)

Each track has a six-dimensional state S (box center, size, center velocity) with a constant-velocity Kalman prediction (Eq. 4, 6). \
For an object in O_on that has an alive track, the 3D centroid of the object is projected into the current image and replaces the predicted center (Eq. 5), so that tracks follow the camera motion even when the 2D motion model fails. \
Points marked Disappeared in the current frame are left out of that centroid. \
Objects in O_on without an alive track are re-activated as candidate tracks with the projected centroid and size, and keep their id if they are matched. \
Association follows ByteTrack: a first pass matches high-score detections to tracks by IoU, with a class term in the cost, and a second pass matches the remaining tracks to low-score detections. \
The track id is the instance id.

### 1.5 SAM2 masks and Unproject (Alg. 1, lines 19-20)

SAM2 produces one mask per track box. \
Before lifting a mask with the depth, the mask is eroded by 2 px (the erosion is cancelled if it would remove more than half of the mask), pixels at depth discontinuities are dropped, depth outliers are removed with a 3-sigma median-absolute-deviation rule, and a pixel claimed by two masks is given to the smaller mask. \
The surviving pixels are unprojected into world-frame points, the set O_add.

### 1.6 ObjMerge and MapMerge (Sec. IV-B)

The paper assigns a unique id to every detection using 3D spatial consistency. \
A detection whose track id already exists adds its points to that instance, with 2 cm voxel de-duplication. \
A detection with a new id is absorbed into the neighboring instance whose voxels contain at least half of its points, whatever the class, and becomes a new instance otherwise. \
After a merge each instance keeps only its largest connected component of points (10 cm voxels, 26-connectivity), which removes stray fragments. \
MapMerge writes the updated neighbor objects back to the global map.

### 1.7 Probabilistic geometric update and pruning (Sec. IV-B3, Eq. 8)

Each point keeps a log-odds value that is updated once per frame with Eq. (8) using P(o|Q) = 0.7 for Observable, 0.3 for Disappeared and 0.5 for Unobservable. \
Objects in O_neg only receive the negative update on their in-view points. \
Points whose log-odds fall below -1 are pruned. \
Because a per-point rule cannot tell that a whole object is gone, each object also carries an existence log-odds: a frame in which it is Disappeared or in O_neg adds logit(0.3), an Observable frame adds logit(0.7), the value is clamped to [-3, 3], and below -1 the object is marked REMOVED and emitted as a removal event. \
A removed instance stays in the map as a record so that the scene graph can express the change.

### 1.8 Bayesian label fusion (Sec. IV-B4, Eq. 10)

Every instance holds a distribution over classes. \
It starts from the first observation and is updated by Bayes' rule with a confusion matrix (diagonal 0.8, off-diagonal uniform) for each new detection of an observable instance (Eq. 10). \
Instances whose best class posterior drops below 0.4 are removed. \
Fusing detections over time lets a label that is wrong in a single frame be overruled by later frames.

### 1.9 4D scene graph and VLM query (Sec. IV-C, IV-D)

Instances are clustered by centroid distance (single linkage, 3 m). \
Inside a cluster, the spatial predicates On (xy overlap above gamma = 0.05 and vertical contact within delta = 0.05 m) and Beside (within 0.5 m) are computed. \
When an object moves by 0.1 m or more, a new temporal node is created and linked to the previous one by a temporal edge, so the graph records where things were and when they moved or disappeared. \
A local subgraph around the robot (5 m) is serialized into a text schema of nodes, edges and temporal cues and sent to the VLM (Gemini) together with the query. \
The answer is read from `<answer>` tags and the ids inside are mapped to 3D waypoints (the object positions). \
Without a Gemini key the VLM call is skipped and the serialization is still written.

---

## 2. Scope decisions

All three layers of Fig. 2 (Geometric / Instance / Topological) are implemented. The scope decisions were made by the human and recorded in DECISIONS.

| Decision | Content |
|---|---|
| D-001 | SuperOdometry (LiDAR-IMU, ROS) is not integrated. A `PoseProvider` interface supplies **dataset poses**, and the extrinsic composition of Eq. (3), P_t = T_WB T_BC, is implemented (D-009: T_WB = dataset camera-to-world pose, T_BC = I). Input is RGB-D video. |
| D-002 | Replica office0 for development and smoke tests, ScanNet v2 as the main benchmark (downloaded by the human after ToS approval, D-109). The paper's real-robot data is not public, so REQ-008 is N/A (D-014). |
| D-003 | The VLM (Gemini) is a pluggable backend. Serialization, the `<answer>` parser and the waypoint conversion are verified deterministically. |

The input stream is RGB `C_t` (H x W x 3 uint8), depth `D_t` (H x W float32, m), intrinsics K and a pose. ScanNet is subsampled from 30 fps with stride 3
to a **10 Hz pose stream** (D-109; D-013: color 1296 x 968 is resized to the depth resolution 640 x 480, depth is uint16 mm), and Replica uses a nominal
10 Hz (D-006). The pose module is the frame clock; frames that arrive faster than 10 Hz are skipped (D-088).

---

## 3. Code layout

The pipeline does not depend on the experiment code: `src/supermap` + `configs/default.yaml` is enough for `python -m supermap.run`. Module
dependencies flow only in the order below (lower to higher), and heavy model loading is deferred to object construction.

### 3.1 Package map

| Path | What it implements | Paper reference |
|---|---|---|
| `types.py`, `map.py`, `camera.py`, `pose.py`, `config.py` | Shared types (Observation, ObjectInstance, GlobalMap), camera projection, `PoseProvider` | Sec. III |
| `datasets/` (`base.py`, `replica.py`, `scannet.py`) | RGB-D sequence loaders, nominal timestamps, invalid-pose rejection | Sec. V-B |
| `geometry/` (`extrinsics.py`, `backproject.py`, `dense_map.py`) | Eq. (3) composition, back-projection (OpenCV axes, z-depth), 2 cm voxel dense map | Sec. IV-A, Eq. 3 |
| `perception/` (`detector.py`, `prompts.py`, `nms.py`, `segmenter.py`, `observation.py`) | GroundingDINO detection with prompts and aliases, class-agnostic NMS, SAM2 box prompts, stream element S | Sec. IV-B, Alg. 1 line 3 |
| `tracking/state.py`, `bytetrack.py` | 6-D Kalman filter (Eq. 4, 6), two-stage ByteTrack association, UpdateFromMap (Eq. 5), re-activation, class cost | Sec. IV-B2, Eq. 4-6 |
| `tracking/neighbors.py` | GetNeighborObjects (sampled-point criterion) | Alg. 1 line 6 |
| `tracking/instance_ops.py`, `mask_ops.py` | Unproject, mask erosion / MAD / overlap resolution, absorb plan (`absorb_targets`), ObjMerge, MapMerge | Alg. 1 lines 19-20, 27-29 |
| `tracking/components.py` | Keep only the largest 26-connected component (`prune_components`, D-118) | choice outside the paper |
| `tracking/depth_edges.py` | Depth-edge (flying pixel) mask (D-116) | choice outside the paper |
| `tracking/gate3d.py`, `footprint.py`, `map_match.py`, `contact.py` | 3D centroid gate, footprint boxes, map matching, 3D contact absorb. All **off by default**, experiment only (section 6) | choice outside the paper |
| `consistency/` (`residual.py`, `classify.py`, `compare_depth.py`, `logodds.py`, `negate.py`, `update.py`, `existence.py`, `params.py`) | Eq. (7) residual, Eq. (9) states, CompareDepth, Eq. (8) log-odds, ObjNegate, pruning, object-level existence log-odds (D-113) | Sec. IV-B3, Eq. 7-9 |
| `semantics/` (`confusion.py`, `fusion.py`, `update.py`) | Eq. (10) Bayesian label fusion, low-confidence removal | Sec. IV-B4, Eq. 10 |
| `graph/` (`cluster.py`, `predicates.py`, `scene_graph.py`, `serialize.py`) | Clustering, On / Beside / Under, temporal nodes and edges, serialization | Sec. IV-C, IV-D |
| `vlm/` (`backend.py`, `parser.py`, `waypoints.py`, `grounding.py`) | Gemini backend with fallbacks, `<answer>` parser, waypoints | Sec. IV-D, V-F |
| `pipeline.py` | Frame processing order (Supp. Alg. 1) | Alg. 1 |
| `scheduler.py` | 10 / 1 / 3 / 5 Hz module rates, simulated | Sec. V-H |
| `run.py` | Entry point (`python -m supermap.run`), output writing | - |

### 3.2 Frame order (`Pipeline._instance_step`, on a segmentation frame)

1. Detect: GroundingDINO produces `B_det` (stream element S = (B_det, I, D_I, Odom); no stage hook, D-095).
2. Neighbors: `O_nbr = GetNeighborObjects(M, Odom)` (D-110).
3. CompareDepth: Eq. (7), (9) give the point states; objects are split into `O_front / O_on / O_behind`. `O_neg <- O_front`.
4. Tracking: tracker predict (Eq. 6) -> UpdateFromMap (alive tracks of `O_on`, Eq. 5) -> re-activation (D-035) -> ByteTrack Associate -> `T_trk`.
5. SAM2: masks from the `T_trk` boxes.
6. Unproject: overlap resolution (D-107), erosion and MAD outlier removal (D-106), depth-edge exclusion (D-116), then back-projection to `O_add`.
7. Absorb plan: the D-111 absorb targets are computed by the pure function `absorb_targets` (D-120).
8. Semantic check: an `O_on` object that is visible, not in `T_trk`, and does not absorb one of this frame's observations joins `O_neg` (Alg. 1 lines 21-25, D-120).
9. ObjNegate + Eq. (8): `O_neg` objects get ObjNegate (p_dis on in-view points), the others the per-state inverse sensor update, one update per point per frame (D-051). The object existence log-odds is updated here too (D-113).
10. ObjMerge: same ids add points (2 cm voxel de-duplication), new ids are absorbed according to the plan of step 7 (D-111), Eq. (10) label fusion (D-089). Then largest-component pruning of the instances that received points (D-118).
11. MapMerge: write `O_nbr` back to the map; same ids outside the neighborhood only get their points added.
12. Pruning / existence / low-confidence removal: points with L < L_prune removed, L_exist < -1 gives REMOVED, max_c P < p_min gives REMOVED, tracks of REMOVED objects dropped (D-089).

Frames without segmentation only run the tracker predict (D-085). A mapping frame integrates the dense map and additionally runs the geometric consistency on
`O_nbr` (Eq. 8, pruning) only ("geometric only", D-052, D-090).

### 3.3 Scheduler

The module rates of Paper Sec. V-H (pose 10 Hz, segmentation 1 Hz, mapping 3 Hz, scene graph 5 Hz) are reproduced as a **timestamp-based simulation** (D-087).
There are no threads: one loop turns modules on with `ModuleScheduler.due(module, stamp)`. Slot grid rule (D-086): slot k = floor((stamp - t0 + 1e-6) / T_m), and a module
runs when it reaches a slot it has not served yet. The rule "T_m elapsed since the last run" gives 25 mapping runs instead of the required 30 on the SPEC's 10 Hz x 100 frame case, so it was not used.
On this grid the 1 Hz slots are a subset of the 3 Hz slots, so a segmentation frame is always a mapping slot. Actual latencies are measured per module in the benchmark (section 8.5).

---

## 4. Parameters

Every value is in `configs/default.yaml` with a comment naming its REQ, paper location and decision, and the defaults below were checked against it. The ScanNet evaluation
uses an evaluation config that differs from the default only in the 19 prompts, the aliases and the name of the VLM key variable. **The paper gives numbers only for the
module rates**; everything else was chosen by us.

### 4.1 Geometry / detection / segmentation / label fusion

| config key | default | meaning | source |
|---|---|---|---|
| `rates.pose / segmentation / mapping / scene_graph` | 10 / 1 / 3 / 5 Hz | module rates | Paper Sec. V-H, Supp. VII-D (the only paper numbers) |
| `extrinsics.T_BC` | I_4 | body-to-camera extrinsic | D-009 |
| `geometry.depth_min / depth_max` | 0.0 (exclusive) / null | valid depth range | D-015 |
| `geometry.dense_map.voxel_size` | 0.02 m | dense map voxel, first point per voxel kept | D-016 |
| `perception.prompts` | 7 office classes (ScanNet config: 19) | label set c | D-025, D-109 |
| `perception.prompt_aliases` | {} (ScanNet config: cabinet, picture, bookshelf, refridgerator) | extra detector phrases, labelled as the class | D-115 |
| `perception.detector.model_id` | `IDEA-Research/grounding-dino-base` | detector | D-114 (D-004 was tiny) |
| `perception.detector.box_threshold` | 0.1 | = ByteTrack `low_thresh` | D-097 (D-020 had 0.35) |
| `perception.detector.nms_iou` | 0.5 | class-agnostic greedy NMS, before the D_high / D_low split | D-116 (revises D-020) |
| `perception.segmenter.model_id` | `facebook/sam2.1-hiera-small` | SAM2, one mask per box, logit threshold 0.0 | D-005, D-023 |
| `semantics.confusion.diag_prob` | 0.8 | confusion-matrix diagonal, off-diagonal uniform | D-061 |
| `semantics.p_min` | 0.4 | instance removed when max_c P(L=c) is below this | D-063 |

### 4.2 Geometric consistency (Eq. 7-9)

| config key | default | meaning | source |
|---|---|---|---|
| `consistency.tau_eps` | 0.1 m | Eq. (9) residual threshold | D-049 |
| `consistency.sigma` | 0.05 m | Gaussian likelihood helper only, not used for classification | D-049 |
| `consistency.p_obs / p_dis / p_unobs` | 0.7 / 0.3 / 0.5 | per-state inverse sensor probability (0.5 = no update) | D-049 |
| `consistency.L_prune` | -1.0 | point pruning threshold (L < L_prune) | D-049 |
| `consistency.L_init / L_clamp` | null / null | initial value = logit(p_obs), no clamp | D-048 |
| `consistency.exist_clamp` | 3.0 | symmetric clamp of the object existence log-odds | D-113 |
| `consistency.exist_prune` | -1.0 | object REMOVED when L_exist falls below this (null disables the rule) | D-113 |
| `consistency.exist_semantic` | true | semantic evidence (O_neg) also feeds L_exist | D-113 |
| `consistency.verdict_window_px` | 0 | window search for the object verdict (off, I-9-4 variant) | iter 10 |
| `consistency.exist_min_points / exist_min_share` | 0 / 0.0 | coverage gate on the existence update (off, I-9-4 variant) | iter 10 |

### 4.3 Tracking and merging

| config key | default | meaning | source |
|---|---|---|---|
| `tracking.kalman` | 6-D, dt 1, Q 25 / 0.390625, R 25, P0 100 / 39.0625 | ByteTrack constants frozen at a reference height of 100 px | D-030, D-031 |
| `tracking.bytetrack.high_thresh / new_track_thresh` | 0.35 / 0.35 | D_high cut, new-track cut | D-097 |
| `tracking.bytetrack.low_thresh` | 0.1 | D_low cut | D-097 |
| `tracking.bytetrack.match_thresh / second_match_thresh` | 0.8 / 0.5 | first / second match (upper bound on 1 - IoU) | D-033 |
| `tracking.bytetrack.track_buffer` | 30 | frames a track may stay unmatched | D-034 |
| `tracking.bytetrack.label_cost` | **1.0** | added to the first-association cost when labels differ (a hard class gate at match_thresh 0.8) | D-117 (revises the 0.0 of D-108) |
| `tracking.bytetrack.gate_3d` | 0.0 (off) | 3D centroid distance gate | tested in D-117, D-119 |
| `tracking.bytetrack.footprint_boxes / map_match_iou` | false / 0.0 | footprint boxes, map matching (off) | iter 12 |
| `tracking.motion_compensation / reactivation` | true / true | Eq. (5) compensation, re-activation of expired tracks | D-030, D-035 |
| `tracking.neighbors.mode / sample_points / max_range` | points / 256 / 10 m | GetNeighborObjects | D-110 |
| `tracking.unproject.min_points` | 1 | minimum points of an observation | D-036 |
| `tracking.unproject.erode_px / erode_min_keep` | 2 px / 0.5 | mask erosion, cancelled if less than 50 % remains | D-106 |
| `tracking.unproject.depth_outlier_k / floor` | 3.0 sigma_MAD / 0.05 m | depth outlier removal | D-106 |
| `tracking.unproject.resolve_overlaps` | smaller | overlapping pixels go to the smaller mask | D-107 |
| `tracking.unproject.edge_jump / edge_radius / edge_rel` | 0.1 m / 1 / 0.0 | a pixel is not lifted when the depth range (max - min) in its 3x3 window exceeds 0.1 m or the window contains an invalid depth | D-116 |
| `tracking.unproject.component_voxel` | null (off) | per-observation connected-component filter | tested in D-118 |
| `tracking.merge.voxel_size` | 0.02 m | de-duplication of merged points | D-101 |
| `tracking.merge.absorb_overlap` | 0.5 | a new id is absorbed into an existing object when this fraction of its points lies in the object's voxels, label-agnostic | D-111 |
| `tracking.merge.absorb_same_label` | false | absorb only into a same-label neighbor (off) | D-119 |
| `tracking.merge.absorb_overlap_rev` | null (off) | the neighbor must also be covered by this fraction | D-119 |
| `tracking.merge.component_voxel` | **0.10 m** | after ObjMerge an instance that received points keeps only its largest 26-connected component | D-118 (revises D-036) |
| `tracking.merge.contact_dist / contact_frac / contact_same_label / contact_exclude` | 0.0 (off) / 0.5 / false / [wall, floor] | 3D contact absorb | iter 13 |

### 4.4 Scene graph, VLM, datasets

| config key | default | meaning | source |
|---|---|---|---|
| `scene_graph.d_clu` | 3.0 m | single-linkage clustering | D-070 |
| `scene_graph.gamma / delta` | 0.05 / 0.05 m | IoU_xy threshold of On, z contact tolerance | D-071 |
| `scene_graph.d_bes` | 0.5 m | horizontal gap limit of Beside | D-073 |
| `scene_graph.d_move` | 0.1 m | movement that creates a temporal node | D-074 |
| `scene_graph.subgraph_radius` | 5.0 m | local subgraph | D-079 |
| `vlm.model / temperature` | gemini-2.0-flash / 0.0 | Paper Sec. V-F | D-080 |
| `vlm.fallback_models` | gemini-flash-lite-latest, gemini-3.8-flash, gemini-flash-latest, gemini-2.5-flash | next model only on 404 | D-099, D-100 |
| `datasets.scannet.depth_scale / nominal_fps` | 1000.0 / 30.0 | millimeters, 30 Hz capture | D-013 |
| `datasets.replica.nominal_fps` | 10.0 | Replica has no timestamps | D-006 |

---

## 5. Details the paper leaves open, and what was chosen

Each topic gives the decision ID, the choice, the measured effect and the alternatives.

### 5.1 Pose and extrinsics (D-001, D-009, D-018)
Without SuperOdometry, the dataset camera pose is T_WB and T_BC = I. The composition lives in one place, `geometry/extrinsics.py`. There is no alternative to compare.
ScanNet poses are stored as float32, so a few poses fail the SE(3) tolerance of 1e-6 and are dropped (3 of 2374 frames, I-8-4, no effect on evaluation).

### 5.2 Prompts and aliases (D-025, D-109, D-115)
The paper's prompt list is not public. ScanNet uses the 19 ScanNet20 classes minus otherfurniture (GT spelling kept, `refridgerator`), with background = wall, floor.
GroundingDINO-base never raised "cabinet" (max token probability 0.31) or "picture" (0.19) above the ByteTrack high threshold 0.35, but fired on "cupboard", "framed picture" and "painting".
`prompt_aliases` therefore puts aliases into the detector text only and labels the result as the class (C and Eq. 10 unchanged). Effect (all aliases): mIoU 48.8 -> 54.8, mAP50 21.2 -> 27.5, cabinet 0 -> 33 IoU,
picture 0 -> 65. Alternatives: an "office desk" alias gives desk 11 -> 48 but table 39 -> 25 and bed 60 -> 50, so it was left out; "curtains" takes shower curtain, also left out. DualMap Appendix I (a GPT-4o indoor vocabulary) was consulted.

### 5.3 Checkpoints (D-114, D-004, D-005)
The first choice was the simplest, tiny / small (D-004, D-005); on the human's instruction (the largest models that fit the paper's hardware, RTX 4090 Laptop 16 GB) they were compared.
GroundingDINO-base alone: mIoU 41.1 -> 48.8, Acc 46.5 -> 57.0, mAP50 18.1 -> 21.2, instances 69 -> 47. SAM2.1-large gives nothing alone (41.6) and -1.9 next to base.
Adopted: detector base, segmenter small. Peak GPU memory 4.9 GB, 79 + 60 ms per segmentation frame.

### 5.4 Detector thresholds and NMS (D-020, D-021, D-097, D-116)
- Label rule (D-021): per query, the class score is the maximum token probability of the class phrase and the label is the argmax. The score is not an Eq. (10) weight (D-024).
- `box_threshold` 0.1 (D-097): equal to `low_thresh` so that ByteTrack can use D_low. 0.35 became the D_high cut.
- Class-agnostic NMS 0.5 (D-116, iter 14): keeps one box when two phrases fire on the same object. 40 scenes: mIoU 54.8 -> 55.9, mAP50 27.5 -> 29.0, chair AP50 17.3 -> 25.6,
  instances 49.3 -> 41.0 per scene (22 scenes better / 15 worse). Cost: desk AP50 -> 0, bed AP50 52.5 -> 32.5. NMS alone gives +1.4 mIoU but no mAP50 gain.
- Threshold sweep (iter 14b): NMS 0.4 loses 1.6 mIoU, 0.6 loses 5 chair AP50, neither recovers desk, so 0.5 stays.

### 5.5 Mask clean-up (D-036, D-106, D-107, D-116 depth edge)
The paper gives no refinement rule, so the first choice was none (D-036). On Replica office0, erosion by 2 px (cancelled if more than 50 % of the mask vanishes) + MAD 3 sigma depth outlier removal (5 cm floor) (D-106)
and giving overlapping pixels to the smaller mask (D-107) took mAP50 from 9.0 to 28.5 (iter 7). Depth-edge exclusion (D-116) is neutral alone (chair AP50 +6.5, bed -14) and contributes to the effect above together with NMS.
Thresholds 0.05 / 0.15 are insensitive (mean change at most 0.6), so 0.1 stays.

### 5.6 Neighbor criterion (D-032 -> D-110)
D-032 made only objects whose centroid lies in the image neighbors. Diagnostic (10 scenes): in 55 % of duplicate new-id events the original was outside the neighborhood (median 5 700 points). D-110 treats an object as a neighbor if any of up to 256 evenly strided sample points
is inside the image and within 10 m (`neighbors.mode: points`). Together with D-111: mIoU 36.1 -> 43.1, mAP50 14.8 -> 18.2, instances 114 -> 85 per scene.

### 5.7 Absorb rule (D-111) and the absorb plan before the semantic check (D-120)
- D-111: when at least 0.5 of the points of a new-id `O_add` fall into the 2 cm voxels of an existing ACTIVE neighbor, it is absorbed regardless of label, the track is remapped to the absorbing id (`Tracker.remap_id`), and the unused id is never reused.
  Diagnostic (10 scenes, 1 278 new ids): 62 % were GT duplicates; a label-agnostic overlap of 0.5 catches 54 % (11 % if same-label only) and would absorb about 9 % of genuinely new objects. 193 absorbs per scene on the 40 scenes.
- D-120 (iter 15): with the D-117 class gate, a relabelled detection of a visible object opens a new track id; the object is in `O_on` but not in `T_trk`, so the Alg. 1 semantic check negates it (L_exist goes down) and only afterwards it is absorbed. A persistently relabelled static object
  was removed after 5 segmentation frames and re-created under a new id (verify_iter_15, I-15-1 P1). Fix: the absorb decision is computed right after Unproject as a pure function, the semantic check treats "absorbs this frame's detection" as supported, and ObjMerge applies the same plan.
  Rejected alternatives: label_cost back to 0 (loses the D-117 gains), a soft class cost (D-108 measured 0.5 as worse than 0 on office0 and it cannot stop the chair / table switches), comparing the detection with the instance's Eq. (10) label (does not cover a persistent relabel).
  Means moved by at most 0.4 against i15_lc_cc10.

### 5.8 Class-aware association (D-108 -> D-117)
D-033 was class-agnostic like ByteTrack and D-108 kept `label_cost` as an experiment only (tiny detector, office0: 0.5 gave mIoU 24.4 but N_T 109 -> 162 and mAP 46.4 / 28.5 -> 42.5 / 25.7).
Iter-15 diagnosis: 36 % of GT chairs are merged with a neighbor through track ID switches between adjacent same-class and different-class boxes. D-117 makes `label_cost` 1.0 the default (40 scenes, on the D-116 base): mIoU 55.9 -> 59.7,
Acc 61.6 -> 63.9, mAP50 29.0 -> 36.9, chair AP50 25.6 -> 36.5, N_active 41.0 -> 44.5, 29 scenes better / 9 worse. The base detector relabels less than tiny, so the counter-evidence of D-108 no longer holds.
On the change batch it hurts the removal side alone (FP strict 0.17 -> 0.28) and recovers together with D-118.

### 5.9 Largest-component pruning (D-118, revises D-036)
Diagnosis: 19 % of GT chairs have bloated boxes from floor / wall points, and a percentile box at evaluation time alone gave chair AP50 +4.5. After ObjMerge, every instance that received points in this frame keeps only the points of its largest
26-connected component on a `component_voxel` 0.10 m grid (points, colors and log-odds are removed together). On top of D-117, 40 scenes: mAP25 54.7 -> 61.1, mAP50 36.9 -> 43.3, chair AP50 36.5 -> 54.4, sofa AP50 62.9 -> 86.4, mIoU 59.7 -> 59.4, Acc 63.9 -> 63.0
(71 k of 11 M instance points per scene dropped). Grid 0.05: mAP50 44.5, chair 57.9 but Acc 62.1 and lower desk / table, so 0.10 was adopted. The per-observation filter (`unproject.component_voxel`) worsens mIoU on the 8 chair-heavy scenes, 51.7 -> 48.5 / 51.4
(thin parts separate after erosion) and stays off. Change batch: remove detection strict 0.43 -> 0.51, REMOVED 8 -> 14 of 29 (the fragments at the old location are dropped).

### 5.10 Point log-odds and pruning (D-048, D-049, D-050, D-051)
Eq. (8) is applied once per point per frame. P(o|Q): Observable 0.7, Disappeared 0.3, Unobservable 0.5. New-point initial log-odds = logit(0.7), L_prune = -1, no clamp, tau_eps = 0.1 m. Out-of-view / invalid-depth points are
NOT_IN_VIEW and get no update (D-046). The object verdict is a majority over in-view points, ties resolved Observable > Unobservable > Disappeared (D-047). `O_neg` objects only receive ObjNegate (p_dis on in-view points). An object that lost all points is kept as REMOVED,
ids are never reused (D-053). D-101: merge voxel de-duplication at 0.02 m prevented office0 instance points from reaching 101 M (24.7x slower than real time).

### 5.11 Object-level existence log-odds (D-113)
The abstract's "existence-and-label confidence update" has no formula, so one was added. Point-wise negative evidence goes only to in-view points and removal needs every point gone, so in the synthetic change detection removed objects survived on their old points
(relaxed FP 0.88, 1 of 29 targets removed, clamping hardly helped). Each instance carries L_exist (initial logit(p_obs)), updated every frame by + logit(p_dis) if DISAPPEARED or in `O_neg`, + logit(p_obs) if Observable, unchanged otherwise, clamped to +-3, and the instance is REMOVED below -1.
Effect: remove FP 0.41 -> 0.20, change recall soft 0.43 -> 0.66, targets reaching REMOVED 1 -> 13 of 29. Cost: static mIoU 43.1 -> 41.1 (about 24 false removals per scene). `exist_prune: null` disables the rule.
Variant `exist_semantic: false` changes almost nothing (FP 0.23, 12 of 29, mIoU 41.1), and `exist_prune -3` is unreachable because of the clamp, so it equals the rule being off.

### 5.12 Label fusion (D-060, D-061, D-062, D-063)
A new instance starts from the uniform prior with one Eq. (10) update from its first observation (D-060), confusion matrix diagonal 0.8 / off-diagonal uniform (D-061), only instances with at least one Observable point are updated (D-062, D-091), and an instance is removed when max_c P < 0.4 (D-063).
Because the first-observation belief is exactly 0.8, a p_min anywhere in (1/C, 0.8] is effectively inert (iter-8 ablation: p_min 0.6 and 0.7 gave identical results).

### 5.13 Scene graph (D-070 to D-079)
Single-linkage clustering with d_clu = 3 m (D-070); On(A,B): |z_min^A - z_max^B| <= delta and IoU_xy > gamma, gamma = 0.05, delta = 0.05 m (D-071); Under is the inverse of On; Beside needs overlapping vertical intervals, IoU_xy <= gamma and a horizontal gap < 0.5 m (D-073);
a centroid move of 0.1 m or more creates a temporal node and edge (D-074); a vanished id keeps its last position and time as a cue (D-075, D-078); local subgraph radius 5 m (D-079). Serialization order is nodes, edges, schema, cues (D-076 to D-078).
The supplementary does not give the "full construction details" either, so all of these are the simplest deterministic choices.

### 5.14 VLM backend and fallbacks (D-080, D-081, D-082, D-098, D-099, D-100)
The model is the paper's `gemini-2.0-flash` at temperature 0, but the API does not serve it (checked 2026-09-30). `vlm.fallback_models` advances to the next model only on a 404; other errors (429, 503) are raised without retry (one call per query).
The free tier allows 20 requests per day per model, so the flash-lite family is placed first (D-100). The key is read from the environment, then from the root `.env` (`KEY=VALUE`), and its value is never logged (D-098).
The `<answer>` parser takes every non-negative integer inside the last tag as an id (D-082). Missing or vanished ids are skipped with a warning (D-083).

---

## 6. Tried and not adopted

The code remains and is off by default. Each row carries the number that rejected it.

| Attempt | Rejecting number | Note |
|---|---|---|
| Segmentation rate 3 / 10 Hz (iter 15, D-119) | mIoU on the 8 chair-heavy scenes 51.7 -> 24.5 / 19.5 | The Eq. (8) and D-113 updates are applied per segmentation event, so the map collapses. The paper's 1 Hz stays |
| ByteTrack thresholds / buffer / p_min / min_points (iter 8, 11 runs) | On 8 scenes bt 0.4 / 0.5 gave +3.4 mIoU, mAP50 11.3 -> 16.1, instances 125 -> 52, but on **the 40 scenes -1.0 mIoU, -1.3 mAP50** (only 17 scenes better) | The 8 scenes are not representative, so defaults are decided on the 40 scenes only. Higher thresholds only block tracks of low-score large structures (wall, cabinet, desk) |
| SAM2.1-large (D-114) | 41.6 alone, -1.9 next to base | The detector is the bottleneck |
| Alias variants (D-115) | "office desk": desk 11 -> 48 but table 39 -> 25, bed 60 -> 50 | "curtains" also rejected (loses shower curtain) |
| I-9-4 mitigations: window search / coverage gate / w3gate (iter 10, 14c) | iter-9 base: window 3 cuts false removals 24 -> 15 per scene but mIoU 40.8 and true removals 13 -> 6 of 29. The gate recovers mIoU 42.1 to 42.6 but remove FP 0.30 to 0.33. D-116 base w3gate: mIoU 55.9 -> 57.1, Acc 61.6 -> 63.4 but mAP50 -0.9, chair AP50 -4.2, change-detection remove change soft 0.49 -> 0.44, REMOVED 8 -> 5 of 29, median 25.5 -> 603 frames | Only moves along the "static accuracy vs change detection" line. The decision stays with the human (I-9-4); the orchestrator recommendation is not to adopt |
| Footprint-box re-activation / map matching (iter 12) | mIoU 51.5 to 52.8, chair AP50 17 -> 11 to 13 | GT check: 25 % of the attached detections belong to another object |
| 3D contact absorb 5 cm (iter 13) | mIoU 52.9, chair AP50 11.6 (same label only: 54.7 / 13.0) | Side-by-side chairs touch within 5 cm and merge |
| D-111 absorb off (D-119) | mIoU 56.7 / mAP50 32.3 / chair 35.3 but instances 41 -> 67.8, change-detection remove change soft 0.49 -> 0.43, FP 0.17 -> 0.24, REMOVED 8 -> 5 | Same-label absorb (`absorb_same_label`) on the 8 scenes: mIoU 51.7 -> 52.4, chair AP50 6.1 -> 8.1, almost no change (chair merges are same-label) |
| Reverse overlap `absorb_overlap_rev` 0.1 / 0.2 (D-119) | Absorbs 1 395 -> 365 / 106, practically "off" | 8-scene mAP50 15.1 / 16.9 |
| 3D centroid gate 0.5 / 1.0 m (D-117, D-119) | 8-scene mIoU +2.1 / +2.8 but chair AP50 6.1 -> 5.8 / 4.1, 1.4x more tracks | Rejected matches become new tracks; ego-motion and depth-median centroids reject true matches |
| Per-observation component filter 0.05 / 0.10 (D-118) | 8-scene mIoU 51.7 -> 48.5 / 51.4 | Thin parts separate after erosion and true points are lost |
| AP ranking score alternatives (D-119) | point count 28.4, existence probability 27.2, observation span 27.4, label x existence 28.6 (baseline 29.0); on the finalist label x existence 41.5 vs 43.3 | Tested at evaluation time only (`--score`). The map keeps no separate score |
| NMS 0.4 / 0.6 (iter 14b) | mIoU 54.3 / 55.8, mAP50 29.1 / 28.9, chair AP50 26.6 / 20.6, desk AP50 6.7 / 3.3 | 0.5 is the optimum, desk not recovered |
| Depth edge 0.05 / 0.15 (iter 14b) | mIoU 56.2 / 55.7, mAP50 29.6 / 28.1 | Mean change at most 0.6, insensitive |
| NMS 0.5 alone (iter 13) | mIoU 56.2 (+1.4), mAP50 flat, desk IoU 11.5 -> 5.1 | Later adopted together with the depth edge as D-116 |
| Depth-edge filter alone (iter 13) | Neutral: chair AP50 +6.5, bed -14 | Loses border points of large objects |

---

## 7. Evaluation protocol

### 7.1 ScanNet GT handling (D-109)
- Scenes: the 40 validation scenes (one `_00` scan per scene id: `scene0011_00`, `scene0030_00`, `scene0046_00`, `scene0086_00`, `scene0222_00`, `scene0378_00`, `scene0389_00`, `scene0435_00`, `scene0050_00`, `scene0064_00`, `scene0084_00`, `scene0100_00`, `scene0131_00`, `scene0164_00`, `scene0193_00`, `scene0207_00`, `scene0231_00`, `scene0256_00`, `scene0300_00`, `scene0329_00`, `scene0355_00`, `scene0406_00`, `scene0423_00`, `scene0474_00`, `scene0500_00`, `scene0549_00`, `scene0568_00`, `scene0580_00`, `scene0599_00`, `scene0616_00`, `scene0629_00`, `scene0645_00`, `scene0653_00`, `scene0664_00`, `scene0678_00`, `scene0685_00`, `scene0695_00`, `scene0700_00`, `scene0025_00`, `scene0088_00`); the first 8 are the scenes shared by open-vocabulary 3D mapping papers. Input is every 3rd frame of the 30 Hz capture (10 Hz).
- GT: NYU40 vertex labels (`_vh_clean_2.labels.ply`), instances from aggregation + segs. 19 classes, background = wall, floor.
- Label transfer: nearest predicted point to each GT vertex within r = 0.05 m (0.10 m is computed too). **All GT vertices** are the denominator.
- Semantic segmentation: mIoU, f-mIoU and Acc are computed both without and with the background. Per-class means are taken over the scenes that contain the class.
- Instance AP: 3D IoU 0.25 / 0.50 between the AABB of the predicted points and the AABB of the GT instance.
- The 40-scene mean has a per-scene standard deviation of 10.8 mIoU, so the 8 common scenes alone are not stable and defaults are decided on all 40.

### 7.2 Protocol variants (re-evaluation only, no pipeline change)
On the final iter-15 run, against plain 59.4: merged (desk>table, shower curtain>curtain) **62.9**, radius-5 cm majority transfer **60.2**, both **63.7**; mAP50 44.6 with merging.
Iter 14 gave 59.3 / 58.8 / 62.4 (plain 55.9), iter 11 gave 58.5 / 59.5 / 64.2 (plain 54.8). Earlier measurements (iter-11 base, not re-measured): a 1 to 99 percentile predicted box raises mAP50 27.5 -> 30.1,
and a 5 cm voxel IoU gives mAP50 14.2. Variants are reported next to the plain numbers and never replace them.

### 7.3 Synthetic change detection (D-112)
The paper's real-robot data for Table IV is not public, so 58 sequences are built from 9 ScanNet scenes (the 8 common ones + scene0193_00) by removing or adding 29 GT instances (`remove`: the target is removed from the second half, `add`: from the first half).
Target classes are chair, toilet, sink, refridgerator, picture, sofa, bed, desk, bathtub (at least 300 vertices, visible in at least 15 frames in both halves; in practice chair 13, desk 6, bed 5, sink 2, refridgerator 2, picture 1).
The object's pixels are replaced by the background ray-cast from the GT mesh with the object's vertices removed (depth = camera z, colour = barycentric vertex colour; mesh holes fall back to nearest-pixel fill).
Metrics are frame based, on segmentation frames where the target location is in view: detection recall (present interval), FP rate (absent interval), change recall soft (1 - FP rate if detected at all) and strict (1 only with zero FP).
Both the paper's strict match (3D IoU > 0.1, centroid < 0.3 m, label match) and a relaxed match (IoU > 0.1 + label, since the point centroid of a large object seen from one side is more than 0.3 m from the GT mesh centroid) are reported.
The log-odds clamp (`L_clamp` 3.0 / 2.0) is an experiment only.

### 7.4 Real-time benchmark protocol (paper Sec. V-H)
Replica office0, 2000 frames (10 Hz = 200 s), VLM step skipped. The wall-clock latency of each module call (CUDA-synchronized) is summarized without the first (warm-up) call. A module passes when mean, p95 and max are all within its period.
Modules run sequentially in one thread and the pose is read from the dataset, so the pose module latency is only the Eq. (3) composition. The GPU is an RTX PRO 6000 Blackwell (stronger than the paper's laptop RTX 4090, so the numbers are optimistic).

---

## 8. Results history

### 8.1 ScanNet v2, 40 scenes (metrics of paper Tables II / III)

| 40-scene mean | mIoU w/o bg | f-mIoU | Acc | mIoU w bg | f-mIoU | Acc | mAP25 | mAP50 | instances/scene (GT 15) |
|---|---|---|---|---|---|---|---|---|---|
| iter 7 (D-101 to 108) | 36.1 | 38.1 | 43.3 | 32.5 | 39.4 | 49.3 | 32.7 | 14.8 | 113.8 |
| iter 8 (+ D-110, D-111) | 43.1 | 44.7 | 49.0 | 39.5 | 46.3 | 54.5 | 32.7 | 18.2 | 85.2 |
| iter 9 (+ D-113) | 41.1 | 42.7 | 46.5 | 38.4 | 45.2 | 52.9 | 31.2 | 18.1 | 68.7 |
| iter 11 (+ D-114, D-115) | 54.8 | 56.8 | 61.2 | 46.8 | 50.0 | 58.8 | 47.3 | 27.5 | 49.3 |
| iter 14 (+ D-116) | 55.9 | 57.1 | 61.6 | 47.1 | 50.2 | 58.3 | 48.2 | 29.0 | 41.0 |
| **iter 15 (+ D-117, D-118, D-120, current)** | **59.4** | **59.9** | **62.8** | **55.6** | **59.0** | **64.0** | **61.3** | **42.9** | 46.3 |
| paper SuperMap (Table II) | 27.4 | 43.5 | 55.5 | 22.6 | 29.1 | 33.0 | - | - | - |

The paper does not publish its scene list, prompts or GT correspondence rule, so this is a reference comparison on the same dataset and the same metrics. Iter 8 improved mIoU in 33 of the 40 scenes (mean +7.0), and
iter 11 is +13.7 mIoU over iter 9 (base detector +7.7, aliases +6.0). Against paper Table III's chair AP50 63.8 and window 42.2, our current values are chair 54.6 and window 27.7.

### 8.2 Per class (iter 14 -> iter 15, mean over the scenes containing the class; wall / floor from the with-background protocol)

| class | scenes | IoU iter 14 | IoU iter 15 | AP50 iter 14 | AP50 iter 15 |
|---|---|---|---|---|---|
| toilet | 9 | 82.2 | 88.0 | 66.7 | 77.8 |
| floor | 40 | 57.5 | 75.4 | - | - |
| picture | 16 | 66.2 | 75.3 | 41.3 | 67.5 |
| curtain | 9 | 72.1 | 74.8 | 37.3 | 63.9 |
| bookshelf | 9 | 68.9 | 73.3 | 48.0 | 58.5 |
| sofa | 11 | 69.2 | 71.9 | 57.6 | 86.4 |
| bathtub | 7 | 61.5 | 69.9 | 28.6 | 33.3 |
| chair | 32 | 61.3 | 68.8 | 25.6 | 54.6 |
| bed | 10 | 52.4 | 62.7 | 32.5 | 50.0 |
| sink | 11 | 61.9 | 61.4 | 63.6 | 58.3 |
| door | 34 | 59.1 | 60.0 | 20.7 | 29.7 |
| window | 23 | 55.6 | 59.0 | 19.8 | 27.7 |
| table | 22 | 44.3 | 51.4 | 25.4 | 40.3 |
| wall | 39 | 42.4 | 48.0 | - | - |
| refridgerator | 8 | 45.5 | 47.0 | 21.9 | 26.0 |
| counter | 6 | 40.1 | 42.0 | 16.7 | 16.7 |
| shower curtain | 7 | 30.7 | 31.9 | 28.6 | 28.6 |
| cabinet | 16 | 32.0 | 29.2 | 11.7 | 25.0 |
| desk | 15 | 2.7 | 1.9 | 0.0 | 0.0 |

### 8.3 Iter-15 chair AP50 lever study
Starting point: the D-116 headline (chair AP50 25.6). Diagnosis: 1 Hz is not the cause; only 7 % of GT chairs are missed, 36 % are merged with a neighbor (D-111 absorb + tracker ID switches), 19 % have bloated boxes, and the label-probability score saturates at 1.0.
Order set by the human: (1) keep 1 Hz (2) absorb rule (3) suppress ID switches (4) remove outlier points (5) score. All switches were added off by default (commit 63cd938) and screened first.

Screening on the 8 chair-heavy scenes (0011 0030 0088 0131 0231 0355 0500 0599). head8 = the D-116 headline restricted to these scenes (chair AP50 6 vs 25.6 over all 40):

| run | override | mIoU | Acc | mAP25 | mAP50 | N_act | chair IoU / AP25 / AP50 | table IoU / AP25 / AP50 |
|---|---|---|---|---|---|---|---|---|
| head8 | (D-116 defaults) | 51.7 | 63.9 | 30.8 | 11.5 | 64.0 | 67.2 / 28.6 / 6.1 | 34.6 / 42.3 / 5.0 |
| chairdiag_noabsorb | absorb_overlap null | 55.5 | 66.5 | 41.7 | 17.3 | 94.6 | 71.9 / 42.5 / 8.4 | 40.9 / 31.5 / 18.0 |
| i15_abs_label | absorb_same_label true | 52.4 | 64.5 | 28.0 | 13.5 | 70.1 | 70.7 / 28.1 / 8.1 | 30.8 / 23.1 / 11.7 |
| i15_abs_rev02 | absorb_overlap_rev 0.2 | 53.2 | 65.9 | 40.7 | 16.9 | 91.9 | 70.6 / 41.1 / 9.1 | 35.3 / 28.7 / 16.2 |
| i15_abs_rev01 | absorb_overlap_rev 0.1 | 53.1 | 64.9 | 36.5 | 15.1 | 87.0 | 68.7 / 32.5 / 7.9 | 33.3 / 33.6 / 13.8 |
| i15_labelcost | label_cost 1.0 | 58.6 | 68.1 | 39.0 | 25.5 | 70.4 | 75.6 / 32.0 / 8.8 | 38.3 / 35.3 / 31.4 |
| i15_lc_noabs | label_cost 1.0 + absorb off | 62.2 | 71.0 | 43.8 | 25.4 | 115.8 | 77.4 / 48.2 / 10.3 | 47.9 / 39.6 / 31.0 |
| i15_gate05 | gate_3d 0.5 | 53.8 | 63.9 | 32.7 | 17.4 | 88.1 | 67.5 / 26.2 / 5.8 | 35.0 / 23.1 / 14.3 |
| i15_gate10 | gate_3d 1.0 | 54.5 | 66.7 | 34.5 | 16.6 | 76.0 | 70.9 / 28.0 / 4.1 | 35.7 / 41.1 / 12.3 |
| i15_cc_unp05 | unproject component_voxel 0.05 | 48.5 | 58.8 | 31.0 | 17.3 | 65.1 | 64.6 / 31.0 / 10.0 | 28.2 / 33.6 / 21.4 |
| i15_cc_unp10 | unproject component_voxel 0.10 | 51.4 | 61.4 | 33.0 | 14.4 | 66.1 | 66.2 / 33.0 / 9.1 | 32.6 / 34.9 / 5.0 |
| i15_cc_merge05 | merge component_voxel 0.05 | 51.3 | 59.1 | 44.9 | 28.9 | 73.0 | 65.8 / 51.5 / 33.8 | 28.3 / 36.4 / 15.0 |
| i15_cc_merge10 | merge component_voxel 0.10 | 54.8 | 62.2 | 46.8 | 25.6 | 67.1 | 66.3 / 49.6 / 21.3 | 30.1 / 38.8 / 20.2 |
| i15_lc_cc05 | label_cost 1.0 + merge cc 0.05 | 56.9 | 62.9 | 45.7 | 31.5 | 79.8 | 69.5 / 63.6 / 39.9 | 36.8 / 30.9 / 22.0 |
| i15_lc_cc10 | label_cost 1.0 + merge cc 0.10 | 58.3 | 64.9 | 48.2 | 29.1 | 76.1 | 72.1 / 59.3 / 31.6 | 39.2 / 38.5 / 26.4 |
| i15_lc_cc05_noabs | + absorb off | 56.7 | 64.3 | 44.1 | 28.5 | 118.9 | 72.6 / 65.6 / 41.3 | 35.4 / 26.9 / 7.6 |
| i15_lc_cc10_noabs | + absorb off | 57.7 | 65.7 | 47.2 | 26.8 | 115.9 | 73.2 / 62.7 / 35.5 | 38.6 / 30.5 / 14.7 |

Counters over the 8 scenes (headline: 1 395 absorbs, N_T 512): `absorb_same_label` 1 153 absorbs, `absorb_overlap_rev` 0.2 / 0.1 give 106 / 365 absorbs, gate_3d 0.5 / 1.0 rejected 8 373 / 2 777 feasible pairs
and the rejected detections became new tracks (N_T 705 / 608, chair AP50 down), merge component_voxel 0.05 / 0.10 dropped 2.16 M / 1.61 M of about 6.6 M instance points.

40 scenes:

| run | mIoU | Acc | mAP25 | mAP50 | N_act | chair IoU / AP25 / AP50 | table AP50 | bed AP50 | sofa AP50 | desk AP50 | scenes better / worse (mIoU, 0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| headline (D-116) | 55.9 | 61.6 | 48.2 | 29.0 | 41.0 | 61.3 / 45.4 / 25.6 | 25.4 | 32.5 | 57.6 | 0.0 | - |
| i15_noabsorb | 56.7 | 62.5 | 53.2 | 32.3 | 67.8 | 64.9 / 59.6 / 35.3 | 32.9 | 39.6 | 75.3 | 2.2 | 21 / 14 |
| i15_labelcost | 59.7 | 63.9 | 54.7 | 36.9 | 44.5 | 70.3 / 62.4 / 36.5 | 42.4 | 56.7 | 62.9 | 3.3 | 29 / 9 |
| i15_lc_cc05 | 59.1 | 62.1 | 59.8 | 44.5 | 47.5 | 65.2 / 72.6 / 57.9 | 45.3 | 45.0 | 86.4 | 7.8 | 26 / 12 |
| i15_lc_cc10 | 59.4 | 63.0 | 61.1 | 43.3 | 46.3 | 65.8 / 72.7 / 54.4 | 40.3 | 50.0 | 86.4 | 0.0 | 27 / 10 |

Reading: (2) the absorb variants do not separate chairs, only "off" helps, at the price of 1.5x (8 scenes) to 1.65x (40 scenes) more instances. (3) The class-aware first association is the strongest single lever; the 3D gate is not.
(4) The per-observation filter removes true points and the instance-level filter after ObjMerge is the one that works. The 0.05 grid wins on chair AP50 alone (57.9 vs 54.4) but costs Acc -0.9 and lowers table / desk / toilet AP50. The shipped defaults are D-117 + D-118 + D-120, and the
**final headline** with D-120 is `i15_d120` (iter-15 row of section 8.1): mIoU 59.4 (55.9), Acc 62.8 (61.6), mAP25 61.3 (48.2), mAP50 42.9 (29.0), chair AP50 54.6 (25.6), chair IoU 68.8 (61.3), N_active 46.3 (41.0).
Iter-14 outputs are archived with the suffix `_iter14`.

### 8.4 Change-detection history (58 sequences, D-112)

| setting | remove: detection strict | FP strict | change soft (strict match) | change soft (relaxed match) | targets reaching REMOVED | add: detection strict | add: change soft |
|---|---|---|---|---|---|---|---|
| iter 8 (no existence rule) | 0.47 | 0.41 | 0.43 | 0.11 | 1 / 29 | 0.38 | 0.55 |
| iter 9 (+ D-113) | 0.47 | 0.20 | 0.66 | 0.31 | 13 / 29 | 0.35 | 0.62 |
| iter 11 (+ D-114, D-115) | 0.52 | 0.22 | 0.62 | 0.47 | 15 / 29 | 0.43 | 0.63 |
| iter 14 (+ D-116) | 0.43 | 0.17 | 0.49 | 0.35 | 8 / 29 | 0.41 | 0.61 |
| iter 14c (D-116 + w3gate, not adopted) | 0.44 | 0.22 | 0.44 | 0.22 | 5 / 29 | 0.38 | 0.59 |
| iter 15: absorb off (not adopted) | 0.47 | 0.24 | 0.43 | 0.23 | 5 / 29 (median 678 frames) | 0.50 | 0.63 |
| iter 15: label_cost 1.0 only | 0.49 | 0.28 | 0.43 | 0.28 | 11 / 29 (300) | 0.47 | 0.56 |
| iter 15: label_cost 1.0 + cc 0.05 (not adopted) | 0.46 | 0.22 | 0.41 | 0.31 | 15 / 29 (411) | 0.40 | 0.62 |
| **iter 15 (D-117 + D-118 + D-120, current)** | **0.51** | **0.23** | **0.46** | **0.28** | **14 / 29 (354)** | **0.51** | **0.64** |
| paper Table IV (6 real objects) | 0.26 to 1.00 | | 0.62 to 1.00 | | | | |

(`i15_d120/` is identical to `i15_lc_cc10/`.) Per class, remove (detection strict / change soft): chair (13) d116 0.55 / 0.62 -> current 0.68 / 0.71, bed (5) 0.47 / 0.70 -> 0.40 / 0.69,
desk (6) 0.06 / 0.17 -> 0.14 / 0.00 (I-14-2: the NMS-suppressed desk box). Desk iter 11 -> iter 14 was 0.42 / 0.54 -> 0.06 / 0.17. Absorb off and the class gate alone leave fragments at the old location, so the removal side is worse;
the D-118 pruning drops those fragments, returning to the d116 level while 14 of 29 targets reach REMOVED.
Conclusion of the D-113 diagnosis: the cause was not the log-odds magnitude (clamp) but the structure "negative evidence only for in-view points, removal only when every point is gone".

### 8.5 Real-time benchmark history (Replica office0, 2000 frames, latency mean / p95 / max in ms)

| module | period | iter 14 | iter 15 (current) | within period |
|---|---|---|---|---|
| pose | 100 ms | 0.1 / 0.1 / 0.3 | 0.1 / 0.1 / 0.7 | yes |
| segmentation | 1000 ms | 187.6 / 248.5 / 386.1 | 189.5 / 259.8 / 330.0 | yes |
| mapping | 333 ms | 64.5 / 76.3 / 162.6 | 64.6 / 77.9 / 128.2 | yes |
| scene graph | 200 ms | 0.6 / 0.7 / 23.9 | 0.6 / 0.8 / 1.1 | yes |

Loop wall time: iter 11 97.3 s, iter 14 100.8 s, iter 15 100.2 s for 200 s of data (0.50x). Iter-11 segmentation was 171.0 / 207.6 / 228.5 ms. Iter-15 breakdown (mean ms, warm-up excluded):
`seg.groundingdino` 94.2, `seg.unproject` 44.3, `seg.sam2` 25.9, `seg.obj_merge` 7.1 (includes the D-118 pruning), `dense_map.integrate` 62.3, `compare_depth` 2.7, `get_neighbor_objects` 0.4.
Two earlier runs the same day, taken while the machine was loaded (load average 40 to 48; loop 164 / 206 s), were discarded and the benchmark was repeated on an idle machine. The GPU is stronger than the paper's laptop 4090, so the numbers are optimistic.

### 8.6 Replica office0 (development)
Iter 7: mIoU 23.6 / 20.1 (without / with background), mAP25 / mAP50 46.4 / 28.5. mAP50 before the mask clean-up (D-106 + D-107) was 9.0. With D-110, D-111 (iter 8) office0 N_T went 73 -> 31 (132 absorbs over 200 segmentation frames).
D-108 test (tiny, on D-106 + D-107): label_cost 0.5 gave mIoU 24.4 but N_T 109 -> 162 and mAP 46.4 / 28.5 -> 42.5 / 25.7. Replica is a reference only, not the paper's benchmark.

---

## 9. Known limitations and open issues

### 9.1 Gaps against the paper
- **window / refrigerator AP**: window AP50 27.7 (paper 42.2), refrigerator AP50 26.0. Thin planar objects whose boxes overlap boxes of other classes remain. Chair AP50 is 54.6, close to the paper's 63.8.
- **desk = 0**: the class-agnostic NMS of D-116 deletes the "desk" box under the "table" box of the same object, so desk never gets a track (desk IoU 1.9, AP50 0.0, change-detection desk remove change soft 0.17 -> 0.00). Thresholds (0.4 / 0.6) do not fix it; pooling the label evidence of a suppressed box into the kept box is the remaining route (I-14-2).
- **Removal recall on narrow revisits**: the paper's robot has a 360-degree LiDAR and deliberate revisits, whereas a ScanNet RGB-D revisit sees the old location only in slivers. Remove change soft is 0.46, only 14 of 29 targets reach REMOVED, with a median of 354 frames.
- **Not comparable**: real-robot change detection (Tables IV, V) and VLM navigation success (Sec. V-F) have no numbers because the data is not public and Gemini has free-tier limits. The paper's gemini-2.0-flash is not served, so a flash-lite family model is used.
- **Dataset ambiguity**: desk / table and curtain / shower curtain have blurry boundaries in the ScanNet GT itself, so whichever is chosen hurts the other (merged evaluation +3.5 mIoU).

### 9.2 Open issues (all P2)
- **I-9-4 w3gate decision**: D-113 falsely removes 17 to 24 instances per static scene from DISAPPEARED verdicts on small slivers (mIoU -2.0). The mitigation variants (window / gate / w3gate) only move along the line between static accuracy and change detection. On the D-116 base w3gate gives +1.2 mIoU but delays removal detection (median 25.5 -> 603 frames). The default is kept; adopting it is the human's decision.
- **I-15-2**: D-118 keeps only the largest connected component of the whole instance, so a larger observation disjoint from the earlier points (for example SAM2 bleeding onto a larger surface) deletes the earlier points regardless of their log-odds; there is no tie rule either. Candidate refinements: restrict the removal to this frame's points, or keep only the component that holds the pre-merge majority.
- **I-15-4**: (a) wording of the `absorb_targets` docstring, (b) with D-117 the track label is the last matched detection label, so a persistently relabelled object opens and drops a new track id every frame and burns track ids (map id, semantic check and Eq. 10 are correct).
- **I-14-2**: desk loss (section 10.1 above).
- Other fidelity notes: I-6-1 (Eq. 5 X_O excludes Disappeared points, median shift 0.24 px), I-7-1 (D-106 / D-107 are not the minimal choice), I-8-4 (3 float32 poses dropped), I-8-7 (D-111 details: O_add order dependence, lowest id on ties),
  I-9-1 (existence log-odds default 0.0 on the direct map-creation path), I-9-2 (the Supp. VII-A2 ablation is reproduced with `exist_prune: null`), I-11-1 (`tolerant_states` uses the interval hull, off), I-11-3 (remaining fragments), I-12-1 / I-12-2 (the off map-matching path),
  I-13-2 / I-13-3 (fidelity notes on the off depth-edge and contact-absorb variants).
- **Instance over-segmentation**: 46.3 instances per scene against 15 GT; what remains are other parts of the same object that do not overlap in 3D, and other-label boxes in the same frame. Proximity merges (2D and 3D) merged neighboring furniture and lost.

---

## 10. Development harness

The implementation was built by harness engineering. Sixty-two requirements (REQ-001 to 062) were extracted from the paper as the
specification, and LLM agents with separate roles (paper analysis, data preparation, implementation, verification, refinement) ran a
build -> verify -> refine loop. The verifier wrote a paper-derived test for every requirement and recorded a PASS / PARTIAL / N/A verdict
(final: 61 PASS / 1 N/A). Completion was decided by an automatic gate rather than by anyone's judgement: a code comment and a test for
every requirement, no stubs, the whole test suite and a smoke run passing, the latest verification report matching the current code, and
no open P0 / P1 issues. Every choice on a detail the paper leaves open is a numbered decision record (D-xxx) and every problem found is an
issue record (I-xx-x); the numbers in this document point to those records. The specification, the decision and issue records, the
verification reports, and the experiment scripts and outputs are kept locally and are not part of the repository.
