# SuperMap

A Python reimplementation of "SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation" (RSS 2026, [arXiv 2608.22896](https://arxiv.org/abs/2608.22896)).

*한국어: [README.ko.md](README.ko.md)*

This repository is an **unofficial**, independent implementation written from the SuperMap paper alone; no official code was used. \
The algorithm and the paper belong to their authors (Shibo Zhao, Guofei Chen et al., Robotics Institute, Carnegie Mellon University). \
The official implementation is at [github.com/superxslam/SuperMap](https://github.com/superxslam/SuperMap) and the project page at [superodometry.com/supermap](https://superodometry.com/supermap). \
It was built by harness engineering: starting from a requirement specification extracted from the paper, LLM agents iterated implement / verify / refine, with a paper-derived test for every requirement and an automatic gate deciding completion. \
The repository contains only the resulting pipeline code and documentation. \
It is evaluated on the 40 validation scenes of ScanNet v2 and on synthetic change sequences built from ScanNet, and it runs in real time at the module rates given in the paper. \
How it was built (implementation details, parameters, experiments and the reasoning behind each default) is documented in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Contents

1. [Algorithm](#algorithm)
2. [Results vs. the paper](#results-vs-the-paper)
3. [Differences from the paper](#differences-from-the-paper)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Repository layout](#repository-layout)
7. [Documentation](#documentation)
8. [Citation](#citation)
9. [Acknowledgements](#acknowledgements)
10. [License](#license)
11. [TODO](#todo)

## Algorithm

Frames are processed in the order of the paper's supplementary Algorithm 1, with four modules running asynchronously at the rates of Sec. V-H (pose 10 Hz, segmentation 1 Hz, mapping 3 Hz, scene graph 5 Hz). \
The per-layer flow and the implementation of each step (Eq. 3-10) are described in [docs/IMPLEMENTATION.md, section 1](docs/IMPLEMENTATION.md#1-algorithm).

## Results vs. the paper

The paper does not publish its scene list, its prompt list or its ground-truth protocol. \
The numbers below are therefore a same-dataset, same-metric comparison and not an identical-condition reproduction. \
Poses come from the dataset (no LiDAR-inertial odometry), and the GPU is an RTX PRO 6000 (paper: RTX 4090 Laptop).

### Class-level segmentation (ScanNet v2, 40 validation scenes, paper Table II)

Predicted points take the label of the nearest ground-truth vertex within r = 0.05 m; values are means over scenes.

| | mIoU | f-mIoU | Acc |
|---|---|---|---|
| Paper, without background | 27.4 | 43.5 | 55.5 |
| **Ours, without background** | **59.4** | **59.9** | **62.8** |
| Paper, with background | 22.6 | 29.1 | 33.0 |
| **Ours, with background** | **55.6** | **59.0** | **64.0** |

### Instance AP (3D axis-aligned box IoU, paper Table III classes; AP50 / AP25)

| Class | Paper | Ours |
|---|---|---|
| chair | 63.8 / 74.7 | 54.6 / 76.6 |
| window | 42.2 / 67.9 | 27.7 / 59.0 |
| refrigerator | 62.5 / 62.5 | 26.0 / 57.3 |
| sofa | 33.4 / 83.4 | 86.4 / 86.4 |
| door | 10.0 / 25.0 | 29.7 / 42.7 |

Over all 17 evaluated classes we reach mAP25 61.3 and mAP50 42.9, with 46 active instances per scene against 15 ground-truth objects. \
The remaining gaps are: window, which is planar and tends to merge with the wall; refrigerator, which appears in only 8 scenes, so a single miss moves the number a lot; and desk, which is 0 because the detector scores the "table" phrase higher on desks and the class-agnostic NMS keeps only one box per object.

### Change detection (paper Table IV)

The paper evaluates 6 real objects on a robot. \
Our data are 58 synthetic ScanNet sequences: 29 objects, each removed or added, where the object pixels are replaced by the background rendered from the ground-truth mesh. \
A strict match requires 3D IoU > 0.1, centroid distance < 0.3 m and the same label.

| | Detection recall | Change recall |
|---|---|---|
| Paper (6 real objects) | 0.26 - 1.00 | 0.62 - 1.00 |
| Ours, added objects | 0.51 | 0.64 |
| Ours, removed objects | 0.51 | 0.46 |

Removals are harder on a narrow RGB-D revisit that sees the old location only in slivers.

### Runtime (paper Sec. V-H)

Replica office0, 2000 frames: 200 s of data are processed in 100.2 s.

| Module | Period (ms) | Mean / p95 / max latency (ms) |
|---|---|---|
| segmentation | 1000 | 189.5 / 259.8 / 330.0 |
| mapping | 333 | 64.6 / 77.9 / 128.2 |
| scene graph | 200 | 0.6 / 0.8 / 1.1 |
| pose | 100 | 0.1 / 0.1 / 0.7 |

All modules stay within the paper's periods. \
The GPU is stronger than the paper's, so these latencies are optimistic.

## Differences from the paper

- **Pose source.** Poses come from the dataset through a `PoseProvider` interface instead of SuperOdometry (LiDAR-IMU, ROS). \
  The extrinsic composition of Eq. (3) is implemented.
- **Depth source.** The depth is the RGB-D camera depth, not LiDAR.
- **Models.** GroundingDINO-base and SAM2.1-hiera-small. \
  The prompt list is the 19 ScanNet benchmark classes plus a few aliases, because the paper's list is unpublished.
- **Timestamps.** They are simulated from the dataset frame rate to reproduce the module rates.
- **VLM.** The backend is pluggable (Gemini flash family, with a fallback when a model is no longer served). \
  The serialization and the answer parser are deterministic.
- **Details the paper leaves open** and this implementation fills in: class-agnostic NMS on detector boxes; depth-edge and depth-outlier filters and mask erosion before lifting; absorbing a new track whose points lie inside an existing instance; a class-aware first association; keeping the largest connected component of an instance after a merge; object-level existence log-odds with removal; and every numeric threshold (Eq. 8, 9, 10 probabilities, scene-graph gamma and delta, ByteTrack thresholds). \
  The measurements behind each choice are in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
- **Evaluation.** ScanNet and synthetic change sequences replace the paper's robot experiments; there is no real-robot or navigation stack.

## Installation

The environment uses uv with Python 3.11 and torch cu128 (an sm_120 GPU needs torch 2.7 or newer). \
Run everything with `.venv/bin/python`.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

- **Models.** Detector: Grounding DINO base, [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base). \
  Segmenter: SAM 2.1 Hiera-S, [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small). \
  The config loads them from the Hugging Face cache (`local_files_only: true`), so download them once:

  ```bash
  .venv/bin/hf download IDEA-Research/grounding-dino-base
  .venv/bin/hf download facebook/sam2.1-hiera-small
  ```

- **Data.** Datasets are not included or redistributed here. \
  Replica RGB-D in the NICE-SLAM format ([Replica.zip](https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip)) goes to `data/replica/<scene>/`. \
  ScanNet requires accepting its [terms of use](http://www.scan-net.org/). \
  Export each `.sens` with the official [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) (`--export_color_images --export_depth_images --export_poses --export_intrinsics`) into `data/scannet/scans/<scene>/{color,depth,pose,intrinsic}/`.
- **VLM (optional).** Put `GEMINI_API_KEY=...` in a `.env` file in the repository root.

## Usage

Run one sequence:

```bash
.venv/bin/python -m supermap.run --config configs/default.yaml --subset configs/subset_replica_office0.yaml --out outputs/office0
```

The subset file selects the dataset, scene and frame range (`configs/subset_replica_office0.yaml`: Replica office0, 50 frames). \
That 50-frame subset is a smoke test that checks the pipeline runs end to end (about 10 s of video, 9 instances). \
To run a whole scene use `experiments/configs/full_office0_subset.yaml` (office0, 2000 frames). \
For another Replica scene change its `scene:`; for ScanNet copy `experiments/configs/scannet_subset_template.yaml` and fill in the scene name.

```bash
.venv/bin/python -m supermap.run --config configs/default.yaml \
  --subset experiments/configs/full_office0_subset.yaml --out outputs/office0_full
```

Runtime (RTX PRO 6000, default config): about 2 min for one 2000-frame Replica scene, 0.5-4 min for one ScanNet scene (stride 3, 300-1500 frames). \
The first run adds a few seconds of model loading.

The output folder contains:

| File | Content |
|---|---|
| `map_objects.json` | instances: id, label distribution, box, state, existence |
| `map_points.npz` | the colorized dense map and instance points |
| `graph.json` | the 4D scene graph (nodes, spatial and temporal edges) |
| `serialized.txt` | the text serialization sent to the VLM |
| `run_summary.json` | frame counts, instance counts, module run counts, absorb / removal statistics, VLM status |

### 3D visualization

Turn an output folder into a single HTML file you can explore in a browser:

```bash
.venv/bin/python scripts/visualize.py outputs/office0   # -> outputs/office0/viz.html
```

It shows the dense map, per-instance points, boxes and labels, spatial (on/beside/under) and temporal edges, VLM waypoints, and the camera trajectory; each can be toggled in the legend (wall / floor / ceiling instances start hidden because they cover the furniture). \
The file is self-contained and opens in any browser. The trajectory needs the subset file (`--subset`; defaults to `subset.yaml` in the output folder, else `configs/subset_replica_office0.yaml`). \
`--png <path>` also saves a static image (needs `uv pip install kaleido` and `.venv/bin/plotly_get_chrome`).

![office0 full 2000-frame run](docs/images/office0_full_orbit.gif)

The animation above is the full office0 run (2000 frames, 21 instances, 45 spatial and 59 temporal edges, 106 s): instance points and boxes, the camera trajectory colored by frame index, and the VLM waypoints (yellow diamonds) for "Where is the chair?". \
Static image with the legend: [docs/images/office0_full_viz.png](docs/images/office0_full_viz.png).

`configs/default.yaml` holds every parameter with a comment per key. \
How each default was chosen is explained in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Repository layout

```
src/supermap/
  datasets/       Replica and ScanNet loaders
  geometry/       Eq. 3 composition, back-projection, dense map
  perception/     GroundingDINO detector, prompts and aliases, SAM2 segmenter
  tracking/       Kalman state, ByteTrack, neighbors, Unproject, ObjMerge / MapMerge
  consistency/    Eq. 7-9 point states, Eq. 8 log-odds, ObjNegate, existence log-odds
  semantics/      Eq. 10 label fusion
  graph/          clustering, predicates, temporal edges, serialization
  vlm/            Gemini backend, <answer> parser, waypoints
  pipeline.py     per-frame processing order (Alg. 1)
  scheduler.py    asynchronous module rates
  run.py          entry point (python -m supermap.run)
configs/          default.yaml (all parameters), example subset
docs/             implementation documentation
```

## Documentation

- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) (한국어: [docs/IMPLEMENTATION.ko.md](docs/IMPLEMENTATION.ko.md)): how it was built, with implementation details, parameters, experiments and the reasoning behind each default.

## Citation

If you use this repository, please cite the original paper. \
Official implementation and project page: [github.com/superxslam/SuperMap](https://github.com/superxslam/SuperMap), [superodometry.com/supermap](https://superodometry.com/supermap).

```bibtex
@inproceedings{supermap2026,
  title     = {SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation},
  author    = {Zhao, Shibo and Chen, Guofei and Zhu, Honghao and Li, Zhiheng and Yao, Changwei and Zantout, Nader and Kim, Seungchan and Wang, Wenshan and Zhang, Ji and Scherer, Sebastian},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026},
  note      = {arXiv:2608.22896},
  url       = {https://arxiv.org/abs/2608.22896}
}
```

## Acknowledgements

This implementation builds on the following datasets and models. \
Please cite them as well if you use this repository. \
No data or model weights are redistributed here; the download scripts fetch them from their original sources.

| | Used for | License / terms |
|---|---|---|
| [ScanNet v2](http://www.scan-net.org/) | Main benchmark (40 validation scenes) and the synthetic change-detection sequences | ScanNet Terms of Use (non-commercial research) |
| [Replica](https://github.com/facebookresearch/Replica-Dataset), RGB-D renders from [NICE-SLAM](https://github.com/cvg/nice-slam) | Development, smoke test, runtime benchmark | Replica research license |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (grounding-dino-base) | Open-vocabulary detection | Apache-2.0 |
| [SAM 2](https://github.com/facebookresearch/sam2) (sam2.1-hiera-small) | Box-prompted instance masks | Apache-2.0 |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Two-stage association (algorithm reimplemented, no code copied) | MIT |

```bibtex
@inproceedings{dai2017scannet,
  title     = {ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
  author    = {Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017}
}

@article{straub2019replica,
  title   = {The {R}eplica Dataset: A Digital Replica of Indoor Spaces},
  author  = {Straub, Julian and Whelan, Thomas and Ma, Lingni and Chen, Yufan and Wijmans, Erik and Green, Simon and Engel, Jakob J. and Mur-Artal, Raul and Ren, Carl and Verma, Shobhit and Clarkson, Anton and Yan, Mingfei and Budge, Brian and Yan, Yajie and Pan, Xiaqing and Yon, June and Zou, Yuyang and Leon, Kimberly and Carter, Nigel and Briales, Jesus and Gillingham, Tyler and Mueggler, Elias and Pesqueira, Luis and Savva, Manolis and Batra, Dhruv and Strasdat, Hauke M. and De Nardi, Renzo and Goesele, Michael and Lovegrove, Steven and Newcombe, Richard},
  journal = {arXiv preprint arXiv:1906.05797},
  year    = {2019}
}

@inproceedings{zhu2022niceslam,
  title     = {NICE-SLAM: Neural Implicit Scalable Encoding for SLAM},
  author    = {Zhu, Zihan and Peng, Songyou and Larsson, Viktor and Xu, Weiwei and Bao, Hujun and Cui, Zhaopeng and Oswald, Martin R. and Pollefeys, Marc},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}

@inproceedings{liu2024groundingdino,
  title     = {Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author    = {Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Jiang, Qing and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}

@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024}
}

@inproceedings{zhang2022bytetrack,
  title     = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author    = {Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```

## License

The code and documentation in this repository are released under the [MIT License](LICENSE). \
The SuperMap algorithm and paper belong to their authors; this license covers only this reimplementation.

## TODO

Where the implementation still falls short of the paper, and what was left out of scope. The numbers and diagnoses behind each item are in [docs/IMPLEMENTATION.md, section 9](docs/IMPLEMENTATION.md#9-known-limitations-and-open-issues).

- [ ] **Instance over-segmentation and mixing.** 46 active instances per ScanNet scene vs 15 GT objects; on office0 two sofas end up as three instances and one table as two. \
  The instance id is the 2D track id, so a lost track opens a new id, and re-activation (Eq. 5) and absorption (D-111) only work on sides already seen. Adjacent objects of the same class get mixed by a single detection box or an ID switch. There is no step that re-merges or splits instances in 3D.
- [ ] **Background leaking into furniture labels.** On office0 a wall plus rug becomes "table" and the blinds become "door". Needs mask clean-up or explicit background handling.
- [ ] **Lost desk class.** The class-agnostic NMS (D-116) suppresses the "desk" box behind the "table" box of the same object, so desk AP50 is 0 (I-14-2).
- [ ] **Thin planar objects.** window AP50 27.7 (paper 42.2), refrigerator 26.0.
- [ ] **Removal recall in change detection.** On narrow-FOV RGB-D revisits only 14 of 29 removed objects reach REMOVED, with a median of 354 frames.
- [ ] **False removals in static scenes.** The object existence log-odds (D-113) deletes 17-24 small fragments per scene (I-9-4).
- [ ] **SuperOdometry / LiDAR input.** Only dataset GT poses and RGB-D depth are used; a `PoseProvider` implementation would add it.
