"""Dataset loaders producing Observation streams (REQ-001, REQ-002)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from supermap.datasets.base import ArraySequence, RGBDSequence, select_frame_ids
from supermap.datasets.replica import ReplicaSequence
from supermap.datasets.scannet import ScanNetSequence

# REQ-008: the paper's real-robot data (Livox Mid-360 LiDAR + 360 deg panoramic
# camera, Sec. V-A) is not public and its modality is outside D-001 (RGB-D
# video). It is not loadable here (D-014); build_sequence rejects it explicitly.
UNSUPPORTED_DATASETS = {
    "real_robot": "paper real-robot data (Livox Mid-360 + panoramic camera, Sec. V-A) "
                  "is not publicly released and is outside the RGB-D scope of D-001 (D-014)",
}


# REQ-008: raised when a dataset the paper used cannot be loaded in this project
class DatasetUnavailableError(RuntimeError):
    pass


# REQ-006, REQ-007, REQ-008: build a sequence from a dataset config section
def build_sequence(ds_cfg: dict, subset: dict | None = None, base_dir: str | Path = ".") -> RGBDSequence:
    """ds_cfg: configs/default.yaml `dataset` entry merged with a dataset-specific section.

    Required keys: name, root, scene. Optional subset keys: start, stride, max_frames.
    Relative roots resolve against base_dir.
    """
    name = ds_cfg["name"]
    if name in UNSUPPORTED_DATASETS:
        raise DatasetUnavailableError(UNSUPPORTED_DATASETS[name])
    sub = dict(subset or {})
    root = Path(ds_cfg["root"])
    if not root.is_absolute():
        root = Path(base_dir) / root
    T_BC = np.asarray(ds_cfg.get("T_BC", np.eye(4)), dtype=np.float64)
    common = dict(start=int(sub.get("start", 0)), stride=int(sub.get("stride", 1)),
                  max_frames=sub.get("max_frames"), T_BC=T_BC)
    if name == "replica":
        return ReplicaSequence(root, ds_cfg["scene"], nominal_fps=float(ds_cfg["nominal_fps"]), **common)
    if name == "scannet":
        return ScanNetSequence(root, ds_cfg["scene"], depth_scale=float(ds_cfg["depth_scale"]),
                               nominal_fps=float(ds_cfg["nominal_fps"]), **common)
    raise ValueError(f"unknown dataset {name!r}")


__all__ = ["ArraySequence", "RGBDSequence", "ReplicaSequence", "ScanNetSequence",
           "DatasetUnavailableError", "build_sequence", "select_frame_ids"]
