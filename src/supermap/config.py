"""YAML config loading. Values and their paper sources live in configs/."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# REQ-006, REQ-007: load config and resolve the dataset section by name
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# REQ-006, REQ-007: merge `datasets.<name>` with scene/root overrides into one dict
def dataset_config(cfg: dict, name: str, **overrides) -> dict:
    sec = dict(cfg["datasets"][name])
    sec["name"] = name
    sec["T_BC"] = cfg["extrinsics"]["T_BC"]
    sec.update({k: v for k, v in overrides.items() if v is not None})
    return sec
