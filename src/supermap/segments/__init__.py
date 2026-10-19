"""Segment-based instance backend (extension mode, outside the paper; D-124).

Observations are kept as per-frame 3D segments and instances are clusters of segments decided by
multi-view consensus (MaskClustering-style support / conflict counting) instead of 2D track ids.
The paper pipeline (track id = instance id, Supp. Alg. 1) stays the default; this package is used
only when `tracking.instance_backend: segments` or by the offline study tool.
"""
from supermap.segments.store import Keyframe, Segment, SegmentStore

__all__ = ["Keyframe", "Segment", "SegmentStore"]
