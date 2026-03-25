"""Canonical per-frame data object that flows through the entire pipeline."""

from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np

from disappointed.detection.models import BoundingBox
from disappointed.lane.models import LaneState
from disappointed.triggers.models import TriggerEvent


@dataclass
class FrameData:
    """Single frame with all accumulated pipeline results."""

    frame: np.ndarray  # BGR image (H, W, 3)
    frame_index: int
    timestamp: float = field(default_factory=time.time)
    detections: list[BoundingBox] = field(default_factory=list)
    lane_state: Optional[LaneState] = None
    trigger_events: list[TriggerEvent] = field(default_factory=list)
    fps: float = 0.0
