"""Data models for lane detection results."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LaneLine:
    """A single detected lane line as a polyline."""

    points: list[tuple[float, float]]  # (x, y) in pixel coordinates
    side: str  # "left" | "right" | "unknown"
    confidence: float = 1.0


@dataclass
class LaneState:
    """Aggregate lane detection state for a single frame."""

    left_line: Optional[LaneLine] = None
    right_line: Optional[LaneLine] = None
    own_offset_from_center: float = 0.0  # Negative = drifting left, positive = right
    confidence: float = 0.0  # Overall lane detection confidence [0, 1]
    departure_detected: bool = False
    departure_side: Optional[str] = None  # "left" | "right"
