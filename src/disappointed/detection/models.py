"""Data models for object detection results."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    """A single detected object with bounding box coordinates."""

    x1: float  # Top-left x (pixels)
    y1: float  # Top-left y (pixels)
    x2: float  # Bottom-right x (pixels)
    y2: float  # Bottom-right y (pixels)
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None  # Assigned by tracker

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def area_ratio(self, frame_width: int, frame_height: int) -> float:
        """Return bbox area as a fraction of the total frame area."""
        return self.area / (frame_width * frame_height)
