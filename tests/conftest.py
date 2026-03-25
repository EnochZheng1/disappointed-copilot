"""Shared test fixtures."""

import numpy as np
import pytest

from disappointed.detection.models import BoundingBox
from disappointed.pipeline.frame_data import FrameData


@pytest.fixture
def sample_frame():
    """A 720p black frame for testing."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_data(sample_frame):
    """A FrameData with a sample frame and no detections."""
    return FrameData(frame=sample_frame, frame_index=0)


def make_car_detection(
    x1: float = 100, y1: float = 200, x2: float = 300, y2: float = 400,
    confidence: float = 0.9, track_id: int | None = None,
) -> BoundingBox:
    """Helper to create a car detection."""
    return BoundingBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        class_id=2, class_name="car",
        confidence=confidence, track_id=track_id,
    )
