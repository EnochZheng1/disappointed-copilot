"""Tests for tailgater trigger."""

import numpy as np

from disappointed.config.schema import TailgaterConfig
from disappointed.detection.models import BoundingBox
from disappointed.pipeline.frame_data import FrameData
from disappointed.triggers.tailgater import TailgaterTrigger


def _make_frame(frame_index, timestamp, detections):
    return FrameData(
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_index=frame_index,
        timestamp=timestamp,
        detections=detections,
    )


def _car(x1, y1, x2, y2, track_id=0):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=2, class_name="car",
                       confidence=0.9, track_id=track_id)


def test_tailgater_triggers_on_rapid_growth():
    config = TailgaterConfig(consecutive_frames=5, bbox_growth_rate_threshold=0.1, cooldown_seconds=0)
    trigger = TailgaterTrigger(config)

    result = None
    for i in range(30):
        # Car centered in frame, growing rapidly (approaching)
        size = 50 + i * 8  # Growing from 50px to ~290px wide
        half = size // 2
        car = _car(640 - half, 400 - half, 640 + half, 400 + half, track_id=1)
        frame = _make_frame(i, i / 30.0, [car])
        result = trigger.evaluate(frame)
        if result is not None:
            break

    assert result is not None
    assert result.trigger_name == "tailgater"


def test_tailgater_does_not_trigger_on_stable_car():
    config = TailgaterConfig(consecutive_frames=5, cooldown_seconds=0)
    trigger = TailgaterTrigger(config)

    result = None
    for i in range(60):
        car = _car(540, 300, 740, 500, track_id=1)  # Constant size
        frame = _make_frame(i, i / 30.0, [car])
        result = trigger.evaluate(frame)

    assert result is None


def test_tailgater_ignores_small_objects():
    config = TailgaterConfig(min_bbox_area_ratio=0.02, cooldown_seconds=0)
    trigger = TailgaterTrigger(config)

    result = None
    for i in range(30):
        # Tiny car (below min_bbox_area_ratio)
        car = _car(630, 350, 650, 370, track_id=1)
        frame = _make_frame(i, i / 30.0, [car])
        result = trigger.evaluate(frame)

    assert result is None


def test_tailgater_ignores_cars_on_sides():
    config = TailgaterConfig(consecutive_frames=5, bbox_growth_rate_threshold=0.1, cooldown_seconds=0)
    trigger = TailgaterTrigger(config)

    result = None
    for i in range(30):
        # Car on far left side (not ahead of us)
        size = 50 + i * 8
        car = _car(10, 400, 10 + size, 400 + size, track_id=1)
        frame = _make_frame(i, i / 30.0, [car])
        result = trigger.evaluate(frame)

    assert result is None
