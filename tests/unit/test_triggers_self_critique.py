"""Tests for self-critique trigger."""

import numpy as np

from disappointed.config.schema import SelfCritiqueConfig
from disappointed.lane.models import LaneState
from disappointed.pipeline.frame_data import FrameData
from disappointed.triggers.self_critique import SelfCritiqueTrigger


def _make_frame(frame_index, timestamp, lane_state):
    return FrameData(
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_index=frame_index,
        timestamp=timestamp,
        lane_state=lane_state,
    )


def test_self_critique_triggers_on_sustained_departure():
    config = SelfCritiqueConfig(departure_threshold=0.15, cooldown_seconds=0)
    trigger = SelfCritiqueTrigger(config)

    result = None
    for i in range(30):
        lane = LaneState(own_offset_from_center=-0.25, confidence=0.8, departure_detected=True, departure_side="left")
        frame = _make_frame(i, i / 30.0, lane)
        result = trigger.evaluate(frame)
        if result is not None:
            break

    assert result is not None
    assert result.trigger_name == "self_critique"
    assert "left" in result.description


def test_self_critique_does_not_trigger_when_centered():
    config = SelfCritiqueConfig(departure_threshold=0.15, cooldown_seconds=0)
    trigger = SelfCritiqueTrigger(config)

    for i in range(60):
        lane = LaneState(own_offset_from_center=0.02, confidence=0.8)
        frame = _make_frame(i, i / 30.0, lane)
        result = trigger.evaluate(frame)

    assert result is None


def test_self_critique_disabled_on_low_lane_confidence():
    config = SelfCritiqueConfig(departure_threshold=0.15, cooldown_seconds=0)
    trigger = SelfCritiqueTrigger(config, lane_confidence_threshold=0.3)

    for i in range(60):
        lane = LaneState(own_offset_from_center=-0.5, confidence=0.1)  # Low confidence
        frame = _make_frame(i, i / 30.0, lane)
        result = trigger.evaluate(frame)

    assert result is None
