"""Tests for centroid tracker."""

from disappointed.detection.models import BoundingBox
from disappointed.detection.tracker import CentroidTracker


def _make_det(x1, y1, x2, y2, cls_name="car"):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=2, class_name=cls_name, confidence=0.9)


def test_tracker_assigns_ids():
    tracker = CentroidTracker()
    dets = [_make_det(100, 100, 200, 200), _make_det(400, 400, 500, 500)]
    result = tracker.update(dets)
    assert result[0].track_id == 0
    assert result[1].track_id == 1


def test_tracker_maintains_ids_across_frames():
    tracker = CentroidTracker()

    # Frame 1
    dets1 = [_make_det(100, 100, 200, 200)]
    tracker.update(dets1)

    # Frame 2 — same object moved slightly
    dets2 = [_make_det(105, 105, 205, 205)]
    result = tracker.update(dets2)
    assert result[0].track_id == 0  # Same ID


def test_tracker_handles_new_objects():
    tracker = CentroidTracker()

    # Frame 1 — one object
    tracker.update([_make_det(100, 100, 200, 200)])

    # Frame 2 — original + new object far away
    dets = [_make_det(105, 105, 205, 205), _make_det(600, 600, 700, 700)]
    result = tracker.update(dets)
    ids = {d.track_id for d in result}
    assert 0 in ids  # Original
    assert len(ids) == 2  # Two distinct IDs


def test_tracker_handles_empty_frames():
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([_make_det(100, 100, 200, 200)])
    tracker.update([])  # Disappeared once
    tracker.update([])  # Disappeared twice
    # Object should still be tracked (max_disappeared=2)
    dets = [_make_det(105, 105, 205, 205)]
    result = tracker.update(dets)
    assert result[0].track_id == 0  # Still same ID
