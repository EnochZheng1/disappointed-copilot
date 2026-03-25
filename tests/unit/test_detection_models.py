"""Tests for detection data models."""

from disappointed.detection.models import BoundingBox


def test_bounding_box_properties():
    box = BoundingBox(x1=10, y1=20, x2=110, y2=120, class_id=2, class_name="car", confidence=0.95)
    assert box.width == 100
    assert box.height == 100
    assert box.area == 10000
    assert box.center == (60.0, 70.0)


def test_bounding_box_area_ratio():
    box = BoundingBox(x1=0, y1=0, x2=128, y2=72, class_id=2, class_name="car", confidence=0.9)
    ratio = box.area_ratio(1280, 720)
    assert abs(ratio - 0.01) < 1e-6


def test_bounding_box_track_id_default():
    box = BoundingBox(x1=0, y1=0, x2=10, y2=10, class_id=0, class_name="person", confidence=0.8)
    assert box.track_id is None
