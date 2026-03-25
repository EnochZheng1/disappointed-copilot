"""Tests for FPS counter."""

import time

from disappointed.utils.fps import FPSCounter


def test_fps_counter_initial():
    counter = FPSCounter()
    assert counter.tick() == 0.0


def test_fps_counter_tracks_rate():
    counter = FPSCounter(window_size=10)
    # Simulate ~100fps (10ms between frames)
    for _ in range(10):
        counter.tick()
        time.sleep(0.01)
    fps = counter.tick()
    # Should be roughly 100fps (allow tolerance for sleep imprecision)
    assert 50 < fps < 200
