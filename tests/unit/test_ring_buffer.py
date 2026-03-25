"""Tests for the recording ring buffer."""

import numpy as np

from disappointed.recording.buffer import RingBuffer


def test_push_and_read():
    buf = RingBuffer(max_seconds=2, fps=10, height=36, width=64)

    # Push 10 frames (1 second)
    for i in range(10):
        frame = np.full((36, 64, 3), i, dtype=np.uint8)
        buf.push(frame, timestamp=i * 0.1)

    frames, timestamps = buf.read_range(5)
    assert frames.shape == (5, 36, 64, 3)
    assert len(timestamps) == 5
    # Last frame pushed had value 9
    assert frames[-1, 0, 0, 0] == 9


def test_ring_wraps_around():
    buf = RingBuffer(max_seconds=1, fps=5, height=10, width=10)
    # Buffer holds 5 frames max. Push 8 frames.
    for i in range(8):
        frame = np.full((10, 10, 3), i * 10, dtype=np.uint8)
        buf.push(frame, timestamp=float(i))

    # Should get the 5 most recent
    frames, ts = buf.read_range(5)
    assert frames.shape == (5, 10, 10, 3)
    assert frames[-1, 0, 0, 0] == 70  # Last pushed: 7*10


def test_read_seconds():
    buf = RingBuffer(max_seconds=5, fps=10, height=10, width=10)
    for i in range(30):
        buf.push(np.zeros((10, 10, 3), dtype=np.uint8), timestamp=i * 0.1)

    frames, ts = buf.read_seconds(1.5)
    assert len(frames) == 15  # 1.5s * 10fps


def test_resizes_input():
    buf = RingBuffer(max_seconds=1, fps=5, height=36, width=64)
    # Push a larger frame — should be resized
    big_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    buf.push(big_frame, timestamp=0.0)

    frames, _ = buf.read_range(1)
    assert frames.shape == (1, 36, 64, 3)


def test_empty_read():
    buf = RingBuffer(max_seconds=1, fps=10, height=10, width=10)
    frames, ts = buf.read_range(5)
    assert len(frames) == 0
