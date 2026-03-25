"""Tests for audio priority queue."""

import time
from pathlib import Path

from disappointed.audio.queue import AudioItem, AudioPriorityQueue


def test_enqueue_and_dequeue():
    q = AudioPriorityQueue(cooldown_seconds=0)
    item = AudioItem(path=Path("test.wav"), trigger_name="tailgater")
    assert q.enqueue(item, priority=0.5) is True
    result = q.dequeue(timeout=0.1)
    assert result is not None
    assert result.trigger_name == "tailgater"


def test_higher_priority_dequeued_first():
    q = AudioPriorityQueue(cooldown_seconds=0)
    low = AudioItem(path=Path("low.wav"), trigger_name="low")
    high = AudioItem(path=Path("high.wav"), trigger_name="high")
    q.enqueue(low, priority=0.3)
    q.enqueue(high, priority=0.9)

    result = q.dequeue(timeout=0.1)
    assert result is not None
    assert result.trigger_name == "high"


def test_cooldown_rejects_fast_enqueue():
    q = AudioPriorityQueue(cooldown_seconds=5.0)
    item1 = AudioItem(path=Path("a.wav"), trigger_name="test")
    q.enqueue(item1, priority=0.5)

    # Dequeue to set last_play_time
    q.dequeue(timeout=0.1)

    # Try to enqueue again immediately — should be rejected by cooldown
    item2 = AudioItem(path=Path("b.wav"), trigger_name="test")
    assert q.enqueue(item2, priority=0.5) is False


def test_dequeue_returns_none_on_empty():
    q = AudioPriorityQueue(cooldown_seconds=0)
    result = q.dequeue(timeout=0.1)
    assert result is None
