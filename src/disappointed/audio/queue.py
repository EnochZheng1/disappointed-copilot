"""Priority audio queue with cooldown and deduplication."""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import PriorityQueue, Empty


@dataclass(order=True)
class AudioItem:
    """An audio clip queued for playback."""

    path: Path = field(compare=False)
    trigger_name: str = field(compare=False)
    priority: float = field(default=0.5, compare=True)  # Higher = play first
    timestamp: float = field(default_factory=time.time, compare=False)

    def __post_init__(self):
        # PriorityQueue is min-heap, so negate priority for max-priority-first
        self.priority = -self.priority


class AudioPriorityQueue:
    """Thread-safe audio queue with global cooldown and per-trigger dedup."""

    def __init__(self, cooldown_seconds: float = 10.0):
        self._queue: PriorityQueue[AudioItem] = PriorityQueue()
        self._cooldown = cooldown_seconds
        self._last_play_time: float = 0.0
        self._last_trigger_times: dict[str, float] = {}
        self._lock = threading.Lock()

    def enqueue(self, item: AudioItem, priority: float = 0.5) -> bool:
        """Add an audio item to the queue. Returns False if rejected by cooldown."""
        now = time.time()
        with self._lock:
            # Global cooldown
            if (now - self._last_play_time) < self._cooldown:
                return False

            # Per-trigger cooldown (same trigger can't fire twice in a row too fast)
            last_trigger = self._last_trigger_times.get(item.trigger_name, 0.0)
            if (now - last_trigger) < self._cooldown:
                return False

        item.priority = -priority  # Negate for min-heap
        self._queue.put(item)
        return True

    def dequeue(self, timeout: float = 0.5) -> AudioItem | None:
        """Get the highest-priority audio item. Returns None on timeout."""
        try:
            item = self._queue.get(timeout=timeout)
        except Empty:
            return None

        with self._lock:
            now = time.time()
            # Final cooldown check at dequeue time
            if (now - self._last_play_time) < self._cooldown:
                return None
            self._last_play_time = now
            self._last_trigger_times[item.trigger_name] = now

        return item
