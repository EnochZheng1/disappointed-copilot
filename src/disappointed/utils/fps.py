"""FPS counter for pipeline performance monitoring."""

import time
from collections import deque


class FPSCounter:
    """Tracks frames-per-second using a sliding window of timestamps."""

    def __init__(self, window_size: int = 60):
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """Record a frame and return the current FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
