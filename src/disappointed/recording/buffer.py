"""In-memory circular ring buffer for recent raw frames."""

import threading

import cv2
import numpy as np


class RingBuffer:
    """Thread-safe circular buffer storing raw BGR frames in pre-allocated numpy memory.

    Stores frames at a reduced resolution to fit within memory budget.
    Supports concurrent read/write: the main loop pushes frames while
    the clip extractor reads ranges from the buffer.
    """

    def __init__(self, max_seconds: int, fps: int, height: int, width: int):
        self._max_frames = max_seconds * fps
        self._height = height
        self._width = width
        self._fps = fps
        self._buffer = np.zeros((self._max_frames, height, width, 3), dtype=np.uint8)
        self._timestamps = np.zeros(self._max_frames, dtype=np.float64)
        self._write_index = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def write_index(self) -> int:
        return self._write_index

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    def push(self, frame: np.ndarray, timestamp: float) -> None:
        """Resize and copy a frame into the ring buffer."""
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            resized = cv2.resize(frame, (self._width, self._height))
        else:
            resized = frame

        idx = self._write_index % self._max_frames
        with self._lock:
            np.copyto(self._buffer[idx], resized)
            self._timestamps[idx] = timestamp
        self._write_index = (self._write_index + 1) % self._max_frames
        self._total_written += 1

    def read_range(self, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the most recent `num_frames` frames. Returns copies safe for background use.

        Returns:
            (frames, timestamps) — frames shape: (N, H, W, 3)
        """
        num_frames = min(num_frames, self._max_frames, self._total_written)
        if num_frames <= 0:
            return np.empty((0, self._height, self._width, 3), dtype=np.uint8), np.empty(0)

        with self._lock:
            # Calculate indices for the most recent num_frames
            end = self._write_index  # One past the last written
            indices = [(end - num_frames + i) % self._max_frames for i in range(num_frames)]
            frames = self._buffer[indices].copy()
            timestamps = self._timestamps[indices].copy()

        return frames, timestamps

    def read_seconds(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Read the most recent `seconds` worth of frames."""
        num_frames = int(seconds * self._fps)
        return self.read_range(num_frames)
