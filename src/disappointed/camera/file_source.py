"""Video file playback source with proper frame timing."""

import logging
import time

import cv2
import numpy as np

from .base import CameraSource

logger = logging.getLogger(__name__)


class FileSource(CameraSource):
    """Plays back a video file at its native frame rate."""

    def __init__(self, file_path: str, loop: bool = False):
        self._file_path = file_path
        self._loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._frame_interval: float = 1.0 / 30.0
        self._last_read_time: float = 0.0
        self._width: int = 0
        self._height: int = 0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._file_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self._file_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval = 1.0 / self._fps
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video opened: {self._file_path} "
            f"({self._width}x{self._height} @ {self._fps:.1f}fps, {total_frames} frames)"
        )

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None

        # Respect frame timing for realistic playback speed
        now = time.perf_counter()
        elapsed = now - self._last_read_time
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        ret, frame = self._cap.read()
        self._last_read_time = time.perf_counter()

        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
                return frame
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video file released")

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self) -> tuple[int, int]:
        return (self._width, self._height)
