"""Webcam camera source using OpenCV VideoCapture."""

import logging

import cv2
import numpy as np

from .base import CameraSource

logger = logging.getLogger(__name__)


class WebcamSource(CameraSource):
    """Captures frames from a webcam via OpenCV."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {self._device_index}")

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Webcam opened: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Webcam released")

    def get_fps(self) -> float:
        if self._cap is None:
            return self._fps
        return self._cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self) -> tuple[int, int]:
        if self._cap is None:
            return (self._width, self._height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
