"""Raspberry Pi camera source using Picamera2."""

import logging

import numpy as np

from .base import CameraSource

logger = logging.getLogger(__name__)


class PicameraSource(CameraSource):
    """Captures frames from the Pi Camera Module via Picamera2.

    Requires: picamera2 (pre-installed on Raspberry Pi OS)
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._picam = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise RuntimeError(
                "picamera2 not available. This source only works on Raspberry Pi. "
                "Install: sudo apt install python3-picamera2"
            )

        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (self._width, self._height), "format": "BGR888"},
            controls={"FrameRate": self._fps},
        )
        self._picam.configure(config)
        self._picam.start()

        logger.info(f"Picamera2 opened: {self._width}x{self._height} @ {self._fps}fps")

    def read(self) -> np.ndarray | None:
        if self._picam is None:
            return None
        return self._picam.capture_array("main")

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None
            logger.info("Picamera2 released")

    def get_fps(self) -> float:
        return float(self._fps)

    def get_resolution(self) -> tuple[int, int]:
        return (self._width, self._height)
