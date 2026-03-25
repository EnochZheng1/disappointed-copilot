"""Abstract camera source interface."""

from abc import ABC, abstractmethod
import numpy as np


class CameraSource(ABC):
    """Base class for all camera/video input sources."""

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the camera/video source."""
        ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return a BGR frame as numpy array, or None if unavailable."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the camera/video source."""
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """Return the source's native FPS."""
        ...

    @abstractmethod
    def get_resolution(self) -> tuple[int, int]:
        """Return (width, height) of the source."""
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
