"""Abstract object detector interface."""

from abc import ABC, abstractmethod
import numpy as np

from .models import BoundingBox


class ObjectDetector(ABC):
    """Base class for all object detection backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model into memory."""
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Run detection on a single BGR frame. Return list of detections."""
        ...

    @abstractmethod
    def get_inference_time_ms(self) -> float:
        """Return the last inference time in milliseconds."""
        ...
