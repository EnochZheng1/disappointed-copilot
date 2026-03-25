"""YOLOv8 object detector using the Ultralytics library (desktop development)."""

import logging
import time

import numpy as np

from .base import ObjectDetector
from .models import BoundingBox

logger = logging.getLogger(__name__)


class UltralyticsDetector(ObjectDetector):
    """Object detection via Ultralytics YOLOv8 (PyTorch backend)."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        target_classes: list[int] | None = None,
        input_size: int = 640,
    ):
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._target_classes = target_classes
        self._input_size = input_size
        self._model = None
        self._last_inference_ms: float = 0.0

    def load_model(self) -> None:
        from ultralytics import YOLO

        logger.info(f"Loading YOLO model: {self._model_path}")
        self._model = YOLO(self._model_path)
        # Warm up with a dummy frame
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False)
        logger.info("YOLO model loaded and warmed up")

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=self._confidence_threshold,
            imgsz=self._input_size,
            verbose=False,
            classes=self._target_classes,
        )
        self._last_inference_ms = (time.perf_counter() - start) * 1000

        detections: list[BoundingBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append(BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                ))

        return detections

    def get_inference_time_ms(self) -> float:
        return self._last_inference_ms
