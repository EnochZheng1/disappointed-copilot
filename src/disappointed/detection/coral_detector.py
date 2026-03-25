"""Google Coral EdgeTPU object detector using PyCoral / TFLite Runtime."""

import logging
import time

import cv2
import numpy as np

from .base import ObjectDetector
from .models import BoundingBox

logger = logging.getLogger(__name__)

# COCO class names for the classes we care about
COCO_LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light",
}


class CoralDetector(ObjectDetector):
    """Object detection via Google Coral EdgeTPU with a TFLite model.

    Requires:
        - libedgetpu1-std
        - pycoral or tflite-runtime
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        target_classes: list[int] | None = None,
        input_size: int = 320,
    ):
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._target_classes = set(target_classes) if target_classes else None
        self._input_size = input_size
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._last_inference_ms: float = 0.0

    def load_model(self) -> None:
        try:
            from pycoral.utils.edgetpu import make_interpreter
            self._interpreter = make_interpreter(self._model_path)
            logger.info(f"Coral EdgeTPU model loaded: {self._model_path}")
        except ImportError:
            # Fall back to tflite_runtime without EdgeTPU
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(
                    model_path=self._model_path,
                    experimental_delegates=[
                        tflite.load_delegate("libedgetpu.so.1")
                    ],
                )
                logger.info(f"TFLite + EdgeTPU delegate loaded: {self._model_path}")
            except Exception:
                logger.exception("Failed to load Coral/TFLite model")
                raise

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Warm up
        input_shape = self._input_details[0]["shape"]
        dummy = np.zeros(input_shape, dtype=self._input_details[0]["dtype"])
        self._interpreter.set_tensor(self._input_details[0]["index"], dummy)
        self._interpreter.invoke()
        logger.info("Coral model warmed up")

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        if self._interpreter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        h, w = frame.shape[:2]

        # Preprocess: resize and normalize for the model
        input_shape = self._input_details[0]["shape"]
        input_h, input_w = input_shape[1], input_shape[2]
        resized = cv2.resize(frame, (input_w, input_h))

        # Handle quantized models (uint8 input)
        input_dtype = self._input_details[0]["dtype"]
        if input_dtype == np.uint8:
            input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

        start = time.perf_counter()
        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()
        self._last_inference_ms = (time.perf_counter() - start) * 1000

        # Parse outputs — format depends on the model export
        # Common TFLite SSD output format: boxes, classes, scores, count
        detections: list[BoundingBox] = []

        try:
            # Try SSD-style output (4 tensors: boxes, classes, scores, count)
            if len(self._output_details) >= 4:
                boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
                classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
                scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]
                count = int(self._interpreter.get_tensor(self._output_details[3]["index"])[0])

                for i in range(count):
                    score = float(scores[i])
                    if score < self._confidence_threshold:
                        continue
                    cls_id = int(classes[i])
                    if self._target_classes and cls_id not in self._target_classes:
                        continue

                    # Boxes are in [y1, x1, y2, x2] normalized format
                    y1, x1, y2, x2 = boxes[i]
                    detections.append(BoundingBox(
                        x1=float(x1 * w), y1=float(y1 * h),
                        x2=float(x2 * w), y2=float(y2 * h),
                        class_id=cls_id,
                        class_name=COCO_LABELS.get(cls_id, f"class_{cls_id}"),
                        confidence=score,
                    ))
            else:
                # YOLO-style single-tensor output
                output = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
                detections = self._parse_yolo_output(output, w, h)
        except Exception:
            logger.exception("Failed to parse detection output")

        return detections

    def _parse_yolo_output(
        self, output: np.ndarray, frame_w: int, frame_h: int
    ) -> list[BoundingBox]:
        """Parse YOLO-format TFLite output tensor."""
        detections = []

        # Output shape varies: could be (N, 6) or (N, 85) etc.
        if output.ndim == 2 and output.shape[1] >= 6:
            for row in output:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                conf = row[4]
                if conf < self._confidence_threshold:
                    continue

                cls_scores = row[5:]
                cls_id = int(np.argmax(cls_scores))
                if self._target_classes and cls_id not in self._target_classes:
                    continue

                x1 = (cx - bw / 2) * frame_w
                y1 = (cy - bh / 2) * frame_h
                x2 = (cx + bw / 2) * frame_w
                y2 = (cy + bh / 2) * frame_h

                detections.append(BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    class_id=cls_id,
                    class_name=COCO_LABELS.get(cls_id, f"class_{cls_id}"),
                    confidence=float(conf),
                ))

        return detections

    def get_inference_time_ms(self) -> float:
        return self._last_inference_ms
