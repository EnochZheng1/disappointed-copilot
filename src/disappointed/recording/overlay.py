"""Overlay renderer — burns bounding boxes, lane lines, and captions onto frames."""

import cv2
import numpy as np

from disappointed.detection.models import BoundingBox
from disappointed.triggers.models import TriggerEvent


class OverlayRenderer:
    """Renders visual overlays onto clip frames for social media export."""

    def draw_detections(self, frame: np.ndarray, detections: list[BoundingBox]) -> np.ndarray:
        """Draw bounding boxes with labels onto a frame."""
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_trigger_marker(self, frame: np.ndarray, event: TriggerEvent) -> np.ndarray:
        """Draw a trigger event marker (red banner at top)."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        text = f"[{event.trigger_name.upper()}] {event.description}"
        cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def draw_caption(self, frame: np.ndarray, caption: str) -> np.ndarray:
        """Draw a caption bar at the bottom of the frame."""
        if not caption:
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, caption, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_watermark(self, frame: np.ndarray) -> np.ndarray:
        """Draw a small watermark in the corner."""
        h, w = frame.shape[:2]
        text = "Disappointed Co-Pilot"
        cv2.putText(frame, text, (w - 220, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        return frame
