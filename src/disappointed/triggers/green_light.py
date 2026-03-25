"""Green Light Lingerer trigger — fires when a green light is detected but we're not moving."""

import cv2
import numpy as np

from disappointed.config.schema import GreenLightConfig
from disappointed.pipeline.frame_data import FrameData
from .base import Trigger
from .models import TriggerEvent


class GreenLightTrigger(Trigger):
    """Detects when we're sitting at a green light without moving.

    Uses YOLO traffic light detection + color histogram analysis to confirm green,
    then sparse optical flow (Lucas-Kanade) on a small set of feature points
    to estimate ego-motion. If ego-motion is near zero for >N seconds while
    green is detected, fires.
    """

    def __init__(self, config: GreenLightConfig):
        super().__init__(name="green_light", cooldown_seconds=config.cooldown_seconds)
        self._config = config
        self._green_start_time: float | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None

    def _evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        if not self._config.enabled:
            return None

        now = frame_data.timestamp

        # Check for traffic light detections
        traffic_lights = [
            d for d in frame_data.detections if d.class_name == "traffic light"
        ]

        green_detected = False
        for tl in traffic_lights:
            if self._is_green(frame_data.frame, tl):
                green_detected = True
                break

        if not green_detected:
            self._green_start_time = None
            return None

        # Estimate ego-motion via sparse optical flow
        ego_speed = self._estimate_ego_motion(frame_data.frame)

        if ego_speed > self._config.speed_threshold:
            # We're moving — reset timer
            self._green_start_time = None
            return None

        # We're stopped at a green light
        if self._green_start_time is None:
            self._green_start_time = now

        elapsed = now - self._green_start_time
        if elapsed >= self._config.green_detected_seconds:
            self._green_start_time = None  # Reset after trigger
            return TriggerEvent(
                trigger_name=self.name,
                severity=min(elapsed / 10.0, 1.0),
                timestamp=now,
                frame_index=frame_data.frame_index,
                description=f"Green light for {elapsed:.1f}s and we haven't moved",
                metadata={"elapsed_seconds": elapsed, "ego_speed": ego_speed},
            )

        return None

    def _is_green(self, frame: np.ndarray, detection) -> bool:
        """Analyze the detected traffic light crop to determine if it's green."""
        x1, y1 = max(0, int(detection.x1)), max(0, int(detection.y1))
        x2, y2 = min(frame.shape[1], int(detection.x2)), min(frame.shape[0], int(detection.y2))

        if x2 <= x1 or y2 <= y1:
            return False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        # Convert to HSV and check for green dominance in the lower third (green light position)
        h_third = crop.shape[0] // 3
        bottom_third = crop[2 * h_third:, :, :]
        if bottom_third.size == 0:
            return False

        hsv = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2HSV)

        # Green hue range in HSV
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        green_ratio = np.count_nonzero(green_mask) / green_mask.size
        return green_ratio > 0.15

    def _estimate_ego_motion(self, frame: np.ndarray) -> float:
        """Estimate ego-motion using sparse optical flow (Lucas-Kanade).

        Returns a scalar speed estimate (mean flow magnitude).
        Low cost: only tracks ~20 feature points.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Focus on the horizon region (middle 40% of frame, upper half)
        roi_y1 = int(h * 0.3)
        roi_y2 = int(h * 0.6)
        roi_x1 = int(w * 0.2)
        roi_x2 = int(w * 0.8)
        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

        if self._prev_gray is None or self._prev_points is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(roi, roi_x1, roi_y1)
            return 0.0

        if self._prev_points is None or len(self._prev_points) < 3:
            self._prev_gray = gray
            self._prev_points = self._detect_features(roi, roi_x1, roi_y1)
            return 0.0

        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None,
            winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        if next_points is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(roi, roi_x1, roi_y1)
            return 0.0

        # Filter good points
        good = status.flatten() == 1
        if np.sum(good) < 2:
            self._prev_gray = gray
            self._prev_points = self._detect_features(roi, roi_x1, roi_y1)
            return 0.0

        prev_good = self._prev_points[good]
        next_good = next_points[good]

        # Mean flow magnitude
        flow = next_good - prev_good
        magnitudes = np.sqrt(flow[:, 0, 0] ** 2 + flow[:, 0, 1] ** 2)
        mean_speed = float(np.mean(magnitudes))

        # Update state — refresh features periodically
        self._prev_gray = gray
        if len(next_good) < 5:
            self._prev_points = self._detect_features(roi, roi_x1, roi_y1)
        else:
            self._prev_points = next_good.reshape(-1, 1, 2)

        return mean_speed

    def _detect_features(
        self, roi: np.ndarray, offset_x: int, offset_y: int
    ) -> np.ndarray | None:
        """Detect feature points in the ROI for optical flow tracking."""
        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self._config.max_feature_points,
            qualityLevel=0.3,
            minDistance=15,
        )
        if points is None:
            return None

        # Offset points back to full-frame coordinates
        points[:, 0, 0] += offset_x
        points[:, 0, 1] += offset_y
        return points
