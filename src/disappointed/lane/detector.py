"""Lane detection pipeline using OpenCV with EMA smoothing and confidence gating."""

import logging
from typing import Optional

import cv2
import numpy as np

from disappointed.config.schema import LaneConfig
from .models import LaneLine, LaneState

logger = logging.getLogger(__name__)


class LaneDetector:
    """Detects lane lines using Canny edge detection + Hough transform.

    Features:
    - Configurable ROI mask to focus on the road ahead
    - EMA (exponential moving average) smoothing on lane endpoints to reduce jitter
    - Confidence score based on line detection quality
    - Graceful degradation: returns low-confidence LaneState when lines aren't visible
    """

    def __init__(self, config: LaneConfig, frame_width: int, frame_height: int):
        self._config = config
        self._frame_width = frame_width
        self._frame_height = frame_height

        # Build the ROI polygon from ratio-based vertices
        self._roi_vertices = np.array([[
            (int(v[0] * frame_width), int(v[1] * frame_height))
            for v in config.roi_vertices_ratio
        ]], dtype=np.int32)

        # EMA state for smoothed lane lines
        self._smoothed_left: Optional[tuple[np.ndarray, np.ndarray]] = None  # (x_coords, y_coords)
        self._smoothed_right: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._alpha = config.ema_alpha

        # Rolling confidence
        self._confidence_history: list[float] = []
        self._confidence_window = 15  # frames

    def detect(self, frame: np.ndarray) -> LaneState:
        """Run lane detection on a single BGR frame."""
        h, w = frame.shape[:2]

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blur, self._config.canny_low, self._config.canny_high)

        # Apply ROI mask
        masked = self._apply_roi_mask(edges)

        # Hough line detection
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=self._config.hough_threshold,
            minLineLength=self._config.hough_min_line_length,
            maxLineGap=self._config.hough_max_line_gap,
        )

        if lines is None:
            return self._build_state(None, None, raw_confidence=0.0)

        # Separate lines into left and right based on slope
        left_lines, right_lines = self._classify_lines(lines, w)

        # Fit and smooth lane lines
        left_lane = self._fit_lane(left_lines, h, side="left")
        right_lane = self._fit_lane(right_lines, h, side="right")

        # Calculate raw confidence based on how many lines were found
        raw_confidence = self._calculate_confidence(left_lines, right_lines)

        return self._build_state(left_lane, right_lane, raw_confidence)

    def _apply_roi_mask(self, edges: np.ndarray) -> np.ndarray:
        """Apply a polygon ROI mask to the edge image."""
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self._roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)

    def _classify_lines(
        self, lines: np.ndarray, frame_width: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Separate Hough lines into left and right lanes based on slope."""
        left_lines = []
        right_lines = []
        mid_x = frame_width / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # Skip vertical lines

            slope = (y2 - y1) / (x2 - x1)

            # Filter out near-horizontal lines (likely noise)
            if abs(slope) < 0.3:
                continue

            # In image coordinates: left lane has negative slope, right has positive
            if slope < 0 and x1 < mid_x and x2 < mid_x:
                left_lines.append(line[0])
            elif slope > 0 and x1 > mid_x and x2 > mid_x:
                right_lines.append(line[0])

        return left_lines, right_lines

    def _fit_lane(
        self,
        lines: list[np.ndarray],
        frame_height: int,
        side: str,
    ) -> Optional[LaneLine]:
        """Fit a single lane line from multiple Hough segments, with EMA smoothing."""
        if not lines:
            # Decay smoothed state
            if side == "left":
                self._smoothed_left = None
            else:
                self._smoothed_right = None
            return None

        # Collect all line segment endpoints
        all_x = []
        all_y = []
        for x1, y1, x2, y2 in lines:
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        # Fit a 1st-degree polynomial (line) to the points
        try:
            coeffs = np.polyfit(all_y, all_x, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Generate two endpoints for the fitted line
        y_bottom = frame_height
        y_top = int(frame_height * 0.6)  # Extend to ~60% of frame height

        x_bottom = np.polyval(coeffs, y_bottom)
        x_top = np.polyval(coeffs, y_top)

        raw_x = np.array([x_bottom, x_top], dtype=np.float64)
        raw_y = np.array([y_bottom, y_top], dtype=np.float64)

        # EMA smoothing
        smoothed_ref = self._smoothed_left if side == "left" else self._smoothed_right
        if smoothed_ref is not None:
            smoothed_x = self._alpha * raw_x + (1 - self._alpha) * smoothed_ref[0]
            smoothed_y = raw_y  # Y coordinates are fixed
        else:
            smoothed_x = raw_x
            smoothed_y = raw_y

        # Store smoothed state
        if side == "left":
            self._smoothed_left = (smoothed_x, smoothed_y)
        else:
            self._smoothed_right = (smoothed_x, smoothed_y)

        points = [
            (float(smoothed_x[0]), float(smoothed_y[0])),
            (float(smoothed_x[1]), float(smoothed_y[1])),
        ]

        return LaneLine(points=points, side=side, confidence=min(len(lines) / 5.0, 1.0))

    def _calculate_confidence(
        self, left_lines: list, right_lines: list
    ) -> float:
        """Calculate detection confidence from line counts."""
        left_score = min(len(left_lines) / 3.0, 1.0) if left_lines else 0.0
        right_score = min(len(right_lines) / 3.0, 1.0) if right_lines else 0.0

        # Both lanes detected = high confidence, one = medium, none = zero
        if left_score > 0 and right_score > 0:
            raw = (left_score + right_score) / 2.0
        elif left_score > 0 or right_score > 0:
            raw = max(left_score, right_score) * 0.5
        else:
            raw = 0.0

        # Rolling average for stability
        self._confidence_history.append(raw)
        if len(self._confidence_history) > self._confidence_window:
            self._confidence_history.pop(0)

        return sum(self._confidence_history) / len(self._confidence_history)

    def _build_state(
        self,
        left: Optional[LaneLine],
        right: Optional[LaneLine],
        raw_confidence: float,
    ) -> LaneState:
        """Build LaneState with ego offset and departure detection."""
        confidence = raw_confidence

        # Calculate ego offset from lane center
        offset = 0.0
        if left and right:
            left_x = left.points[0][0]  # Bottom x of left lane
            right_x = right.points[0][0]  # Bottom x of right lane
            lane_center = (left_x + right_x) / 2.0
            frame_center = self._frame_width / 2.0
            lane_width = right_x - left_x
            if lane_width > 0:
                offset = (frame_center - lane_center) / lane_width  # Normalized [-0.5, 0.5]

        # Departure detection
        departure = False
        departure_side = None
        if confidence >= self._config.confidence_threshold and abs(offset) > 0.15:
            departure = True
            departure_side = "left" if offset < 0 else "right"

        return LaneState(
            left_line=left,
            right_line=right,
            own_offset_from_center=offset,
            confidence=confidence,
            departure_detected=departure,
            departure_side=departure_side,
        )
