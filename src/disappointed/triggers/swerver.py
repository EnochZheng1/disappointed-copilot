"""Swerver detection trigger — fires when another car crosses lane lines."""

from collections import deque

from disappointed.config.schema import SwerverConfig
from disappointed.pipeline.frame_data import FrameData
from .base import Trigger
from .models import TriggerEvent


class SwerverTrigger(Trigger):
    """Detects when another vehicle's bbox center crosses detected lane lines.

    Auto-disables when lane detection confidence is below threshold.
    """

    def __init__(self, config: SwerverConfig, lane_confidence_threshold: float = 0.3):
        super().__init__(name="swerver", cooldown_seconds=config.cooldown_seconds)
        self._config = config
        self._lane_conf_threshold = lane_confidence_threshold
        # Track which side of the lane each vehicle is on: track_id -> deque of side ("left"/"right"/"center")
        self._side_history: dict[int, deque[str]] = {}

    def _evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        if not self._config.enabled:
            return None

        lane = frame_data.lane_state
        if lane is None or lane.confidence < self._lane_conf_threshold:
            return None  # Lane detection unreliable — auto-disable

        if not lane.left_line or not lane.right_line:
            return None

        # Get lane boundaries at the bottom of the frame
        left_x = lane.left_line.points[0][0]
        right_x = lane.right_line.points[0][0]
        lane_width = right_x - left_x
        if lane_width <= 0:
            return None

        now = frame_data.timestamp

        # Check each tracked vehicle
        active_ids = set()
        for det in frame_data.detections:
            if det.class_name not in ("car", "truck", "bus", "motorcycle"):
                continue
            tid = det.track_id
            if tid is None:
                continue

            active_ids.add(tid)
            cx, cy = det.center

            # Determine which side of the lane the car center is on
            # Use the lane line x-position interpolated to the car's y-position
            # For simplicity, use the bottom-of-frame lane positions
            if cx < left_x:
                side = "left"
            elif cx > right_x:
                side = "right"
            else:
                side = "center"

            if tid not in self._side_history:
                self._side_history[tid] = deque(maxlen=30)
            self._side_history[tid].append(side)

            # Check for lane crossing: was outside, now center (or vice versa)
            history = list(self._side_history[tid])
            if len(history) < self._config.consecutive_frames:
                continue

            recent = history[-self._config.consecutive_frames:]
            sides_set = set(recent)

            # Crossing detected: vehicle has been on both sides of a lane boundary
            if len(sides_set) > 1 and "center" in sides_set and ("left" in sides_set or "right" in sides_set):
                crossing_from = "left" if "left" in sides_set else "right"
                return TriggerEvent(
                    trigger_name=self.name,
                    severity=0.7,
                    timestamp=now,
                    frame_index=frame_data.frame_index,
                    description=f"A vehicle is swerving into our lane from the {crossing_from}",
                    metadata={"track_id": tid, "crossing_from": crossing_from},
                )

        # Clean up stale tracks
        for tid in list(self._side_history.keys()):
            if tid not in active_ids:
                del self._side_history[tid]

        return None
