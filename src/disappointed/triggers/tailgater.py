"""Tailgater detection trigger — fires when a car ahead is closing distance rapidly."""

import time
from collections import deque

from disappointed.config.schema import TailgaterConfig
from disappointed.pipeline.frame_data import FrameData
from .base import Trigger
from .models import TriggerEvent


class TailgaterTrigger(Trigger):
    """Detects tailgating by tracking bbox area growth rate of cars ahead.

    A car that's getting closer will have a rapidly growing bounding box.
    We track the area ratio over a sliding window and fire if the growth rate
    exceeds the threshold for consecutive frames.
    """

    def __init__(self, config: TailgaterConfig):
        super().__init__(name="tailgater", cooldown_seconds=config.cooldown_seconds)
        self._config = config
        # Track area ratios per tracked object: track_id -> deque of (timestamp, area_ratio)
        self._area_history: dict[int, deque[tuple[float, float]]] = {}

    def _evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        if not self._config.enabled:
            return None

        h, w = frame_data.frame.shape[:2]
        frame_area = w * h
        now = frame_data.timestamp

        # Focus on vehicles (cars, trucks, buses) in the lower-center of the frame (ahead of us)
        center_x = w / 2
        relevant_dets = []
        for det in frame_data.detections:
            if det.class_name not in ("car", "truck", "bus"):
                continue
            # Must be roughly centered (within middle 60% of frame)
            cx, cy = det.center
            if abs(cx - center_x) > w * 0.3:
                continue
            area_ratio = det.area / frame_area
            if area_ratio < self._config.min_bbox_area_ratio:
                continue
            relevant_dets.append((det, area_ratio))

        # Update area histories
        active_ids = set()
        for det, area_ratio in relevant_dets:
            tid = det.track_id
            if tid is None:
                continue
            active_ids.add(tid)
            if tid not in self._area_history:
                self._area_history[tid] = deque(maxlen=60)  # ~2s at 30fps
            self._area_history[tid].append((now, area_ratio))

        # Clean up stale tracks
        for tid in list(self._area_history.keys()):
            if tid not in active_ids:
                del self._area_history[tid]

        # Check each tracked vehicle for rapid growth
        for tid, history in self._area_history.items():
            if len(history) < self._config.consecutive_frames:
                continue

            # Compute growth rate over the window
            recent = list(history)[-self._config.consecutive_frames:]
            t_start, area_start = recent[0]
            t_end, area_end = recent[-1]

            dt = t_end - t_start
            if dt <= 0 or area_start <= 0:
                continue

            growth_rate = (area_end - area_start) / (area_start * dt)

            if growth_rate >= self._config.bbox_growth_rate_threshold:
                return TriggerEvent(
                    trigger_name=self.name,
                    severity=min(growth_rate / 0.5, 1.0),  # Normalize to [0, 1]
                    timestamp=now,
                    frame_index=frame_data.frame_index,
                    description=f"A vehicle is closing in fast (growth rate: {growth_rate:.1%}/s)",
                    metadata={"track_id": tid, "growth_rate": growth_rate},
                )

        return None
