"""Hard brake detection trigger — fires when all tracked vehicles grow rapidly (ego deceleration)."""

from collections import deque

from disappointed.config.schema import HardBrakeConfig
from disappointed.pipeline.frame_data import FrameData
from .base import Trigger
from .models import TriggerEvent


class HardBrakeTrigger(Trigger):
    """Detects hard braking by the ego vehicle using bbox growth rate.

    When the driver brakes hard, all vehicles ahead appear to grow rapidly
    in the frame (everything rushes toward the camera). This is cheaper and
    more reliable than dense optical flow.

    Requires at least `min_tracked_vehicles` to be visible and all growing
    simultaneously above the threshold.
    """

    def __init__(self, config: HardBrakeConfig):
        super().__init__(name="hard_brake", cooldown_seconds=config.cooldown_seconds)
        self._config = config
        self._area_history: dict[int, deque[tuple[float, float]]] = {}

    def _evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        if not self._config.enabled:
            return None

        h, w = frame_data.frame.shape[:2]
        frame_area = w * h
        now = frame_data.timestamp

        # Track all vehicles
        vehicles = [
            d for d in frame_data.detections
            if d.class_name in ("car", "truck", "bus") and d.track_id is not None
        ]

        active_ids = set()
        for det in vehicles:
            tid = det.track_id
            active_ids.add(tid)
            area_ratio = det.area / frame_area
            if tid not in self._area_history:
                self._area_history[tid] = deque(maxlen=30)
            self._area_history[tid].append((now, area_ratio))

        # Clean up stale tracks
        for tid in list(self._area_history.keys()):
            if tid not in active_ids:
                del self._area_history[tid]

        # Need enough tracked vehicles
        if len(self._area_history) < self._config.min_tracked_vehicles:
            return None

        # Check if ALL tracked vehicles are growing rapidly (= we're decelerating)
        growth_rates = []
        for tid, history in self._area_history.items():
            if len(history) < self._config.consecutive_frames:
                return None  # Not enough data for any vehicle

            recent = list(history)[-self._config.consecutive_frames:]
            t_start, area_start = recent[0]
            t_end, area_end = recent[-1]

            dt = t_end - t_start
            if dt <= 0 or area_start <= 0:
                return None

            growth_rate = (area_end - area_start) / (area_start * dt)
            growth_rates.append(growth_rate)

        if not growth_rates:
            return None

        # All vehicles must be growing above threshold simultaneously
        if all(gr >= self._config.bbox_growth_rate_threshold for gr in growth_rates):
            avg_rate = sum(growth_rates) / len(growth_rates)
            return TriggerEvent(
                trigger_name=self.name,
                severity=min(avg_rate / 0.5, 1.0),
                timestamp=now,
                frame_index=frame_data.frame_index,
                description=f"Hard braking detected ({len(growth_rates)} vehicles rushing toward us)",
                metadata={"avg_growth_rate": avg_rate, "vehicle_count": len(growth_rates)},
            )

        return None
