"""Self-critique trigger — fires when the ego vehicle drifts out of lane."""

from disappointed.config.schema import SelfCritiqueConfig
from disappointed.pipeline.frame_data import FrameData
from .base import Trigger
from .models import TriggerEvent


class SelfCritiqueTrigger(Trigger):
    """Detects when the driver's own vehicle is departing from lane center.

    Uses LaneState.own_offset_from_center and the lane detector's departure flag.
    Auto-disables when lane detection confidence is below threshold.
    """

    def __init__(self, config: SelfCritiqueConfig, lane_confidence_threshold: float = 0.3):
        super().__init__(name="self_critique", cooldown_seconds=config.cooldown_seconds)
        self._config = config
        self._lane_conf_threshold = lane_confidence_threshold
        self._departure_frames = 0  # Consecutive frames with departure

    def _evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        if not self._config.enabled:
            return None

        lane = frame_data.lane_state
        if lane is None or lane.confidence < self._lane_conf_threshold:
            self._departure_frames = 0
            return None

        offset = abs(lane.own_offset_from_center)
        self._report_diagnostics("lane_offset", offset, self._config.departure_threshold)
        if offset > self._config.departure_threshold:
            self._departure_frames += 1
        else:
            self._departure_frames = 0

        # Require sustained departure (at least ~0.5s at 30fps)
        if self._departure_frames >= 15:
            side = lane.departure_side or ("left" if lane.own_offset_from_center < 0 else "right")
            severity = min(offset / 0.4, 1.0)
            self._departure_frames = 0  # Reset counter after trigger
            return TriggerEvent(
                trigger_name=self.name,
                severity=severity,
                timestamp=frame_data.timestamp,
                frame_index=frame_data.frame_index,
                description=f"You're drifting {side} (offset: {lane.own_offset_from_center:+.3f})",
                metadata={"offset": lane.own_offset_from_center, "side": side},
            )

        return None
