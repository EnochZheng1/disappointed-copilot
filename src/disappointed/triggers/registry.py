"""Trigger registry — evaluates all active triggers per frame."""

from disappointed.pipeline.frame_data import FrameData
from .base import Trigger, TriggerDiagnostics
from .models import TriggerEvent


class TriggerRegistry:
    """Manages all active triggers and evaluates them each frame."""

    def __init__(self):
        self._triggers: list[Trigger] = []

    def register(self, trigger: Trigger) -> None:
        self._triggers.append(trigger)

    def set_tune_mode(self, enabled: bool) -> None:
        for trigger in self._triggers:
            trigger.tune_mode = enabled

    def evaluate(self, frame_data: FrameData) -> list[TriggerEvent]:
        events = []
        for trigger in self._triggers:
            event = trigger.evaluate(frame_data)
            if event is not None:
                events.append(event)
        return events

    def get_diagnostics(self) -> list[TriggerDiagnostics]:
        """Get the latest diagnostics from all triggers (for tuning display)."""
        return [t.diagnostics for t in self._triggers if t.diagnostics is not None]

    @property
    def trigger_names(self) -> list[str]:
        return [t.name for t in self._triggers]
