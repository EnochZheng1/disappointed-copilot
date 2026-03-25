"""Abstract trigger interface with cooldown and history management."""

from abc import ABC, abstractmethod
from collections import deque
import time

from disappointed.pipeline.frame_data import FrameData
from .models import TriggerEvent


class Trigger(ABC):
    """Base class for all driving behavior triggers."""

    def __init__(self, name: str, cooldown_seconds: float = 15.0):
        self.name = name
        self.cooldown_seconds = cooldown_seconds
        self._last_triggered: float = 0.0
        self._history: deque[FrameData] = deque(maxlen=300)  # ~10s at 30fps

    @abstractmethod
    def _evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        """Core evaluation logic. Subclasses implement this.

        Return a TriggerEvent if triggered, else None.
        """
        ...

    def evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        """Evaluate trigger with cooldown enforcement and history tracking."""
        self._history.append(frame_data)

        if self._is_in_cooldown(frame_data.timestamp):
            return None

        event = self._evaluate(frame_data)
        if event is not None:
            self._last_triggered = frame_data.timestamp
        return event

    def _is_in_cooldown(self, now: float) -> bool:
        return (now - self._last_triggered) < self.cooldown_seconds
