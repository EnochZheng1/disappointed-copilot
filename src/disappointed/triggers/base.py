"""Abstract trigger interface with cooldown, history, and tuning diagnostics."""

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from disappointed.pipeline.frame_data import FrameData
from .models import TriggerEvent

logger = logging.getLogger(__name__)


@dataclass
class TriggerDiagnostics:
    """Diagnostic data for trigger tuning — how close we came to firing."""

    trigger_name: str
    metric_name: str  # e.g. "growth_rate", "offset", "green_seconds"
    current_value: float
    threshold: float
    progress: float  # 0.0 = nowhere close, 1.0 = at threshold, >1.0 = would fire

    @property
    def pct(self) -> str:
        return f"{self.progress:.0%}"


class Trigger(ABC):
    """Base class for all driving behavior triggers."""

    def __init__(self, name: str, cooldown_seconds: float = 15.0):
        self.name = name
        self.cooldown_seconds = cooldown_seconds
        self._last_triggered: float = 0.0
        self._history: deque[FrameData] = deque(maxlen=300)  # ~10s at 30fps
        self.tune_mode: bool = False  # Set externally to enable diagnostics
        self._last_diagnostics: TriggerDiagnostics | None = None

    @abstractmethod
    def _evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        """Core evaluation logic. Subclasses implement this."""
        ...

    def _report_diagnostics(self, metric_name: str, value: float, threshold: float) -> None:
        """Report how close the trigger is to firing. Call from _evaluate()."""
        if threshold == 0:
            progress = 0.0
        else:
            progress = abs(value) / abs(threshold)

        self._last_diagnostics = TriggerDiagnostics(
            trigger_name=self.name,
            metric_name=metric_name,
            current_value=value,
            threshold=threshold,
            progress=progress,
        )

        if self.tune_mode and progress >= 0.5:
            logger.info(
                f"[TUNE] {self.name}: {metric_name}={value:.4f} "
                f"(threshold={threshold:.4f}, progress={progress:.0%})"
            )

    def evaluate(self, frame_data: FrameData) -> TriggerEvent | None:
        """Evaluate trigger with cooldown enforcement and history tracking."""
        self._history.append(frame_data)
        self._last_diagnostics = None

        if self._is_in_cooldown(frame_data.timestamp):
            return None

        event = self._evaluate(frame_data)
        if event is not None:
            self._last_triggered = frame_data.timestamp
        return event

    def _is_in_cooldown(self, now: float) -> bool:
        return (now - self._last_triggered) < self.cooldown_seconds

    @property
    def diagnostics(self) -> TriggerDiagnostics | None:
        return self._last_diagnostics
