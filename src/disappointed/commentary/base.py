"""Abstract commentary engine interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from disappointed.triggers.models import TriggerEvent


class CommentaryEngine(ABC):
    """Base class for audio commentary generation."""

    @abstractmethod
    def get_audio(self, event: TriggerEvent) -> Path | None:
        """Return a path to a WAV/MP3 file for the given event, or None."""
        ...
