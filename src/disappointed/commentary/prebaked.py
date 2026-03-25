"""Pre-baked commentary engine — instant audio from voice pack WAV files."""

import logging
import random
from pathlib import Path

from disappointed.triggers.models import TriggerEvent
from .base import CommentaryEngine

logger = logging.getLogger(__name__)


class PrebakedEngine(CommentaryEngine):
    """Selects a random pre-baked audio file from the active voice pack's trigger folder.

    Directory structure expected:
        assets/audio/{voice_pack}/{trigger_name}/*.wav

    Falls back to the "general" folder if no trigger-specific audio exists.
    """

    def __init__(self, audio_dir: Path, voice_pack: str):
        self._audio_dir = Path(audio_dir)
        self._voice_pack = voice_pack
        self._cache: dict[str, list[Path]] = {}
        self._scan_audio_files()

    def _scan_audio_files(self) -> None:
        """Scan the voice pack directory and cache available audio files per trigger."""
        pack_dir = self._audio_dir / self._voice_pack
        if not pack_dir.exists():
            logger.warning(f"Voice pack directory not found: {pack_dir}")
            return

        for trigger_dir in pack_dir.iterdir():
            if not trigger_dir.is_dir():
                continue
            files = list(trigger_dir.glob("*.wav")) + list(trigger_dir.glob("*.mp3"))
            if files:
                self._cache[trigger_dir.name] = files
                logger.info(f"Voice pack '{self._voice_pack}/{trigger_dir.name}': {len(files)} audio files")

    def get_audio(self, event: TriggerEvent) -> Path | None:
        """Get a random audio file matching the trigger, or from 'general' fallback."""
        # Try trigger-specific folder first
        files = self._cache.get(event.trigger_name)
        if not files:
            # Fall back to general
            files = self._cache.get("general")
        if not files:
            return None
        return random.choice(files)

    @property
    def available_triggers(self) -> list[str]:
        return list(self._cache.keys())
