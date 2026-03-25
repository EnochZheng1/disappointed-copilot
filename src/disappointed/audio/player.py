"""Non-blocking audio playback using pygame mixer."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_mixer_initialized = False


def _ensure_mixer():
    """Lazy-init pygame mixer on first use."""
    global _mixer_initialized
    if _mixer_initialized:
        return
    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
        _mixer_initialized = True
        logger.info("Audio mixer initialized")
    except Exception:
        logger.exception("Failed to initialize audio mixer")


class AudioPlayer:
    """Plays audio files via pygame mixer."""

    def __init__(self, volume: float = 0.8):
        self._volume = max(0.0, min(1.0, volume))

    def play_blocking(self, path: Path) -> None:
        """Play an audio file and block until playback finishes."""
        _ensure_mixer()
        if not _mixer_initialized:
            return

        try:
            import pygame
            path = Path(path)
            if not path.exists():
                logger.warning(f"Audio file not found: {path}")
                return

            sound = pygame.mixer.Sound(str(path))
            sound.set_volume(self._volume)
            channel = sound.play()
            if channel:
                while channel.get_busy():
                    pygame.time.wait(50)
        except Exception:
            logger.exception(f"Failed to play audio: {path}")

    def play_nonblocking(self, path: Path) -> None:
        """Play an audio file without blocking."""
        _ensure_mixer()
        if not _mixer_initialized:
            return

        try:
            import pygame
            path = Path(path)
            if not path.exists():
                logger.warning(f"Audio file not found: {path}")
                return

            sound = pygame.mixer.Sound(str(path))
            sound.set_volume(self._volume)
            sound.play()
        except Exception:
            logger.exception(f"Failed to play audio: {path}")
