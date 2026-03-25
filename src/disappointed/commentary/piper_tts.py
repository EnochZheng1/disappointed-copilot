"""Piper TTS wrapper for local text-to-speech synthesis."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PiperTTS:
    """Text-to-speech using Piper (https://github.com/rhasspy/piper).

    Requires piper-tts to be installed: pip install piper-tts
    Or the piper binary available on PATH.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._use_python = True  # Try Python module first, fall back to CLI

    def synthesize(self, text: str, output_path: Path) -> None:
        """Synthesize text to a WAV file."""
        if self._use_python:
            try:
                self._synthesize_python(text, output_path)
                return
            except ImportError:
                logger.info("piper-tts Python module not available, trying CLI")
                self._use_python = False
            except Exception:
                logger.exception("Piper Python synthesis failed, trying CLI")
                self._use_python = False

        self._synthesize_cli(text, output_path)

    def _synthesize_python(self, text: str, output_path: Path) -> None:
        """Synthesize using the piper-tts Python module."""
        import wave
        from piper import PiperVoice

        voice = PiperVoice.load(self._model_path)
        with wave.open(str(output_path), "wb") as wav_file:
            voice.synthesize(text, wav_file)
        logger.debug(f"Piper TTS (Python): {output_path}")

    def _synthesize_cli(self, text: str, output_path: Path) -> None:
        """Synthesize using the piper CLI binary."""
        try:
            result = subprocess.run(
                [
                    "piper",
                    "--model", self._model_path,
                    "--output_file", str(output_path),
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"Piper CLI failed: {result.stderr}")
            else:
                logger.debug(f"Piper TTS (CLI): {output_path}")
        except FileNotFoundError:
            logger.error("Piper binary not found on PATH. Install: pip install piper-tts")
        except subprocess.TimeoutExpired:
            logger.error("Piper TTS synthesis timed out")
