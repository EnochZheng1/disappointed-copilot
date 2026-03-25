"""LLM-based dynamic commentary engine using Ollama."""

import json
import logging
import tempfile
from pathlib import Path

from disappointed.triggers.models import TriggerEvent
from .base import CommentaryEngine
from .personas import get_persona, get_roast_prompt

logger = logging.getLogger(__name__)


class LLMEngine(CommentaryEngine):
    """Generates dynamic sarcastic roasts via Ollama + Piper TTS.

    This is the secondary "bonus follow-up" commentary engine.
    Latency: 3-10 seconds depending on hardware.
    """

    def __init__(
        self,
        voice_pack: str,
        ollama_model: str = "llama3.2:1b",
        ollama_base_url: str = "http://localhost:11434",
        piper_model_path: str | None = None,
    ):
        self._voice_pack = voice_pack
        self._ollama_model = ollama_model
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._piper_model_path = piper_model_path
        self._persona = get_persona(voice_pack)
        self._tts = None

    def get_audio(self, event: TriggerEvent) -> Path | None:
        """Generate a roast via LLM and synthesize to audio."""
        text = self._generate_text(event)
        if not text:
            return None

        audio_path = self._synthesize_speech(text)
        return audio_path

    def _generate_text(self, event: TriggerEvent) -> str | None:
        """Call Ollama to generate a sarcastic roast."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed — LLM engine requires: pip install httpx")
            return None

        prompt = get_roast_prompt(self._persona, event.trigger_name, event.description)

        try:
            response = httpx.post(
                f"{self._ollama_base_url}/api/generate",
                json={
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "system": self._persona.system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "num_predict": 80,  # Keep it short
                    },
                },
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            if text:
                logger.info(f"LLM roast: {text}")
            return text or None
        except Exception:
            logger.exception("Ollama request failed")
            return None

    def _synthesize_speech(self, text: str) -> Path | None:
        """Convert text to speech using Piper TTS. Returns path to WAV file."""
        if not self._piper_model_path:
            logger.debug("No Piper model configured — skipping TTS")
            return None

        try:
            from .piper_tts import PiperTTS

            if self._tts is None:
                self._tts = PiperTTS(self._piper_model_path)

            output_path = Path(tempfile.mktemp(suffix=".wav", prefix="roast_"))
            self._tts.synthesize(text, output_path)
            return output_path
        except Exception:
            logger.exception("Piper TTS synthesis failed")
            return None
