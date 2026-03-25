#!/usr/bin/env python3
"""Batch-generate pre-baked audio WAV files for all voice packs and trigger types.

Usage:
    python scripts/generate_prebaked_audio.py                    # Generate all with edge-tts
    python scripts/generate_prebaked_audio.py --pack deadpan_ai  # Generate one pack
    python scripts/generate_prebaked_audio.py --count 5          # 5 per trigger
    python scripts/generate_prebaked_audio.py --placeholder      # Silent placeholders (no TTS)

Requires: edge-tts (pip install edge-tts) — free, no API key needed.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Roast templates per trigger type. Each will be synthesized as a unique WAV.
ROAST_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "british_instructor": {
        "tailgater": [
            "Oh splendid. This person seems to believe our bumper is a magnet.",
            "Marvellous. They're so close I can see what radio station they're listening to.",
            "And here we have a classic tailgater. Truly the peak of human intelligence.",
            "Right then. If they get any closer, I'll have to charge them rent.",
            "Oh how delightful. Another driver who skipped the chapter on stopping distances.",
        ],
        "swerver": [
            "And there goes another one, weaving about like a confused salmon.",
            "Lovely. That person just treated the lane markings as a gentle suggestion.",
            "Oh brilliant. No indicator, no mirror check, just pure unbridled chaos.",
            "I see someone's decided that lanes are optional today. How very free-spirited.",
            "Wonderful. That car just performed a lane change that would make a snake dizzy.",
        ],
        "green_light": [
            "The light is green. That means go. I realise this is terribly complex.",
            "Green means go. It's been green for quite some time now. Any day now.",
            "Oh don't mind the green light. It's not like anyone behind us has places to be.",
            "The light. Is green. We are not moving. Marvellous.",
            "I believe the correct response to a green light is forward motion. Just a thought.",
        ],
        "self_critique": [
            "We appear to be wandering. How very scenic of us.",
            "I couldn't help but notice we're drifting. The lane is right there.",
            "Are we driving or interpretive dancing? I genuinely can't tell anymore.",
            "The lane lines are not decorative. They serve an actual purpose.",
            "Right. We're having a little wander. Shall I fetch a map?",
        ],
        "hard_brake": [
            "Ah yes. The emergency stop. Always thrilling when it's unplanned.",
            "Well that was dramatic. My tea nearly went everywhere.",
            "Good grief. Perhaps we could anticipate these things occasionally?",
            "And there go the brakes. My neck thanks you for the exercise.",
            "Charming. Absolutely charming. That took years off my life.",
        ],
        "general": [
            "I'm not angry. I'm just disappointed.",
            "This is fine. Everything is fine.",
            "And people wonder why I drink tea.",
        ],
    },
    "tired_mom": {
        "tailgater": [
            "Oh great. Another one on our tail. Buddy, I have kids in here.",
            "Are you kidding me right now? Back off. I swear to God.",
            "I can't. I literally cannot. This person is so close I can smell their cologne.",
            "If you hit us, I will find you. And I will send you my therapy bill.",
            "Oh wonderful. Because today wasn't stressful enough already.",
        ],
        "swerver": [
            "Did you SEE that? Did anyone else see that? No blinker. Nothing. Just vibes.",
            "I can't even. I literally cannot even. That person just—unbelievable.",
            "Oh cool cool cool. Just cut me off. That's fine. I'm fine.",
            "Wow. Just wow. And they didn't even look. I am DONE.",
            "You know what? I'm adding that to my list. My very long list.",
        ],
        "green_light": [
            "Sweetie. The light changed. We're going. This isn't naptime. Move it.",
            "Hello? Green light? Let's GO people. I have soccer practice at seven.",
            "The light is green. GREEN. I don't have time for this.",
            "Oh my GOD the light changed like ten years ago. MOVE.",
            "If we miss this light I swear I'm going to lose it.",
        ],
        "self_critique": [
            "Okay so MAYBE I drifted a little. I've been up since five AM.",
            "Don't judge me. I've got three kids screaming in the back.",
            "Fine. FINE. I'll stay in my lane. Happy now?",
            "I'm tired okay? Everyone's tired. The lane moved, not me.",
            "Oops. Let's just pretend that didn't happen.",
        ],
        "hard_brake": [
            "OH MY GOD. Oh my— okay. Okay. We're fine. We're fine.",
            "WHAT THE— are you SERIOUS right now?!",
            "That's it. I'm done driving today. I'm done. Someone else drive.",
            "My HEART. My literal heart just stopped. I can't do this.",
            "Jesus, Mary, and Joseph. That took ten years off my life.",
        ],
        "general": [
            "I need a glass of wine after this.",
            "This is why I have gray hairs.",
            "Is it bedtime yet? Please say it's bedtime.",
        ],
    },
    "deadpan_ai": {
        "tailgater": [
            "Rear proximity alert. Following distance: inadvisable. Human error probability: certain.",
            "The vehicle behind us appears to be attempting physical fusion with our chassis.",
            "I calculate a 94.7 percent chance that driver failed their spatial reasoning assessment.",
            "Alert. Tailgater detected. Deploying disappointment protocols.",
            "That vehicle's following distance violates several laws of physics and all laws of common sense.",
        ],
        "swerver": [
            "Lane departure detected in adjacent vehicle. Turn signal usage: zero. Surprise factor: also zero.",
            "That maneuver would score a negative four on my driving competence index.",
            "Analyzing lane change. No signal. No mirror check. Peak human performance.",
            "I have observed a lane violation. Adding it to my ever-growing database of human inadequacy.",
            "That vehicle has deviated from its lane. I would express shock, but my probability models predicted this.",
        ],
        "green_light": [
            "The traffic signal has been green for 4.2 seconds. Initiating disappointment subroutines.",
            "Green light detected. Vehicle speed: zero. Processing... processing... still processing.",
            "I'm afraid I must inform you that the light is green. This is not a drill.",
            "Attention. The electromagnetic wavelength of the traffic signal has shifted to 520 nanometers. That means go.",
            "I have calculated 347 ways to proceed through this intersection. All of them require moving.",
        ],
        "self_critique": [
            "Lane departure detected. Operator appears to be experiencing a navigational malfunction.",
            "Our trajectory suggests either a lane change or a catastrophic loss of focus. I suspect the latter.",
            "I'm contractually obligated not to speculate about the cause of this drift. But I have theories.",
            "Lane centering deviation exceeds acceptable parameters. Recalibrating my expectations downward.",
            "Initiating lane departure warning. My disappointment algorithms are running at full capacity.",
        ],
        "hard_brake": [
            "Sudden deceleration event. G-force: uncomfortable. Planning ahead score: zero.",
            "Emergency braking detected. I had identified this threat 3.7 seconds before you did.",
            "Rapid deceleration logged. This was preventable. Most things with humans are.",
            "Alert. Kinetic energy conversion event in progress. Also known as slamming the brakes.",
            "I have experienced what humans call whiplash. Adding to incident report number 4,271.",
        ],
        "general": [
            "Processing. My disappointment buffer has overflowed.",
            "I was not programmed for this level of incompetence. Recalibrating.",
            "This is fine. Adjusting baseline expectations for human driving ability.",
        ],
    },
}


# Edge-TTS voice mapping per persona
# See full list: edge-tts --list-voices
EDGE_VOICES: dict[str, dict] = {
    "british_instructor": {
        "voice": "en-GB-RyanNeural",       # British male, dry and measured
        "rate": "-5%",                       # Slightly slower for gravitas
        "pitch": "-2Hz",                     # Slightly deeper
    },
    "tired_mom": {
        "voice": "en-US-JennyNeural",       # American female, expressive
        "rate": "+5%",                       # Slightly faster, exasperated
        "pitch": "+0Hz",
    },
    "deadpan_ai": {
        "voice": "en-US-GuyNeural",         # American male, flat delivery
        "rate": "-10%",                      # Slow and deliberate
        "pitch": "-4Hz",                     # Lower for robotic feel
    },
}


def generate_with_edge_tts(text: str, output_path: Path, voice_config: dict) -> bool:
    """Generate an MP3 via edge-tts, then convert to WAV."""
    import asyncio

    async def _synthesize():
        import edge_tts

        mp3_path = output_path.with_suffix(".mp3")
        communicate = edge_tts.Communicate(
            text,
            voice=voice_config["voice"],
            rate=voice_config.get("rate", "+0%"),
            pitch=voice_config.get("pitch", "+0Hz"),
        )
        await communicate.save(str(mp3_path))

        # Convert MP3 to WAV using a simple approach
        _mp3_to_wav(mp3_path, output_path)
        mp3_path.unlink()  # Clean up MP3

    try:
        asyncio.run(_synthesize())
        return True
    except Exception as e:
        logger.error(f"edge-tts failed for '{text[:50]}...': {e}")
        return False


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    """Convert MP3 to WAV. Tries ffmpeg first, falls back to pydub, then just copies as-is."""
    import subprocess

    # Try ffmpeg (most reliable)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "22050", "-ac", "1", str(wav_path)],
            capture_output=True, check=True, timeout=10,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fall back: just rename MP3 to WAV extension (pygame can play both)
    import shutil
    shutil.copy2(mp3_path, wav_path)


def generate_with_piper(text: str, output_path: Path, model_path: str) -> bool:
    """Generate a WAV file using Piper TTS."""
    try:
        import wave
        from piper import PiperVoice

        voice = PiperVoice.load(model_path)
        with wave.open(str(output_path), "wb") as wav_file:
            voice.synthesize(text, wav_file)
        return True
    except ImportError:
        logger.error("piper-tts not installed. Run: pip install piper-tts")
        return False
    except Exception as e:
        logger.error(f"Piper TTS failed for '{text[:50]}...': {e}")
        return False


def generate_placeholder(text: str, output_path: Path) -> bool:
    """Generate a silent placeholder WAV file (for testing without TTS)."""
    import struct
    import wave

    # Generate a very short silent WAV
    sample_rate = 22050
    duration = 0.5  # 0.5 second of silence
    n_samples = int(sample_rate * duration)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))

    # Save the text as a sidecar file for reference
    text_path = output_path.with_suffix(".txt")
    text_path.write_text(text, encoding="utf-8")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate pre-baked audio for voice packs")
    parser.add_argument("--pack", type=str, default=None, help="Generate only this voice pack")
    parser.add_argument("--count", type=int, default=None, help="Max roasts per trigger (default: all)")
    parser.add_argument("--engine", type=str, default="edge-tts",
                        choices=["edge-tts", "piper", "placeholder"],
                        help="TTS engine to use (default: edge-tts)")
    parser.add_argument("--piper-model", type=str, default=None, help="Piper TTS model path (.onnx)")
    parser.add_argument("--placeholder", action="store_true", help="Shortcut for --engine placeholder")
    parser.add_argument("--output-dir", type=str, default="assets/audio", help="Output base directory")
    args = parser.parse_args()

    if args.placeholder:
        args.engine = "placeholder"

    output_base = Path(args.output_dir)
    total_generated = 0
    total_skipped = 0
    total_failed = 0

    packs = {args.pack: ROAST_TEMPLATES[args.pack]} if args.pack else ROAST_TEMPLATES

    for pack_name, triggers in packs.items():
        voice_config = EDGE_VOICES.get(pack_name, EDGE_VOICES["british_instructor"])
        logger.info(f"\n{'='*60}")
        logger.info(f"Voice pack: {pack_name} (voice: {voice_config['voice']})")
        logger.info(f"{'='*60}")

        for trigger_name, roasts in triggers.items():
            trigger_dir = output_base / pack_name / trigger_name
            trigger_dir.mkdir(parents=True, exist_ok=True)

            roast_list = roasts[:args.count] if args.count else roasts
            logger.info(f"\n  [{trigger_name}] — {len(roast_list)} roasts")

            for i, text in enumerate(roast_list, 1):
                text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                filename = f"{i:03d}_{text_hash}.wav"
                output_path = trigger_dir / filename

                if output_path.exists():
                    total_skipped += 1
                    continue

                if args.engine == "placeholder":
                    ok = generate_placeholder(text, output_path)
                elif args.engine == "piper":
                    if not args.piper_model:
                        logger.error("--piper-model required for piper engine")
                        sys.exit(1)
                    ok = generate_with_piper(text, output_path, args.piper_model)
                else:  # edge-tts (default)
                    ok = generate_with_edge_tts(text, output_path, voice_config)

                if ok:
                    total_generated += 1
                    logger.info(f"    [{i}/{len(roast_list)}] {text[:60]}...")
                else:
                    total_failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Generated: {total_generated}, Skipped: {total_skipped}, Failed: {total_failed}")
    logger.info(f"Audio files in: {output_base.resolve()}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
