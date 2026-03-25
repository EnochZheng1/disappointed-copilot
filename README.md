# Disappointed Co-Pilot 😤🚗

An AI dashcam that watches the road and delivers sarcastic commentary about bad drivers — yours included.

Built for Raspberry Pi 5 + Google Coral, but runs on any desktop with a webcam or video file.

https://github.com/user-attachments/assets/placeholder

## What It Does

- **Sees the road** — YOLOv8 object detection tracks cars, trucks, pedestrians, and traffic lights in real-time
- **Detects bad driving** — 5 rule-based triggers fire on tailgating, swerving, green light lingering, lane departure, and hard braking
- **Roasts accordingly** — Instant pre-baked audio commentary from one of 3 voice personas, with optional LLM-generated dynamic roasts via Ollama
- **Clips for content** — Auto-saves 45-second MP4 clips with bounding box overlays when triggers fire, ready for TikTok/Reels

## Voice Personas

| Persona | Style |
|---------|-------|
| **Nigel** (British Instructor) | *"Oh splendid. This person seems to believe our bumper is a magnet."* |
| **Karen** (Tired Soccer Mom) | *"Are you kidding me right now? Back off. I have kids in here, buddy."* |
| **DAVE-9000** (Deadpan AI) | *"Following distance: inadvisable. Human error probability: certain."* |

84 voiced audio files across all personas and trigger types, generated with edge-tts.

## Quick Start

```bash
# Clone and install
git clone https://github.com/EnochZheng1/disappointed-copilot.git
cd disappointed-copilot
pip install -e ".[dev]"

# Generate voice audio (requires internet, ~2 min)
python scripts/generate_prebaked_audio.py

# Run with webcam
python -m disappointed

# Run with a video file
python -m disappointed --source path/to/dashcam.mp4

# Switch voice persona
python -m disappointed --voice-pack deadpan_ai

# Demo mode — fires random triggers every 8s to test audio
python -m disappointed --demo 8

# Tuning mode — shows how close each trigger is to firing
python -m disappointed --source dashcam.mp4 --tune
```

## Triggers

| Trigger | What It Detects | How |
|---------|----------------|-----|
| **Tailgater** | Vehicle closing in fast behind you | Bounding box area growth rate tracking |
| **Swerver** | Another car cutting into your lane | Bbox center crossing detected lane lines |
| **Green Light Lingerer** | Sitting at a green light not moving | Traffic light color analysis + sparse optical flow |
| **Self-Critique** | Your own lane departure | Lane center offset from OpenCV lane detection |
| **Hard Brake** | Sudden deceleration | All tracked vehicles growing simultaneously |

All triggers have configurable thresholds, cooldowns, and auto-disable when detection confidence is low (e.g. lane detection in rain).

## Architecture

```
Camera → Detection (YOLOv8) → Lane Detection (OpenCV) → Trigger Evaluation
                                                              │
                                          ┌───────────────────┼──────────────┐
                                          ▼                   ▼              ▼
                                    Pre-baked Audio     LLM Roast      Clip Extraction
                                    (<100ms latency)    (Ollama+TTS)   (45s MP4+overlay)
                                          │                   │
                                          ▼                   ▼
                                       Audio Priority Queue → Speaker
```

- **Main thread**: capture → detect → lanes → triggers → dispatch
- **Audio worker**: dequeues and plays clips with cooldown
- **Commentary worker**: async LLM generation via Ollama
- **Clip worker**: waits for post-trigger window, extracts from ring buffer, encodes MP4

## Configuration

Everything is YAML-driven. Swap between desktop and Pi with zero code changes:

```bash
# Desktop development (default)
python -m disappointed --config config/default.yaml config/desktop_dev.yaml

# Raspberry Pi deployment
python -m disappointed --config config/default.yaml config/pi_deploy.yaml
```

Key config options:
- `camera.backend`: `webcam`, `file`, or `picamera2`
- `detector.backend`: `ultralytics` (desktop) or `coral` (Pi + EdgeTPU)
- `commentary.voice_pack`: `british_instructor`, `tired_mom`, or `deadpan_ai`
- `triggers.*`: per-trigger enable/disable, thresholds, cooldowns
- `recording.*`: clip duration, buffer size, output directory

## Hardware (Raspberry Pi Build)

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 5 (8GB) | Main compute |
| Google Coral USB Accelerator | AI coprocessor for 30+ FPS inference |
| Pi Camera Module 3 (Wide) | Dashcam eye |
| Pi 5 Active Cooler | **Mandatory** — prevents thermal throttling |
| 5V/5A car power supply | Hardwired, stable power |
| USB speaker | Audio output |

```bash
# One-line Pi setup
sudo bash deploy/install_pi.sh
```

## Project Structure

```
src/disappointed/
├── camera/          # Webcam, video file, Picamera2 sources
├── detection/       # YOLOv8 (Ultralytics + Coral EdgeTPU) + centroid tracker
├── lane/            # OpenCV lane detection with EMA smoothing
├── triggers/        # 5 trigger types with cooldown + diagnostics
├── commentary/      # Pre-baked audio, Ollama LLM, Piper TTS, 3 personas
├── audio/           # Pygame playback + priority queue with cooldown
├── recording/       # Ring buffer + clip extraction + overlay renderer
├── pipeline/        # Main loop coordinator + worker threads
├── config/          # Pydantic schema + YAML loader
└── utils/           # FPS counter, math helpers, thermal monitor
```

## Scripts

```bash
python scripts/generate_prebaked_audio.py          # Generate all voice audio
python scripts/generate_prebaked_audio.py --pack tired_mom  # One persona only
python scripts/calibrate_roi.py --source video.mp4 # Set lane detection ROI
python scripts/benchmark.py --source video.mp4     # Profile FPS + latency
python scripts/export_model.py                     # Export YOLOv8n for Coral EdgeTPU
```

## Known Bottlenecks

| Issue | Mitigation |
|-------|-----------|
| **Thermal throttling** on Pi dashboard | Active cooler mandatory; thermal monitor auto-reduces detection frequency at 75°C |
| **Dense optical flow** too expensive | Uses sparse Lucas-Kanade (~20 points) or bbox-based detection instead |
| **LLM latency** (6-10s on Pi) | Pre-baked audio is the core UX (<100ms); LLM is a bonus follow-up |
| **Lane detection** fails in rain/glare | EMA smoothing + confidence gating auto-disables lane triggers when unreliable |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full product roadmap, from MVP to commercial product.

**Next up:** Social media export (9:16 vertical crop, styled subtitles), more roast lines, vehicle make/model detection ("Of course it's a BMW").

## License

MIT
