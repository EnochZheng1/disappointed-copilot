# Disappointed Co-Pilot — Product Roadmap

## Current State (v0.1.0)

**38 source modules | 34 passing tests | 84 voice-acted audio files | 3 personas**

The core pipeline is functional end-to-end on desktop:
- YOLOv8n object detection with centroid tracking
- Lane detection (Canny + HoughLinesP + EMA smoothing)
- 5 trigger types (tailgater, swerver, green light, self-critique, hard brake)
- Pre-baked audio commentary with 3 voice packs (edge-tts generated)
- LLM dynamic roasts via Ollama (async follow-up)
- Ring buffer + clip extractor (45s MP4 with overlays)
- Raspberry Pi 5 + Coral EdgeTPU deployment pipeline
- Demo mode for testing (`--demo N`)

**Not yet validated in real driving conditions.** Current testing is desktop-only with POV footage.

---

## Milestone 1: Road-Ready MVP (v0.2.0)

**Goal**: A system you can mount in a car and drive with. Triggers fire accurately on real roads, audio timing feels natural, and clips are auto-saved.

### 1.1 Trigger Calibration
- [ ] Collect 30+ minutes of real dashcam footage across conditions (highway, city, night, rain)
- [ ] Tune tailgater thresholds — current `bbox_growth_rate_threshold: 0.15` needs validation against actual approaching vehicles at various speeds
- [ ] Tune swerver `lane_cross_ratio` and `consecutive_frames` to avoid false positives from road curvature
- [ ] Calibrate green light HSV color ranges across different traffic light styles (LED vs incandescent, sun glare)
- [ ] Validate hard brake trigger — may need to lower `min_tracked_vehicles` to 1 with a higher growth rate threshold as a fallback
- [ ] Add a **trigger tuning mode** that logs all near-miss events (events that came close to threshold but didn't fire) so you can review and adjust
- [ ] Per-trigger false positive / false negative tracking with CSV export

### 1.2 Lane Detection Hardening
- [ ] Implement `scripts/calibrate_roi.py` — click 4 points on a live frame to define the lane ROI for your specific camera mount angle
- [ ] Test lane detection across: fresh paint, faded lines, dashed lines, night, rain, direct sun glare
- [ ] Add alternative lane detection using color filtering (white/yellow line isolation in HSV) as a fallback when edge detection fails
- [ ] Consider lightweight lane detection model (Ultra-Fast-Lane-Detection-V2) as a more robust alternative to classical CV — benchmark FPS vs accuracy tradeoff

### 1.3 Audio Timing Polish
- [ ] Reduce global audio cooldown from 10s to 5s for more active commentary
- [ ] Add trigger-specific cooldown overrides (e.g., self-critique every 20s, tailgater every 10s)
- [ ] Add "sigh" and "hmm" transition sounds between triggers for more natural flow
- [ ] Add volume ducking — lower audio volume when triggers fire in rapid succession to avoid overwhelming the driver
- [ ] Test audio latency end-to-end: trigger detection → audio queue → mixer playback. Target: <200ms for pre-baked

### 1.4 Clip Recording Validation
- [ ] End-to-end test: drive with recording enabled, verify 45s MP4 clips are saved on trigger events
- [ ] Verify overlay rendering at 640×360 looks acceptable when upscaled on a phone screen
- [ ] Add clip deduplication — if two triggers fire within 10s, merge into one longer clip instead of two overlapping clips
- [ ] Add clip metadata sidecar (JSON) with trigger info, timestamps, GPS coordinates (if available)

### 1.5 First Git Commit + CI
- [ ] Initial commit of all current code
- [ ] Add GitHub Actions CI: lint (ruff), type check (mypy), pytest on push
- [ ] Add pre-commit hooks (ruff format, ruff check)

---

## Milestone 2: Content Creator Pipeline (v0.3.0)

**Goal**: The system produces scroll-stopping, ready-to-upload clips for TikTok/Reels/YouTube Shorts without manual editing.

### 2.1 Social Media Export
- [ ] **Vertical crop (9:16)** — auto-crop the 16:9 dashcam frame to a phone-friendly 9:16 centered on the action (follow the trigger's bounding box)
- [ ] **Styled text overlays** — burn the roast text as large, readable subtitles (white text, black outline, bottom third) like TikTok captions
- [ ] **Trigger type badge** — visual indicator in corner (e.g., red "TAILGATER" badge with icon)
- [ ] **Intro/outro frames** — 1-second title card at the start ("Disappointed Co-Pilot") and end card with social handles
- [ ] **Auto-subtitle generation** — transcribe the audio commentary and burn as synced subtitles (using whisper.cpp or similar)
- [ ] **Export presets** — `tiktok` (9:16, 1080x1920, 60s max), `youtube_shorts` (9:16, 1080x1920), `instagram_reel` (9:16), `youtube` (16:9, 1920x1080)

### 2.2 Clip Compilation
- [ ] **Daily highlight reel** — at end of each drive, auto-compile the top 5 trigger clips (by severity) into a single 60-second compilation with transitions
- [ ] **"Best of" export** — script to scan all saved clips, rank by severity/entertainment value, and compile a highlight reel
- [ ] **Clip rating system** — after each trigger, briefly display a "rate this roast" prompt on screen (optional, for training data)

### 2.3 More Roast Content
- [ ] Expand from 5 to **25-30 roasts per trigger per persona** to avoid repetition
- [ ] Add **situational context** to roasts — "The BMW cut us off" vs "A truck cut us off" (use class_name in the roast template)
- [ ] Add **severity-scaled responses** — mild annoyance (low severity) vs full outrage (high severity) with different audio pools per level
- [ ] Add **combo reactions** — if two triggers fire within 30s, play a special "Are you KIDDING me?! Again?!" escalation audio
- [ ] Add **time-of-day awareness** — morning commute roasts vs late-night roasts (different energy levels)

### 2.4 New Voice Packs
- [ ] **Backseat Grandma** — loving but terrified, gasps and clutches pearls
- [ ] **Sports Commentator** — narrates driving incidents like a boxing match
- [ ] **Drill Sergeant** — aggressive, barking orders at other drivers
- [ ] **Zen Monk** — calm, philosophical disappointment ("All things are temporary... including that driver's license")
- [ ] **Community voice pack support** — documented format + submission pipeline for user-created packs
- [ ] Integrate higher-quality TTS (ElevenLabs API for premium packs, keep edge-tts for free tier)

---

## Milestone 3: Intelligence Upgrade (v0.4.0)

**Goal**: Smarter detection, fewer false positives, context-aware commentary.

### 3.1 Advanced Detection
- [ ] **Vehicle make/model classification** — fine-tune a classifier on the YOLO crop to identify BMW, Tesla, pickup truck, etc. This unlocks "Of course it's a BMW" roasts
- [ ] **Turn signal detection** — detect whether the swerving car actually used a blinker (bounding box + flashing light analysis)
- [ ] **Speed estimation** — use optical flow + camera calibration to estimate relative speeds. "That car is doing at least 90 in a 60 zone"
- [ ] **Multi-lane tracking** — detect which lane each vehicle is in, track lane changes over time, detect undertaking
- [ ] **Pedestrian awareness** — detect pedestrians near crosswalks, trigger "watch out for the pedestrian" safety alerts (non-sarcastic)

### 3.2 Context-Aware Commentary
- [ ] **Scene understanding** — classify the driving context (highway, city, parking lot, residential) and adjust trigger sensitivity + roast style accordingly
- [ ] **Weather detection** — rain/fog/night detection via frame analysis. Adjust lane confidence thresholds and add weather-specific roasts ("Driving in the rain without headlights. Bold strategy.")
- [ ] **Repeated offender tracking** — if the same tracked vehicle triggers multiple events, escalate the roasts ("Oh look, it's our friend from earlier. Still can't drive, I see.")
- [ ] **Self-improvement mode** — track and log the driver's own trigger events over time, generate a "driving report card" with trends

### 3.3 LLM Enhancement
- [ ] **Vision LLM integration** — when a trigger fires, send the cropped trigger frame to a multimodal LLM (Llava, Phi-3-vision) for richer context ("I can see it's a white BMW X5 with a dented bumper")
- [ ] **Conversation memory** — LLM remembers previous roasts in the session and builds a narrative ("And for the THIRD time today...")
- [ ] **Roast quality scoring** — generate multiple roasts, score them locally, pick the funniest one before speaking
- [ ] Benchmark Llama 3.2 1B vs Phi-3-mini vs Gemma 2B on Pi 5 for roast quality and latency

---

## Milestone 4: Hardware & Deployment (v0.5.0)

**Goal**: Production-grade hardware build that runs reliably in a car 24/7.

### 4.1 Raspberry Pi Optimization
- [ ] **Profile and optimize** — run `scripts/benchmark.py` on Pi 5 + Coral, identify the bottleneck stage, optimize the hot path
- [ ] **Thermal stress test** — run the full pipeline in a parked car under direct sunlight for 2 hours. Log CPU temp, throttle frequency, trigger accuracy degradation
- [ ] **Integrate ThermalMonitor** into the pipeline coordinator — auto-reduce `detect_every_n_frames` when hot, pause LLM at 80°C
- [ ] **Memory optimization** — profile memory usage under sustained operation, tune ring buffer size to stay under 5GB total
- [ ] **Startup time** — optimize cold start to <10 seconds (pre-load model, defer non-critical init)
- [ ] **Watchdog** — add a hardware watchdog timer to auto-reboot if the process hangs

### 4.2 Hardware Build Guide
- [ ] **Parts list with links** — Pi 5 8GB, Coral USB, Pi Camera Module 3 Wide, Active Cooler, specific case, specific power supply, USB speaker
- [ ] **3D-printable mount** — design a dash-mount enclosure that holds Pi + camera + speaker, with ventilation slots. Publish STL files
- [ ] **Wiring diagram** — 12V car → 5V/5A buck converter → Pi, speaker connection, camera ribbon cable routing
- [ ] **Assembly guide with photos** — step-by-step build instructions
- [ ] **Safety notes** — don't obstruct airbags, secure mounting to prevent projectiles in a crash, don't let it block the driver's view

### 4.3 Boot & Auto-Start
- [ ] **Read-only filesystem** — configure Pi to boot from a read-only root partition to prevent SD card corruption from sudden power loss (car ignition off)
- [ ] **Auto-start on boot** — systemd service (already implemented), but add graceful handling of "no camera connected" and "no Coral connected"
- [ ] **Power management** — detect ignition off (voltage drop on 5V rail) and trigger graceful shutdown with clip save
- [ ] **USB drive auto-mount** — detect USB drive insertion, export clips to it, eject safely

### 4.4 Alternative Hardware
- [ ] **Android phone build** — investigate running the pipeline on an old Android phone via Termux + Python. Camera access via OpenCV, audio via built-in speaker. Evaluate thermal limits.
- [ ] **Jetson Nano / Orin Nano** — for users who want higher FPS and can run larger models. Add a config profile.
- [ ] **Orange Pi 5** — cheaper Pi alternative with NPU. Evaluate RKNN support.

---

## Milestone 5: Connected Features (v1.0.0)

**Goal**: Cloud sync, community features, and a companion app.

### 5.1 Companion Mobile App
- [ ] **Live view** — stream the debug display to a phone via local WiFi (WebSocket + MJPEG)
- [ ] **Clip review** — browse saved clips on the phone, delete or star favorites
- [ ] **Settings UI** — adjust trigger sensitivity, switch voice packs, toggle recording from the phone
- [ ] **Push notifications** — "Your dashcam captured 3 clips today" with thumbnails

### 5.2 Cloud Sync
- [ ] **Auto-upload starred clips** — sync to Google Drive / Dropbox / S3 over WiFi when parked
- [ ] **Remote access** — view the dashcam live from anywhere (Tailscale / Cloudflare Tunnel)
- [ ] **OTA updates** — push new voice packs, trigger models, and software updates to the Pi remotely

### 5.3 Community Platform
- [ ] **Clip sharing** — upload anonymized clips to a community feed (blur license plates automatically)
- [ ] **Voice pack marketplace** — community-created voice packs with ratings and downloads
- [ ] **Leaderboard** — "Most disappointed co-pilot" stats (most triggers per mile, most clips, etc.)
- [ ] **Roast crowdsourcing** — community submits roast text, top-voted ones get synthesized into official voice packs

### 5.4 Data & Analytics
- [ ] **Driving analytics dashboard** — track trigger frequency over time, most common trigger types, time-of-day patterns
- [ ] **Heatmap** — GPS-tagged trigger locations on a map ("this intersection is where everyone runs red lights")
- [ ] **Insurance-ready export** — option to export non-sarcastic, timestamped incident logs for insurance claims (serious mode)

---

## Milestone 6: Commercial Product (v2.0.0)

**Goal**: A purchasable product with professional packaging and support.

### 6.1 Product SKUs
- [ ] **DIY Kit** — Pi 5 + Coral + camera + case + SD card with pre-installed software. User assembles.
- [ ] **Pre-Built Unit** — fully assembled, plug-and-play dashcam unit. Mount, plug in 12V, drive.
- [ ] **Software-Only** — downloadable image for users who already have a Pi 5 + Coral.

### 6.2 Monetization
- [ ] **Free tier** — 1 voice pack, basic triggers, 10 clips per day
- [ ] **Pro tier ($5/month)** — all voice packs, LLM dynamic roasts, unlimited clips, cloud sync, priority support
- [ ] **Voice pack store** — premium celebrity-voiced packs ($2.99 each) via ElevenLabs voice cloning (with licensing)
- [ ] **Affiliate partnerships** — partner with dashcam mount / car accessory brands

### 6.3 Legal & Compliance
- [ ] **Privacy audit** — ensure no personal data is stored or transmitted without consent. License plates in clips must be blurrable.
- [ ] **Distracted driving disclaimer** — prominent warning that audio commentary should not distract the driver. Volume auto-limits. Option to disable audio and only record clips.
- [ ] **Regional compliance** — dashcam recording laws vary by country/state. Add a config for recording consent requirements (some jurisdictions require notification).
- [ ] **Terms of service + privacy policy** for cloud features

---

## Technical Debt & Infrastructure (Ongoing)

### Testing
- [ ] Increase unit test coverage to 80%+ (currently ~35-40%)
- [ ] Add integration tests: full pipeline on recorded dashcam video with expected trigger events
- [ ] Add trigger regression tests: known scenarios that must always fire (or not fire)
- [ ] Performance regression tests: ensure FPS doesn't degrade across releases
- [ ] Load testing on Pi 5: 1-hour sustained operation without memory leaks or crashes

### Code Quality
- [ ] Add type hints to all public interfaces (mypy strict mode)
- [ ] Add ruff linting + formatting to CI
- [ ] Implement OpenCV DNN detector as a zero-dependency fallback (no ultralytics/pycoral needed)
- [ ] Refactor trigger history tracking — currently each trigger manages its own deque, could be centralized

### Documentation
- [ ] API documentation for all public classes (auto-generated from docstrings)
- [ ] Architecture decision records (ADRs) for key design choices
- [ ] Contributing guide for community voice pack creation
- [ ] Video tutorial: "Build Your Own Disappointed Co-Pilot in 30 Minutes"

### DevOps
- [ ] GitHub Actions CI: lint → type check → unit tests → integration tests
- [ ] Automated Pi image builds (Packer + custom Raspberry Pi OS)
- [ ] Release automation: tag → build → publish to GitHub Releases
- [ ] Nightly benchmarks on Pi hardware (if self-hosted runner available)

---

## Priority Matrix

| Milestone | Impact | Effort | Priority |
|-----------|--------|--------|----------|
| M1: Road-Ready MVP | Critical | Medium | **NOW** |
| M2: Content Pipeline | High | Medium | Next |
| M3: Intelligence | High | High | After M2 |
| M4: Hardware Build | Medium | Medium | Parallel with M2 |
| M5: Connected | Medium | High | After M3 |
| M6: Commercial | Variable | Very High | Long-term |

---

## Timeline (Rough Estimates)

| Milestone | Target |
|-----------|--------|
| v0.2.0 — Road-Ready MVP | 4-6 weeks |
| v0.3.0 — Content Creator Pipeline | 6-8 weeks after M1 |
| v0.4.0 — Intelligence Upgrade | 8-12 weeks after M2 |
| v0.5.0 — Hardware Build | Parallel with M2-M3 |
| v1.0.0 — Connected Features | 12-16 weeks after M4 |
| v2.0.0 — Commercial | 6-12 months after v1.0 |
