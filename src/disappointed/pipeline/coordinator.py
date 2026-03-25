"""Pipeline coordinator — main loop, worker threads, and display rendering."""

import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from disappointed.audio.player import AudioPlayer
from disappointed.audio.queue import AudioPriorityQueue, AudioItem
from disappointed.camera.base import CameraSource
from disappointed.commentary.base import CommentaryEngine
from disappointed.config.schema import AppConfig
from disappointed.detection.base import ObjectDetector
from disappointed.detection.models import BoundingBox
from disappointed.detection.tracker import CentroidTracker
from disappointed.lane.detector import LaneDetector
from disappointed.lane.models import LaneState
from disappointed.pipeline.frame_data import FrameData
from disappointed.recording.buffer import RingBuffer
from disappointed.recording.clip_extractor import ClipExtractor, ClipRequest
from disappointed.triggers.models import TriggerEvent
from disappointed.triggers.registry import TriggerRegistry
from disappointed.utils.fps import FPSCounter

logger = logging.getLogger(__name__)

# Color palette for tracked objects (BGR)
COLORS = [
    (255, 100, 100),  # blue
    (100, 255, 100),  # green
    (100, 100, 255),  # red
    (255, 255, 100),  # cyan
    (255, 100, 255),  # magenta
    (100, 255, 255),  # yellow
    (200, 200, 200),  # gray
    (255, 180, 100),  # light blue
]


def _get_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return COLORS[track_id % len(COLORS)]


def _draw_detections(frame: np.ndarray, detections: list[BoundingBox]) -> None:
    """Draw bounding boxes, labels, and track IDs onto the frame."""
    for det in detections:
        color = _get_color(det.track_id)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        track_str = f" #{det.track_id}" if det.track_id is not None else ""
        label = f"{det.class_name}{track_str} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 6),
            (x1 + label_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            frame, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )


def _draw_lane_state(frame: np.ndarray, lane_state: LaneState) -> None:
    """Draw detected lane lines and offset indicator on the frame."""
    h, w = frame.shape[:2]

    # Draw lane lines
    for lane_line in [lane_state.left_line, lane_state.right_line]:
        if lane_line is None:
            continue
        color = (0, 255, 0) if not lane_state.departure_detected else (0, 0, 255)
        pts = [(int(x), int(y)) for x, y in lane_line.points]
        if len(pts) >= 2:
            cv2.line(frame, pts[0], pts[1], color, 3)

    # Draw lane center offset indicator
    if lane_state.left_line and lane_state.right_line:
        left_x = lane_state.left_line.points[0][0]
        right_x = lane_state.right_line.points[0][0]
        lane_center = int((left_x + right_x) / 2)
        frame_center = w // 2

        # Draw offset arrow at bottom of frame
        y_indicator = h - 40
        cv2.arrowedLine(
            frame, (frame_center, y_indicator), (lane_center, y_indicator),
            (0, 255, 255), 2, tipLength=0.3,
        )

    # Offset + confidence text
    offset_text = f"Offset: {lane_state.own_offset_from_center:+.3f} | Lane conf: {lane_state.confidence:.2f}"
    if lane_state.departure_detected:
        offset_text += f" | DEPARTURE {lane_state.departure_side}"
    cv2.putText(
        frame, offset_text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )


def _draw_trigger_status(frame: np.ndarray, events: list[TriggerEvent]) -> None:
    """Draw active trigger events on the frame."""
    y = 60
    for event in events:
        severity_color = (
            (0, 255, 0) if event.severity < 0.3
            else (0, 255, 255) if event.severity < 0.7
            else (0, 0, 255)
        )
        text = f"TRIGGER: {event.trigger_name} ({event.severity:.2f}) - {event.description}"
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, severity_color, 2, cv2.LINE_AA,
        )
        y += 25


def _draw_fps(frame: np.ndarray, fps: float, inference_ms: float) -> None:
    """Draw FPS and inference time on the top-left corner."""
    text = f"FPS: {fps:.1f} | Inference: {inference_ms:.1f}ms"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )


class PipelineCoordinator:
    """Orchestrates the full pipeline: capture -> detect -> lanes -> triggers -> commentary -> audio."""

    def __init__(
        self,
        config: AppConfig,
        camera: CameraSource,
        detector: ObjectDetector | None = None,
        lane_detector: LaneDetector | None = None,
        trigger_registry: TriggerRegistry | None = None,
        prebaked_engine: CommentaryEngine | None = None,
        llm_engine: CommentaryEngine | None = None,
        demo_interval: float | None = None,
    ):
        self.config = config
        self.camera = camera
        self.detector = detector
        self.lane_detector = lane_detector
        self.trigger_registry = trigger_registry
        self.prebaked_engine = prebaked_engine
        self.llm_engine = llm_engine
        self._demo_interval = demo_interval

        self.tracker = CentroidTracker() if detector else None
        self._fps_counter = FPSCounter()
        self._frame_index = 0
        self._running = threading.Event()
        self._last_demo_trigger: float = 0.0

        # Last known detection/lane results (for skip-frame interpolation)
        self._last_detections: list[BoundingBox] = []
        self._last_lane_state: Optional[LaneState] = None

        # Audio system
        self._audio_queue = AudioPriorityQueue(
            cooldown_seconds=config.audio.cooldown_seconds
        ) if config.audio.enabled else None
        self._commentary_queue: queue.Queue[TriggerEvent] = queue.Queue()

        # Recording system
        self._ring_buffer: Optional[RingBuffer] = None
        self._clip_extractor: Optional[ClipExtractor] = None
        self._clip_queue: queue.Queue[ClipRequest] = queue.Queue()
        if config.recording.enabled:
            buf_w, buf_h = config.recording.buffer_resolution
            self._ring_buffer = RingBuffer(
                max_seconds=config.recording.hot_buffer_seconds,
                fps=config.camera.fps,
                height=buf_h,
                width=buf_w,
            )
            self._clip_extractor = ClipExtractor(config.recording)

    def run(self) -> None:
        """Enter the main processing loop."""
        self._running.set()
        logger.info("Pipeline starting...")

        with self.camera:
            if self.detector:
                self.detector.load_model()

            # Start worker threads
            workers = []
            if self._audio_queue:
                t = threading.Thread(target=self._audio_worker, daemon=True, name="audio_worker")
                t.start()
                workers.append(t)

            if self.llm_engine and self.config.commentary.llm_enabled:
                t = threading.Thread(target=self._commentary_worker, daemon=True, name="commentary_worker")
                t.start()
                workers.append(t)

            if self._clip_extractor and self._ring_buffer:
                t = threading.Thread(target=self._clip_worker, daemon=True, name="clip_worker")
                t.start()
                workers.append(t)

            logger.info("Pipeline running. Press 'q' to quit.")

            while self._running.is_set():
                raw_frame = self.camera.read()
                if raw_frame is None:
                    logger.info("No more frames. Stopping.")
                    break

                frame_data = FrameData(
                    frame=raw_frame,
                    frame_index=self._frame_index,
                    timestamp=time.time(),
                )

                # --- Ring buffer ---
                if self._ring_buffer:
                    self._ring_buffer.push(raw_frame, frame_data.timestamp)

                # --- Object detection ---
                inference_ms = 0.0
                if self.detector and (self._frame_index % self.config.detector.detect_every_n_frames == 0):
                    frame_data.detections = self.detector.detect(raw_frame)
                    inference_ms = self.detector.get_inference_time_ms()
                    if self.tracker:
                        frame_data.detections = self.tracker.update(frame_data.detections)
                    self._last_detections = frame_data.detections
                else:
                    frame_data.detections = self._last_detections

                # --- Lane detection ---
                if self.lane_detector and self.config.lane.enabled:
                    if self._frame_index % self.config.lane.lane_every_n_frames == 0:
                        frame_data.lane_state = self.lane_detector.detect(raw_frame)
                        self._last_lane_state = frame_data.lane_state
                    else:
                        frame_data.lane_state = self._last_lane_state

                # --- Trigger evaluation ---
                if self.trigger_registry:
                    frame_data.trigger_events = self.trigger_registry.evaluate(frame_data)
                    for event in frame_data.trigger_events:
                        self._dispatch_event(event)

                # --- Demo mode: fire random triggers on a timer ---
                if self._demo_interval:
                    now = time.time()
                    if now - self._last_demo_trigger >= self._demo_interval:
                        self._last_demo_trigger = now
                        demo_event = self._make_demo_trigger(frame_data)
                        frame_data.trigger_events.append(demo_event)
                        self._dispatch_event(demo_event)

                # --- FPS ---
                frame_data.fps = self._fps_counter.tick()

                # --- Debug display ---
                if self.config.debug_display:
                    display_frame = raw_frame.copy()
                    if frame_data.detections:
                        _draw_detections(display_frame, frame_data.detections)
                    if frame_data.lane_state:
                        _draw_lane_state(display_frame, frame_data.lane_state)
                    if frame_data.trigger_events:
                        _draw_trigger_status(display_frame, frame_data.trigger_events)
                    _draw_fps(display_frame, frame_data.fps, inference_ms)

                    cv2.imshow("Disappointed Co-Pilot", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested by user.")
                        break

                self._frame_index += 1

        self._running.clear()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")

    def _dispatch_event(self, event: TriggerEvent) -> None:
        """Fan out a trigger event to audio and commentary systems."""
        logger.info(f"TRIGGER: {event.trigger_name} (severity={event.severity:.2f}) - {event.description}")

        # Pre-baked audio — instant response
        if self.prebaked_engine and self._audio_queue:
            audio_path = self.prebaked_engine.get_audio(event)
            if audio_path:
                self._audio_queue.enqueue(
                    AudioItem(path=audio_path, trigger_name=event.trigger_name),
                    priority=event.severity,
                )

        # LLM commentary — queued for async generation
        if self.llm_engine and self.config.commentary.llm_enabled:
            self._commentary_queue.put(event)

        # Clip extraction — queued for background processing
        if self._clip_extractor and self._ring_buffer:
            self._clip_queue.put(ClipRequest(
                event=event,
                pre_seconds=self.config.recording.pre_trigger_seconds,
                post_seconds=self.config.recording.post_trigger_seconds,
            ))

    def _audio_worker(self) -> None:
        """Dequeue and play audio clips, respecting cooldown."""
        player = AudioPlayer(volume=self.config.audio.volume)
        while self._running.is_set():
            item = self._audio_queue.dequeue(timeout=0.5)
            if item:
                player.play_blocking(item.path)

    def _commentary_worker(self) -> None:
        """Generate dynamic roasts via LLM and queue the resulting audio."""
        while self._running.is_set():
            try:
                event = self._commentary_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                audio_path = self.llm_engine.get_audio(event)
                if audio_path and self._audio_queue:
                    self._audio_queue.enqueue(
                        AudioItem(path=audio_path, trigger_name=event.trigger_name),
                        priority=event.severity * 0.8,  # Slightly lower than pre-baked
                    )
            except Exception:
                logger.exception("LLM commentary generation failed")

    def _clip_worker(self) -> None:
        """Wait for post-trigger window, then extract clip from ring buffer."""
        while self._running.is_set():
            try:
                request = self._clip_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            # Wait for the post-trigger footage to accumulate
            wait_time = request.post_seconds
            logger.info(f"Clip worker: waiting {wait_time}s for post-trigger footage...")
            deadline = time.time() + wait_time
            while time.time() < deadline and self._running.is_set():
                time.sleep(0.5)
            # Extract and encode the clip
            try:
                self._clip_extractor.extract_clip(self._ring_buffer, request)
            except Exception:
                logger.exception("Clip extraction failed")

    @staticmethod
    def _make_demo_trigger(frame_data: FrameData) -> TriggerEvent:
        """Generate a random trigger event for demo/testing purposes."""
        import random
        trigger_names = ["tailgater", "swerver", "green_light", "self_critique", "hard_brake"]
        descriptions = {
            "tailgater": "DEMO: A car is riding our bumper",
            "swerver": "DEMO: Someone just cut into our lane",
            "green_light": "DEMO: The light is green and we're not moving",
            "self_critique": "DEMO: We're drifting out of our lane",
            "hard_brake": "DEMO: We just slammed the brakes",
        }
        name = random.choice(trigger_names)
        return TriggerEvent(
            trigger_name=name,
            severity=random.uniform(0.4, 0.9),
            timestamp=frame_data.timestamp,
            frame_index=frame_data.frame_index,
            description=descriptions[name],
        )

    def stop(self) -> None:
        self._running.clear()
