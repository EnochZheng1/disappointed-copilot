"""Clip extractor — extracts, overlays, and encodes trigger event clips to MP4."""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from disappointed.config.schema import RecordingConfig
from disappointed.triggers.models import TriggerEvent
from .buffer import RingBuffer
from .overlay import OverlayRenderer

logger = logging.getLogger(__name__)


class ClipRequest:
    """A request to extract a clip from the ring buffer."""

    def __init__(self, event: TriggerEvent, pre_seconds: int, post_seconds: int):
        self.event = event
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.requested_at = time.time()


class ClipExtractor:
    """Extracts clip segments from the ring buffer and encodes them as MP4 with overlays."""

    def __init__(self, config: RecordingConfig, dedup_window: float = 10.0):
        self._config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._overlay = OverlayRenderer()
        self._dedup_window = dedup_window
        self._last_clip_time: float = 0.0

    def should_extract(self, request: ClipRequest) -> bool:
        """Check if this clip request should be processed (dedup)."""
        if (request.requested_at - self._last_clip_time) < self._dedup_window:
            logger.info(
                f"Clip dedup: skipping {request.event.trigger_name} "
                f"({request.requested_at - self._last_clip_time:.1f}s since last clip)"
            )
            return False
        return True

    def extract_clip(self, ring_buffer: RingBuffer, request: ClipRequest) -> Path | None:
        """Extract frames from the ring buffer and encode to an MP4 clip.

        Should be called AFTER the post-trigger window has elapsed.
        """
        if not self.should_extract(request):
            return None

        total_seconds = request.pre_seconds + request.post_seconds
        frames, timestamps = ring_buffer.read_seconds(total_seconds)

        if len(frames) == 0:
            logger.warning("No frames in ring buffer for clip extraction")
            return None

        # Generate output filename
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(request.event.timestamp))
        filename = f"{request.event.trigger_name}_{ts_str}.mp4"
        output_path = self._output_dir / filename

        # Determine which frame index is the trigger point
        trigger_frame_idx = max(0, len(frames) - int(request.post_seconds * ring_buffer.fps))

        # Encode
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self._config.output_fps, (w, h))

        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {output_path}")
            return None

        for i, frame in enumerate(frames):
            frame = frame.copy()

            # Draw trigger marker on the trigger frame and a few after it
            if trigger_frame_idx <= i <= trigger_frame_idx + int(ring_buffer.fps * 3):
                frame = self._overlay.draw_trigger_marker(frame, request.event)

            # Caption on all frames
            frame = self._overlay.draw_caption(frame, request.event.description)

            # Watermark
            frame = self._overlay.draw_watermark(frame)

            writer.write(frame)

        writer.release()
        self._last_clip_time = time.time()

        duration = len(frames) / ring_buffer.fps
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Clip saved: {output_path} ({duration:.1f}s, {file_size_mb:.1f}MB, "
            f"trigger: {request.event.trigger_name})"
        )
        return output_path
