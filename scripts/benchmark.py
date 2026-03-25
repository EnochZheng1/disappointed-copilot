#!/usr/bin/env python3
"""Benchmark pipeline FPS and per-stage latencies.

Usage:
    python scripts/benchmark.py                              # Webcam benchmark
    python scripts/benchmark.py --source test_video/clip.mp4 # Video file
    python scripts/benchmark.py --frames 300                 # Run for 300 frames
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline performance")
    parser.add_argument("--source", type=str, default=None, help="Video file path (default: webcam)")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to benchmark")
    parser.add_argument("--config", type=str, nargs="+",
                        default=["config/default.yaml", "config/desktop_dev.yaml"])
    args = parser.parse_args()

    from disappointed.config.loader import load_config
    from disappointed.config.schema import CameraBackend
    from disappointed.app import _create_camera, _create_detector, _create_lane_detector

    config = load_config(*[Path(p) for p in args.config])
    if args.source:
        config.camera.backend = CameraBackend.FILE
        config.camera.file_path = args.source
    config.debug_display = False  # No display during benchmark

    camera = _create_camera(config)
    detector = _create_detector(config)
    lane_detector = _create_lane_detector(config)

    camera.open()
    if detector:
        detector.load_model()

    # Warmup
    logger.info("Warming up (10 frames)...")
    for _ in range(10):
        frame = camera.read()
        if frame is not None and detector:
            detector.detect(frame)

    # Benchmark
    logger.info(f"Benchmarking {args.frames} frames...")
    timings = {"capture": [], "detect": [], "lane": [], "total": []}

    for i in range(args.frames):
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        frame = camera.read()
        timings["capture"].append((time.perf_counter() - t0) * 1000)

        if frame is None:
            break

        if detector:
            t0 = time.perf_counter()
            detector.detect(frame)
            timings["detect"].append((time.perf_counter() - t0) * 1000)

        if lane_detector:
            t0 = time.perf_counter()
            lane_detector.detect(frame)
            timings["lane"].append((time.perf_counter() - t0) * 1000)

        timings["total"].append((time.perf_counter() - t_total_start) * 1000)

    camera.close()

    # Report
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for stage, times in timings.items():
        if not times:
            continue
        arr = np.array(times)
        print(f"\n{stage.upper():>10s}:")
        print(f"  Mean:   {arr.mean():.1f}ms")
        print(f"  Median: {np.median(arr):.1f}ms")
        print(f"  P95:    {np.percentile(arr, 95):.1f}ms")
        print(f"  P99:    {np.percentile(arr, 99):.1f}ms")
        print(f"  Min:    {arr.min():.1f}ms")
        print(f"  Max:    {arr.max():.1f}ms")

    if timings["total"]:
        total = np.array(timings["total"])
        effective_fps = 1000.0 / total.mean()
        print(f"\nEffective FPS: {effective_fps:.1f}")
        print(f"Frames benchmarked: {len(timings['total'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
