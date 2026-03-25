#!/usr/bin/env python3
"""Interactive ROI calibration for lane detection.

Click 4 points on a video frame to define the lane detection region of interest.
The points are saved as ratio-based coordinates to your config file.

Usage:
    python scripts/calibrate_roi.py                              # Use webcam
    python scripts/calibrate_roi.py --source test_video/clip.mp4 # Use video file
    python scripts/calibrate_roi.py --output config/desktop_dev.yaml

Controls:
    Left-click:  Place a point (4 points needed)
    Right-click: Undo last point
    R:           Reset all points
    S:           Save to config and exit
    Q:           Quit without saving
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

points: list[tuple[int, int]] = []
frame_display = None
frame_original = None


def mouse_callback(event, x, y, flags, param):
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw()


def redraw():
    global frame_display
    frame_display = frame_original.copy()
    h, w = frame_display.shape[:2]

    # Draw existing points
    for i, (px, py) in enumerate(points):
        color = (0, 255, 0) if i < 4 else (0, 0, 255)
        cv2.circle(frame_display, (px, py), 6, color, -1)
        cv2.putText(frame_display, str(i + 1), (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw polygon if we have enough points
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(frame_display, points[i], points[i + 1], (0, 255, 0), 2)
    if len(points) == 4:
        cv2.line(frame_display, points[3], points[0], (0, 255, 0), 2)
        # Fill polygon with transparency
        overlay = frame_display.copy()
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, frame_display, 0.8, 0, frame_display)

    # Instructions
    if len(points) < 4:
        text = f"Click point {len(points) + 1}/4 (bottom-left, top-left, top-right, bottom-right)"
    else:
        text = "Press S to save, R to reset, Q to quit"
    cv2.putText(frame_display, text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show ratios
    for i, (px, py) in enumerate(points):
        ratio_text = f"({px/w:.2f}, {py/h:.2f})"
        cv2.putText(frame_display, ratio_text, (px + 10, py + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv2.imshow("ROI Calibration", frame_display)


def save_to_config(config_path: Path, width: int, height: int):
    """Save the 4 ROI points as ratios to a YAML config file."""
    ratios = [[round(px / width, 3), round(py / height, 3)] for px, py in points]

    config_data = {}
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    if "lane" not in config_data:
        config_data["lane"] = {}
    config_data["lane"]["roi_vertices_ratio"] = ratios

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nROI saved to {config_path}:")
    print(f"  roi_vertices_ratio: {ratios}")


def main():
    global frame_original, frame_display

    parser = argparse.ArgumentParser(description="Calibrate lane detection ROI")
    parser.add_argument("--source", type=str, default=None, help="Video file (default: webcam)")
    parser.add_argument("--output", type=str, default="config/desktop_dev.yaml", help="Config file to save ROI")
    args = parser.parse_args()

    # Open video source
    if args.source:
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source")
        sys.exit(1)

    # Grab a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        sys.exit(1)

    # Skip ahead for video files to get a representative frame
    if args.source:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 4, 100))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

    cap.release()

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    print("Click 4 points: bottom-left → top-left → top-right → bottom-right")
    print("These define the trapezoidal region where lane lines are expected.")

    frame_original = frame
    frame_display = frame.copy()

    cv2.namedWindow("ROI Calibration")
    cv2.setMouseCallback("ROI Calibration", mouse_callback)
    redraw()

    while True:
        cv2.imshow("ROI Calibration", frame_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit without saving.")
            break
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == ord("s"):
            if len(points) == 4:
                save_to_config(Path(args.output), w, h)
                print("Saved! Restart the pipeline to use the new ROI.")
                break
            else:
                print(f"Need 4 points, have {len(points)}. Keep clicking.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
