#!/usr/bin/env python3
"""Export YOLOv8n to TensorFlow Lite format for Google Coral EdgeTPU.

This script exports the model in two steps:
1. YOLOv8n (PyTorch) -> TFLite (full integer quantization)
2. Compile for EdgeTPU using the edgetpu_compiler

Usage:
    python scripts/export_model.py

Prerequisites:
    pip install ultralytics
    # For EdgeTPU compilation (run on Debian/Ubuntu):
    # curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    # echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    # sudo apt update && sudo apt install edgetpu-compiler
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Export YOLOv8n to TFLite with int8 quantization
    logger.info("Step 1: Exporting YOLOv8n to TFLite (int8 quantization)...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.export(
            format="tflite",
            imgsz=320,
            int8=True,
            data="coco128.yaml",  # Calibration dataset
        )
        logger.info("TFLite export complete.")
    except Exception:
        logger.exception("Failed to export model")
        sys.exit(1)

    # Find the exported TFLite file
    tflite_files = list(Path(".").rglob("*_full_integer_quant.tflite"))
    if not tflite_files:
        tflite_files = list(Path(".").rglob("*.tflite"))

    if not tflite_files:
        logger.error("No TFLite file found after export!")
        sys.exit(1)

    tflite_path = tflite_files[0]
    logger.info(f"Found TFLite model: {tflite_path}")

    # Move to models directory
    dest = output_dir / tflite_path.name
    tflite_path.rename(dest)
    logger.info(f"Moved to: {dest}")

    # Step 2: Compile for EdgeTPU
    logger.info("Step 2: Compiling for EdgeTPU...")
    try:
        result = subprocess.run(
            ["edgetpu_compiler", "-s", "-o", str(output_dir), str(dest)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("EdgeTPU compilation successful!")
            logger.info(f"Output: {output_dir}")
            # List the compiled files
            for f in output_dir.glob("*edgetpu*"):
                logger.info(f"  {f}")
        else:
            logger.warning(f"EdgeTPU compiler failed: {result.stderr}")
            logger.info("You can compile later on a Debian machine with:")
            logger.info(f"  edgetpu_compiler -s -o models/ {dest}")
    except FileNotFoundError:
        logger.info("edgetpu_compiler not found. That's OK for desktop development.")
        logger.info("To compile for Coral, install edgetpu_compiler on a Linux machine:")
        logger.info("  sudo apt install edgetpu-compiler")
        logger.info(f"  edgetpu_compiler -s -o models/ {dest}")

    logger.info("\nDone! Model files are in the models/ directory.")


if __name__ == "__main__":
    main()
