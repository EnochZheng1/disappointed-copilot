"""Entry point: python -m disappointed"""

import argparse
import logging
import sys
from pathlib import Path

from disappointed.config.loader import load_config
from disappointed.app import create_and_run


def main():
    parser = argparse.ArgumentParser(
        description="Disappointed Co-Pilot — AI dashcam that roasts bad drivers"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=["config/default.yaml", "config/desktop_dev.yaml"],
        help="Config file paths (later files override earlier ones)",
    )
    parser.add_argument(
        "--voice-pack",
        type=str,
        default=None,
        help="Override voice pack (e.g. british_instructor, tired_mom, deadpan_ai)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file path (overrides camera config to use file source)",
    )
    parser.add_argument(
        "--demo",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Demo mode: fire a random trigger every N seconds to test audio",
    )
    args = parser.parse_args()

    config = load_config(*[Path(p) for p in args.config])

    if args.voice_pack:
        config.commentary.voice_pack = args.voice_pack

    if args.source:
        from disappointed.config.schema import CameraBackend
        config.camera.backend = CameraBackend.FILE
        config.camera.file_path = args.source

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        create_and_run(config, demo_interval=args.demo)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
