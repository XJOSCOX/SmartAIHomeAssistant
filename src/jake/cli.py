"""Development CLI with optional vision imports and explicit configuration."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from jake.config import load_app_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jake local webcam preview (no recording)")
    parser.add_argument("--config", type=Path, default=Path("config/local.toml"))
    parser.add_argument("--detect", action="store_true", help="Enable local person detection")
    args = parser.parse_args(argv)
    try:
        config = load_app_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        import cv2

        from jake.adapters.opencv_camera import CameraError
        from jake.adapters.yolo_detector import DetectorError, YoloPersonDetector
        from jake.preview import preview
    except ImportError:
        print("Vision dependencies unavailable. Run: uv sync --extra vision", file=sys.stderr)
        return 2
    try:
        detector = (
            YoloPersonDetector(config.detector, config.pipeline.min_person_confidence)
            if args.detect
            else None
        )
        preview(config, detector)
    except KeyboardInterrupt:
        print("\nCamera preview stopped.")
        return 0
    except (CameraError, DetectorError, cv2.error) as exc:
        print(f"Camera preview error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
