"""Development CLI with optional vision imports and explicit configuration."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from jake.adapters.person_events import EventError, PersonEventGenerator
from jake.config import load_app_config
from jake.event_console import log_event
from jake.ports import PersonTracker


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jake local webcam preview (no recording)")
    parser.add_argument("--config", type=Path, default=Path("config/local.toml"))
    parser.add_argument("--detect", action="store_true", help="Enable local person detection")
    parser.add_argument(
        "--track", action="store_true", help="Enable Jake tracking (implies --detect)"
    )
    parser.add_argument("--tracker", choices=("iou", "kalman"), default="iou")
    parser.add_argument("--assignment", choices=("greedy", "hungarian"), default=None)
    parser.add_argument("--events", action="store_true", help="Log semantic person events locally")
    parser.add_argument(
        "--debug-tracks", action="store_true", help="Show Kalman predictions and misses"
    )
    args = parser.parse_args(argv)
    if args.events and not args.track:
        parser.error("--events requires --track")
    if args.assignment is not None and not args.track:
        parser.error("--assignment requires --track")
    if args.tracker == "kalman" and not args.track:
        parser.error("--tracker kalman requires --track")
    if args.debug_tracks and not (args.track and args.tracker == "kalman"):
        parser.error("--debug-tracks requires --track --tracker kalman")
    try:
        config = load_app_config(args.config)
        if args.assignment is not None:
            config = replace(config, tracking=replace(config.tracking, assignment=args.assignment))
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        import cv2

        from jake.adapters.iou_tracker import IoUPersonTracker, TrackerError
        from jake.adapters.kalman_tracker import KalmanPersonTracker
        from jake.adapters.opencv_camera import CameraError
        from jake.adapters.yolo_detector import DetectorError, YoloPersonDetector
        from jake.preview import preview
    except ImportError:
        print("Vision dependencies unavailable. Run: uv sync --extra vision", file=sys.stderr)
        return 2
    try:
        detector = (
            YoloPersonDetector(config.detector, config.pipeline.min_person_confidence)
            if args.detect or args.track
            else None
        )
        tracker: PersonTracker | None = None
        diagnostics = None
        if args.track:
            if args.tracker == "kalman":
                motion_tracker = KalmanPersonTracker(config.tracking)
                tracker = motion_tracker
                if args.debug_tracks:
                    diagnostics = motion_tracker.diagnostics
            else:
                tracker = IoUPersonTracker(config.tracking)
        preview(
            config,
            detector,
            tracker,
            track_diagnostics=diagnostics,
            events=PersonEventGenerator(config.events) if args.events else None,
            event_sink=log_event if args.events else None,
        )
    except KeyboardInterrupt:
        print("\nCamera preview stopped.")
        return 0
    except (CameraError, DetectorError, TrackerError, EventError, cv2.error) as exc:
        print(f"Camera preview error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
