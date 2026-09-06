"""Development CLI with optional vision imports and explicit configuration."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from jake.adapters.person_events import EventError, PersonEventGenerator
from jake.appearance import AppearanceError
from jake.config import load_app_config
from jake.event_console import log_event
from jake.identity import IdentityError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jake local webcam preview (no recording)")
    parser.add_argument("--config", type=Path, default=Path("config/local.toml"))
    parser.add_argument("--detect", action="store_true", help="Enable local person detection")
    parser.add_argument(
        "--track", action="store_true", help="Enable Jake tracking (implies --detect)"
    )
    parser.add_argument("--tracker", choices=("iou", "kalman", "kalman-baseline"), default="iou")
    parser.add_argument("--assignment", choices=("greedy", "hungarian"), default=None)
    parser.add_argument("--events", action="store_true", help="Log semantic person events locally")
    parser.add_argument(
        "--debug-tracks", action="store_true", help="Show Kalman predictions and misses"
    )
    parser.add_argument(
        "--appearance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable short-term local appearance encoding with --track --tracker kalman",
    )
    parser.add_argument(
        "--reid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain recently-lost appearance tracks",
    )
    parser.add_argument(
        "--identity", action="store_true", help="Enable enrolled local face identity"
    )
    parser.add_argument(
        "--visitors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt in to encrypted anonymous visitor memory (requires --track)",
    )
    parser.add_argument(
        "--debug-visitors",
        action="store_true",
        help="Print visitor evidence/rejection diagnostics without vectors",
    )
    args = parser.parse_args(argv)
    if args.identity and not args.track:
        parser.error("--identity requires --track")
    if args.appearance is True and not (args.track and args.tracker == "kalman"):
        parser.error("--appearance requires --track --tracker kalman")
    if args.events and not args.track:
        parser.error("--events requires --track")
    if args.assignment is not None and not args.track:
        parser.error("--assignment requires --track")
    if args.tracker in {"kalman", "kalman-baseline"} and not args.track:
        parser.error("--tracker kalman requires --track")
    if args.debug_tracks and not (args.track and args.tracker in {"kalman", "kalman-baseline"}):
        parser.error("--debug-tracks requires --track --tracker kalman")
    try:
        config = load_app_config(args.config)
        args.identity = args.identity or config.identity.enabled
        if args.identity and not args.track:
            raise ValueError("identity requires --track")
        if args.visitors is not None:
            config = replace(config, visitors=replace(config.visitors, enabled=args.visitors))
        if config.visitors.enabled:
            if not args.track:
                raise ValueError("visitor memory requires --track")
            args.identity = True
        if args.debug_visitors and not config.visitors.enabled:
            raise ValueError("--debug-visitors requires visitor memory enabled")
        if args.reid is not None:
            config = replace(
                config,
                tracking=replace(
                    config.tracking, reid=replace(config.tracking.reid, enabled=args.reid)
                ),
            )
        if args.reid is True and not (
            args.track and args.tracker == "kalman" and args.appearance is not False
        ):
            raise ValueError("--reid requires --track --tracker kalman and appearance")
        if args.reid is True and args.appearance is None:
            args.appearance = True
        if args.appearance is not None:
            config = replace(
                config,
                tracking=replace(
                    config.tracking,
                    appearance=replace(config.tracking.appearance, enabled=args.appearance),
                ),
            )
        if args.assignment is not None:
            config = replace(config, tracking=replace(config.tracking, assignment=args.assignment))
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        import cv2

        from jake.adapters.iou_tracker import TrackerError
        from jake.adapters.opencv_camera import CameraError
        from jake.adapters.yolo_detector import DetectorError
        from jake.preview import preview
    except ImportError:
        print("Vision dependencies unavailable. Run: uv sync --extra vision", file=sys.stderr)
        return 2
    try:
        from jake.application.composition import compose

        components = compose(
            config,
            detect=args.detect,
            track=args.track,
            tracker_mode=args.tracker,
            identify=args.identity,
            debug_tracks=args.debug_tracks,
        )
        detector, tracker = components.detector, components.tracker
        encoder, identity, visitors = components.encoder, components.identity, components.visitors
        diagnostics = components.diagnostics
        preview(
            config,
            detector,
            tracker,
            track_diagnostics=diagnostics,
            events=PersonEventGenerator(config.events)
            if args.events or visitors is not None
            else None,
            event_sink=log_event
            if args.events
            else (lambda _: None)
            if visitors is not None
            else None,
            encoder=encoder,
            identity=identity,
            visitors=visitors,
            debug_visitors=args.debug_visitors,
        )
    except KeyboardInterrupt:
        print("\nCamera preview stopped.")
        return 0
    except ImportError:
        print(
            "Optional dependency unavailable; install the requested extras (including identity).",
            file=sys.stderr,
        )
        return 1
    except (
        CameraError,
        DetectorError,
        TrackerError,
        EventError,
        AppearanceError,
        IdentityError,
        cv2.error,
    ) as exc:
        print(f"Camera preview error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
