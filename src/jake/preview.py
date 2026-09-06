"""Local development display; never imported by the perception core."""

from collections.abc import Callable
from contextlib import suppress
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from jake.adapters.opencv_camera import OpenCVCamera
from jake.appearance import encode_detections
from jake.config import AppConfig
from jake.diagnostics import TrackDiagnostic, measure_detection
from jake.domain import FrameContext, PersonDetection, PersonEvent, PersonTrack
from jake.face_identity import FaceIdentityService
from jake.ports import AppearanceEncoder, EventGenerator, PersonDetector, PersonTracker

WINDOW = "Jake local camera - q/Q to quit"


def draw_people(
    display: NDArray[np.uint8], detections: tuple[PersonDetection | PersonTrack, ...]
) -> None:
    """Render structured detections on the preview copy, never on a Frame."""
    height, width = display.shape[:2]
    for detection in detections:
        box = detection.box
        left, right = (min(width - 1, int(x * width)) for x in (box.left, box.right))
        top, bottom = (min(height - 1, int(y * height)) for y in (box.top, box.bottom))
        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            display,
            (f"ID {detection.track_id} | " if isinstance(detection, PersonTrack) else "")
            + f"PERSON {detection.confidence:.1%}",
            (left, max(15, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def preview(
    config: AppConfig,
    detector: PersonDetector | None = None,
    tracker: PersonTracker | None = None,
    *,
    track_diagnostics: Callable[[], tuple[TrackDiagnostic, ...]] | None = None,
    events: EventGenerator | None = None,
    event_sink: Callable[[PersonEvent], None] | None = None,
    encoder: AppearanceEncoder | None = None,
    identity: FaceIdentityService | None = None,
) -> None:
    """Display frames on the main thread, with no recording or persistence."""
    if tracker is not None and detector is None:
        raise ValueError("tracking preview requires a detector")
    if events is not None and (tracker is None or event_sink is None):
        raise ValueError("event preview requires a tracker and event sink")
    if encoder is not None and tracker is None:
        raise ValueError("appearance preview requires a tracker")
    if identity is not None and tracker is None:
        raise ValueError("identity preview requires tracking")
    with OpenCVCamera(config.pipeline.camera_id, config.camera) as camera:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            start = perf_counter()
            for count, frame in enumerate(camera, start=1):
                measurement = measure_detection(detector, frame) if detector is not None else None
                rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
                    frame.height, frame.width, 3
                )
                display = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
                if measurement is not None:
                    if tracker is not None:
                        context = FrameContext(frame.camera_id, frame.sequence, frame.captured_at)
                        detections = measurement.detections
                        if encoder is not None:
                            encoding_start = perf_counter()
                            detections = encode_detections(frame, detections, encoder)
                            encoding_ms = (perf_counter() - encoding_start) * 1000
                            cv2.putText(
                                display,
                                f"appearance {encoding_ms:.1f} ms",
                                (10, frame.height - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                        tracks = tracker.update(context, detections)
                        if identity is not None:
                            identities, identity_events = identity.process(frame, tracks)
                            for item in identity_events:
                                print(
                                    f"{item.kind} track={item.track_id} "
                                    f"resident_id={item.match.resident_id} state={item.match.state}"
                                )
                            for index, (track_id, match) in enumerate(identities.items()):
                                label = (
                                    f"ID {track_id} | {match.display_name or ''} | {match.state}"
                                )
                                cv2.putText(
                                    display,
                                    label,
                                    (10, frame.height - 35 - index * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 0),
                                    1,
                                )
                        if events is not None and event_sink is not None:
                            for event in events.generate(context, tracks):
                                event_sink(event)
                        draw_people(display, tuple(t for t in tracks if not t.recently_lost))
                        if track_diagnostics is not None:
                            cv2.putText(
                                display,
                                f"assignment {config.tracking.assignment}",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 165, 255),
                                1,
                            )
                            draw_track_diagnostics(display, track_diagnostics())
                        cv2.putText(
                            display,
                            f"active tracks {sum(not t.recently_lost for t in tracks)} "
                            "(includes missed)",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                    else:
                        draw_people(display, measurement.detections)
                    cv2.putText(
                        display,
                        f"people {len(measurement.detections)} | "
                        f"inference {measurement.inference_ms:.1f} ms | "
                        f"detection {measurement.detection_fps:.1f} FPS",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                elapsed = perf_counter() - start
                fps = count / elapsed if elapsed > 0 else 0.0
                cv2.putText(
                    display,
                    f"seq {frame.sequence} | {fps:.1f} FPS | {frame.width}x{frame.height}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(WINDOW, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
        finally:
            # A failed window creation or already-destroyed window must not mask
            # an acquisition/display error or interrupt.
            with suppress(cv2.error):
                cv2.destroyWindow(WINDOW)


def draw_track_diagnostics(
    display: NDArray[np.uint8], diagnostics: tuple[TrackDiagnostic, ...]
) -> None:
    """Optional debug rendering: orange predictions, blue measurements, missed counts."""
    height, width = display.shape[:2]
    for index, item in enumerate(diagnostics):
        for box, color in (
            ()
            if item.lifecycle == "RECENTLY_LOST"
            else ((item.predicted_box, (0, 165, 255)), (item.measured_box, (255, 0, 0)))
        ):
            if box is not None:
                cv2.rectangle(
                    display,
                    (min(width - 1, int(box.left * width)), min(height - 1, int(box.top * height))),
                    (
                        min(width - 1, int(box.right * width)),
                        min(height - 1, int(box.bottom * height)),
                    ),
                    color,
                    1,
                )
        label = f"ID {item.track_id} | missed {item.missed_frames}"
        if item.lifecycle is not None:
            label = f"ID {item.track_id} | {item.lifecycle}"
            if item.lifecycle == "TENTATIVE":
                label += f" {item.visible_hits}/{item.confirmation_hits}"
            elif item.lifecycle == "LOST":
                label += f" {item.missed_seconds:.1f}s"
        if item.lifecycle == "RECENTLY_LOST":
            label = f"RECENTLY_LOST ID {item.track_id} age={item.missed_seconds:.1f}s"
        if item.lifecycle == "REACTIVATED":
            label = f"REACTIVATED ID {item.track_id}"
        if item.appearance_similarity is not None:
            label += f" | app {item.appearance_similarity:.2f}"
        cv2.putText(
            display,
            label,
            (10, 125 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
        )
