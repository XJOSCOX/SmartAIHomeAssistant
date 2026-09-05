"""Local development display; never imported by the perception core."""

from contextlib import suppress
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from jake.adapters.opencv_camera import OpenCVCamera
from jake.config import AppConfig
from jake.diagnostics import measure_detection
from jake.domain import FrameContext, PersonDetection, PersonTrack
from jake.ports import PersonDetector, PersonTracker

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
) -> None:
    """Display frames on the main thread, with no recording or persistence."""
    if tracker is not None and detector is None:
        raise ValueError("tracking preview requires a detector")
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
                        tracks = tracker.update(
                            FrameContext(frame.camera_id, frame.sequence, frame.captured_at),
                            measurement.detections,
                        )
                        draw_people(display, tracks)
                        cv2.putText(
                            display,
                            f"active tracks {len(tracks)} (includes missed)",
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
