"""Local development display; never imported by the perception core."""

from contextlib import suppress
from time import perf_counter

import cv2
import numpy as np

from jake.adapters.opencv_camera import OpenCVCamera
from jake.config import AppConfig

WINDOW = "Jake local camera - q to quit"


def preview(config: AppConfig) -> None:
    """Display frames on the main thread, with no recording or persistence."""
    with OpenCVCamera(config.pipeline.camera_id, config.camera) as camera:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            start = perf_counter()
            for count, frame in enumerate(camera, start=1):
                rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
                    frame.height, frame.width, 3
                )
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            # A failed window creation or already-destroyed window must not mask
            # an acquisition/display error or interrupt.
            with suppress(cv2.error):
                cv2.destroyWindow(WINDOW)
