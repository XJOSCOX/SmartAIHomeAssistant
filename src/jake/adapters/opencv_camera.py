"""Local OpenCV acquisition; install Jake's vision extra to use this adapter."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from types import TracebackType
from typing import Protocol, Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from jake.config import CameraConfig
from jake.domain import Frame


class CameraError(RuntimeError):
    """Camera acquisition failed; discard the current session."""


class Capture(Protocol):
    """The small OpenCV boundary, injectable without camera hardware."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...
    def release(self) -> None: ...


class OpenCVCamera:
    """Single-use, context-managed FrameSource for one local camera session.

    Iteration is synchronous and unbuffered by Jake. Always use a with block so
    early loop exits and consumer exceptions release the device immediately.
    Native camera drivers may buffer or block reads. Instances are not thread-safe.
    """

    def __init__(
        self,
        camera_id: str,
        config: CameraConfig,
        *,
        capture_factory: Callable[[int], Capture] | None = None,
    ) -> None:
        if not camera_id.strip():
            raise ValueError("camera_id must not be blank")
        self._camera_id = camera_id
        self._config = config
        self._factory = capture_factory
        self._capture: Capture | None = None
        self._used = False
        self._sequence = 0

    def __enter__(self) -> Self:
        if self._used:
            raise CameraError("camera sessions are single-use; create a new adapter")
        self._used = True
        try:
            self._capture = (
                self._factory(self._config.device)
                if self._factory is not None
                else cast(Capture, cv2.VideoCapture(self._config.device))
            )
            if not self._capture.isOpened():
                raise CameraError(
                    f"Cannot open local camera index {self._config.device}; "
                    "check the index, camera permissions, and other applications using it"
                )
        except BaseException as exc:
            self.close()
            if isinstance(exc, cv2.error):
                raise CameraError(f"Cannot open local camera index {self._config.device}") from exc
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release at most once, including after a failed read or repeated cleanup."""
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        if self._capture is None:
            raise CameraError("camera is not open; iterate inside its with block")
        try:
            ok, bgr = self._capture.read()
            captured_at = datetime.now(UTC)
            if not ok or bgr is None or bgr.size == 0:
                raise CameraError(f"Failed to read local camera index {self._config.device}")
            if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise CameraError("camera must supply an 8-bit, three-channel BGR image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = bgr.shape[:2]
            frame = Frame(
                self._camera_id, self._sequence, captured_at, width, height, rgb.tobytes(order="C")
            )
            self._sequence += 1
            return frame
        except BaseException as exc:
            self.close()
            if isinstance(exc, cv2.error):
                raise CameraError(
                    "OpenCV failed while reading or converting a camera frame"
                ) from exc
            raise
