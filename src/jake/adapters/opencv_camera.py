"""Local OpenCV acquisition; install Jake's vision extra to use this adapter."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite
from time import perf_counter
from types import TracebackType
from typing import Protocol, Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from jake.camera_info import CameraInfo
from jake.config import CameraConfig
from jake.domain import Frame


class CameraError(RuntimeError):
    """Camera acquisition failed; discard the current session."""


class Capture(Protocol):
    """The small OpenCV boundary, injectable without camera hardware."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...
    def release(self) -> None: ...
    def set(self, prop: int, value: float) -> bool: ...
    def get(self, prop: int) -> float: ...
    def getBackendName(self) -> str: ...


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
        capture_factory: Callable[..., Capture] | None = None,
    ) -> None:
        if not camera_id.strip():
            raise ValueError("camera_id must not be blank")
        self._camera_id = camera_id
        self._config = config
        self._factory = capture_factory
        self._capture: Capture | None = None
        self._used = False
        self._sequence = 0
        self._first_capture: float | None = None
        self.info = CameraInfo(config.width, config.height, config.fps)

    def __enter__(self) -> Self:
        if self._used:
            raise CameraError("camera sessions are single-use; create a new adapter")
        self._used = True
        try:
            backends = {
                "auto": cv2.CAP_ANY,
                "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF,
                "v4l2": cv2.CAP_V4L2,
                "gstreamer": cv2.CAP_GSTREAMER,
            }
            args = (
                (self._config.device,)
                if self._config.backend == "auto"
                else (self._config.device, backends[self._config.backend])
            )
            self._capture = (
                self._factory(*args)
                if self._factory is not None
                else cast(Capture, cv2.VideoCapture(*args))
            )
            if not self._capture.isOpened():
                raise CameraError(
                    f"Cannot open local camera index {self._config.device}; "
                    "check the index, camera permissions, and other applications using it"
                )
            self._negotiate()
        except BaseException as exc:
            self.close()
            if isinstance(exc, cv2.error):
                raise CameraError(f"Cannot open local camera index {self._config.device}") from exc
            raise
        return self

    def _negotiate(self) -> None:
        assert self._capture is not None
        config = self._config
        warnings = []
        # Codec first: compressed UVC modes can enable higher USB resolutions.
        requests = (
            (
                "fourcc",
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter.fourcc(*config.fourcc) if config.fourcc else None,
            ),
            ("width", cv2.CAP_PROP_FRAME_WIDTH, config.width),
            ("height", cv2.CAP_PROP_FRAME_HEIGHT, config.height),
            ("fps", cv2.CAP_PROP_FPS, config.fps),
        )
        for name, prop, value in requests:
            if value is not None:
                try:
                    accepted = self._capture.set(prop, float(value))
                except (cv2.error, NotImplementedError):
                    accepted = False
                if not accepted:
                    warnings.append(f"{name} request unsupported")

        def read(prop: int) -> float | None:
            try:
                value = self._capture.get(prop)  # type: ignore[union-attr]
                if isinstance(value, (int, float)) and isfinite(value) and value > 0:
                    return float(value)
            except (cv2.error, NotImplementedError):
                pass
            return None

        width, height = read(cv2.CAP_PROP_FRAME_WIDTH), read(cv2.CAP_PROP_FRAME_HEIGHT)
        codec_value = read(cv2.CAP_PROP_FOURCC)
        codec = None
        if codec_value is not None:
            decoded = "".join(chr((int(codec_value) >> (8 * i)) & 255) for i in range(4))
            if decoded.isascii() and decoded.isprintable():
                codec = decoded
        try:
            backend = self._capture.getBackendName()
        except (cv2.error, NotImplementedError):
            backend = self._config.backend
        self.info = replace(
            self.info,
            actual_width=int(width) if width else None,
            actual_height=int(height) if height else None,
            actual_fps=read(cv2.CAP_PROP_FPS),
            backend=backend if isinstance(backend, str) else self._config.backend,
            codec=codec,
            warnings=tuple(warnings),
        )

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
            completed = perf_counter()
            if self._first_capture is None:
                self._first_capture = completed
            elapsed = completed - self._first_capture
            self.info = replace(
                self.info,
                actual_width=width,
                actual_height=height,
                capture_fps=self._sequence / elapsed if elapsed > 0 else None,
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
