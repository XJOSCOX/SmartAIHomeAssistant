"""Optional, model-independent session metadata; FrameSource stays minimal."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraInfo:
    requested_width: int | None = None
    requested_height: int | None = None
    requested_fps: float | None = None
    actual_width: int | None = None
    actual_height: int | None = None
    actual_fps: float | None = None
    backend: str = "unknown"
    codec: str | None = None
    capture_fps: float | None = None
    warnings: tuple[str, ...] = ()

    @property
    def mismatch(self) -> bool:
        return any(
            requested is not None and actual is not None and abs(requested - actual) > tolerance
            for requested, actual, tolerance in (
                (self.requested_width, self.actual_width, 0),
                (self.requested_height, self.actual_height, 0),
                (self.requested_fps, self.actual_fps, 0.5),
            )
        )

    @property
    def summary(self) -> str:
        requested = (
            f"{self.requested_width or 'auto'}x{self.requested_height or 'auto'}"
            f" @ {self.requested_fps or 'auto'} FPS requested"
        )
        actual = (
            f"{self.actual_width or '?'}x{self.actual_height or '?'}"
            f" @ {self.actual_fps or '?'} FPS driver-reported"
        )
        measured = "warming up" if self.capture_fps is None else f"{self.capture_fps:.1f} FPS"
        return (
            f"Camera: {requested} | Actual: {actual} | {self.backend} {self.codec or ''}"
            f" | delivered capture {measured}"
            + (" | REQUEST NOT HONORED" if self.mismatch else "")
            + (" | " + "; ".join(self.warnings) if self.warnings else "")
        )
