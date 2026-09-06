"""Immutable, model-independent contracts shared by perception components."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import hypot, isclose, isfinite


def _identifier(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _timestamp(value: datetime) -> None:
    if value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")


def _confidence(value: float) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized image coordinates; origin is the top-left corner."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("bounding box coordinates must be finite and normalized")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("bounding box must have positive area")


@dataclass(frozen=True, slots=True)
class Frame:
    """Packed RGB8 pixels in row-major order, with no padding between rows.

    Adapters translate their native image representation at the boundary.
    Pixel data is excluded from repr to avoid incidental image disclosure.
    """

    camera_id: str
    sequence: int
    captured_at: datetime
    width: int
    height: int
    pixels: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.camera_id, "camera_id")
        _timestamp(self.captured_at)
        if self.sequence < 0:
            raise ValueError("frame sequence must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not isinstance(self.pixels, bytes):
            raise TypeError("pixels must be immutable bytes")
        if len(self.pixels) != self.width * self.height * 3:
            raise ValueError("pixel buffer size must match packed RGB8 dimensions")


@dataclass(frozen=True, slots=True)
class AppearanceEmbedding:
    """Immutable unit vector, session-local metadata; never an identity or event payload."""

    values: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("embedding must be a non-empty immutable tuple")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in self.values
        ):
            raise ValueError("embedding values must be finite numbers")
        if not isclose(hypot(*self.values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("embedding must be L2 normalized")


@dataclass(frozen=True, slots=True)
class PersonDetection:
    box: BoundingBox
    confidence: float
    appearance: AppearanceEmbedding | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class PersonTrack:
    """A session-local track, not a resident identity or biometric identifier."""

    track_id: str
    box: BoundingBox
    confidence: float
    missed_frames: int = 0
    confirmed: bool = True
    recently_lost: bool = False
    continuity_epoch: int = 0

    def __post_init__(self) -> None:
        _identifier(self.track_id, "track_id")
        _confidence(self.confidence)
        if (
            isinstance(self.missed_frames, bool)
            or not isinstance(self.missed_frames, int)
            or self.missed_frames < 0
        ):
            raise ValueError("missed_frames must be a non-negative integer")
        if not isinstance(self.recently_lost, bool) or (
            self.recently_lost and (not self.confirmed or self.missed_frames == 0)
        ):
            raise ValueError("recently_lost must describe a confirmed missed track")
        if (
            isinstance(self.continuity_epoch, bool)
            or not isinstance(self.continuity_epoch, int)
            or self.continuity_epoch < 0
        ):
            raise ValueError("continuity_epoch must be a non-negative integer")
        if not isinstance(self.confirmed, bool):
            raise ValueError("confirmed must be a boolean")

    @property
    def visible(self) -> bool:
        """True only when this update has a matched or newly observed detection."""
        return self.missed_frames == 0


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Metadata available to tracking and event generation; contains no pixels."""

    camera_id: str
    sequence: int
    captured_at: datetime

    def __post_init__(self) -> None:
        _identifier(self.camera_id, "camera_id")
        _timestamp(self.captured_at)
        if self.sequence < 0:
            raise ValueError("frame sequence must be non-negative")


class EventKind(StrEnum):
    PERSON_ENTERED = "person_entered"
    PERSON_PRESENT = "person_present"
    PERSON_UPDATED = "person_updated"
    PERSON_LEFT = "person_left"


@dataclass(frozen=True, slots=True)
class PersonEvent:
    """Minimal event metadata; no image, embedding, name, or audio payload."""

    event_id: str
    kind: EventKind
    context: FrameContext
    track_id: str
    entered_at: datetime | None = None

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.track_id, "track_id")
        if self.entered_at is not None:
            _timestamp(self.entered_at)
            if self.entered_at.astimezone(UTC) > self.context.captured_at.astimezone(UTC):
                raise ValueError("entered_at must not follow the event timestamp")

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed camera-timeline time; absent on legacy events without entry time."""
        if self.entered_at is None:
            return None
        return (
            self.context.captured_at.astimezone(UTC) - self.entered_at.astimezone(UTC)
        ).total_seconds()

    @property
    def left_at(self) -> datetime | None:
        return self.context.captured_at if self.kind == EventKind.PERSON_LEFT else None
