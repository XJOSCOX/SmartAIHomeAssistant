"""Immutable, model-independent contracts shared by perception components."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite


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
class PersonDetection:
    box: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        _confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class PersonTrack:
    """A session-local track, not a resident identity or biometric identifier."""

    track_id: str
    box: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        _identifier(self.track_id, "track_id")
        _confidence(self.confidence)


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
    PERSON_UPDATED = "person_updated"
    PERSON_LEFT = "person_left"


@dataclass(frozen=True, slots=True)
class PersonEvent:
    """Minimal event metadata; no image, embedding, name, or audio payload."""

    event_id: str
    kind: EventKind
    context: FrameContext
    track_id: str

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.track_id, "track_id")
