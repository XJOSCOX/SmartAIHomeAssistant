"""Face identity contracts, deliberately separate from body appearance and PersonTrack."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import hypot, isclose, isfinite
from uuid import UUID

from jake.domain import BoundingBox, FrameContext


@dataclass(frozen=True, slots=True)
class FaceDetection:
    box: BoundingBox
    confidence: float
    landmarks: tuple[tuple[float, float], ...] = ()
    clipped: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid face confidence")
        if self.landmarks and (
            len(self.landmarks) != 5
            or any(len(point) != 2 for point in self.landmarks)
            or any(not isfinite(v) or not 0 <= v <= 1 for point in self.landmarks for v in point)
        ):
            raise ValueError("face landmarks must be five normalized points")


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    model_id: str
    values: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_id, str)
            or not self.model_id
            or len(self.model_id) > 200
            or not isinstance(self.values, tuple)
            or not self.values
        ):
            raise ValueError("invalid face embedding schema")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
            for v in self.values
        ) or not isclose(hypot(*self.values), 1.0, abs_tol=1e-6):
            raise ValueError("face embedding must be finite and normalized")


@dataclass(frozen=True, slots=True)
class ResidentProfile:
    resident_id: str
    display_name: str
    templates: tuple[FaceEmbedding, ...] = field(repr=False)
    sample_count: int
    enrolled_at: datetime

    def __post_init__(self) -> None:
        UUID(self.resident_id)
        if (
            not self.display_name.strip()
            or self.display_name != self.display_name.strip()
            or len(self.display_name) > 80
            or not self.display_name.isprintable()
        ):
            raise ValueError("resident name must be printable and 1-80 characters")
        if (
            type(self.sample_count) is not int
            or not 3 <= self.sample_count <= 100
            or not isinstance(self.templates, tuple)
            or not 1 <= len(self.templates) <= 10
        ):
            raise ValueError("resident profile requires multiple enrollment observations")
        if self.enrolled_at.utcoffset() is None:
            raise ValueError("enrollment timestamp must be aware")
        if len({(t.model_id, len(t.values)) for t in self.templates}) != 1:
            raise ValueError("mixed face templates")


class IdentityState(StrEnum):
    UNKNOWN = "UNKNOWN"
    CANDIDATE = "CANDIDATE"
    RESIDENT = "RESIDENT"


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    state: IdentityState
    resident_id: str | None = None
    display_name: str | None = None
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class IdentityEvent:
    event_id: str
    context: FrameContext
    track_id: str
    match: IdentityMatch
    kind: str = "IDENTITY_STATE_CHANGED"


@dataclass(frozen=True, slots=True)
class FaceQuality:
    accepted: bool
    reason: str
    pose: str = "center"

    face_width_px: int | None = None
    face_height_px: int | None = None
    minimum_pixels: int | None = None

    @property
    def summary(self) -> str:
        if self.face_width_px is None or self.face_height_px is None:
            return self.reason
        size = f"{self.face_width_px}x{self.face_height_px} px"
        if self.reason == "face too small":
            return f"face too small: {size}; minimum {self.minimum_pixels} px"
        return f"{self.reason}; Face {size}"
