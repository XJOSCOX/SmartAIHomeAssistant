"""Anonymous visitor metadata; never a resident or physical track identity."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from uuid import UUID

from jake.domain import FrameContext
from jake.identity_domain import FaceEmbedding


class VisitorState(StrEnum):
    UNKNOWN = "UNKNOWN"
    VISITOR_CANDIDATE = "VISITOR_CANDIDATE"
    FIRST_TIME_VISITOR = "FIRST_TIME_VISITOR"
    RECURRING_VISITOR = "RECURRING_VISITOR"
    KNOWN_VISITOR = "KNOWN_VISITOR"


@dataclass(frozen=True, slots=True)
class VisitStatistics:
    completed: int = 0
    mean_seconds: float = 0.0
    m2_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.completed) is not int
            or self.completed < 0
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0
                for v in (self.mean_seconds, self.m2_seconds)
            )
            or (self.completed == 0 and (self.mean_seconds != 0 or self.m2_seconds != 0))
        ):
            raise ValueError("invalid visitor duration statistics")

    def add(self, seconds: float) -> "VisitStatistics":
        if not isfinite(seconds) or seconds < 0:
            raise ValueError("invalid visit duration")
        n = self.completed + 1
        delta = seconds - self.mean_seconds
        mean = self.mean_seconds + delta / n
        return VisitStatistics(n, mean, max(0.0, self.m2_seconds + delta * (seconds - mean)))

    @property
    def variance_seconds(self) -> float:
        return self.m2_seconds / (self.completed - 1) if self.completed > 1 else 0.0


@dataclass(frozen=True, slots=True)
class VisitorProfile:
    visitor_id: str
    template: FaceEmbedding = field(repr=False)
    created_at: datetime
    last_seen_at: datetime
    last_visit_at: datetime
    visit_count: int = 1
    statistics: VisitStatistics = VisitStatistics()
    display_name: str | None = None
    explicitly_labeled: bool = False

    def __post_init__(self) -> None:
        UUID(self.visitor_id)
        if any(
            t.utcoffset() is None for t in (self.created_at, self.last_seen_at, self.last_visit_at)
        ):
            raise ValueError("visitor timestamps must be timezone-aware")
        if not self.created_at <= self.last_visit_at <= self.last_seen_at:
            raise ValueError("invalid visitor timeline")
        if (
            type(self.visit_count) is not int
            or self.visit_count < 1
            or self.statistics.completed > self.visit_count
        ):
            raise ValueError("invalid visitor count")
        if type(self.explicitly_labeled) is not bool or self.explicitly_labeled != (
            self.display_name is not None
        ):
            raise ValueError("visitor label requires explicit labeling")
        if self.display_name is not None and (
            not isinstance(self.display_name, str)
            or not self.display_name.isprintable()
            or not 1 <= len(self.display_name) <= 80
            or self.display_name != self.display_name.strip()
        ):
            raise ValueError("invalid visitor label")


@dataclass(frozen=True, slots=True)
class VisitorMatch:
    state: VisitorState = VisitorState.UNKNOWN
    visitor_id: str | None = None
    visit_count: int = 0
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class VisitorEvent:
    event_id: str
    kind: str
    context: FrameContext
    track_id: str
    match: VisitorMatch


@dataclass(frozen=True, slots=True)
class VisitorObservation:
    """Transient quality-approved face evidence; never serialized into events."""

    embedding: FaceEmbedding = field(repr=False)
    detector_confidence: float

    def __post_init__(self) -> None:
        if not isfinite(self.detector_confidence) or not 0 <= self.detector_confidence <= 1:
            raise ValueError("invalid visitor observation confidence")


@dataclass(frozen=True, slots=True)
class ResidentEvidence:
    """Fresh quality-approved resident comparison; no biometric vector."""

    resident_id: str
    similarity: float
    strong: bool
