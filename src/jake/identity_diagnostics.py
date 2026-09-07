"""Immutable identity diagnostics; no images, templates or embeddings."""

from dataclasses import dataclass

from jake.identity_domain import IdentityMatch


@dataclass(frozen=True, slots=True)
class MatchDiagnostic:
    reason: str
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class TemporalDiagnostic:
    required_confirmations: int
    support_count: int
    window_seconds: float
    last_valid_age_seconds: float | None


@dataclass(frozen=True, slots=True)
class IdentityDiagnostic:
    track_id: str
    match: IdentityMatch
    similarity: float | None
    reason: str
    face_summary: str
    face_width_px: int | None
    face_height_px: int | None
    detector_confidence: float | None
    temporal: TemporalDiagnostic
