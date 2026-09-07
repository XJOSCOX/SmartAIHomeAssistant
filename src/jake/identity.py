"""Face-only matching, explicit enrollment, and per-track temporal verification."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import fsum, hypot
from uuid import uuid4

from jake.domain import FrameContext, PersonTrack
from jake.identity_config import IdentityConfig
from jake.identity_diagnostics import MatchDiagnostic, TemporalDiagnostic
from jake.identity_domain import (
    FaceEmbedding,
    FaceQuality,
    IdentityEvent,
    IdentityMatch,
    IdentityState,
    ResidentProfile,
)


class IdentityError(ValueError):
    """Invalid identity input, unsafe storage, or local face adapter failure."""


def face_normalize(model_id: str, values: tuple[float, ...]) -> FaceEmbedding:
    norm = hypot(*values)
    if not norm > 0:
        raise IdentityError("invalid face vector")
    return FaceEmbedding(model_id, tuple(v / norm for v in values))


def face_cosine(a: FaceEmbedding, b: FaceEmbedding) -> float:
    if a.model_id != b.model_id or len(a.values) != len(b.values):
        raise IdentityError("face template/model mismatch; re-enrollment required")
    return max(-1.0, min(1.0, fsum(x * y for x, y in zip(a.values, b.values, strict=True))))


class CosineIdentityMatcher:
    def __init__(self, config: IdentityConfig) -> None:
        self.config = config
        self._diagnostic = MatchDiagnostic("not observed")

    @property
    def diagnostic(self) -> MatchDiagnostic:
        return self._diagnostic

    def match(
        self, embedding: FaceEmbedding, profiles: tuple[ResidentProfile, ...]
    ) -> IdentityMatch:
        scores = sorted(
            ((max(face_cosine(embedding, t) for t in p.templates), p) for p in profiles),
            key=lambda item: (-item[0], item[1].resident_id),
        )
        if not scores or scores[0][0] < self.config.candidate_similarity:
            self._diagnostic = MatchDiagnostic(
                "no resident profiles loaded"
                if not scores
                else f"similarity below candidate threshold {self.config.candidate_similarity:.2f}",
                scores[0][0] if scores else None,
            )
            return IdentityMatch(IdentityState.UNKNOWN)
        score, profile = scores[0]
        if len(scores) > 1 and score - scores[1][0] < self.config.ambiguity_margin:
            self._diagnostic = MatchDiagnostic(
                f"ambiguous resident match; margin < {self.config.ambiguity_margin:.2f}", score
            )
            return IdentityMatch(IdentityState.UNKNOWN)
        self._diagnostic = MatchDiagnostic(
            "candidate; resident support requires similarity >= "
            f"{self.config.resident_similarity:.2f}",
            score,
        )
        # A single embedding is only ever a candidate, even above resident threshold.
        return IdentityMatch(
            IdentityState.CANDIDATE, profile.resident_id, profile.display_name, score
        )


class Enrollment:
    def __init__(self, config: IdentityConfig) -> None:
        self.config = config
        self.samples: list[FaceEmbedding] = []
        self.poses: set[str] = set()
        self.last_at: datetime | None = None

    @property
    def ready(self) -> bool:
        return len(self.samples) >= self.config.enrollment_samples and len(self.poses) >= 2

    def accept(self, quality: FaceQuality, embedding: FaceEmbedding | None, at: datetime) -> str:
        if at.utcoffset() is None:
            raise IdentityError("enrollment timestamp must be aware")
        if not quality.accepted or embedding is None:
            return quality.summary
        if self.ready:
            return "complete"
        if (
            self.last_at is not None
            and (at.astimezone(UTC) - self.last_at.astimezone(UTC)).total_seconds() < 0.5
        ):
            return "wait between samples"
        if (
            self.samples
            and max(face_cosine(embedding, s) for s in self.samples)
            >= self.config.duplicate_similarity
        ):
            return "duplicate sample; vary pose slightly"
        if (
            self.samples
            and min(face_cosine(embedding, s) for s in self.samples)
            < self.config.resident_similarity
        ):
            return "inconsistent face; keep the same consenting resident in view"
        if (
            len(self.samples) >= self.config.enrollment_samples - 1
            and len(self.poses | {quality.pose}) < 2
        ):
            return "turn slightly for a second pose"
        self.samples.append(embedding)
        self.poses.add(quality.pose)
        self.last_at = at
        return "accepted"

    def profile(self, name: str) -> ResidentProfile:
        if not self.ready or self.last_at is None:
            raise IdentityError("enrollment needs multiple quality samples and poses")
        centroid = face_normalize(
            self.samples[0].model_id,
            tuple(
                fsum(s.values[i] for s in self.samples) / len(self.samples)
                for i in range(len(self.samples[0].values))
            ),
        )
        return ResidentProfile(
            str(uuid4()), name.strip(), (centroid,), len(self.samples), self.last_at
        )


@dataclass
class _Evidence:
    epoch: int
    match: IdentityMatch = field(default_factory=lambda: IdentityMatch(IdentityState.UNKNOWN))
    support: list[datetime] = field(default_factory=list)
    last_valid: datetime | None = None
    last_counted: datetime | None = None


class TemporalIdentity:
    """No biometric vectors are retained here; body continuity is not face evidence."""

    def __init__(self, config: IdentityConfig) -> None:
        self.config = config
        self._state: dict[str, _Evidence] = {}
        self._last: FrameContext | None = None

    def diagnostics(self) -> dict[str, TemporalDiagnostic]:
        """Copy current evidence metadata without mutating or extending support."""
        if self._last is None:
            return {}
        now = self._last.captured_at.astimezone(UTC)
        return {
            key: TemporalDiagnostic(
                self.config.required_confirmations,
                sum(
                    (now - t).total_seconds() <= self.config.confirmation_window_seconds
                    for t in state.support
                ),
                self.config.confirmation_window_seconds,
                (now - state.last_valid).total_seconds() if state.last_valid else None,
            )
            for key, state in self._state.items()
        }

    def update(
        self,
        context: FrameContext,
        tracks: tuple[PersonTrack, ...],
        observations: dict[str, IdentityMatch],
    ) -> tuple[dict[str, IdentityMatch], tuple[IdentityEvent, ...]]:
        now = context.captured_at.astimezone(UTC)
        if self._last and (
            context.camera_id != self._last.camera_id
            or context.sequence <= self._last.sequence
            or now < self._last.captured_at.astimezone(UTC)
        ):
            raise IdentityError("invalid identity camera timeline")
        if len({t.track_id for t in tracks}) != len(tracks):
            raise IdentityError("duplicate identity track IDs")
        self._last = context
        alive = {t.track_id for t in tracks}
        self._state = {k: v for k, v in self._state.items() if k in alive}
        events = []
        results = {}
        for track in tracks:
            state = self._state.setdefault(track.track_id, _Evidence(track.continuity_epoch))
            before = state.match
            if track.continuity_epoch != state.epoch or track.recently_lost:
                state = _Evidence(track.continuity_epoch)
                self._state[track.track_id] = state
            if (
                state.last_valid
                and (now - state.last_valid).total_seconds() > self.config.carry_seconds
            ):
                state.match = IdentityMatch(IdentityState.UNKNOWN)
                state.support.clear()
            observation = (
                observations.get(track.track_id) if track.visible and track.confirmed else None
            )
            if observation is not None:
                if (
                    observation.resident_id != state.match.resident_id
                    or observation.similarity is None
                    or observation.similarity < self.config.resident_similarity
                ):
                    state.support.clear()
                    # A replaceable matcher proposes an identity; only this temporal
                    # layer may grant RESIDENT, even if an adapter returns that state.
                    state.match = IdentityMatch(
                        IdentityState.CANDIDATE
                        if observation.resident_id is not None
                        else IdentityState.UNKNOWN,
                        observation.resident_id,
                        observation.display_name,
                        observation.similarity,
                    )
                state.last_valid = now
                if (
                    observation.resident_id is not None
                    and observation.similarity is not None
                    and observation.similarity >= self.config.resident_similarity
                ):
                    state.support = [
                        t
                        for t in state.support
                        if (now - t).total_seconds() <= self.config.confirmation_window_seconds
                    ]
                    if (
                        state.last_counted is None
                        or (now - state.last_counted).total_seconds()
                        >= self.config.observation_interval_seconds
                    ):
                        state.support.append(now)
                        state.last_counted = now
                    verified = (
                        state.match.state == IdentityState.RESIDENT
                        or len(state.support) >= self.config.required_confirmations
                    )
                    state.match = IdentityMatch(
                        IdentityState.RESIDENT if verified else IdentityState.CANDIDATE,
                        observation.resident_id,
                        observation.display_name,
                        observation.similarity,
                    )
            results[track.track_id] = state.match
            if (before.state, before.resident_id) != (state.match.state, state.match.resident_id):
                events.append(
                    IdentityEvent(
                        str(uuid4()),
                        context,
                        track.track_id,
                        state.match,
                        "IDENTITY_RESOLVED"
                        if state.match.state == IdentityState.RESIDENT
                        else "IDENTITY_STATE_CHANGED",
                    )
                )
        return results, tuple(events)
