"""Conservative visitor confirmation driven by semantic presence sessions."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import uuid4

from jake.domain import EventKind, FrameContext, PersonEvent, PersonTrack
from jake.identity import IdentityError, face_cosine, face_normalize
from jake.identity_domain import FaceEmbedding, IdentityMatch, IdentityState
from jake.visitor_config import VisitorConfig
from jake.visitor_domain import (
    VisitorEvent,
    VisitorMatch,
    VisitorObservation,
    VisitorProfile,
    VisitorState,
)
from jake.visitor_ports import VisitorStore


def _replace_with(profile: VisitorProfile) -> Callable[[VisitorProfile], VisitorProfile]:
    return lambda _: profile


def visitor_match(profile: VisitorProfile, recurring_count: int) -> VisitorMatch:
    state = (
        VisitorState.KNOWN_VISITOR
        if profile.explicitly_labeled
        else (
            VisitorState.RECURRING_VISITOR
            if profile.visit_count >= recurring_count
            else VisitorState.FIRST_TIME_VISITOR
        )
    )
    return VisitorMatch(state, profile.visitor_id, profile.visit_count, profile.display_name)


@dataclass
class _Visit:
    entered_at: datetime
    epoch: int = 0
    samples: list[tuple[datetime, FaceEmbedding]] = field(default_factory=list)
    visitor_id: str | None = None
    verified: bool = False
    resident_possible: bool = False
    last_proof: datetime | None = None
    target: str | None = None


class VisitorMemory:
    def __init__(
        self,
        config: VisitorConfig,
        store: VisitorStore,
        *,
        is_nonresident: Callable[[FaceEmbedding], bool],
        new_id: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        self.config, self.store, self.new_id = config, store, new_id
        self.is_nonresident = is_nonresident
        self._visits: dict[str, _Visit] = {}
        self._last: FrameContext | None = None

    def process(
        self,
        context: FrameContext,
        tracks: tuple[PersonTrack, ...],
        events: tuple[PersonEvent, ...],
        observations: dict[str, VisitorObservation],
        blocked: set[str],
        identities: dict[str, IdentityMatch],
    ) -> tuple[dict[str, VisitorMatch], tuple[VisitorEvent, ...]]:
        if not self.config.enabled:
            return {}, ()
        now = context.captured_at.astimezone(UTC)
        if self._last and (
            context.camera_id != self._last.camera_id
            or context.sequence <= self._last.sequence
            or now < self._last.captured_at.astimezone(UTC)
        ):
            raise IdentityError("invalid visitor timeline")
        self._last = context
        self.store.expire(now, self.config.retention_days)
        profiles = {p.visitor_id: p for p in self.store.profiles()}
        emitted: list[VisitorEvent] = []

        def emit(kind: str, track_id: str, profile: VisitorProfile) -> None:
            emitted.append(
                VisitorEvent(
                    self.new_id(),
                    kind,
                    context,
                    track_id,
                    visitor_match(profile, self.config.recurring_visit_count),
                )
            )

        for event in events:
            if event.context != context:
                raise IdentityError("visitor event timeline mismatch")
            if event.kind == EventKind.PERSON_ENTERED:
                self._visits.setdefault(
                    event.track_id, _Visit((event.entered_at or now).astimezone(UTC))
                )
            elif event.kind == EventKind.PERSON_LEFT:
                visit = self._visits.pop(event.track_id, None)
                if visit and visit.visitor_id in profiles:
                    profile = profiles[visit.visitor_id]
                    if not visit.resident_possible:
                        duration = (now - visit.entered_at).total_seconds()
                        profile = replace(
                            profile,
                            last_seen_at=max(
                                profile.last_seen_at, visit.last_proof or profile.last_seen_at
                            ),
                            statistics=profile.statistics.add(duration),
                        )
                        self.store.update(profile.visitor_id, _replace_with(profile))
                        profiles[profile.visitor_id] = profile
                    emit("VISITOR_LEFT", event.track_id, profile)

        results: dict[str, VisitorMatch] = {}
        ready: dict[str, tuple[FaceEmbedding, str | None]] = {}
        for track in sorted(tracks, key=lambda t: t.track_id):
            track_id = track.track_id
            results[track_id] = VisitorMatch()
            visit = self._visits.get(track_id)
            if visit is None:
                continue
            if (
                track_id in blocked
                or identities.get(track_id, IdentityMatch(IdentityState.UNKNOWN)).state
                != IdentityState.UNKNOWN
            ):
                visit.resident_possible = True
            if track.continuity_epoch != visit.epoch or track.recently_lost:
                visit.epoch, visit.verified = track.continuity_epoch, False
                visit.samples.clear()
            if visit.resident_possible or not track.visible or not track.confirmed:
                visit.samples.clear()
                continue
            observation = observations.get(track_id)
            if (
                observation is None
                or observation.detector_confidence < self.config.min_face_quality
            ):
                if (
                    visit.verified
                    and visit.visitor_id in profiles
                    and visit.last_proof
                    and (now - visit.last_proof).total_seconds() <= self.config.carry_seconds
                ):
                    results[track_id] = visitor_match(
                        profiles[visit.visitor_id], self.config.recurring_visit_count
                    )
                continue
            embedding = observation.embedding
            if not self.is_nonresident(embedding):
                visit.resident_possible = True
                visit.samples.clear()
                continue
            scores = sorted(
                ((face_cosine(embedding, p.template), p.visitor_id) for p in profiles.values()),
                key=lambda v: (-v[0], v[1]),
            )
            selected: str | None = None
            if (
                scores
                and scores[0][0] >= self.config.match_similarity - self.config.ambiguity_margin
            ):
                if scores[0][0] < self.config.match_similarity or (
                    len(scores) > 1 and scores[0][0] - scores[1][0] < self.config.ambiguity_margin
                ):
                    visit.samples.clear()
                    visit.verified = False
                    continue
                selected = scores[0][1]
            if visit.visitor_id is not None and selected != visit.visitor_id:
                visit.samples.clear()
                visit.verified = False
                continue
            if selected is not None and any(
                v.visitor_id == selected for k, v in self._visits.items() if k != track_id
            ):
                visit.samples.clear()
                continue
            if visit.verified and visit.visitor_id in profiles:
                visit.last_proof = now
                profile = profiles[visit.visitor_id]
                # Bound writes during long visits while keeping retention tied to evidence.
                if (now - profile.last_seen_at).total_seconds() >= 3600:
                    profile = replace(profile, last_seen_at=now)
                    self.store.update(profile.visitor_id, _replace_with(profile))
                    profiles[profile.visitor_id] = profile
                results[track_id] = visitor_match(
                    profiles[visit.visitor_id], self.config.recurring_visit_count
                )
                continue
            visit.samples = [
                (t, e)
                for t, e in visit.samples
                if (now - t).total_seconds() <= self.config.observation_window_seconds
            ]
            if selected != visit.target:
                visit.samples.clear()
                visit.target = selected
            if any(
                face_cosine(embedding, e) < self.config.match_similarity for _, e in visit.samples
            ):
                visit.samples.clear()
            if (
                not visit.samples
                or (now - visit.samples[-1][0]).total_seconds()
                >= self.config.observation_interval_seconds
            ):
                visit.samples.append((now, embedding))
            results[track_id] = VisitorMatch(VisitorState.VISITOR_CANDIDATE)
            if len(visit.samples) >= self.config.required_observations:
                # Every observation, not just its centroid, must agree with the proposed match.
                if selected and any(
                    face_cosine(e, profiles[selected].template) < self.config.match_similarity
                    for _, e in visit.samples
                ):
                    visit.samples.clear()
                    results[track_id] = VisitorMatch()
                    continue
                centroid = face_normalize(
                    embedding.model_id,
                    tuple(
                        sum(e.values[i] for _, e in visit.samples) / len(visit.samples)
                        for i in range(len(embedding.values))
                    ),
                )
                if not self.is_nonresident(centroid):
                    visit.resident_possible = True
                    visit.samples.clear()
                    results[track_id] = VisitorMatch()
                    continue
                if selected is None and any(
                    face_cosine(centroid, p.template)
                    >= self.config.match_similarity - self.config.ambiguity_margin
                    for p in profiles.values()
                ):
                    visit.samples.clear()
                    results[track_id] = VisitorMatch()
                    continue
                ready[track_id] = centroid, selected

        # Resolve batch conflicts before any writes. Input ordering cannot select a winner.
        for track_id, (centroid, selected) in ready.items():
            if any(
                other != track_id
                and (
                    selected is not None
                    and selected == other_id
                    or face_cosine(centroid, other_center)
                    >= self.config.match_similarity - self.config.ambiguity_margin
                )
                for other, (other_center, other_id) in ready.items()
            ):
                self._visits[track_id].samples.clear()
                results[track_id] = VisitorMatch()
                continue
            visit = self._visits[track_id]
            if selected is None:
                profile = VisitorProfile(
                    self.new_id(), centroid, visit.entered_at, now, visit.entered_at
                )
                self.store.add(profile)
                emit("VISITOR_FIRST_SEEN", track_id, profile)
            else:
                profile = profiles[selected]
                if visit.visitor_id is None:
                    profile = replace(
                        profile,
                        visit_count=profile.visit_count + 1,
                        last_visit_at=max(profile.last_visit_at, visit.entered_at),
                        last_seen_at=max(profile.last_seen_at, now),
                    )
                    self.store.update(selected, _replace_with(profile))
                    emit("VISITOR_RECOGNIZED", track_id, profile)
                    if profile.visit_count == self.config.recurring_visit_count:
                        emit("VISITOR_BECAME_RECURRING", track_id, profile)
            profiles[profile.visitor_id] = profile
            visit.visitor_id, visit.verified, visit.last_proof = profile.visitor_id, True, now
            visit.samples.clear()
            results[track_id] = visitor_match(profile, self.config.recurring_visit_count)
        return results, tuple(emitted)
