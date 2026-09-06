"""Conservative visitor confirmation driven by semantic presence sessions."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from jake.domain import EventKind, FrameContext, PersonEvent, PersonTrack
from jake.identity import IdentityError, face_cosine, face_normalize
from jake.identity_domain import FaceEmbedding, IdentityMatch, IdentityState
from jake.visitor_config import VisitorConfig
from jake.visitor_domain import (
    ResidentEvidence,
    VisitorEvent,
    VisitorMatch,
    VisitorObservation,
    VisitorProfile,
    VisitorState,
)
from jake.visitor_ports import VisitorStore


def _replace_with(profile: VisitorProfile) -> Callable[[VisitorProfile], VisitorProfile]:
    return lambda _: profile


def visitor_match(profile: VisitorProfile, config: VisitorConfig) -> VisitorMatch:
    if profile.explicitly_labeled:
        state = VisitorState.KNOWN_VISITOR
    elif profile.distinct_visit_days >= config.frequent_distinct_days:
        state = VisitorState.FREQUENT_VISITOR
    elif profile.distinct_visit_days >= config.recurring_distinct_days:
        state = VisitorState.RECURRING_VISITOR
    else:
        state = VisitorState.FIRST_TIME_VISITOR
    return VisitorMatch(
        state,
        profile.visitor_id,
        profile.session_count,
        profile.display_name,
        profile.distinct_visit_days,
    )


@dataclass
class _Visit:
    entered_at: datetime
    epoch: int = 0
    samples: list[tuple[datetime, FaceEmbedding]] = field(default_factory=list)
    visitor_id: str | None = None
    verified: bool = False
    resident_verified: bool = False
    resident_candidate_count: int = 0
    resident_candidate_id: str | None = None
    resident_candidate_times: list[datetime] = field(default_factory=list)
    resident_paused: bool = False
    nonresident_support: int = 0
    last_nonresident_at: datetime | None = None
    last_proof: datetime | None = None
    target: str | None = None

    def pause(self) -> None:
        self.resident_paused = True
        self.verified = False
        self.samples.clear()
        self.nonresident_support = 0
        self.last_nonresident_at = None


class VisitorMemory:
    def __init__(
        self,
        config: VisitorConfig,
        store: VisitorStore,
        *,
        timezone: str = "UTC",
        is_nonresident: Callable[[FaceEmbedding], bool],
        new_id: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        self.config, self.store, self.new_id = config, store, new_id
        self.timezone = ZoneInfo(timezone)
        self.is_nonresident = is_nonresident
        self._visits: dict[str, _Visit] = {}
        self._last: FrameContext | None = None
        self.diagnostics: dict[str, str] = {}

    def process(
        self,
        context: FrameContext,
        tracks: tuple[PersonTrack, ...],
        events: tuple[PersonEvent, ...],
        observations: dict[str, VisitorObservation],
        blocked: set[str],
        identities: dict[str, IdentityMatch],
        *,
        resident_evidence: dict[str, ResidentEvidence] | None = None,
        face_diagnostics: dict[str, str] | None = None,
    ) -> tuple[dict[str, VisitorMatch], tuple[VisitorEvent, ...]]:
        if not self.config.enabled:
            return {}, ()
        self.diagnostics = {
            t.track_id: (face_diagnostics or {}).get(t.track_id, "waiting for face observation")
            for t in tracks
        }
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
                    visitor_match(profile, self.config),
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
                    if not visit.resident_verified and not visit.resident_paused:
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
                self.diagnostics[track_id] = "waiting for semantic ENTERED"
                continue
            identity = identities.get(track_id, IdentityMatch(IdentityState.UNKNOWN))
            if identity.state == IdentityState.RESIDENT:
                visit.resident_verified = True
                visit.pause()
            if visit.resident_verified:
                self.diagnostics[track_id] = "blocked resident confirmed"
                continue
            if visit.resident_candidate_count >= self.config.resident_candidate_confirmations:
                self.diagnostics[track_id] = "blocked repeated strong resident evidence"
                continue
            if track.continuity_epoch != visit.epoch or track.recently_lost:
                visit.epoch, visit.verified = track.continuity_epoch, False
                visit.samples.clear()
            if not track.visible or not track.confirmed:
                visit.samples.clear()
                self.diagnostics[track_id] = "waiting for visible confirmed track"
                continue
            evidence = (resident_evidence or {}).get(track_id)
            if (
                track_id in blocked
                or identity.state == IdentityState.CANDIDATE
                or evidence is not None
            ):
                visit.pause()
                # Only fresh, spaced comparisons count; carried CANDIDATE states do not.
                if evidence is not None:
                    if not evidence.strong or visit.resident_candidate_id != evidence.resident_id:
                        visit.resident_candidate_times.clear()
                    visit.resident_candidate_id = evidence.resident_id
                    visit.resident_candidate_times = [
                        t
                        for t in visit.resident_candidate_times
                        if (now - t).total_seconds()
                        <= self.config.resident_candidate_window_seconds
                    ]
                    if evidence.strong and (
                        not visit.resident_candidate_times
                        or (now - visit.resident_candidate_times[-1]).total_seconds()
                        >= self.config.observation_interval_seconds
                    ):
                        visit.resident_candidate_times.append(now)
                    visit.resident_candidate_count = len(visit.resident_candidate_times)
                detail = f" similarity {evidence.similarity:.2f}" if evidence else ""
                self.diagnostics[track_id] = (
                    "blocked repeated strong resident evidence"
                    if visit.resident_candidate_count
                    >= self.config.resident_candidate_confirmations
                    else f"paused possible resident{detail}"
                )
                continue
            observation = observations.get(track_id)
            if (
                observation is None
                or observation.detector_confidence < self.config.min_detector_confidence
            ):
                if observation is not None:
                    self.diagnostics[track_id] = (
                        f"rejected detector confidence {observation.detector_confidence:.2f} "
                        f"< {self.config.min_detector_confidence:.2f}"
                    )
                if (
                    visit.verified
                    and visit.visitor_id in profiles
                    and visit.last_proof
                    and (now - visit.last_proof).total_seconds() <= self.config.carry_seconds
                ):
                    results[track_id] = visitor_match(profiles[visit.visitor_id], self.config)
                continue
            embedding = observation.embedding
            if not self.is_nonresident(embedding):
                visit.pause()
                self.diagnostics[track_id] = "paused possible resident comparison"
                continue
            if visit.resident_paused:
                if (
                    visit.last_nonresident_at is not None
                    and (now - visit.last_nonresident_at).total_seconds()
                    > self.config.observation_window_seconds
                ):
                    visit.nonresident_support = 0
                if (
                    visit.last_nonresident_at is None
                    or (now - visit.last_nonresident_at).total_seconds()
                    >= self.config.observation_interval_seconds
                ):
                    visit.nonresident_support += 1
                    visit.last_nonresident_at = now
                self.diagnostics[track_id] = (
                    f"recovering nonresident {visit.nonresident_support}/"
                    f"{self.config.nonresident_recovery_observations}"
                )
                if visit.nonresident_support >= self.config.nonresident_recovery_observations:
                    visit.resident_paused = False
                    visit.resident_candidate_times.clear()
                    visit.resident_candidate_count = 0
                # Recovery evidence is not reused as visitor confirmation evidence.
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
                    self.diagnostics[track_id] = "paused ambiguous visitor similarity"
                    continue
                selected = scores[0][1]
            if visit.visitor_id is not None and selected != visit.visitor_id:
                self.diagnostics[track_id] = "paused conflicting visitor identity"
                visit.samples.clear()
                visit.verified = False
                continue
            if selected is not None and any(
                v.visitor_id == selected for k, v in self._visits.items() if k != track_id
            ):
                self.diagnostics[track_id] = (
                    "paused visitor profile already active on another track"
                )
                visit.samples.clear()
                continue
            if visit.verified and visit.visitor_id in profiles:
                state = visitor_match(profiles[visit.visitor_id], self.config).state
                self.diagnostics[track_id] = f"confirmed {state}"
                visit.last_proof = now
                profile = profiles[visit.visitor_id]
                # Bound writes during long visits while keeping retention tied to evidence.
                if (now - profile.last_seen_at).total_seconds() >= 3600:
                    profile = replace(profile, last_seen_at=now)
                    self.store.update(profile.visitor_id, _replace_with(profile))
                    profiles[profile.visitor_id] = profile
                results[track_id] = visitor_match(profiles[visit.visitor_id], self.config)
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
            self.diagnostics[track_id] = (
                f"candidate {len(visit.samples)}/{self.config.required_observations}"
            )
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
                    visit.pause()
                    self.diagnostics[track_id] = "paused final centroid plausibly resident"
                    results[track_id] = VisitorMatch()
                    continue
                if selected is None and any(
                    face_cosine(centroid, p.template)
                    >= self.config.match_similarity - self.config.ambiguity_margin
                    for p in profiles.values()
                ):
                    self.diagnostics[track_id] = "paused ambiguous visitor centroid"
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
                self.diagnostics[track_id] = "paused simultaneous visitor conflict"
                results[track_id] = VisitorMatch()
                continue
            visit = self._visits[track_id]
            if selected is None:
                profile = VisitorProfile(
                    self.new_id(),
                    centroid,
                    visit.entered_at,
                    now,
                    visit.entered_at,
                    last_visit_local_date=now.astimezone(self.timezone).date(),
                )
                self.store.add(profile)
                emit("VISITOR_FIRST_SEEN", track_id, profile)
            else:
                profile = profiles[selected]
                if visit.visitor_id is None:
                    local_date = now.astimezone(self.timezone).date()
                    previous_days = profile.distinct_visit_days
                    profile = replace(
                        profile,
                        session_count=profile.session_count + 1,
                        distinct_visit_days=previous_days
                        + (local_date > profile.last_visit_local_date),
                        last_visit_local_date=max(local_date, profile.last_visit_local_date),
                        last_visit_at=max(profile.last_visit_at, visit.entered_at),
                        last_seen_at=max(profile.last_seen_at, now),
                    )
                    self.store.update(selected, _replace_with(profile))
                    emit("VISITOR_RECOGNIZED", track_id, profile)
                    for threshold, kind in (
                        (self.config.recurring_distinct_days, "VISITOR_BECAME_RECURRING"),
                        (self.config.frequent_distinct_days, "VISITOR_BECAME_FREQUENT"),
                    ):
                        if previous_days < threshold <= profile.distinct_visit_days:
                            emit(kind, track_id, profile)
            profiles[profile.visitor_id] = profile
            visit.visitor_id, visit.verified, visit.last_proof = profile.visitor_id, True, now
            visit.samples.clear()
            results[track_id] = visitor_match(profile, self.config)
            self.diagnostics[track_id] = f"confirmed {results[track_id].state}"
        return results, tuple(emitted)
