"""Metadata-only semantic events observing the complete active track set."""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4, uuid5

from jake.config import EventConfig
from jake.domain import EventKind, FrameContext, PersonEvent, PersonTrack


class EventError(ValueError):
    """Invalid input to a single-camera event session; no event state was changed."""


@dataclass(frozen=True, slots=True)
class _Presence:
    entered_at: datetime
    last_emitted_at: datetime


class PersonEventGenerator:
    """Observe track transitions, independently of matching and expiration policy.

    Call serially once per processed frame, including empty active sets. A missing
    ID means expiration, never a missed detection. Track IDs must not be reused.
    UUID namespaces isolate sessions; inject a fixed namespace for repeatable replay.
    """

    def __init__(self, config: EventConfig, *, session_id: UUID | None = None) -> None:
        self._config = config
        self._session_id = session_id if session_id is not None else uuid4()
        self._presence: dict[str, _Presence] = {}
        self._closed: set[str] = set()
        self._last_context: FrameContext | None = None

    def generate(
        self, context: FrameContext, tracks: tuple[PersonTrack, ...]
    ) -> tuple[PersonEvent, ...]:
        now = context.captured_at.astimezone(UTC)
        previous = self._last_context
        if previous is not None:
            if context.camera_id != previous.camera_id:
                raise EventError("event generator cannot mix camera sessions")
            if context.sequence <= previous.sequence:
                raise EventError("event frame sequences must be strictly increasing")
            if now < previous.captured_at.astimezone(UTC):
                raise EventError("event capture timestamps must not move backwards")
        current = {track.track_id: track for track in tracks}
        if len(current) != len(tracks):
            raise EventError("duplicate active track ID")
        if self._closed.intersection(current):
            raise EventError("expired track IDs must not be reused within an event session")

        presence = self._presence.copy()
        expired = presence.keys() - current.keys()
        events = []
        # Lexical ID ordering makes output independent of input tuple ordering.
        for track_id in sorted(expired):
            departed = presence.pop(track_id)
            events.append(
                self._event(EventKind.PERSON_LEFT, context, track_id, departed.entered_at)
            )
        for track_id, track in sorted(current.items()):
            if not track.confirmed or not track.visible:
                continue
            state = presence.get(track_id)
            if state is None:
                presence[track_id] = _Presence(context.captured_at, now)
                events.append(
                    self._event(EventKind.PERSON_ENTERED, context, track_id, context.captured_at)
                )
            elif (
                now - state.last_emitted_at
            ).total_seconds() >= self._config.present_interval_seconds:
                presence[track_id] = _Presence(state.entered_at, now)
                events.append(
                    self._event(EventKind.PERSON_PRESENT, context, track_id, state.entered_at)
                )
        self._presence = presence
        self._closed.update(expired)
        self._last_context = context
        return tuple(events)

    def _event(
        self, kind: EventKind, context: FrameContext, track_id: str, entered_at: datetime
    ) -> PersonEvent:
        # repr of a tuple preserves boundaries even for arbitrary camera/track names.
        name = repr((context.camera_id, context.sequence, track_id, kind.value))
        return PersonEvent(str(uuid5(self._session_id, name)), kind, context, track_id, entered_at)
