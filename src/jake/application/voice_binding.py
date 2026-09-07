"""Perception-to-voice metadata bridge. Never exports frames or embeddings."""

from datetime import datetime

from jake.domain import EventKind, PersonEvent, PersonTrack
from jake.identity_domain import IdentityMatch, IdentityState
from jake.visitor_domain import VisitorMatch
from jake.voice_domain import VisualContext, VisualPerson


class VisualVoiceBridge:
    def __init__(self) -> None:
        self._arrivals: set[str] = set()

    def observe(
        self,
        at: datetime,
        tracks: tuple[PersonTrack, ...],
        events: tuple[PersonEvent, ...],
        identities: dict[str, IdentityMatch],
        visitors: dict[str, VisitorMatch],
    ) -> VisualContext:
        active = {t.track_id for t in tracks}
        self._arrivals.intersection_update(active)
        for event in events:
            if event.kind == EventKind.PERSON_ENTERED:
                self._arrivals.add(event.track_id)
            elif event.kind == EventKind.PERSON_LEFT:
                self._arrivals.discard(event.track_id)
        people = []
        for track in tracks:
            identity = identities.get(track.track_id, IdentityMatch(IdentityState.UNKNOWN))
            visitor = visitors.get(track.track_id)
            people.append(
                VisualPerson(
                    track.track_id,
                    track.visible,
                    track.confirmed,
                    track.track_id in self._arrivals,
                    str(identity.state),
                    identity.resident_id if identity.state == IdentityState.RESIDENT else None,
                    visitor.visitor_id if visitor else None,
                    str(visitor.state) if visitor else "UNKNOWN",
                    identity.display_name if identity.state == IdentityState.RESIDENT else None,
                )
            )
        return VisualContext(at, tuple(people))
