"""Deterministic, session-memory conversation and conservative visual policies."""

import re
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from jake.voice_config import InteractionConfig
from jake.voice_domain import (
    SpeechRecognitionResult,
    SpeechRequest,
    VisualContext,
    VisualPerson,
    VoiceEvent,
)

RESPONSES = {
    "GREETING": "Hello. How can I help you?",
    "DELIVERY": "Thank you. You can leave the package by the door.",
    "WHO_ARE_YOU": "I'm Jake, the local home assistant.",
    "IS_ANYONE_HOME": "I can't share household occupancy information.",
    "GOODBYE": "Goodbye.",
    "UNKNOWN": "I can help with greetings and deliveries. What do you need?",
}


def intent(text: str) -> str:
    words = re.findall(r"[a-z]+", text.casefold())
    normalized = " ".join(words)
    if "who are you" in normalized:
        return "WHO_ARE_YOU"
    if "anyone home" in normalized or "anybody home" in normalized:
        return "IS_ANYONE_HOME"
    if set(words) & {"package", "delivery", "parcel"}:
        return "DELIVERY"
    if set(words) & {"goodbye", "bye"}:
        return "GOODBYE"
    if set(words) & {"hello", "hi", "hey"}:
        return "GREETING"
    return "UNKNOWN"


def associate(context: VisualContext | None, at: datetime, max_age: float) -> VisualPerson | None:
    if context is None or not 0 <= (at - context.at).total_seconds() <= max_age:
        return None
    visible = [p for p in context.people if p.visible]
    # An extra tentative person also makes acoustic attribution ambiguous.
    return visible[0] if len(visible) == 1 and visible[0].confirmed else None


@dataclass
class ConversationSession:
    conversation_id: str
    started_at: datetime
    last_activity_at: datetime
    association: VisualPerson | None
    turns: deque[tuple[str, str]] = field(repr=False)


class ConversationManager:
    def __init__(self, config: InteractionConfig) -> None:
        self.config = config
        self.session: ConversationSession | None = None
        self.events: deque[VoiceEvent] = deque(maxlen=128)

    def expire(self, at: datetime) -> None:
        if (
            self.session
            and (at - self.session.last_activity_at).total_seconds()
            >= self.config.conversation_timeout_seconds
        ):
            self.end(at)

    def end(self, at: datetime) -> None:
        if self.session:
            self.events.append(VoiceEvent("CONVERSATION_ENDED", at, self.session.conversation_id))
            self.session.turns.clear()
            self.session = None

    def respond(
        self, result: SpeechRecognitionResult, person: VisualPerson | None
    ) -> SpeechRequest | None:
        self.expire(result.ended_at)
        text = " ".join(result.text.split())
        if (
            not text
            or len(text) > 2000
            or not any(c.isalpha() for c in text)
            or not text.isprintable()
        ):
            return None
        if self.session and self.session.association != person:
            self.end(result.started_at)
        if self.session is None:
            self.session = ConversationSession(
                str(uuid4()),
                result.started_at,
                result.ended_at,
                person,
                deque(maxlen=self.config.max_turns),
            )
            self.events.append(
                VoiceEvent(
                    "CONVERSATION_STARTED",
                    result.started_at,
                    self.session.conversation_id,
                    person.track_id if person else None,
                )
            )
        self.session.last_activity_at = result.ended_at
        response = RESPONSES[intent(text)]
        self.session.turns.extend((("user", text), ("jake", response)))
        self.events.append(
            VoiceEvent(
                "SPEECH_RECOGNIZED",
                result.ended_at,
                self.session.conversation_id,
                person.track_id if person else None,
            )
        )
        if intent(text) == "GOODBYE":
            self.end(result.ended_at)
        return SpeechRequest(response)


class GreetingPolicy:
    def __init__(self, config: InteractionConfig) -> None:
        self.config = config
        self._greeted: set[str] = set()
        self._cooldowns: OrderedDict[str, datetime] = OrderedDict()

    def observe(self, context: VisualContext, now: datetime) -> tuple[SpeechRequest, ...]:
        if (
            not 0
            <= (now - context.at).total_seconds()
            <= self.config.visual_context_max_age_seconds
        ):
            return ()
        self._greeted.intersection_update(p.track_id for p in context.people)
        for key, at in list(self._cooldowns.items()):
            if (now - at).total_seconds() >= self.config.greeting_cooldown_minutes * 60:
                del self._cooldowns[key]
        greetings = []
        for person in context.people:
            if (
                not (person.visible and person.confirmed and person.semantic_arrival)
                or person.track_id in self._greeted
            ):
                continue
            key, text = "", ""
            if (
                self.config.resident_greeting_enabled
                and person.identity_state == "RESIDENT"
                and person.resident_id
                and person.display_name
            ):
                key = "resident:" + person.resident_id
                text = f"Welcome home, {person.display_name}."
            elif (
                self.config.visitor_greeting_enabled
                and person.identity_state == "UNKNOWN"
                and person.visitor_id
                and person.visitor_state
                in ("FIRST_TIME_VISITOR", "RECURRING_VISITOR", "FREQUENT_VISITOR", "KNOWN_VISITOR")
            ):
                key = "visitor:" + person.visitor_id
                text = "Hello. I'm Jake, the home assistant. How can I help you?"
            if not key:
                continue
            self._greeted.add(person.track_id)
            if key in self._cooldowns:
                continue
            # Fail closed when the bounded cooldown cache is full: never evict a
            # live cooldown and inadvertently greet that person again.
            if len(self._cooldowns) >= 1024:
                continue
            self._cooldowns[key] = now
            greetings.append(SpeechRequest(text))
        return tuple(greetings)
