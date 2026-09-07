"""Model-independent immutable conversation metadata and local reasoning port."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SpeakerContext:
    association_status: str = "unresolved"
    track_id: str | None = None
    identity_state: str = "UNKNOWN"
    resident_id: str | None = None
    resident_name: str | None = field(default=None, repr=False)
    visitor_id: str | None = None
    visitor_state: str = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ScenePerson:
    track_id: str
    identity_state: str
    visitor_state: str


@dataclass(frozen=True, slots=True)
class ConversationContext:
    conversation_id: str
    timestamp: datetime
    speaker: SpeakerContext
    visible_people: tuple[ScenePerson, ...]
    visible_resident_count: int
    visible_visitor_count: int
    scene_observed_at: datetime | None
    history: tuple[tuple[str, str], ...] = field(repr=False)
    supported_actions: tuple[str, ...] = ("local conversation",)
    unsupported_actions: tuple[str, ...] = (
        "device control",
        "locks",
        "emergency calls",
        "internet",
        "historical memory",
        "occupancy disclosure",
        "acoustic speaker identification",
    )


@dataclass(frozen=True, slots=True)
class ConversationRequest:
    context: ConversationContext = field(repr=False)
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ConversationResponse:
    text: str = field(repr=False)
    generation_ms: float = 0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class ModelStatus:
    state: str = "not_loaded"
    load_ms: float | None = None


class ConversationModel(Protocol):
    @property
    def status(self) -> ModelStatus: ...
    def generate(self, request: ConversationRequest) -> ConversationResponse: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...
