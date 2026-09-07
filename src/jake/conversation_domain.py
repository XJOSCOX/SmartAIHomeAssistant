"""Model-independent immutable conversation metadata and local reasoning port."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from threading import Event
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


class ConversationErrorCode(StrEnum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    BACKEND_IMPORT_FAILED = "BACKEND_IMPORT_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_LOAD_TIMEOUT = "MODEL_LOAD_TIMEOUT"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    CHAT_TEMPLATE_FAILED = "CHAT_TEMPLATE_FAILED"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    TOKEN_BUDGET_FAILED = "TOKEN_BUDGET_FAILED"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    GENERATION_FAILED = "GENERATION_FAILED"
    INVALID_JSON = "INVALID_JSON"
    INVALID_RESPONSE_SHAPE = "INVALID_RESPONSE_SHAPE"
    INVALID_REPLY = "INVALID_REPLY"
    POLICY_REJECTED = "POLICY_REJECTED"
    CANCELLED = "CANCELLED"
    RESET_FAILED = "RESET_FAILED"
    SCHEMA_FAILED = "SCHEMA_FAILED"


class ConversationFailure(RuntimeError):
    """Only enum metadata is retained; never wrap unsafe native exception payloads."""

    def __init__(self, code: ConversationErrorCode) -> None:
        self.code = code
        super().__init__(code.value)

    @property
    def permanent(self) -> bool:
        return self.code in {
            ConversationErrorCode.MODEL_NOT_FOUND,
            ConversationErrorCode.BACKEND_IMPORT_FAILED,
            ConversationErrorCode.MODEL_LOAD_FAILED,
            ConversationErrorCode.MODEL_LOAD_TIMEOUT,
            ConversationErrorCode.CHAT_TEMPLATE_FAILED,
            ConversationErrorCode.RESET_FAILED,
            ConversationErrorCode.SCHEMA_FAILED,
        }


@dataclass(frozen=True, slots=True)
class ModelStatus:
    state: str = "not_loaded"
    load_ms: float | None = None
    last_error_code: ConversationErrorCode | None = None
    chat_template_ok: bool = False


class ConversationModel(Protocol):
    @property
    def status(self) -> ModelStatus: ...
    def load(self) -> None: ...
    def generate(
        self, request: ConversationRequest, *, cancel: Event | None = None
    ) -> ConversationResponse: ...
    def cancel_current(self) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...
