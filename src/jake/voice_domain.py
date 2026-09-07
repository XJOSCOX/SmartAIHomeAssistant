"""Local audio and conversation data. Payloads are excluded from repr/events."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite


class VoiceError(RuntimeError):
    """Local audio/speech failed; callers expose categories, not model payloads."""


def aware(at: datetime) -> None:
    if at.utcoffset() is None:
        raise ValueError("voice timestamps must be timezone-aware")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    sequence: int
    started_at: datetime
    pcm: bytes = field(repr=False)
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        aware(self.started_at)
        if (
            type(self.sequence) is not int
            or self.sequence < 0
            or self.sample_rate != 16000
            or not isinstance(self.pcm, bytes)
            or len(self.pcm) != 960
        ):
            raise ValueError("audio chunks must be 30ms of mono PCM16LE at 16kHz")


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    started_at: datetime
    ended_at: datetime
    pcm: bytes = field(repr=False)
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        aware(self.started_at)
        aware(self.ended_at)
        if (
            self.ended_at <= self.started_at
            or self.sample_rate != 16000
            or not isinstance(self.pcm, bytes)
            or not 0 < len(self.pcm) <= 960000
            or len(self.pcm) % 2
        ):
            raise ValueError("invalid bounded speech segment")

    @property
    def duration_seconds(self) -> float:
        return len(self.pcm) / (2 * self.sample_rate)


@dataclass(frozen=True, slots=True)
class SpeechRecognitionResult:
    text: str = field(repr=False)
    started_at: datetime
    ended_at: datetime
    processing_ms: float
    language: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        aware(self.started_at)
        aware(self.ended_at)
        if (
            self.ended_at < self.started_at
            or not isfinite(self.processing_ms)
            or self.processing_ms < 0
        ):
            raise ValueError("invalid recognition timing")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("invalid recognition confidence")


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text) > 500
            or not self.text.isprintable()
        ):
            raise ValueError("speech requests require 1-500 printable characters")


@dataclass(frozen=True, slots=True)
class SpeechResult:
    pcm: bytes = field(repr=False)
    sample_rate: int
    processing_ms: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pcm, bytes)
            or len(self.pcm) % 2
            or type(self.sample_rate) is not int
            or not 1 <= self.sample_rate <= 192000
            or not self.pcm
            or len(self.pcm) > self.sample_rate * 2 * 60
        ):
            raise ValueError("invalid bounded mono PCM16LE synthesis result")
        if not isfinite(self.processing_ms) or self.processing_ms < 0:
            raise ValueError("invalid synthesis timing")


@dataclass(frozen=True, slots=True)
class VisualPerson:
    track_id: str
    visible: bool = True
    confirmed: bool = True
    semantic_arrival: bool = False
    identity_state: str = "UNKNOWN"
    resident_id: str | None = None
    visitor_id: str | None = None
    visitor_state: str = "UNKNOWN"
    display_name: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class VisualContext:
    at: datetime
    people: tuple[VisualPerson, ...] = ()


@dataclass(frozen=True, slots=True)
class VoiceEvent:
    kind: str
    at: datetime
    conversation_id: str | None = None
    track_id: str | None = None
