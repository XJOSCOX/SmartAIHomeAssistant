"""Strict opt-in local speech settings; no model initialization."""

from dataclasses import dataclass
from math import isfinite


def number(value: object, name: str, low: float, high: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite and between {low} and {high}")


def integer(value: object, name: str, low: int, high: int) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    number(value, name, low, high)


@dataclass(frozen=True, slots=True)
class AudioConfig:
    enabled: bool = False
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 16000
    channels: int = 1
    queue_chunks: int = 16

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("audio.enabled must be boolean")
        for value in (self.input_device, self.output_device):
            if value is not None and not (
                type(value) is int
                and value >= 0
                or isinstance(value, str)
                and value.strip()
                and value.isprintable()
            ):
                raise ValueError("audio devices must be nonnegative indexes or descriptions")
        if type(self.sample_rate) is not int or self.sample_rate != 16000:
            raise ValueError("Phase 3A audio.sample_rate must be 16000")
        if type(self.channels) is not int or self.channels != 1:
            raise ValueError("Phase 3A requires mono audio")
        integer(self.queue_chunks, "audio.queue_chunks", 1, 64)


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    language: str = "en"
    vad_mode: int = 2
    minimum_speech_ms: int = 240
    silence_timeout_ms: int = 600
    max_utterance_seconds: float = 15.0
    pre_roll_ms: int = 180
    echo_guard_ms: int = 300

    def __post_init__(self) -> None:
        if (
            not isinstance(self.language, str)
            or not self.language.isascii()
            or not self.language.isalpha()
            or not 2 <= len(self.language) <= 3
        ):
            raise ValueError("speech.language must be a short language code")
        integer(self.vad_mode, "speech.vad_mode", 0, 3)
        integer(self.minimum_speech_ms, "minimum_speech_ms", 30, 5000)
        integer(self.silence_timeout_ms, "silence_timeout_ms", 30, 5000)
        integer(self.pre_roll_ms, "pre_roll_ms", 0, 1000)
        integer(self.echo_guard_ms, "echo_guard_ms", 0, 3000)
        number(self.max_utterance_seconds, "max_utterance_seconds", 1, 30)
        if (
            self.minimum_speech_ms + self.pre_roll_ms + self.silence_timeout_ms
            > self.max_utterance_seconds * 1000
        ):
            raise ValueError("utterance limit must accommodate speech, pre-roll and silence")


@dataclass(frozen=True, slots=True)
class STTConfig:
    model: str = "models/speech/faster-whisper-base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip() or "://" in self.model:
            raise ValueError("stt.model must be a local model directory")
        if self.device not in ("cpu", "cuda") or self.compute_type not in (
            "int8",
            "float32",
            "float16",
            "int8_float16",
        ):
            raise ValueError("unsupported STT device/compute type")
        if self.device == "cpu" and self.compute_type not in ("int8", "float32"):
            raise ValueError("CPU STT requires int8 or float32")
        integer(self.cpu_threads, "stt.cpu_threads", 1, 32)


@dataclass(frozen=True, slots=True)
class TTSConfig:
    engine: str = "piper"
    voice: str = "models/speech/en_US-lessac-medium.onnx"

    def __post_init__(self) -> None:
        if self.engine != "piper":
            raise ValueError("unsupported tts.engine")
        if (
            not isinstance(self.voice, str)
            or "://" in self.voice
            or not self.voice.endswith(".onnx")
        ):
            raise ValueError("tts.voice must be a local ONNX file")


@dataclass(frozen=True, slots=True)
class InteractionConfig:
    resident_greeting_enabled: bool = False
    visitor_greeting_enabled: bool = False
    greeting_cooldown_minutes: float = 60
    conversation_timeout_seconds: float = 60
    visual_context_max_age_seconds: float = 2
    max_turns: int = 20

    def __post_init__(self) -> None:
        if any(
            type(v) is not bool
            for v in (self.resident_greeting_enabled, self.visitor_greeting_enabled)
        ):
            raise ValueError("greeting options must be boolean")
        number(self.greeting_cooldown_minutes, "greeting cooldown", 1, 1440)
        number(self.conversation_timeout_seconds, "conversation timeout", 1, 3600)
        number(self.visual_context_max_age_seconds, "visual context age", 0.1, 10)
        integer(self.max_turns, "max_turns", 2, 100)
