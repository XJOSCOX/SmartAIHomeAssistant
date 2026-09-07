"""Replaceable local audio/model boundaries; no native packages in these ports."""

from threading import Event
from typing import Protocol

from jake.voice_domain import (
    AudioChunk,
    SpeechRecognitionResult,
    SpeechRequest,
    SpeechResult,
    SpeechSegment,
)


class AudioInput(Protocol):
    def start(self) -> None: ...
    def read(self, timeout: float) -> AudioChunk | None: ...
    def discard(self) -> None: ...
    def close(self) -> None: ...


class VoiceActivityDetector(Protocol):
    def is_speech(self, chunk: AudioChunk) -> bool: ...


class SpeechRecognizer(Protocol):
    def transcribe(self, segment: SpeechSegment) -> SpeechRecognitionResult: ...


class SpeechSynthesizer(Protocol):
    def synthesize(self, request: SpeechRequest) -> SpeechResult: ...


class AudioOutput(Protocol):
    def play(self, result: SpeechResult, cancel: Event) -> None: ...
    def stop(self) -> None: ...
