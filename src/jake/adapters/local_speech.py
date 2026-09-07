"""Offline speech engines. No engine types escape these adapters."""

import importlib
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from jake.voice_config import SpeechConfig, STTConfig, TTSConfig
from jake.voice_domain import (
    AudioChunk,
    SpeechRecognitionResult,
    SpeechRequest,
    SpeechResult,
    SpeechSegment,
    VoiceError,
)


class VadModel(Protocol):
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool: ...


class WebRtcVAD:
    def __init__(self, config: SpeechConfig) -> None:
        try:
            self.model = cast(VadModel, importlib.import_module("webrtcvad").Vad(config.vad_mode))
        except Exception as exc:
            raise VoiceError("Local WebRTC VAD unavailable; install voice extra") from exc

    def is_speech(self, chunk: AudioChunk) -> bool:
        return self.model.is_speech(chunk.pcm, chunk.sample_rate)


class DecodedSegment(Protocol):
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


class DecodeInfo(Protocol):
    language: str


class WhisperEngine(Protocol):
    def transcribe(
        self, audio: NDArray[np.float32], **options: object
    ) -> tuple[Iterable[DecodedSegment], DecodeInfo]: ...


class FasterWhisperRecognizer:
    def __init__(self, config: STTConfig, speech: SpeechConfig) -> None:
        path = Path(config.model).expanduser().absolute()
        if not all(
            (path / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")
        ):
            raise VoiceError("STT requires a complete existing local CTranslate2 model directory")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        logging.getLogger("faster_whisper").setLevel(logging.ERROR)
        self.language = speech.language
        try:
            self.model = cast(
                WhisperEngine,
                importlib.import_module("faster_whisper").WhisperModel(
                    str(path),
                    device=config.device,
                    compute_type=config.compute_type,
                    cpu_threads=config.cpu_threads,
                    num_workers=1,
                    local_files_only=True,
                ),
            )
        except Exception as exc:
            raise VoiceError("Cannot initialize local STT model") from exc

    def transcribe(self, segment: SpeechSegment) -> SpeechRecognitionResult:
        started = perf_counter()
        audio = np.frombuffer(segment.pcm, dtype="<i2").astype(np.float32) / 32768.0
        try:
            decoded, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                max_new_tokens=256,
            )
            text = " ".join(
                item.text.strip()
                for item in decoded
                if item.no_speech_prob <= 0.6
                and item.avg_logprob >= -1.0
                and item.compression_ratio <= 2.4
            ).strip()
            if len(text) > 2000 or not text.isprintable():
                text = ""
            return SpeechRecognitionResult(
                text,
                segment.started_at,
                segment.ended_at,
                (perf_counter() - started) * 1000,
                info.language,
            )
        except Exception as exc:
            raise VoiceError("Local transcription failed") from exc


class PiperChunk(Protocol):
    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_int16_bytes: bytes


class PiperEngine(Protocol):
    use_tashkeel: bool

    def synthesize(self, text: str) -> Iterable[PiperChunk]: ...


class PiperSynthesizer:
    def __init__(self, config: TTSConfig) -> None:
        path = Path(config.voice).expanduser().absolute()
        if not path.is_file() or not Path(str(path) + ".json").is_file():
            raise VoiceError("Piper requires existing local .onnx and .onnx.json voice files")
        try:
            with Path(str(path) + ".json").open(encoding="utf-8") as stream:
                metadata = json.load(stream)
            if not isinstance(metadata, dict) or metadata.get("phoneme_type", "espeak") not in (
                "espeak",
                "text",
            ):
                raise VoiceError("Only local eSpeak/text Piper phonemizers are supported")
            importlib.import_module("onnxruntime").disable_telemetry_events()
            self.model = cast(
                PiperEngine,
                importlib.import_module("piper").PiperVoice.load(str(path), use_cuda=False),
            )
            # Some Piper language helpers fetch resources on first use. Keep only
            # bundled eSpeak/text paths, and disable optional Arabic diacritization.
            self.model.use_tashkeel = False
        except Exception as exc:
            raise VoiceError("Cannot initialize local Piper voice") from exc

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        started = perf_counter()
        chunks = []
        size, rate = 0, 0
        try:
            for chunk in self.model.synthesize(request.text):
                if (
                    chunk.sample_width != 2
                    or chunk.sample_channels != 1
                    or chunk.sample_rate <= 0
                    or rate not in (0, chunk.sample_rate)
                ):
                    raise VoiceError("Piper output must use a consistent mono PCM16 format")
                rate = chunk.sample_rate
                size += len(chunk.audio_int16_bytes)
                if size > rate * 2 * 60:
                    raise VoiceError("Speech output exceeds the 60-second safety bound")
                chunks.append(chunk.audio_int16_bytes)
            if not rate:
                raise VoiceError("Piper returned no audio")
            return SpeechResult(b"".join(chunks), rate, (perf_counter() - started) * 1000)
        except Exception as exc:
            raise VoiceError("Local speech synthesis failed") from exc
