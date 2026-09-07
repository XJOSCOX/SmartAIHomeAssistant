"""PortAudio devices and bounded in-memory PCM capture/playback; never record files."""

import importlib
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from queue import Empty, Full, Queue
from threading import Event, Lock
from typing import Protocol, cast

from jake.voice_config import AudioConfig
from jake.voice_domain import AudioChunk, SpeechResult, VoiceError


@dataclass(frozen=True)
class AudioDevice:
    index: int
    description: str
    host_api: str
    input_channels: int
    output_channels: int
    default_sample_rate: float


def devices() -> tuple[AudioDevice, ...]:
    try:
        sd = importlib.import_module("sounddevice")
        hosts = sd.query_hostapis()
        return tuple(
            AudioDevice(
                i,
                d["name"],
                hosts[d["hostapi"]]["name"],
                d["max_input_channels"],
                d["max_output_channels"],
                d["default_samplerate"],
            )
            for i, d in enumerate(sd.query_devices())
        )
    except Exception as exc:
        raise VoiceError(
            "Audio device discovery failed; install voice extra and PortAudio"
        ) from exc


class NativeStream(Protocol):
    def start(self) -> object: ...
    def stop(self) -> object: ...
    def abort(self) -> object: ...
    def close(self) -> object: ...
    def write(self, data: bytes) -> object: ...


class SoundDeviceInput:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.queue: Queue[AudioChunk] = Queue(maxsize=config.queue_chunks)
        self.stream: NativeStream | None = None
        self.sequence = 0
        self.dropped_chunks = 0
        self._closed = Event()
        self._finished = Event()

    def _callback(self, data: bytes, frames: int, timing: object, status: object) -> None:
        self.sequence += 1
        if self._closed.is_set():
            return
        if status:
            self.sequence += 1  # Signal driver discontinuity to the segmenter.
        if frames != 480 or len(data) != 960:
            self.dropped_chunks += 1
            return
        chunk = AudioChunk(
            self.sequence, datetime.now(UTC) - timedelta(milliseconds=30), bytes(data)
        )
        try:
            self.queue.put_nowait(chunk)
        except Full:
            self.dropped_chunks += 1
            with suppress(Empty):
                self.queue.get_nowait()
            with suppress(Full):
                self.queue.put_nowait(chunk)

    def start(self) -> None:
        if self.stream is not None or self._closed.is_set():
            raise VoiceError("Audio input is single-use")
        if not self.config.enabled:
            raise VoiceError("Audio is disabled")
        try:
            self.stream = cast(
                NativeStream,
                importlib.import_module("sounddevice").RawInputStream(
                    samplerate=16000,
                    channels=1,
                    dtype="int16",
                    blocksize=480,
                    device=self.config.input_device,
                    callback=self._callback,
                    finished_callback=self._finished.set,
                ),
            )
            self.stream.start()
        except Exception as exc:
            self.close()
            raise VoiceError("Cannot open configured microphone at 16kHz mono") from exc

    def read(self, timeout: float) -> AudioChunk | None:
        if self._finished.is_set() and not self._closed.is_set():
            raise VoiceError("Microphone stream stopped unexpectedly")
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def discard(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def close(self) -> None:
        self._closed.set()
        stream, self.stream = self.stream, None
        try:
            if stream is not None:
                try:
                    stream.abort()
                finally:
                    stream.close()
        finally:
            self.discard()


class SoundDeviceOutput:
    def __init__(self, config: AudioConfig) -> None:
        self.device = config.output_device
        self._lock = Lock()
        self._stream: NativeStream | None = None

    def play(self, result: SpeechResult, cancel: Event) -> None:
        stream = cast(
            NativeStream,
            importlib.import_module("sounddevice").RawOutputStream(
                samplerate=result.sample_rate,
                channels=1,
                dtype="int16",
                device=self.device,
            ),
        )
        try:
            with self._lock:
                self._stream = stream
            if cancel.is_set():
                return
            stream.start()
            block = max(2, result.sample_rate // 50 * 2)
            for offset in range(0, len(result.pcm), block):
                if cancel.is_set():
                    break
                stream.write(result.pcm[offset : offset + block])
            if not cancel.is_set():
                stream.stop()  # Drain playback before lifting recognition suppression.
        except Exception as exc:
            if not cancel.is_set():
                raise VoiceError("Speaker playback failed") from exc
        finally:
            with self._lock:
                self._stream = None
            stream.close()

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.abort()
