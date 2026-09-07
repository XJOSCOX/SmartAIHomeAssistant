"""Bounded PCM utterance segmentation; speech decisions belong to a VAD port."""

from collections import deque
from datetime import timedelta

from jake.voice_config import SpeechConfig
from jake.voice_domain import AudioChunk, SpeechSegment
from jake.voice_ports import VoiceActivityDetector


class UtteranceSegmenter:
    def __init__(self, config: SpeechConfig, vad: VoiceActivityDetector) -> None:
        self.config, self.vad = config, vad
        self._pre: deque[AudioChunk] = deque(maxlen=config.pre_roll_ms // 30 + 1)
        self._chunks: list[AudioChunk] = []
        self._last: AudioChunk | None = None
        self._speech_ms = self._silence_ms = 0
        self.state = "IDLE"
        self.started_this_chunk = False

    @property
    def buffered_chunks(self) -> int:
        return len(self._chunks) + len(self._pre)

    def reset(self) -> None:
        self._pre.clear()
        self._chunks.clear()
        self._last = None
        self._speech_ms = self._silence_ms = 0
        self.state = "IDLE"
        self.started_this_chunk = False

    def push(self, chunk: AudioChunk) -> SpeechSegment | None:
        self.started_this_chunk = False
        if self._last is not None and (
            chunk.sequence != self._last.sequence + 1
            or abs((chunk.started_at - self._last.started_at).total_seconds() - 0.03) > 0.1
        ):
            self.reset()  # A dropped chunk is a discontinuity, never splice speech across it.
        self._last = chunk
        speech = self.vad.is_speech(chunk)
        if self.state == "IDLE":
            self._pre.append(chunk)
            if not speech:
                return None
            self._chunks = list(self._pre)
            self._pre.clear()
            self.state = "LISTENING"
            self.started_this_chunk = True
        else:
            self._chunks.append(chunk)
        if speech:
            self._speech_ms += 30
            self._silence_ms = 0
        else:
            self._silence_ms += 30
        if self._silence_ms < self.config.silence_timeout_ms and len(self._chunks) < int(
            self.config.max_utterance_seconds / 0.03
        ):
            return None
        self.state = "FINALIZING"
        result = None
        if self._speech_ms >= self.config.minimum_speech_ms:
            result = SpeechSegment(
                self._chunks[0].started_at,
                chunk.started_at + timedelta(milliseconds=30),
                b"".join(c.pcm for c in self._chunks),
            )
        self.reset()
        return result
