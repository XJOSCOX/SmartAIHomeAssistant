"""Bounded voice workers, half-duplex playback and metadata-only vision handoff."""

from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic

from jake.conversation import ConversationManager, GreetingPolicy, associate
from jake.voice_config import AudioConfig, InteractionConfig, SpeechConfig
from jake.voice_domain import SpeechRequest, SpeechResult, VisualContext, VoiceError, VoiceEvent
from jake.voice_ports import (
    AudioInput,
    AudioOutput,
    SpeechRecognizer,
    SpeechSynthesizer,
    VoiceActivityDetector,
)
from jake.voice_segmentation import UtteranceSegmenter


class AsyncSpeech:
    """Four pending requests maximum, including synthesis/playback in progress."""

    def __init__(self, synthesizer: SpeechSynthesizer, output: AudioOutput) -> None:
        self.synthesizer, self.output = synthesizer, output
        self._queue: Queue[tuple[SpeechRequest, Event]] = Queue(maxsize=4)
        self._pending: set[Event] = set()
        self._lock = Lock()
        self._closed = Event()
        self._thread: Thread | None = None
        self.generation_ms = 0.0
        self.completed = 0
        self.error: str | None = None

    def start(self) -> None:
        if self._thread is not None or self._closed.is_set():
            raise VoiceError("Speech worker is single-use")
        self._thread = Thread(target=self._run, name="jake-tts")
        self._thread.start()

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def speak(self, text: str) -> bool:
        request = SpeechRequest(text)
        with self._lock:
            if self._closed.is_set() or self._thread is None or len(self._pending) >= 4:
                return False
            cancel = Event()
            self._pending.add(cancel)
            self._queue.put_nowait((request, cancel))
            return True

    def stop(self) -> None:
        with self._lock:
            for cancel in self._pending:
                cancel.set()
            while True:
                try:
                    _, cancel = self._queue.get_nowait()
                    self._pending.discard(cancel)
                except Empty:
                    break
        self.output.stop()

    def close(self) -> None:
        self._closed.set()
        try:
            self.stop()
        finally:
            if self._thread is not None:
                self._thread.join()  # Native synthesis must finish before ownership is dropped.

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                request, cancel = self._queue.get(timeout=0.05)
            except Empty:
                continue
            result: SpeechResult | None = None
            try:
                if not cancel.is_set():
                    result = self.synthesizer.synthesize(request)
                    self.generation_ms = result.processing_ms
                    if not cancel.is_set():
                        self.output.play(result, cancel)
                        if not cancel.is_set():
                            self.completed += 1
            except Exception as exc:
                self.error = type(exc).__name__  # Never retain exception payloads.
            finally:
                with self._lock:
                    self._pending.discard(cancel)
                result = None
                del request


@dataclass(frozen=True)
class VoiceMetrics:
    input_status: str = "stopped"
    vad_state: str = "IDLE"
    utterance_seconds: float = 0
    stt_ms: float = 0
    tts_ms: float = 0
    stt_real_time_factor: float | None = None


class VoiceService:
    """Owns conversation state on one worker; vision publishes metadata only.

    Capturing uses the AudioInput adapter's bounded native callback queue. STT is
    serial on the voice worker. Synthesis/playback runs on its own worker. During
    STT pending chunks are discarded; TTS plus its tail guard drains/suppresses VAD.
    """

    def __init__(
        self,
        audio: AudioConfig,
        speech: SpeechConfig,
        interaction: InteractionConfig,
        source: AudioInput,
        vad: VoiceActivityDetector,
        recognizer: SpeechRecognizer,
        synthesizer: SpeechSynthesizer,
        output: AudioOutput,
    ) -> None:
        self.audio, self.speech, self.interaction = audio, speech, interaction
        self.source, self.recognizer = source, recognizer
        self.segmenter = UtteranceSegmenter(speech, vad)
        self.conversations = ConversationManager(interaction)
        self.greetings = GreetingPolicy(interaction)
        self.speaker = AsyncSpeech(synthesizer, output)
        self.metrics = VoiceMetrics()
        self._lock = Lock()
        self._context: VisualContext | None = None
        self._events: deque[VoiceEvent] = deque(maxlen=128)
        self._cancel = Event()
        self._thread: Thread | None = None
        self._echo_until = 0.0

    def publish(self, context: VisualContext) -> None:
        # Latest snapshot, not a queue. No audio/model work on the perception thread.
        with self._lock:
            self._context = context

    def drain_events(self) -> tuple[VoiceEvent, ...]:
        with self._lock:
            events = tuple(self._events)
            self._events.clear()
            return events

    def _emit(self, event: VoiceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def start(self) -> None:
        if not self.audio.enabled:
            raise VoiceError("Audio is disabled; opt in with audio.enabled=true")
        if self._thread is not None or self._cancel.is_set():
            raise VoiceError("Voice sessions are single-use")
        try:
            self.source.start()
            self.speaker.start()
            self._thread = Thread(target=self._run, name="jake-voice")
            self._thread.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._cancel.set()
        try:
            self.source.close()
        finally:
            try:
                self.speaker.close()
            finally:
                if self._thread is not None:
                    self._thread.join()
                self.segmenter.reset()
                self.conversations.end(datetime.now(UTC))
                with self._lock:
                    self._context = None
                    self._events.clear()
                self.metrics = VoiceMetrics()
                self.greetings = GreetingPolicy(self.interaction)

    def _run(self) -> None:
        person = None
        completed = 0
        self.metrics = VoiceMetrics(input_status="listening")
        try:
            while not self._cancel.is_set():
                now = datetime.now(UTC)
                self.conversations.expire(now)
                while self.conversations.events:
                    self._emit(self.conversations.events.popleft())
                if self.speaker.error:
                    raise VoiceError("Local speech playback worker failed")
                if self.speaker.completed != completed:
                    for _ in range(self.speaker.completed - completed):
                        self._emit(VoiceEvent("JAKE_SPOKE", now))
                    completed = self.speaker.completed
                    self._echo_until = monotonic() + self.speech.echo_guard_ms / 1000
                    self.source.discard()
                    self.segmenter.reset()
                self.metrics = replace(self.metrics, tts_ms=self.speaker.generation_ms)
                if self.speaker.is_speaking:
                    self._echo_until = monotonic() + self.speech.echo_guard_ms / 1000
                if self.speaker.is_speaking or monotonic() < self._echo_until:
                    self.source.discard()
                    self.segmenter.reset()
                    self.metrics = replace(self.metrics, vad_state="SUPPRESSED")
                    self._cancel.wait(0.02)
                    continue
                with self._lock:
                    context = self._context
                if self.segmenter.state == "IDLE" and context:
                    for greeting in self.greetings.observe(context, now):
                        self.speaker.speak(greeting.text)
                    if self.speaker.is_speaking:
                        continue
                self.metrics = replace(self.metrics, vad_state=self.segmenter.state)
                chunk = self.source.read(0.05)
                if chunk is None:
                    continue
                # Do not transcribe audio captured during STT, playback or the tail.
                if self.speaker.is_speaking:
                    self.segmenter.reset()
                    continue
                segment = self.segmenter.push(chunk)
                if self.segmenter.started_this_chunk:
                    person = associate(
                        context, chunk.started_at, self.interaction.visual_context_max_age_seconds
                    )
                    self._emit(
                        VoiceEvent(
                            "SPEECH_STARTED",
                            chunk.started_at,
                            track_id=person.track_id if person else None,
                        )
                    )
                self.metrics = replace(self.metrics, vad_state=self.segmenter.state)
                if segment is not None:
                    self.metrics = replace(
                        self.metrics,
                        vad_state="TRANSCRIBING",
                        utterance_seconds=segment.duration_seconds,
                    )
                    result = self.recognizer.transcribe(segment)
                    self.source.discard()
                    if self._cancel.is_set():
                        break
                    self.metrics = replace(
                        self.metrics,
                        stt_ms=result.processing_ms,
                        stt_real_time_factor=result.processing_ms / 1000 / segment.duration_seconds,
                        vad_state="IDLE",
                    )
                    response = self.conversations.respond(result, person)
                    if response is not None:
                        self.speaker.speak(response.text)
                    del result, response, segment
        except Exception as exc:
            self.metrics = replace(self.metrics, input_status="failed:" + type(exc).__name__)
            self._cancel.set()
        finally:
            with suppress(Exception):
                self.source.close()
            with suppress(Exception):
                self.speaker.stop()
            self.source.discard()
            self.segmenter.reset()
            self.conversations.end(datetime.now(UTC))
