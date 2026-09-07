"""Bounded voice workers, half-duplex playback and metadata-only vision handoff."""

from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic

from jake.application.conversation_worker import ConversationWorker
from jake.conversation import ConversationManager, GreetingPolicy, associate
from jake.conversation_ai_config import ConversationAIConfig
from jake.conversation_context import (
    FALLBACK,
    build_context,
    personalize,
    priority_response,
    validate_response,
)
from jake.conversation_domain import ConversationModel, ConversationRequest, SpeakerContext
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
        *,
        conversation_model: ConversationModel | None = None,
        conversation_ai: ConversationAIConfig | None = None,
        timezone: str = "UTC",
        capabilities: tuple[str, ...] = ("local conversation",),
    ) -> None:
        self.audio, self.speech, self.interaction = audio, speech, interaction
        ai = conversation_ai or ConversationAIConfig()
        self.ai = (
            ConversationWorker(conversation_model, ai.timeout_seconds)
            if ai.enabled and conversation_model is not None
            else None
        )
        self.timezone, self.capabilities = timezone, capabilities
        self._pending_ai = False
        self.ai_fallback_used = False
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
            if self.ai is not None:
                self.ai.start()
            self.source.start()
            self.speaker.start()
            self._thread = Thread(target=self._run, name="jake-voice")
            self._thread.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._cancel.set()
        if self.ai is not None:
            self.ai.model.cancel()
        try:
            self.source.close()
        finally:
            try:
                self.speaker.close()
            finally:
                if self.ai is not None:
                    self.ai.close()
                if self._thread is not None:
                    self._thread.join()
                self.segmenter.reset()
                self.conversations.end(datetime.now(UTC))
                with self._lock:
                    self._context = None
                    self._events.clear()
                self.metrics = VoiceMetrics()
                self.greetings = GreetingPolicy(self.interaction)

    def _finish_ai(
        self, request: ConversationRequest, generated: str | None, *, guarded: bool = False
    ) -> None:
        if self._cancel.is_set():
            return
        context = request.context
        response = (
            generated
            if guarded and generated is not None
            else validate_response(generated or "", request)
        )
        self.ai_fallback_used = response == FALLBACK
        with self._lock:
            latest = self._context
        current = associate(
            latest, datetime.now(UTC), self.interaction.visual_context_max_age_seconds
        )
        if (
            current is None
            or current.track_id != context.speaker.track_id
            or current.identity_state != "RESIDENT"
            or current.resident_id != context.speaker.resident_id
        ):
            context = replace(context, speaker=SpeakerContext())
        personalized = personalize(
            response, context, already_named=self.conversations.named_in_session
        )
        result = self.conversations.finish(
            request.context.conversation_id, request.text, personalized, request.context.timestamp
        )
        if result is not None:
            if personalized != response:
                self.conversations.named_in_session = True
            self.speaker.speak(result.text)

    def _run(self) -> None:
        person = None
        utterance_context: VisualContext | None = None
        utterance_at = datetime.now(UTC)
        completed = 0
        self.metrics = VoiceMetrics(input_status="listening")
        try:
            while not self._cancel.is_set():
                now = datetime.now(UTC)
                self.conversations.expire(now)
                if self.ai is not None:
                    if self._pending_ai and self.conversations.session is None:
                        self.ai.cancel_active()
                    outcome = self.ai.poll()
                    if outcome is not None:
                        self._pending_ai = False
                        self._finish_ai(
                            outcome.request, outcome.response.text if outcome.response else None
                        )
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
                if self.segmenter.state == "IDLE" and context and not self._pending_ai:
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
                    if self.ai is not None:
                        with self._lock:
                            context = self._context
                    utterance_context = context
                    utterance_at = chunk.started_at
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
                if segment is not None and self._pending_ai:
                    # VAD continues, but no second STT/LLM request queues behind active reasoning.
                    del segment
                    continue
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
                    if self.ai is None:
                        response = self.conversations.respond(result, person)
                        if response is not None:
                            self.speaker.speak(response.text)
                        del response
                    else:
                        text = self.conversations.begin(result, person)
                        session = self.conversations.session
                        if text is not None and session is not None:
                            request = ConversationRequest(
                                build_context(
                                    session.conversation_id,
                                    result.ended_at,
                                    person,
                                    utterance_context,
                                    tuple(session.turns),
                                    timezone=self.timezone,
                                    max_age=self.interaction.visual_context_max_age_seconds,
                                    supported=self.capabilities,
                                    association_at=utterance_at,
                                ),
                                text,
                            )
                            guarded = priority_response(text)
                            if guarded is not None:
                                self._finish_ai(request, guarded, guarded=True)
                            else:
                                self._pending_ai = self.ai.submit(request)
                                if not self._pending_ai:
                                    self._finish_ai(request, None)
                            del request
                        del text, session
                    del result, segment
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
