"""Hardware/model-free voice state, policy, buffering and worker tests."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from time import monotonic, sleep
from unittest.mock import Mock

import pytest

from jake.application.voice import AsyncSpeech, VoiceService
from jake.application.voice_binding import VisualVoiceBridge
from jake.config import AppConfig, PipelineConfig, load_app_config
from jake.conversation import ConversationManager, GreetingPolicy, associate, intent
from jake.domain import BoundingBox, EventKind, FrameContext, PersonEvent, PersonTrack
from jake.identity_domain import IdentityMatch, IdentityState
from jake.visitor_domain import VisitorMatch, VisitorState
from jake.voice_config import AudioConfig, InteractionConfig, SpeechConfig
from jake.voice_domain import (
    AudioChunk,
    SpeechRecognitionResult,
    SpeechResult,
    VisualContext,
    VisualPerson,
    VoiceError,
)
from jake.voice_segmentation import UtteranceSegmenter

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def chunk(i: int, speech: bool = True, at: datetime = NOW) -> AudioChunk:
    return AudioChunk(i, at + timedelta(milliseconds=30 * i), bytes([int(speech), 0]) * 480)


def result(text: str, at: datetime = NOW) -> SpeechRecognitionResult:
    return SpeechRecognitionResult(text, at, at + timedelta(seconds=1), 50, "en")


class VAD:
    def is_speech(self, audio: AudioChunk) -> bool:
        return bool(audio.pcm[0])


def until(predicate: Callable[[], bool]) -> None:
    deadline = monotonic() + 3
    while not predicate() and monotonic() < deadline:
        sleep(0.005)
    assert predicate()


def test_default_opt_in_and_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "voice.toml"
    path.write_text('[pipeline]\ncamera_id="test"', encoding="utf-8")
    assert not load_app_config(path).audio.enabled
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[audio]\nenabled=true\n'
        'input_device="USB Mic"\noutput_device=2\n'
        '[speech]\nvad_mode=3\n[stt]\ndevice="cpu"\n[tts]\nengine="piper"',
        encoding="utf-8",
    )
    assert load_app_config(path).audio == AudioConfig(True, "USB Mic", 2)


@pytest.mark.parametrize(
    "table,value",
    [
        ("audio", "enabled=1"),
        ("audio", "sample_rate=48000"),
        ("audio", "sample_rate=true"),
        ("audio", "channels=2"),
        ("audio", "input_device=-1"),
        ("audio", "output_device=true"),
        ("audio", 'input_device=""'),
        ("audio", "queue_chunks=0"),
        ("audio", "queue_chunks=65"),
        ("speech", "vad_mode=4"),
        ("speech", "vad_mode=true"),
        ("speech", "silence_timeout_ms=-1"),
        ("speech", "pre_roll_ms=2000"),
        ("speech", "max_utterance_seconds=inf"),
        ("speech", "minimum_speech_ms=5000\nmax_utterance_seconds=1"),
        ("speech", 'language=""'),
        ("stt", 'device="cloud"'),
        ("stt", 'model="https://remote"'),
        ("stt", 'compute_type="float16"'),
        ("stt", "cpu_threads=0"),
        ("tts", 'engine="cloud"'),
        ("tts", 'voice="https://remote.onnx"'),
        ("interaction", "greeting_cooldown_minutes=0"),
        ("interaction", "max_turns=1000"),
        ("interaction", "resident_greeting_enabled=1"),
        ("audio", "typo=true"),
        ("speech", "vad_enabled=false"),
    ],
)
def test_strict_voice_config(tmp_path: Path, table: str, value: str) -> None:
    path = tmp_path / "voice.toml"
    path.write_text(f'[pipeline]\ncamera_id="test"\n[{table}]\n{value}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_app_config(path)


def test_segmentation_silence_preroll_and_minimum() -> None:
    cfg = SpeechConfig(minimum_speech_ms=60, silence_timeout_ms=60, pre_roll_ms=60)
    segmenter = UtteranceSegmenter(cfg, VAD())
    assert segmenter.push(chunk(0, False)) is None and segmenter.state == "IDLE"
    assert segmenter.push(chunk(1)) is None and segmenter.state == "LISTENING"
    assert segmenter.push(chunk(2)) is None
    assert segmenter.push(chunk(3, False)) is None
    segment = segmenter.push(chunk(4, False))
    assert segment is not None and segment.started_at == NOW
    assert segment.duration_seconds == pytest.approx(0.15)
    assert segmenter.state == "IDLE" and segmenter.buffered_chunks == 0
    for i, speech in enumerate((True, False, False), start=5):
        assert segmenter.push(chunk(i, speech)) is None
    assert segmenter.state == "IDLE"


def test_utterance_maximum_and_bounded_idle_buffer() -> None:
    segmenter = UtteranceSegmenter(SpeechConfig(max_utterance_seconds=1, pre_roll_ms=0), VAD())
    segments = [s for i in range(1000) if (s := segmenter.push(chunk(i))) is not None]
    assert segments and all(s.duration_seconds <= 1 for s in segments)
    segmenter.reset()
    for i in range(1000):
        assert segmenter.push(chunk(i, False)) is None
    assert segmenter.buffered_chunks <= 1


def test_queue_gap_discards_partial_utterance() -> None:
    segmenter = UtteranceSegmenter(SpeechConfig(), VAD())
    segmenter.push(chunk(0))
    segmenter.push(chunk(1))
    segmenter.push(chunk(10, False))
    assert segmenter.state == "IDLE"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello", "GREETING"),
        ("I have a package.", "DELIVERY"),
        ("I have a delivery", "DELIVERY"),
        ("who are you?", "WHO_ARE_YOU"),
        ("is anyone home?", "IS_ANYONE_HOME"),
        ("goodbye", "GOODBYE"),
        ("something unexpected", "UNKNOWN"),
        ("shipping", "UNKNOWN"),
    ],
)
def test_intents(text: str, expected: str) -> None:
    assert intent(text) == expected


def test_conversation_empty_timeout_bounded_turns_and_private_events() -> None:
    manager = ConversationManager(InteractionConfig(conversation_timeout_seconds=2, max_turns=4))
    assert manager.respond(result("  "), None) is None and manager.session is None
    assert manager.respond(result("123"), None) is None
    response = manager.respond(result("I have a package"), None)
    assert response and "package by the door" in response.text
    assert manager.session is not None
    session = manager.session
    for _ in range(10):
        manager.respond(result("private words"), None)
    assert len(session.turns) == 4
    assert "private words" not in repr(session) and "private words" not in repr(manager.events)
    manager.expire(NOW + timedelta(seconds=3))
    assert manager.session is None and not session.turns
    assert manager.events[-1].kind == "CONVERSATION_ENDED"


def test_single_visible_association_and_ambiguity() -> None:
    one = VisualPerson("1")
    assert associate(VisualContext(NOW, (one,)), NOW, 2) == one
    assert associate(VisualContext(NOW, (one, VisualPerson("2"))), NOW, 2) is None
    assert associate(VisualContext(NOW, (one, VisualPerson("2", confirmed=False))), NOW, 2) is None
    assert associate(VisualContext(NOW, (replace(one, visible=False),)), NOW, 2) is None
    assert associate(VisualContext(NOW, (one,)), NOW + timedelta(seconds=3), 2) is None
    assert associate(None, NOW, 2) is None


def test_association_change_closes_previous_conversation() -> None:
    manager = ConversationManager(InteractionConfig())
    manager.respond(result("hi"), VisualPerson("1"))
    old = manager.session
    manager.respond(result("hello"), None)
    assert old is not None and not old.turns
    assert manager.session is not old and manager.session is not None
    assert manager.session.association is None


def test_resident_cooldown_flicker_and_late_identity() -> None:
    policy = GreetingPolicy(
        InteractionConfig(resident_greeting_enabled=True, greeting_cooldown_minutes=1)
    )
    p = VisualPerson("1", semantic_arrival=True, identity_state="CANDIDATE")
    assert not policy.observe(VisualContext(NOW, (p,)), NOW)
    p = replace(p, identity_state="RESIDENT", resident_id="r1", display_name="Joseph")
    assert policy.observe(VisualContext(NOW, (p,)), NOW)[0].text == "Welcome home, Joseph."
    assert not policy.observe(VisualContext(NOW, (replace(p, visible=False),)), NOW)
    assert not policy.observe(VisualContext(NOW, (p,)), NOW)
    policy.observe(VisualContext(NOW, ()), NOW)
    assert not policy.observe(VisualContext(NOW, (replace(p, track_id="2"),)), NOW)
    later = NOW + timedelta(seconds=61)
    assert policy.observe(VisualContext(later, (replace(p, track_id="3"),)), later)
    assert not GreetingPolicy(InteractionConfig()).observe(VisualContext(NOW, (p,)), NOW)


@pytest.mark.parametrize(
    "state", ["FIRST_TIME_VISITOR", "RECURRING_VISITOR", "FREQUENT_VISITOR", "KNOWN_VISITOR"]
)
def test_visitor_greeting_requires_confirmed_semantic_arrival_and_cooldown(state: str) -> None:
    policy = GreetingPolicy(InteractionConfig(visitor_greeting_enabled=True))
    p = VisualPerson("1", visitor_id="v", visitor_state=state)
    assert not policy.observe(VisualContext(NOW, (p,)), NOW)
    p = replace(p, semantic_arrival=True)
    assert not policy.observe(VisualContext(NOW, (replace(p, identity_state="CANDIDATE"),)), NOW)
    request = policy.observe(VisualContext(NOW, (p,)), NOW)[0]
    assert state not in request.text
    policy.observe(VisualContext(NOW, ()), NOW)
    assert not policy.observe(VisualContext(NOW, (replace(p, track_id="2"),)), NOW)
    assert not policy.observe(VisualContext(NOW, (p,)), NOW + timedelta(seconds=10))


def test_visual_bridge_preserves_arrival_during_missed_tracks() -> None:
    bridge = VisualVoiceBridge()
    box = BoundingBox(0, 0, 1, 1)
    track = PersonTrack("1", box, 0.9)
    context = FrameContext("camera", 0, NOW)
    entered = PersonEvent("e", EventKind.PERSON_ENTERED, context, "1", NOW)
    snapshot = bridge.observe(NOW, (track,), (entered,), {}, {})
    assert snapshot.people[0].semantic_arrival
    resident = {"1": IdentityMatch(IdentityState.RESIDENT, "r", "Joseph", 0.9)}
    snapshot = bridge.observe(NOW, (replace(track, missed_frames=1),), (), resident, {})
    assert snapshot.people[0].semantic_arrival and not snapshot.people[0].visible
    snapshot = bridge.observe(NOW, (track,), (), resident, {})
    assert snapshot.people[0].resident_id == "r"
    visitor = {"1": VisitorMatch(VisitorState.FIRST_TIME_VISITOR, "v")}
    assert bridge.observe(NOW, (track,), (), {}, visitor).people[0].visitor_id == "v"
    bridge.observe(NOW, (), (replace(entered, kind=EventKind.PERSON_LEFT),), {}, {})
    assert not bridge.observe(NOW, (track,), (), {}, {}).people[0].semantic_arrival


class Input:
    def __init__(self) -> None:
        self.queue: Queue[AudioChunk] = Queue(maxsize=64)
        self.started = self.closed = False

    def start(self) -> None:
        self.started = True

    def read(self, timeout: float) -> AudioChunk | None:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def discard(self) -> None:
        while not self.queue.empty():
            self.read(0.001)

    def close(self) -> None:
        self.closed = True
        self.discard()


def test_async_speech_bounded_stop_and_shutdown() -> None:
    entered, release = Event(), Event()

    def synthesize(request: object) -> SpeechResult:
        entered.set()
        release.wait(2)
        return SpeechResult(bytes(320), 16000, 2)

    output = Mock()
    speaker = AsyncSpeech(Mock(synthesize=synthesize), output)
    speaker.start()
    try:
        assert speaker.speak("hello") and entered.wait(1)
        assert speaker.is_speaking
        assert all(speaker.speak("hello") for _ in range(3))
        assert not speaker.speak("queue full")
        speaker.stop()
        release.set()
        until(lambda: not speaker.is_speaking)
        output.play.assert_not_called()
    finally:
        release.set()
        speaker.close()
    assert not speaker.is_speaking


def test_end_to_end_and_half_duplex_without_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Input()
    entered, release = Event(), Event()

    def playback(audio: SpeechResult, cancel: Event) -> None:
        entered.set()
        while not release.wait(0.005) and not cancel.is_set():
            pass

    recognizer = Mock(transcribe=lambda segment: result("I have a package", datetime.now(UTC)))
    recognizer.transcribe = Mock(side_effect=recognizer.transcribe)
    service = VoiceService(
        AudioConfig(enabled=True),
        SpeechConfig(minimum_speech_ms=60, silence_timeout_ms=60, pre_roll_ms=0),
        InteractionConfig(),
        source,
        VAD(),
        recognizer,
        Mock(synthesize=Mock(return_value=SpeechResult(bytes(320), 16000, 2))),
        Mock(play=playback),
    )
    service.start()
    try:
        for i, speech in enumerate((True, True, False, False)):
            source.queue.put(chunk(i, speech, datetime.now(UTC)))
        assert entered.wait(2)
        assert service.metrics.stt_ms == 50
        for i in range(20):
            source.queue.put(chunk(i, True, datetime.now(UTC)))
        until(lambda: service.metrics.vad_state == "SUPPRESSED")
        sleep(0.08)
        assert recognizer.transcribe.call_count == 1
        assert service.metrics.stt_real_time_factor == pytest.approx(0.05 / 0.12)
        release.set()
        until(lambda: service.speaker.completed == 1)
    finally:
        release.set()
        service.close()
    assert source.closed and service.conversations.session is None
    assert not service.speaker.is_speaking
    assert not list(tmp_path.iterdir())


def test_disabled_service_never_opens_audio() -> None:
    source = Input()
    service = VoiceService(
        AudioConfig(), SpeechConfig(), InteractionConfig(), source, VAD(), Mock(), Mock(), Mock()
    )
    with pytest.raises(VoiceError, match="disabled"):
        service.start()
    assert not source.started
    service.close()


def test_disabled_composition_never_loads_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.application.voice_composition import compose_voice

    config = AppConfig(PipelineConfig("test"))
    with pytest.raises(VoiceError, match="disabled"):
        compose_voice(config)


@pytest.mark.parametrize("preview", [False, True])
def test_integrated_camera_publishes_only_metadata(
    monkeypatch: pytest.MonkeyPatch, preview: bool
) -> None:
    from unittest.mock import MagicMock

    from jake.application.composition import Components
    from jake.application.voice_camera import VoiceCamera
    from jake.domain import Frame, PersonDetection

    box = BoundingBox(0, 0, 1, 1)
    detector = Mock(detect=Mock(return_value=(PersonDetection(box, 0.9),)))
    tracker = Mock(update=Mock(return_value=(PersonTrack("1", box, 0.9),)))
    compose = Mock(return_value=Components(detector, tracker, None, None, None, None))
    monkeypatch.setattr("jake.application.composition.compose", compose)
    source = MagicMock()
    frame = Frame("test", 0, NOW, 1, 1, b"abc")
    source.__enter__.return_value = [frame]
    monkeypatch.setattr("jake.adapters.opencv_camera.OpenCVCamera", Mock(return_value=source))
    voice = Mock()
    camera = VoiceCamera(AppConfig(PipelineConfig("test")), voice, preview=preview)
    camera.start()
    until(lambda: voice.publish.called)
    assert camera.finished.wait(2)
    image = camera.preview_snapshot()
    assert (image is not None) == preview
    if image is not None:
        assert image.frame is frame
        assert image.context is voice.publish.call_args.args[0]
    detector.detect.assert_called_once()
    tracker.update.assert_called_once()
    compose.assert_called_once()
    camera.close()
    assert camera.preview_snapshot() is None
    assert camera.error is None
    snapshot = voice.publish.call_args.args[0]
    assert isinstance(snapshot, VisualContext)
    assert snapshot.people[0].semantic_arrival and snapshot.people[0].track_id == "1"
    assert not hasattr(snapshot, "pixels")
    source.__exit__.assert_called_once()


def test_voice_composition_does_not_open_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.application.voice_composition import compose_voice

    replacements = {}
    for module, name in (
        ("local_speech", "WebRtcVAD"),
        ("local_speech", "FasterWhisperRecognizer"),
        ("local_speech", "PiperSynthesizer"),
        ("sounddevice_audio", "SoundDeviceInput"),
        ("sounddevice_audio", "SoundDeviceOutput"),
    ):
        replacements[name] = Mock()
        monkeypatch.setattr(f"jake.adapters.{module}.{name}", replacements[name])
    config = AppConfig(PipelineConfig("test"), audio=AudioConfig(enabled=True))
    service = compose_voice(config)
    replacements["SoundDeviceInput"].assert_called_once_with(config.audio)
    replacements["FasterWhisperRecognizer"].assert_called_once_with(config.stt, config.speech)
    replacements["SoundDeviceInput"].return_value.start.assert_not_called()
    service.close()


def test_voice_worker_error_closes_microphone_without_logging_payload() -> None:
    source = Input()
    service = VoiceService(
        AudioConfig(enabled=True),
        SpeechConfig(),
        InteractionConfig(),
        source,
        VAD(),
        Mock(),
        Mock(synthesize=Mock(side_effect=VoiceError("private transcript"))),
        Mock(),
    )
    service.start()
    try:
        service.speaker.speak("hello")
        until(lambda: service.metrics.input_status.startswith("failed"))
        until(lambda: source.closed)
        assert "private" not in repr(service.metrics)
    finally:
        service.close()


def test_cli_shutdown_with_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.voice_cli import main

    path = tmp_path / "voice.toml"
    path.write_text('[pipeline]\ncamera_id="test"\n[audio]\nenabled=true', encoding="utf-8")
    voice, camera = Mock(), Mock()
    monkeypatch.setattr(
        "jake.application.voice_composition.compose_voice", Mock(return_value=voice)
    )
    monkeypatch.setattr("jake.application.voice_camera.VoiceCamera", Mock(return_value=camera))
    monkeypatch.setattr(
        "jake.voice_cli.Event", Mock(return_value=Mock(wait=Mock(side_effect=KeyboardInterrupt)))
    )
    assert main(["--config", str(path), "--with-camera"]) == 0
    voice.start.assert_called_once()
    camera.start.assert_called_once()
    camera.request_stop.assert_called_once()
    voice.close.assert_called_once()
    camera.close.assert_called_once()


@pytest.mark.parametrize("data", [b"", bytes(2), bytes(962)])
def test_audio_chunk_contract_rejects_invalid_pcm(data: bytes) -> None:
    with pytest.raises(ValueError):
        AudioChunk(0, NOW, data)


def test_audio_pipeline_progresses_while_preview_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Thread

    from jake.application.voice_camera import VoicePreviewSnapshot
    from jake.domain import Frame
    from jake.voice_preview import VoicePreview

    entered, release, played = Event(), Event(), Event()

    def render(*args: object) -> None:
        entered.set()
        release.wait(3)

    for name in ("namedWindow", "destroyWindow"):
        monkeypatch.setattr(f"jake.voice_preview.cv2.{name}", Mock())
    monkeypatch.setattr("jake.voice_preview.cv2.imshow", render)
    monkeypatch.setattr("jake.voice_preview.cv2.waitKey", Mock(return_value=-1))
    source = Input()
    recognizer = Mock(transcribe=Mock(return_value=result("hello", datetime.now(UTC))))
    synthesizer = Mock(synthesize=Mock(return_value=SpeechResult(bytes(320), 16000, 2)))
    service = VoiceService(
        AudioConfig(enabled=True),
        SpeechConfig(minimum_speech_ms=60, silence_timeout_ms=60, pre_roll_ms=0),
        InteractionConfig(),
        source,
        VAD(),
        recognizer,
        synthesizer,
        Mock(play=lambda audio, cancel: played.set()),
    )
    display = VoicePreview(2)
    snapshot = VoicePreviewSnapshot(
        Frame("test", 0, NOW, 1, 1, b"abc"),
        (),
        VisualContext(NOW, ()),
        30,
    )
    renderer = Thread(target=display.update, args=(snapshot,))
    service.start()
    renderer.start()
    try:
        assert entered.wait(2)
        at = datetime.now(UTC)
        for i, speech in enumerate((True, True, False, False)):
            source.queue.put(chunk(i, speech, at))
        assert played.wait(2)
        assert renderer.is_alive()
        recognizer.transcribe.assert_called_once()
        synthesizer.synthesize.assert_called_once()
    finally:
        release.set()
        renderer.join(3)
        service.close()
        display.close()
    assert source.closed
