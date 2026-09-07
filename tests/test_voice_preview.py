"""Voice preview tests use fake HighGUI; no camera, audio device or weights."""

from datetime import UTC, datetime, timedelta
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from jake.application.voice_camera import VoicePreviewSnapshot
from jake.domain import BoundingBox, Frame, PersonTrack
from jake.voice_domain import VisualContext, VisualPerson
from jake.voice_preview import VoicePreview, context_label

NOW = datetime.now(UTC)


def snapshot(*people: VisualPerson) -> VoicePreviewSnapshot:
    return VoicePreviewSnapshot(
        Frame("test", 1, NOW, 640, 480, bytes(640 * 480 * 3)),
        tuple(PersonTrack(p.track_id, BoundingBox(0.1, 0.1, 0.4, 0.8), 0.9) for p in people),
        VisualContext(NOW, people),
        24.5,
    )


@pytest.mark.parametrize(
    ("people", "expected"),
    [
        (
            (VisualPerson("4", identity_state="RESIDENT", display_name="Joseph"),),
            "VOICE CONTEXT: Joseph | RESIDENT | track=4",
        ),
        (
            (VisualPerson("5", visitor_id="abcd123", visitor_state="FIRST_TIME_VISITOR"),),
            "VOICE CONTEXT: VISITOR ABCD | FIRST_TIME_VISITOR | track=5",
        ),
        ((VisualPerson("1"),), "VOICE CONTEXT: UNKNOWN | track=1"),
        ((VisualPerson("1"), VisualPerson("2")), "VOICE CONTEXT: unresolved"),
        ((VisualPerson("1", confirmed=False),), "VOICE CONTEXT: unresolved"),
        ((VisualPerson("1", visible=False),), "VOICE CONTEXT: unresolved"),
    ],
)
def test_context(people: tuple[VisualPerson, ...], expected: str) -> None:
    context = VisualContext(NOW, people)
    assert context_label(context, NOW, 2) == expected
    assert context_label(context, NOW + timedelta(seconds=3), 2) == "VOICE CONTEXT: unresolved"
    assert context_label(context, NOW - timedelta(seconds=1), 2) == "VOICE CONTEXT: unresolved"


@pytest.fixture
def gui(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    mocks = {}
    for name in ("namedWindow", "imshow", "waitKey", "destroyWindow", "putText"):
        mocks[name] = Mock(return_value=-1)
        monkeypatch.setattr(f"jake.voice_preview.cv2.{name}", mocks[name])
    return mocks


@pytest.mark.parametrize("key", ["q", "Q"])
def test_render_and_quit(gui: dict[str, Mock], key: str) -> None:
    gui["waitKey"].return_value = ord(key)
    item = snapshot(VisualPerson("4", identity_state="RESIDENT", display_name="Joseph"))
    preview = VoicePreview(2)
    assert not preview.update(item)
    labels = [c.args[1] for c in gui["putText"].call_args_list]
    assert "Joseph | RESIDENT" in labels
    assert any("ID 4 | PERSON" in line for line in labels)
    assert any("640x480 | processing 24.5 FPS" in line for line in labels)
    assert any("not acoustic speaker identification" in line for line in labels)
    assert item.frame.pixels == bytes(640 * 480 * 3)
    preview.close()
    preview.close()
    gui["destroyWindow"].assert_called_once()


def test_preview_requires_camera(capsys: pytest.CaptureFixture[str]) -> None:
    from jake.voice_cli import main

    with pytest.raises(SystemExit) as error:
        main(["--preview"])
    assert error.value.code == 2
    assert "--preview requires --with-camera" in capsys.readouterr().err


@pytest.mark.parametrize("interrupt", [False, True])
def test_cli_preview_shutdown(monkeypatch: pytest.MonkeyPatch, interrupt: bool) -> None:
    from jake.application.perception_session import PerceptionOverrides
    from jake.config import AppConfig, PipelineConfig
    from jake.voice_cli import main

    voice, camera, display = Mock(), Mock(), Mock()
    voice.metrics.input_status = "listening"
    camera.error = None
    display.update.return_value = False
    if interrupt:
        display.update.side_effect = KeyboardInterrupt
    monkeypatch.setattr(
        "jake.voice_cli.load_app_config", Mock(return_value=AppConfig(PipelineConfig("test")))
    )
    monkeypatch.setattr(
        "jake.application.voice_composition.compose_voice", Mock(return_value=voice)
    )
    factory = Mock(return_value=camera)
    monkeypatch.setattr("jake.application.voice_camera.VoiceCamera", factory)
    monkeypatch.setattr("jake.voice_preview.VoicePreview", Mock(return_value=display))
    assert main(["--with-camera", "--preview"]) == 0
    assert factory.call_args.kwargs == {"preview": True, "overrides": PerceptionOverrides()}
    factory.assert_called_once()
    camera.start.assert_called_once()
    display.update.assert_called_once_with(camera.preview_snapshot.return_value)
    camera.request_stop.assert_called_once()
    voice.close.assert_called_once()
    camera.close.assert_called_once()
    display.close.assert_called_once()


def test_blocked_render_does_not_hold_voice_context_lock(gui: dict[str, Mock]) -> None:
    from jake.application.voice import VoiceService
    from jake.voice_config import AudioConfig, InteractionConfig, SpeechConfig

    entered, release = Event(), Event()
    service = VoiceService(
        AudioConfig(), SpeechConfig(), InteractionConfig(), Mock(), Mock(), Mock(), Mock(), Mock()
    )
    item = snapshot(VisualPerson("1"))
    service.publish(item.context)

    def blocked_display(*args: object) -> None:
        entered.set()
        assert release.wait(2)

    gui["imshow"].side_effect = blocked_display
    preview = VoicePreview(2)
    renderer = Thread(target=preview.update, args=(item,))
    renderer.start()
    try:
        assert entered.wait(2)
        # The voice context mailbox remains writable during a blocked HighGUI call.
        service.publish(VisualContext(NOW, ()))
        assert service.drain_events() == ()
    finally:
        release.set()
        renderer.join(2)
        preview.close()
    assert not renderer.is_alive()
