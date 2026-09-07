"""Regression coverage for shared camera and integrated voice composition."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest

from jake.application.perception_session import PerceptionOverrides, configure_perception
from jake.application.voice_camera import VoiceCamera
from jake.config import AppConfig, PipelineConfig
from jake.domain import BoundingBox, Frame, PersonDetection, PersonTrack
from jake.identity import face_normalize
from jake.identity_domain import FaceDetection, FaceQuality, ResidentProfile
from jake.voice_preview import context_label


@pytest.mark.parametrize(
    "enabled,override,resident",
    [(True, None, True), (False, None, False), (False, True, True), (True, False, False)],
)
def test_real_identity_pipeline_to_voice_preview(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    override: bool | None,
    resident: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = datetime.now(UTC)
    box = BoundingBox(0.1, 0.1, 0.9, 0.9)
    embedding = face_normalize("fake", (1.0, 0.0))
    profile = ResidentProfile(str(UUID(int=1)), "Joseph", (embedding,), 10, now)
    store = Mock(return_value=Mock(profiles=Mock(return_value=(profile,))))
    monkeypatch.setattr("jake.adapters.local_identity_store.LocalIdentityStore", store)
    face = Mock(detect=Mock(return_value=(FaceDetection(box, 0.99),)))
    monkeypatch.setattr("jake.adapters.opencv_faces.YuNetFaceDetector", Mock(return_value=face))
    monkeypatch.setattr(
        "jake.adapters.opencv_faces.SFaceEncoder",
        Mock(return_value=Mock(encode=Mock(return_value=embedding))),
    )
    monkeypatch.setattr(
        "jake.adapters.opencv_faces.face_quality", Mock(return_value=FaceQuality(True, "accepted"))
    )
    detector = Mock(detect=Mock(return_value=(PersonDetection(box, 0.9),)))
    monkeypatch.setattr(
        "jake.adapters.yolo_detector.YoloPersonDetector", Mock(return_value=detector)
    )
    tracker = Mock(update=Mock(return_value=(PersonTrack("4", box, 0.9),)))
    monkeypatch.setattr(
        "jake.adapters.kalman_tracker.StabilizedKalmanPersonTracker", Mock(return_value=tracker)
    )
    frames = [Frame("test", i, now + timedelta(seconds=i), 1, 1, b"abc") for i in range(4)]
    source = MagicMock()
    source.__enter__.return_value = frames
    factory = Mock(return_value=source)
    monkeypatch.setattr("jake.adapters.opencv_camera.OpenCVCamera", factory)
    config = AppConfig(PipelineConfig("test"))
    config = replace(config, identity=replace(config.identity, enabled=enabled))
    voice = Mock()
    camera = VoiceCamera(
        config, voice, preview=True, overrides=PerceptionOverrides(identity=override)
    )
    camera.start()
    try:
        assert camera.finished.wait(3)
        assert camera.error is None
        snapshot = camera.preview_snapshot()
        assert snapshot is not None
        context = voice.publish.call_args.args[0]
        assert snapshot.context is context
        assert snapshot.resident_profile_count == (1 if resident else None)
        if resident:
            assert "Resident profiles loaded: 1" in capsys.readouterr().out
            assert snapshot.identity_diagnostics[0].match.display_name == "Joseph"
            assert face.detect.call_count == len(frames)
        assert context.people[0].identity_state == ("RESIDENT" if resident else "UNKNOWN")
        label = context_label(context, context.at, 2)
        assert ("Joseph | RESIDENT | track=4" in label) == resident
        assert detector.detect.call_count == len(frames)
        assert tracker.update.call_count == len(frames)
        assert store.call_count == int(resident)
        factory.assert_called_once()
    finally:
        camera.close()
    source.__exit__.assert_called_once()


def test_config_and_overrides_use_same_session_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(PipelineConfig("test"))
    overrides = PerceptionOverrides("kalman", "hungarian", True, True, True, True)
    resolved = configure_perception(config, overrides)
    configured = configure_perception(resolved.config)
    assert configured == resolved
    composer = Mock()
    monkeypatch.setattr("jake.application.composition.compose", composer)
    resolved.compose()
    configured.compose()
    assert composer.call_args_list[0] == composer.call_args_list[1]
    assert composer.call_args.kwargs["identify"] is True
    assert resolved.config.tracking.appearance.enabled
    assert resolved.config.tracking.reid.enabled
    assert resolved.config.tracking.assignment == "hungarian"
    assert resolved.event_generator() is not None
    assert not config.identity.enabled  # resolution never mutates the source config


@pytest.mark.parametrize("identity", [None, False, True])
def test_visitors_always_require_resident_first_support(identity: bool | None) -> None:
    session = configure_perception(
        AppConfig(PipelineConfig("test")),
        PerceptionOverrides(identity=identity, visitors=True),
        events=False,
    )
    assert session.config.identity.enabled and session.config.visitors.enabled
    assert session.event_generator() is not None


@pytest.mark.parametrize("mode", ["iou", "kalman-baseline", "kalman"])
def test_tracker_modes_preserved(mode: str) -> None:
    assert (
        configure_perception(
            AppConfig(PipelineConfig("test")), PerceptionOverrides(tracker=mode)
        ).tracker
        == mode
    )


@pytest.mark.parametrize(
    "options",
    [
        PerceptionOverrides(tracker="invalid"),
        PerceptionOverrides(tracker="iou", appearance=True),
        PerceptionOverrides(reid=True, appearance=False),
    ],
)
def test_invalid_overrides_fail_before_composition(options: PerceptionOverrides) -> None:
    with pytest.raises(ValueError):
        configure_perception(AppConfig(PipelineConfig("test")), options)


def test_voice_cli_overrides_reach_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.voice_cli import main

    config = AppConfig(PipelineConfig("test"))
    monkeypatch.setattr("jake.voice_cli.load_app_config", Mock(return_value=config))
    voice, camera = Mock(), Mock()
    monkeypatch.setattr(
        "jake.application.voice_composition.compose_voice", Mock(return_value=voice)
    )
    factory = Mock(return_value=camera)
    monkeypatch.setattr("jake.application.voice_camera.VoiceCamera", factory)
    monkeypatch.setattr(
        "jake.voice_cli.Event", Mock(return_value=Mock(wait=Mock(side_effect=KeyboardInterrupt)))
    )
    assert (
        main(
            [
                "--with-camera",
                "--tracker",
                "kalman",
                "--assignment",
                "hungarian",
                "--appearance",
                "--reid",
                "--identity",
                "--visitors",
            ]
        )
        == 0
    )
    assert factory.call_args.kwargs["overrides"] == PerceptionOverrides(
        "kalman", "hungarian", True, True, True, True
    )


def test_normal_cli_uses_shared_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.cli import main

    config = AppConfig(PipelineConfig("test"))
    monkeypatch.setattr("jake.cli.load_app_config", Mock(return_value=config))
    from jake.application.composition import Components

    components = Components(Mock(), Mock(), Mock(), Mock(), Mock(), None)
    composer = Mock(return_value=components)
    monkeypatch.setattr("jake.application.composition.compose", composer)
    preview = Mock()
    monkeypatch.setattr("jake.preview.preview", preview)
    assert (
        main(
            [
                "--track",
                "--tracker",
                "kalman",
                "--assignment",
                "hungarian",
                "--appearance",
                "--reid",
                "--identity",
                "--visitors",
            ]
        )
        == 0
    )
    expected = configure_perception(
        config, PerceptionOverrides("kalman", "hungarian", True, True, True, True)
    )
    assert composer.call_args.args[0] == expected.config
    assert preview.call_args.args[0] == expected.config
    assert preview.call_args.kwargs["events"] is not None
