"""Safe identity metadata and cross-entry-point store resolution regression tests."""

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest

from jake.domain import BoundingBox, Frame, PersonTrack
from jake.face_identity import FaceIdentityService
from jake.identity import face_normalize
from jake.identity_config import IdentityConfig
from jake.identity_domain import FaceDetection, FaceQuality, ResidentProfile
from jake.voice_preview import identity_lines

NOW = datetime(2026, 9, 6, tzinfo=UTC)
BOX = BoundingBox(0.1, 0.1, 0.9, 0.9)
TRACK = PersonTrack("3", BOX, 0.99)
EMBEDDING = face_normalize("fake", (1.0, 0.0))
PROFILE = ResidentProfile(str(UUID(int=1)), "Joseph", (EMBEDDING,), 10, NOW)


def frame(sequence: int, seconds: float) -> Frame:
    return Frame("test", sequence, NOW + timedelta(seconds=seconds), 1, 1, b"abc")


def service(quality: FaceQuality, score: float = 1.0) -> FaceIdentityService:
    embedding = face_normalize("fake", (score, (1 - score * score) ** 0.5))
    return FaceIdentityService(
        IdentityConfig(),
        Mock(detect=Mock(return_value=(FaceDetection(BOX, 0.99),))),
        Mock(encode=Mock(return_value=embedding)),
        Mock(profiles=Mock(return_value=(PROFILE,))),
        Mock(return_value=quality),
    )


@pytest.mark.parametrize(
    "quality,reason",
    [
        (
            FaceQuality(
                False, "face too small", face_width_px=48, face_height_px=55, minimum_pixels=64
            ),
            "48x55 px; minimum 64 px",
        ),
        (FaceQuality(False, "blurred face", face_width_px=92, face_height_px=110), "blurred face"),
        (FaceQuality(True, "accepted", face_width_px=92, face_height_px=110), "92x110 px"),
    ],
)
def test_face_diagnostics(quality: FaceQuality, reason: str) -> None:
    identity = service(quality)
    identity.process(frame(0, 0), (TRACK,))
    (diagnostic,) = identity.diagnostics()
    assert reason in diagnostic.face_summary
    assert diagnostic.face_width_px == quality.face_width_px
    assert diagnostic.detector_confidence == 0.99
    assert identity.profile_count == 1
    assert not any(key in repr(asdict(diagnostic)) for key in ("templates", "values", "embedding"))
    identity.detector.detect.assert_called_once()  # type: ignore[attr-defined]
    assert identity.encoder.encode.call_count == int(quality.accepted)  # type: ignore[attr-defined]


def test_no_face_and_low_detector_confidence() -> None:
    identity = service(FaceQuality(False, "low detector confidence"))
    identity.detector.detect.return_value = ()  # type: ignore[attr-defined]
    identity.process(frame(0, 0), (TRACK,))
    assert identity.diagnostics()[0].reason == "no face detected"
    identity.detector.detect.return_value = (FaceDetection(BOX, 0.5),)  # type: ignore[attr-defined]
    identity.process(frame(1, 1), (TRACK,))
    assert identity.diagnostics()[0].reason == "rejected detector confidence 0.50 < 0.90"
    identity.encoder.encode.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "score,state,reason",
    [
        (0.2, "UNKNOWN", "below candidate threshold"),
        (0.57, "CANDIDATE", "resident support requires"),
    ],
)
def test_match_explanation(score: float, state: str, reason: str) -> None:
    identity = service(FaceQuality(True, "accepted"), score)
    identity.process(frame(0, 0), (TRACK,))
    (diagnostic,) = identity.diagnostics()
    assert diagnostic.match.state == state
    assert diagnostic.similarity == pytest.approx(score)
    assert reason in diagnostic.reason
    if state == "CANDIDATE":
        assert "CANDIDATE Joseph" in identity_lines(diagnostic)[0]
    assert diagnostic.temporal.support_count == 0


def test_temporal_progress_read_only_age_and_final_resident() -> None:
    identity = service(FaceQuality(True, "accepted"))
    for i in range(3):
        identity.process(frame(i, i * 0.5), (TRACK,))
        (diagnostic,) = identity.diagnostics()
        assert diagnostic.temporal.support_count == i + 1
        assert diagnostic.temporal.required_confirmations == identity.config.required_confirmations
        assert diagnostic.temporal.window_seconds == identity.config.confirmation_window_seconds
        assert diagnostic.temporal.last_valid_age_seconds == 0
        assert f"confirmations {i + 1}/3" in identity_lines(diagnostic)[3]
        assert diagnostic.match.state == ("RESIDENT" if i == 2 else "CANDIDATE")
    detached = identity.temporal.diagnostics()
    detached.clear()
    assert identity.temporal.diagnostics()
    identity.process(frame(3, 1.1), (TRACK,))
    (diagnostic,) = identity.diagnostics()
    assert diagnostic.temporal.last_valid_age_seconds == pytest.approx(0.1)
    assert diagnostic.face_summary.startswith("last observation:")
    assert identity.detector.detect.call_count == 3  # type: ignore[attr-defined]


@pytest.mark.parametrize("absolute", [False, True])
def test_enrollment_camera_voice_resolve_same_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absolute: bool,
) -> None:
    import json

    from jake.adapters.local_identity_store import LocalIdentityStore
    from jake.application.voice_camera import VoiceCamera
    from jake.cli import main as camera_cli
    from jake.config import load_app_config
    from jake.enroll_cli import main as enroll

    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "identities"
    root = expected if absolute else Path("identities")
    path = tmp_path / "config.toml"
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[identity]\nenabled=true\nstore_path='
        + json.dumps(str(root)),
        encoding="utf-8",
    )
    observed = []

    def store_factory(path: Path) -> Mock:
        store = LocalIdentityStore(path)
        observed.append(store.root)
        # No key access or store reads; path resolution uses the real constructor.
        return Mock(profiles=Mock(return_value=()))

    monkeypatch.setattr("jake.adapters.local_identity_store.LocalIdentityStore", store_factory)
    for target in (
        "jake.adapters.yolo_detector.YoloPersonDetector",
        "jake.adapters.opencv_faces.YuNetFaceDetector",
        "jake.adapters.opencv_faces.SFaceEncoder",
    ):
        monkeypatch.setattr(target, Mock())
    monkeypatch.setattr("jake.preview.preview", Mock())
    source = MagicMock()
    source.__enter__.return_value = []
    monkeypatch.setattr("jake.adapters.opencv_camera.OpenCVCamera", Mock(return_value=source))
    assert enroll(["--config", str(path), "--list"]) == 0
    assert camera_cli(["--config", str(path), "--track"]) == 0
    camera = VoiceCamera(load_app_config(path), Mock())
    camera.start()
    try:
        assert camera.finished.wait(3)
        assert camera.error is None
    finally:
        camera.close()
    assert observed == [expected, expected, expected]
    assert not expected.exists()


def test_empty_store_and_ambiguous_match_diagnostics() -> None:
    identity = service(FaceQuality(True, "accepted"))
    identity.profiles = ()
    identity.process(frame(0, 0), (TRACK,))
    assert identity.profile_count == 0
    assert identity.diagnostics()[0].reason == "no resident profiles loaded"
    assert identity.diagnostics()[0].similarity is None
    identity.profiles = (
        PROFILE,
        replace(PROFILE, resident_id=str(UUID(int=2)), display_name="Other"),
    )
    identity.process(frame(1, 1), (TRACK,))
    assert "ambiguous resident match" in identity.diagnostics()[0].reason
    assert identity.diagnostics()[0].match.state == "UNKNOWN"
