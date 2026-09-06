import json
import math
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import numpy as np
import pytest

from jake.adapters.local_identity_store import LocalIdentityStore
from jake.adapters.opencv_faces import SFaceEncoder, YuNetFaceDetector, crop, face_quality
from jake.domain import BoundingBox, Frame, FrameContext, PersonEvent, PersonTrack
from jake.face_identity import FaceIdentityService
from jake.identity import (
    CosineIdentityMatcher,
    Enrollment,
    IdentityError,
    TemporalIdentity,
    face_cosine,
    face_normalize,
)
from jake.identity_config import IdentityConfig
from jake.identity_domain import (
    FaceDetection,
    FaceQuality,
    IdentityMatch,
    IdentityState,
    ResidentProfile,
)
from jake.identity_encryption import decrypt, encrypt

START = datetime(2026, 1, 1, tzinfo=UTC)
BOX = BoundingBox(0.1, 0.1, 0.9, 0.9)
A = face_normalize("test", (1.0, 0.0))
B = face_normalize("test", (0.0, 1.0))
PROFILE = ResidentProfile(str(UUID(int=1)), "Joseph", (A,), 10, START)
OTHER = ResidentProfile(str(UUID(int=2)), "Other", (B,), 10, START)
TRACK = PersonTrack("1", BOX, 0.9)
CONFIG = IdentityConfig()


def context(sequence: int, seconds: float) -> FrameContext:
    return FrameContext("test", sequence, START + timedelta(seconds=seconds))


def candidate(profile: ResidentProfile = PROFILE) -> IdentityMatch:
    return IdentityMatch(IdentityState.CANDIDATE, profile.resident_id, profile.display_name, 0.9)


def test_cosine_and_single_match_only_candidate() -> None:
    matcher = CosineIdentityMatcher(CONFIG)
    assert face_cosine(A, A) == 1 and face_cosine(A, B) == 0
    assert matcher.match(A, (PROFILE,)).state == IdentityState.CANDIDATE
    assert matcher.match(B, (PROFILE,)).state == IdentityState.UNKNOWN
    assert matcher.match(A, ()).state == IdentityState.UNKNOWN
    assert (
        matcher.match(A, (PROFILE, replace(OTHER, templates=(A,)))).state == IdentityState.UNKNOWN
    )
    with pytest.raises(IdentityError):
        face_cosine(A, face_normalize("other", (1.0, 0.0)))


def test_confirmation_conflict_carry_and_no_transfer() -> None:
    identity = TemporalIdentity(CONFIG)
    for seq, at in enumerate((0.0, 0.3, 0.6)):
        matches, events = identity.update(context(seq, at), (TRACK,), {"1": candidate()})
        assert matches["1"].state == (
            IdentityState.RESIDENT if seq == 2 else IdentityState.CANDIDATE
        )
    assert events[0].kind == "IDENTITY_RESOLVED"
    matches, _ = identity.update(context(3, 1), (TRACK,), {})
    assert matches["1"].state == IdentityState.RESIDENT
    matches, _ = identity.update(context(4, 1.1), (TRACK, replace(TRACK, track_id="2")), {})
    assert matches["2"].state == IdentityState.UNKNOWN
    matches, _ = identity.update(context(5, 1.2), (TRACK,), {"1": candidate(OTHER)})
    assert (
        matches["1"].state == IdentityState.CANDIDATE
        and matches["1"].resident_id == OTHER.resident_id
    )
    matches, _ = identity.update(
        context(6, 1.5), (TRACK,), {"1": IdentityMatch(IdentityState.UNKNOWN)}
    )
    assert matches["1"].state == IdentityState.UNKNOWN


def test_reactivation_needs_fresh_face_but_preserves_resident_when_verified() -> None:
    identity = TemporalIdentity(CONFIG)
    for seq in range(3):
        identity.update(context(seq, seq * 0.3), (TRACK,), {"1": candidate()})
    # Epoch protects even when no intervening pool snapshot was delivered.
    restored = replace(TRACK, continuity_epoch=1)
    matches, _ = identity.update(context(3, 2.5), (restored,), {})
    assert matches["1"].state == IdentityState.UNKNOWN
    for seq in range(4, 7):
        matches, _ = identity.update(
            context(seq, 2.5 + (seq - 3) * 0.3), (restored,), {"1": candidate()}
        )
    assert (
        matches["1"].state == IdentityState.RESIDENT
        and matches["1"].resident_id == PROFILE.resident_id
    )
    matches, _ = identity.update(
        context(7, 4), (replace(restored, missed_frames=1, recently_lost=True),), {}
    )
    assert matches["1"].state == IdentityState.UNKNOWN
    identity.update(context(8, 5), (), {})
    assert identity._state == {}


def test_temporal_expiry_and_confirmation_window() -> None:
    identity = TemporalIdentity(CONFIG)
    identity.update(context(0, 0), (TRACK,), {"1": candidate()})
    matches, _ = identity.update(context(1, 0.01), (TRACK,), {"1": candidate()})
    assert matches["1"].state == IdentityState.CANDIDATE
    matches, _ = identity.update(context(2, 4), (TRACK,), {})
    assert matches["1"].state == IdentityState.UNKNOWN
    with pytest.raises(IdentityError):
        identity.update(context(1, 3), (TRACK,), {})


def test_enrollment_quality_duplicates_and_centroid() -> None:
    enrollment = Enrollment(replace(CONFIG, enrollment_samples=3))
    assert enrollment.accept(FaceQuality(False, "blurred"), A, START) == "blurred"
    with pytest.raises(IdentityError):
        enrollment.profile("Joseph")
    assert enrollment.accept(FaceQuality(True, "ok"), A, START) == "accepted"
    assert enrollment.accept(FaceQuality(True, "ok"), A, START + timedelta(seconds=1)).startswith(
        "duplicate"
    )
    second = face_normalize("test", (math.cos(0.15), math.sin(0.15)))
    third = face_normalize("test", (math.cos(-0.15), math.sin(-0.15)))
    assert (
        enrollment.accept(FaceQuality(True, "ok", "left"), second, START + timedelta(seconds=2))
        == "accepted"
    )
    assert (
        enrollment.accept(FaceQuality(True, "ok", "right"), third, START + timedelta(seconds=3))
        == "accepted"
    )
    profile = enrollment.profile("Joseph")
    assert profile.sample_count == 3 and profile.templates[0].values == pytest.approx((1.0, 0.0))
    assert "values" not in repr(profile)


def test_enrollment_requires_pose_variation_and_consistency() -> None:
    enrollment = Enrollment(replace(CONFIG, enrollment_samples=3))
    enrollment.accept(FaceQuality(True, "ok"), A, START)
    assert "wait" in enrollment.accept(FaceQuality(True, "ok"), B, START + timedelta(seconds=0.1))
    assert "inconsistent" in enrollment.accept(
        FaceQuality(True, "ok"), B, START + timedelta(seconds=1)
    )
    for seq, angle in enumerate((0.15, 0.3), start=2):
        reason = enrollment.accept(
            FaceQuality(True, "ok"),
            face_normalize("test", (math.cos(angle), math.sin(angle))),
            START + timedelta(seconds=seq),
        )
    assert reason == "turn slightly for a second pose" and not enrollment.ready


def test_store_atomic_validation_delete_and_no_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("jake.adapters.local_identity_store.private_permissions", Mock())
    store = LocalIdentityStore(tmp_path / "identities")
    assert store.profiles() == ()
    store.add(PROFILE)
    assert store.profiles() == (PROFILE,)
    document, _ = decrypt(json.loads(store.path.read_bytes()), store.provider)
    assert isinstance(document, dict)
    assert set(document) == {"version", "residents"}
    assert set(document["residents"][0]) == {
        "resident_id",
        "display_name",
        "templates",
        "sample_count",
        "enrolled_at",
    }
    with pytest.raises(IdentityError, match="already"):
        store.add(replace(OTHER, display_name="JOSEPH"))
    original = store.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(
            "jake.adapters.local_identity_store.os.replace", Mock(side_effect=OSError("failed"))
        )
        with pytest.raises(OSError):
            store.add(OTHER)
    assert store.path.read_bytes() == original
    assert sorted(p.name for p in store.root.iterdir()) == ["residents.json"]
    store.delete(PROFILE.resident_id)
    assert store.profiles() == ()
    with pytest.raises(IdentityError):
        store.delete(PROFILE.resident_id)
    store.path.write_text('{"version":2,"residents":[]}', encoding="utf-8")
    with pytest.raises(IdentityError, match="corrupt"):
        store.add(PROFILE)
    assert '"version":2' in store.path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "payload", ["{", "[]", '{"version":true,"residents":[]}', '{"version":1,"residents":[{}]}']
)
def test_corrupt_store_rejected(tmp_path: Path, payload: str) -> None:
    store = LocalIdentityStore(tmp_path)
    store.path.write_text(payload, encoding="utf-8")
    with pytest.raises(IdentityError):
        store.profiles()


def sample_frame() -> Frame:
    pixels = np.random.default_rng(1).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    return Frame("test", 0, START, 200, 200, pixels.tobytes())


FACE = FaceDetection(BOX, 0.99, ((0.35, 0.35), (0.65, 0.35), (0.5, 0.5), (0.4, 0.65), (0.6, 0.65)))


def test_crop_quality_and_poor_observation_does_not_encode() -> None:
    frame = sample_frame()
    bgr, x, y = crop(frame, BoundingBox(0.5, 0.5, 1, 1))
    assert bgr.shape == (100, 100, 3) and (x, y) == (100, 100)
    assert face_quality(frame, FACE, CONFIG).accepted
    for face, reason in (
        (replace(FACE, confidence=0.1), "confidence"),
        (replace(FACE, clipped=True), "clipped"),
        (replace(FACE, box=BoundingBox(0.4, 0.4, 0.5, 0.5)), "small"),
        (replace(FACE, landmarks=()), "landmarks"),
    ):
        assert reason in face_quality(frame, face, CONFIG).reason
    blank = replace(frame, pixels=bytes(200 * 200 * 3))
    assert face_quality(blank, FACE, CONFIG).reason == "blurred face"
    detector, encoder, store = Mock(), Mock(), Mock()
    store.profiles.return_value = (PROFILE,)
    detector.detect.return_value = (FACE,)
    service = FaceIdentityService(
        CONFIG, detector, encoder, store, lambda *_: FaceQuality(False, "blurred")
    )
    matches, _ = service.process(frame, (TRACK,))
    assert matches["1"].state == IdentityState.UNKNOWN
    encoder.encode.assert_not_called()


def test_face_model_adapters_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detector_path, encoder_path = tmp_path / "detector.onnx", tmp_path / "encoder.onnx"
    config = replace(CONFIG, detector_model=str(detector_path), encoder_model=str(encoder_path))
    with pytest.raises(IdentityError):
        YuNetFaceDetector(config)
    with pytest.raises(IdentityError):
        SFaceEncoder(config)
    detector_path.write_bytes(b"model")
    encoder_path.write_bytes(b"model")
    detector, encoder = Mock(), Mock()
    monkeypatch.setattr("cv2.FaceDetectorYN.create", Mock(return_value=detector))
    monkeypatch.setattr("cv2.FaceRecognizerSF.create", Mock(return_value=encoder))
    detector.detect.return_value = (
        1,
        np.array(
            [[10, 10, 80, 80, 30, 30, 70, 30, 50, 50, 35, 70, 65, 70, 0.99]], dtype=np.float32
        ),
    )
    frame = sample_frame()
    face = YuNetFaceDetector(config).detect(frame, BoundingBox(0.25, 0.25, 0.75, 0.75))[0]
    assert face.box.left == 0.3 and face.landmarks[0] == (0.4, 0.4)
    encoder.alignCrop.return_value = np.ones((112, 112, 3), dtype=np.uint8)
    encoder.feature.return_value = np.ones((1, 128), dtype=np.float32)
    output = SFaceEncoder(config).encode(frame, face)
    assert len(output.values) == 128 and np.linalg.norm(output.values) == pytest.approx(1)
    assert output.model_id.startswith("opencv-sface")
    assert encoder.alignCrop.call_args.args[0].shape == (80, 80, 3)
    assert set(f.name for f in fields(PersonEvent)) == {
        "event_id",
        "kind",
        "context",
        "track_id",
        "entered_at",
    }
    assert not {"resident_id", "display_name", "templates", "embedding"} & {
        f.name for f in fields(PersonTrack)
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(required_confirmations=1),
        dict(enrollment_samples=1),
        dict(resident_similarity=0.1),
        dict(candidate_similarity=float("nan")),
        dict(min_face_pixels=True),
        dict(encoder_model="https://bad.onnx"),
        dict(carry_seconds=0),
    ],
)
def test_invalid_identity_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IdentityConfig(**kwargs)  # type: ignore[arg-type]


def test_enrollment_cli_mocked_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from jake.enroll_cli import main

    path = tmp_path / "local.toml"
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[camera]\nwidth=1920\nheight=1080\nfps=30\n'
        "[identity]\nenrollment_samples=3",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(["--name", "Joseph", "--config", str(path)])
    store, detector, encoder = Mock(), Mock(), Mock()
    store.profiles.return_value = ()
    detector.detect.return_value = (FACE,)
    encoder.encode.side_effect = [
        A,
        face_normalize("test", (math.cos(0.15), math.sin(0.15))),
        face_normalize("test", (math.cos(-0.15), math.sin(-0.15))),
    ]
    camera = MagicMock()
    camera.__enter__.return_value = [
        replace(sample_frame(), sequence=i, captured_at=START + timedelta(seconds=i))
        for i in range(3)
    ]
    monkeypatch.setattr(
        "jake.adapters.local_identity_store.LocalIdentityStore", Mock(return_value=store)
    )
    monkeypatch.setattr("jake.adapters.opencv_faces.YuNetFaceDetector", Mock(return_value=detector))
    monkeypatch.setattr("jake.adapters.opencv_faces.SFaceEncoder", Mock(return_value=encoder))
    monkeypatch.setattr(
        "jake.adapters.opencv_faces.face_quality",
        Mock(side_effect=[FaceQuality(True, "ok", pose) for pose in ("center", "left", "right")]),
    )
    camera_factory = Mock(return_value=camera)
    monkeypatch.setattr("jake.adapters.opencv_camera.OpenCVCamera", camera_factory)
    for method in ("namedWindow", "imshow", "putText", "destroyWindow"):
        monkeypatch.setattr(f"cv2.{method}", Mock())
    monkeypatch.setattr("cv2.waitKey", Mock(return_value=-1))
    assert main(["--name", "Joseph", "--consent", "--config", str(path)]) == 0
    assert camera_factory.call_args.args[1].width == 1920
    assert camera_factory.call_args.args[1].height == 1080
    assert camera_factory.call_args.args[1].fps == 30
    assert store.add.call_args.args[0].sample_count == 3
    camera.__exit__.assert_called_once()
    assert main(["--list", "--config", str(path)]) == 0
    assert main(["--delete", PROFILE.resident_id, "--config", str(path)]) == 0
    store.delete.assert_called_once_with(PROFILE.resident_id)


def test_service_valid_faces_and_ambiguous_regions() -> None:
    detector, encoder, store = Mock(), Mock(), Mock()
    detector.detect.return_value = (FACE,)
    encoder.encode.return_value = A
    store.profiles.return_value = (PROFILE,)
    service = FaceIdentityService(
        CONFIG, detector, encoder, store, lambda *_: FaceQuality(True, "ok")
    )
    for seq in range(3):
        matches, events = service.process(
            replace(sample_frame(), sequence=seq, captured_at=START + timedelta(seconds=0.3 * seq)),
            (TRACK,),
        )
    assert matches["1"].state == IdentityState.RESIDENT and events[0].kind == "IDENTITY_RESOLVED"
    encoder.reset_mock()
    service.process(
        replace(sample_frame(), sequence=3, captured_at=START + timedelta(seconds=1)),
        (TRACK, replace(TRACK, track_id="2")),
    )
    encoder.encode.assert_not_called()


def test_quality_extreme_pose() -> None:
    landmarks = list(FACE.landmarks)
    landmarks[2] = (0.8, 0.5)
    assert (
        face_quality(sample_frame(), replace(FACE, landmarks=tuple(landmarks)), CONFIG).reason
        == "extreme pose"
    )


def test_store_lock_and_invalid_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jake.adapters.local_identity_store.private_permissions", Mock())
    store = LocalIdentityStore(tmp_path)
    (tmp_path / ".writer.lock").touch()
    with pytest.raises(IdentityError, match="locked"):
        store.add(PROFILE)
    (tmp_path / ".writer.lock").unlink()
    store.add(PROFILE)
    data, key_id = decrypt(json.loads(store.path.read_bytes()), store.provider)
    assert isinstance(data, dict)
    data["residents"][0]["templates"][0]["values"] = [2.0, 0.0]
    store.path.write_bytes(encrypt(data, key_id, store.provider))
    with pytest.raises(IdentityError, match="corrupt"):
        store.profiles()


def test_private_permissions_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from jake.adapters.local_identity_store import private_permissions

    path = tmp_path / "private"
    path.mkdir()
    if os.name == "nt":
        run = Mock(return_value=Mock(stdout='"user","S-1-5-21-123"\n'))
        monkeypatch.setattr("jake.adapters.local_identity_store.subprocess.run", run)
        private_permissions(path, True)
        assert "/inheritance:r" in run.call_args.args[0]
        assert "*S-1-5-21-123:(OI)(CI)F" in run.call_args.args[0]
    else:
        private_permissions(path, True)
        assert path.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("exit_mode", ["q", "Q", "interrupt", "model_error"])
def test_enrollment_cancel_and_failure_release_without_saving(
    exit_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from jake.enroll_cli import main

    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id="test"', encoding="utf-8")
    store, detector, encoder = Mock(), Mock(), Mock()
    store.profiles.return_value = ()
    detector.detect.return_value = ()
    if exit_mode == "interrupt":
        detector.detect.side_effect = KeyboardInterrupt
    elif exit_mode == "model_error":
        detector.detect.side_effect = IdentityError("local model failed")
    camera = MagicMock()
    camera.__enter__.return_value = [sample_frame()]
    monkeypatch.setattr(
        "jake.adapters.local_identity_store.LocalIdentityStore", Mock(return_value=store)
    )
    monkeypatch.setattr("jake.adapters.opencv_faces.YuNetFaceDetector", Mock(return_value=detector))
    monkeypatch.setattr("jake.adapters.opencv_faces.SFaceEncoder", Mock(return_value=encoder))
    monkeypatch.setattr("jake.adapters.opencv_camera.OpenCVCamera", Mock(return_value=camera))
    destroy = Mock()
    for method in ("namedWindow", "imshow", "putText"):
        monkeypatch.setattr(f"cv2.{method}", Mock())
    monkeypatch.setattr("cv2.destroyWindow", destroy)
    monkeypatch.setattr("cv2.waitKey", Mock(return_value=ord(exit_mode[0])))
    assert main(["--name", "Joseph", "--consent", "--config", str(path)]) == (
        1 if exit_mode == "model_error" else 0
    )
    camera.__exit__.assert_called_once()
    destroy.assert_called_once()
    store.add.assert_not_called()
    encoder.encode.assert_not_called()


def test_pipeline_keeps_identity_and_person_events_separate() -> None:
    from jake.config import PipelineConfig
    from jake.pipeline import PerceptionPipeline

    detector, tracker, events, identity = Mock(), Mock(), Mock(), Mock()
    detector.detect.return_value = ()
    tracker.update.return_value = (TRACK,)
    events.generate.return_value = ()
    identity.process.return_value = ({"1": IdentityMatch(IdentityState.UNKNOWN)}, ())
    pipeline = PerceptionPipeline(
        PipelineConfig("test"), detector, tracker, events, identity=identity
    )
    frame = sample_frame()
    assert pipeline.process(frame) == ()
    identity.process.assert_called_once_with(frame, (TRACK,))
    assert pipeline.identity_matches["1"].state == IdentityState.UNKNOWN
    assert pipeline.identity_events == ()


def test_identity_domain_rejects_malformed_profiles_and_landmarks() -> None:
    for change in ({"sample_count": 3.5}, {"display_name": "Joseph\x7f"}):
        with pytest.raises(ValueError):
            replace(PROFILE, **change)
    with pytest.raises(ValueError, match="five normalized points"):
        FaceDetection(BOX, 0.9, ((0.1, 0.2, 0.3),) * 5)  # type: ignore[arg-type]


def test_matcher_cannot_bypass_temporal_confirmation() -> None:
    identity = TemporalIdentity(CONFIG)
    proposed = IdentityMatch(IdentityState.RESIDENT, PROFILE.resident_id, "Joseph", 0.9)
    results, _ = identity.update(context(0, 0), (TRACK,), {"1": proposed})
    assert results["1"].state == IdentityState.CANDIDATE
    for seq in (1, 2):
        results, _ = identity.update(context(seq, seq), (TRACK,), {"1": proposed})
    assert results["1"].state == IdentityState.RESIDENT
    conflict = IdentityMatch(IdentityState.RESIDENT, OTHER.resident_id, "Other", 0.9)
    results, _ = identity.update(context(3, 3), (TRACK,), {"1": conflict})
    assert results["1"].state == IdentityState.CANDIDATE
