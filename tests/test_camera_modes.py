from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from jake.adapters.opencv_camera import OpenCVCamera
from jake.adapters.opencv_faces import crop, face_quality
from jake.adapters.yolo_detector import YoloPersonDetector
from jake.config import CameraConfig, DetectorConfig, PipelineConfig, load_app_config
from jake.domain import BoundingBox, Frame, PersonTrack
from jake.face_identity import FaceIdentityService
from jake.identity import Enrollment, face_normalize
from jake.identity_config import IdentityConfig
from jake.identity_domain import FaceDetection, FaceQuality
from jake.pipeline import PerceptionPipeline


def test_full_mode_and_device_only_configs(tmp_path: Path) -> None:
    path = tmp_path / "camera.toml"
    path.write_text('[pipeline]\ncamera_id="test"\n[camera]\ndevice=2\n', encoding="utf-8")
    assert load_app_config(path).camera == CameraConfig(2)
    with path.open("a", encoding="utf-8") as f:
        f.write('width=1920\nheight=1080\nfps=30\nbackend="dshow"\nfourcc="MJPG"')
    assert load_app_config(path).camera == CameraConfig(2, 1920, 1080, 30, "dshow", "MJPG")


@pytest.mark.parametrize(
    "setting",
    [
        "width=0",
        "width=-1",
        "width=true",
        "width=1.5",
        'width="1920"',
        "height=0",
        "height=false",
        "height=2.5",
        "fps=0",
        "fps=-1",
        "fps=nan",
        "fps=inf",
        "fps=true",
        'fps="30"',
        'backend="bad"',
        'fourcc="MJPEG"',
        'fourcc=""',
        "fourcc=1234",
    ],
)
def test_invalid_mode(tmp_path: Path, setting: str) -> None:
    path = tmp_path / "camera.toml"
    path.write_text('[pipeline]\ncamera_id="test"\n[camera]\n' + setting, encoding="utf-8")
    with pytest.raises(ValueError):
        load_app_config(path)


def device() -> Mock:
    capture = Mock()
    capture.isOpened.return_value = True
    capture.set.return_value = True
    values = {
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
        cv2.CAP_PROP_FPS: 29.97,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter.fourcc(*"MJPG"),
    }
    capture.get.side_effect = values.get
    capture.getBackendName.return_value = "DSHOW"
    capture.read.return_value = True, np.zeros((720, 1280, 3), dtype=np.uint8)
    return capture


def test_requests_and_reports_actual_mode_and_measures_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = device()
    monkeypatch.setattr("jake.adapters.opencv_camera.perf_counter", Mock(side_effect=[10.0, 10.1]))
    with OpenCVCamera(
        "test",
        CameraConfig(width=1920, height=1080, fps=30, fourcc="MJPG"),
        capture_factory=lambda _: capture,
    ) as camera:
        assert [c.args[0] for c in capture.set.call_args_list] == [
            cv2.CAP_PROP_FOURCC,
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        ]
        assert [c.args[1] for c in capture.set.call_args_list][1:] == [1920, 1080, 30]
        assert camera.info.actual_width == 1280 and camera.info.actual_height == 720
        assert camera.info.actual_fps == 29.97 and camera.info.codec == "MJPG"
        assert camera.info.mismatch and "REQUEST NOT HONORED" in camera.info.summary
        frame = next(camera)
        assert (frame.width, frame.height) == (1280, 720)
        assert camera.info.capture_fps is None
        next(camera)
        assert camera.info.capture_fps == pytest.approx(10)
    capture.release.assert_called_once()


@pytest.mark.parametrize(
    "backend,api",
    [
        ("auto", None),
        ("dshow", cv2.CAP_DSHOW),
        ("msmf", cv2.CAP_MSMF),
        ("v4l2", cv2.CAP_V4L2),
        ("gstreamer", cv2.CAP_GSTREAMER),
    ],
)
def test_explicit_backend(backend: str, api: int | None) -> None:
    factory = Mock(return_value=device())
    with OpenCVCamera("test", CameraConfig(2, backend=backend), capture_factory=factory):
        pass
    factory.assert_called_once_with(*((2,) if api is None else (2, api)))
    factory.return_value.set.assert_not_called()


def test_unsupported_properties_are_best_effort() -> None:
    capture = device()
    capture.set.side_effect = cv2.error("unsupported")
    capture.get.side_effect = cv2.error("unsupported")
    capture.getBackendName.side_effect = cv2.error("unsupported")
    with OpenCVCamera(
        "test",
        CameraConfig(width=3840, height=2160, fps=30, fourcc="MJPG"),
        capture_factory=lambda _: capture,
    ) as camera:
        assert camera.info.actual_width is None
        assert len(camera.info.warnings) == 4
        assert next(camera).width == camera.info.actual_width == 1280
        assert camera.info.actual_fps is None


@pytest.mark.parametrize(
    "size", [(640, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (800, 600)]
)
def test_source_dimensions_and_detector_normalization(
    size: tuple[int, int], tmp_path: Path
) -> None:
    width, height = size
    capture = device()
    capture.read.return_value = True, np.zeros((height, width, 3), dtype=np.uint8)
    with OpenCVCamera("test", CameraConfig(), capture_factory=lambda _: capture) as camera:
        frame = next(camera)
        assert (frame.width, frame.height) == (width, height)
        assert (camera.info.actual_width, camera.info.actual_height) == size
    weights = tmp_path / "fake.pt"
    weights.touch()
    model = Mock()
    model.names = {0: "person"}
    result = Mock()
    result.boxes.data.cpu.return_value.tolist.return_value = [
        [width / 4, height / 4, width * 3 / 4, height * 3 / 4, 0.9, 0]
    ]
    model.predict.return_value = [result]
    detection = YoloPersonDetector(
        DetectorConfig(str(weights)), 0.5, model_factory=lambda _: model
    ).detect(frame)[0]
    assert detection.box == BoundingBox(0.25, 0.25, 0.75, 0.75)
    assert model.predict.call_args.kwargs["source"].shape == (height, width, 3)
    assert model.predict.call_args.kwargs["imgsz"] == 640


def test_high_resolution_identity_and_visitor_evidence_preserve_frame_and_cadence() -> None:
    frame = Frame("test", 0, datetime(2026, 1, 1, tzinfo=UTC), 1920, 1080, bytes(1920 * 1080 * 3))
    box = BoundingBox(0.1, 0.1, 0.9, 0.9)
    detector, face_detector, encoder, store, events, tracker, visitors = (Mock() for _ in range(7))
    detector.detect.return_value = ()
    tracker.update.return_value = (PersonTrack("1", box, 0.9),)
    events.generate.return_value = ()
    face_detector.detect.return_value = (FaceDetection(box, 0.99),)
    encoder.encode.return_value = face_normalize("fake", (1.0, 0.0))
    store.profiles.return_value = ()
    visitors.process.return_value = {}, ()
    identity = FaceIdentityService(
        IdentityConfig(),
        face_detector,
        encoder,
        store,
        lambda *_: FaceQuality(True, "accepted"),
        collect_visitors=True,
    )
    pipeline = PerceptionPipeline(
        PipelineConfig("test"), detector, tracker, events, identity=identity, visitors=visitors
    )
    pipeline.process(frame)
    assert detector.detect.call_args.args[0] is frame
    assert face_detector.detect.call_args.args[0] is frame
    assert encoder.encode.call_args.args[0] is frame
    assert visitors.process.called
    pipeline.process(
        replace(frame, sequence=1, captured_at=frame.captured_at + timedelta(seconds=0.1))
    )
    face_detector.detect.assert_called_once()
    encoder.encode.assert_called_once()


def test_crop_and_face_size_use_native_pixels() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[100:153, 200:247] = [10, 20, 30]
    frame = Frame("test", 0, datetime(2026, 1, 1, tzinfo=UTC), 1920, 1080, image.tobytes())
    box = BoundingBox(200 / 1920, 100 / 1080, 247 / 1920, 153 / 1080)
    bgr, x, y = crop(frame, box)
    assert (x, y) == (200, 100) and bgr.shape == (53, 47, 3)
    assert np.all(bgr == [30, 20, 10])
    quality = face_quality(frame, FaceDetection(box, 0.99), IdentityConfig())
    assert not quality.accepted
    assert (quality.face_width_px, quality.face_height_px) == (47, 53)
    assert quality.summary == "face too small: 47x53 px; minimum 64 px"
    assert Enrollment(IdentityConfig()).accept(quality, None, frame.captured_at) == quality.summary


def test_yunet_and_sface_use_original_roi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.adapters.opencv_faces import SFaceEncoder, YuNetFaceDetector

    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake")
    config = IdentityConfig(detector_model=str(path), encoder_model=str(path))
    detector, encoder = Mock(), Mock()
    monkeypatch.setattr("cv2.FaceDetectorYN.create", Mock(return_value=detector))
    monkeypatch.setattr("cv2.FaceRecognizerSF.create", Mock(return_value=encoder))
    detector.detect.return_value = (
        1,
        np.array(
            [[100, 100, 200, 180, 140, 140, 250, 140, 200, 180, 150, 230, 240, 230, 0.99]],
            dtype=np.float32,
        ),
    )
    image = np.full((1080, 1920, 3), [10, 20, 30], dtype=np.uint8)
    frame = Frame("test", 0, datetime(2026, 1, 1, tzinfo=UTC), 1920, 1080, image.tobytes())
    face = YuNetFaceDetector(config).detect(frame, BoundingBox(0.25, 0.25, 0.75, 0.75))[0]
    assert detector.detect.call_args.args[0].shape == (540, 960, 3)
    assert np.all(detector.detect.call_args.args[0] == [30, 20, 10])
    assert face.box.left == pytest.approx(580 / 1920)
    encoder.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
    encoder.feature.return_value = np.ones((1, 128), dtype=np.float32)
    SFaceEncoder(config).encode(frame, face)
    assert encoder.alignCrop.call_args.args[0].shape == (180, 200, 3)
    assert np.all(encoder.alignCrop.call_args.args[0] == [30, 20, 10])
    assert np.array_equal(np.frombuffer(frame.pixels, dtype=np.uint8).reshape(image.shape), image)


def test_visitor_size_diagnostic_and_no_embedding() -> None:
    frame = Frame("test", 0, datetime(2026, 1, 1, tzinfo=UTC), 1920, 1080, bytes(1920 * 1080 * 3))
    face = FaceDetection(BoundingBox(200 / 1920, 100 / 1080, 247 / 1920, 153 / 1080), 0.99)
    detector = Mock(detect=Mock(return_value=(face,)))
    encoder = Mock()
    service = FaceIdentityService(
        IdentityConfig(),
        detector,
        encoder,
        Mock(profiles=Mock(return_value=())),
        face_quality,
        collect_visitors=True,
    )
    service.process(frame, (PersonTrack("1", BoundingBox(0, 0, 1, 1), 0.9),))
    assert service.visitor_diagnostics["1"] == "rejected face too small: 47x53 px; minimum 64 px"
    assert service.face_qualities["1"].face_width_px == 47
    encoder.encode.assert_not_called()
    assert service.visitor_observations == {}


def test_set_false_and_unknown_readback() -> None:
    capture = device()
    capture.set.return_value = False
    capture.get.return_value = float("nan")
    capture.get.side_effect = None
    with OpenCVCamera("test", CameraConfig(fps=30), capture_factory=lambda _: capture) as camera:
        assert camera.info.warnings == ("fps request unsupported",)
        assert camera.info.actual_fps is None
        assert "? FPS driver-reported" in camera.info.summary
