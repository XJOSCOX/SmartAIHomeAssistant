import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from jake.adapters.yolo_detector import DetectorError, YoloPersonDetector, _load_model
from jake.config import DetectorConfig
from jake.domain import BoundingBox, Frame, PersonDetection
from jake.ports import PersonDetector


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("detector tests must never use the network")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def config(tmp_path: Path) -> DetectorConfig:
    path = tmp_path / "fake.pt"
    path.write_bytes(b"fake weights; never loaded by a real model")
    return DetectorConfig(str(path))


@pytest.fixture
def model() -> Mock:
    model = Mock()
    model.names = {0: "cat", 7: "person"}
    model.predict.return_value = [Mock()]
    set_rows(model, [])
    return model


def set_rows(model: Mock, rows: list[list[float]]) -> None:
    model.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = rows


def frame() -> Frame:
    return Frame("test", 10, datetime.now(UTC), 100, 50, bytes(100 * 50 * 3))


def test_only_people_confidence_and_normalization(config: DetectorConfig, model: Mock) -> None:
    set_rows(model, [[10, 5, 90, 45, 0.974, 7], [0, 0, 30, 30, 0.99, 0]])
    detector: PersonDetector = YoloPersonDetector(config, 0.5, model_factory=lambda _: model)
    assert detector.detect(frame()) == (PersonDetection(BoundingBox(0.1, 0.1, 0.9, 0.9), 0.974),)
    options = model.predict.call_args.kwargs
    assert options["classes"] == [7]
    assert options["conf"] == 0.5
    assert options["device"] == "cpu"
    assert options["imgsz"] == 640
    for flag in ("save", "save_txt", "save_crop", "show", "visualize", "verbose", "stream"):
        assert options[flag] is False


def test_rgb_reconstructed_as_owned_bgr_for_ultralytics(
    config: DetectorConfig, model: Mock
) -> None:
    original = Frame("test", 0, datetime.now(UTC), 2, 1, bytes([1, 2, 3, 4, 5, 6]))
    detector = YoloPersonDetector(config, 0.5, model_factory=lambda _: model)
    detector.detect(original)
    image = model.predict.call_args.kwargs["source"]
    np.testing.assert_array_equal(image, np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8))
    assert image.flags.c_contiguous and image.flags.owndata
    image[:] = 255
    assert original.pixels == bytes([1, 2, 3, 4, 5, 6])


def test_multiple_people_and_independent_frames(config: DetectorConfig, model: Mock) -> None:
    set_rows(model, [[0, 0, 50, 50, 0.9, 7], [50, 0, 100, 50, 0.8, 7]])
    detector = YoloPersonDetector(config, 0.5, model_factory=lambda _: model)
    first = detector.detect(frame())
    assert len(first) == 2
    assert detector.detect(frame()) == first
    assert model.predict.call_count == 2


def test_zero_people(config: DetectorConfig, model: Mock) -> None:
    assert YoloPersonDetector(config, 0.5, model_factory=lambda _: model).detect(frame()) == ()


@pytest.mark.parametrize(
    "row",
    [
        [10, 0, 10, 50, 0.9, 7],
        [20, 0, 10, 50, 0.9, 7],
        [0, 20, 10, 10, 0.9, 7],
        [101, 0, 110, 50, 0.9, 7],
        [-20, 0, -10, 50, 0.9, 7],
        [0, 0, float("nan"), 50, 0.9, 7],
        [0, 0, 10, float("inf"), 0.9, 7],
        [0, 0, 10, 50, float("nan"), 7],
        [0, 0, 10, 50, 1.1, 7],
        [0, 0, 10, 50, -0.1, 7],
        [0, 0, 10, 50, 0.49, 7],
        [0, 0, 10, 50, 0.9, 7.5],
    ],
)
def test_invalid_or_low_confidence_rows_skipped(
    config: DetectorConfig, model: Mock, row: list[float]
) -> None:
    set_rows(model, [row])
    assert YoloPersonDetector(config, 0.5, model_factory=lambda _: model).detect(frame()) == ()


def test_clips_partial_boxes_and_accepts_threshold(config: DetectorConfig, model: Mock) -> None:
    set_rows(model, [[-10, -5, 120, 55, 0.5, 7]])
    result = YoloPersonDetector(config, 0.5, model_factory=lambda _: model).detect(frame())
    assert result == (PersonDetection(BoundingBox(0, 0, 1, 1), 0.5),)


def test_predict_error_wrapped_with_cause(config: DetectorConfig, model: Mock) -> None:
    failure = RuntimeError("backend failed")
    model.predict.side_effect = failure
    detector = YoloPersonDetector(config, 0.5, model_factory=lambda _: model)
    with pytest.raises(DetectorError, match="inference") as raised:
        detector.detect(frame())
    assert raised.value.__cause__ is failure


def test_model_setup_error_wrapped(config: DetectorConfig) -> None:
    factory = Mock(side_effect=RuntimeError("invalid model"))
    with pytest.raises(DetectorError, match="initialize"):
        YoloPersonDetector(config, 0.5, model_factory=factory)


def test_missing_weights_never_call_factory(tmp_path: Path) -> None:
    factory = Mock()
    with pytest.raises(DetectorError, match="Download weights explicitly"):
        YoloPersonDetector(DetectorConfig(str(tmp_path / "missing.pt")), 0.5, model_factory=factory)
    factory.assert_not_called()


def test_missing_person_class_fails(config: DetectorConfig, model: Mock) -> None:
    model.names = {0: "cat"}
    with pytest.raises(DetectorError, match="no class named 'person'"):
        YoloPersonDetector(config, 0.5, model_factory=lambda _: model)


def test_unexpected_output_is_clear(config: DetectorConfig, model: Mock) -> None:
    set_rows(model, [[0, 0, 50, 50, 0.9, 7, 99]])
    detector = YoloPersonDetector(config, 0.5, model_factory=lambda _: model)
    with pytest.raises(DetectorError, match="untracked"):
        detector.detect(frame())
    model.predict.return_value = []
    with pytest.raises(DetectorError, match="one detection result"):
        detector.detect(frame())


def test_privacy_settings_applied_before_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    package, utils = Mock(), Mock()

    def fake_import(name: str) -> Mock:
        assert os.environ["YOLO_OFFLINE"] == "true"
        assert os.environ["YOLO_AUTOINSTALL"] == "false"
        return utils if name.endswith(".utils") else package

    for key in ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE"):
        monkeypatch.setenv(key, "override-me")
    monkeypatch.setattr("jake.adapters.yolo_detector.importlib.import_module", fake_import)
    _load_model("local.pt")
    assert utils.ONLINE is False
    assert utils.AUTOINSTALL is False
    utils.SETTINGS.update.assert_called_once_with({"sync": False})
    package.YOLO.assert_called_once_with("local.pt", task="detect")


def test_missing_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE"):
        monkeypatch.setenv(key, "false")
    monkeypatch.setattr(
        "jake.adapters.yolo_detector.importlib.import_module", Mock(side_effect=ImportError)
    )
    with pytest.raises(DetectorError, match="uv sync --extra detection"):
        _load_model("local.pt")
