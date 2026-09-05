from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from jake.adapters.iou_tracker import IoUPersonTracker, TrackerError
from jake.adapters.yolo_detector import DetectorError
from jake.cli import main
from jake.config import AppConfig, CameraConfig, PipelineConfig, TrackingConfig
from jake.domain import BoundingBox, PersonDetection
from jake.preview import preview


@pytest.fixture
def desktop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = True, np.zeros((40, 80, 3), dtype=np.uint8)
    mocks = {"capture": capture, "VideoCapture": Mock(return_value=capture)}
    for name in ("namedWindow", "imshow", "putText", "destroyWindow"):
        mocks[name] = Mock()
    mocks["waitKey"] = Mock(return_value=ord("q"))
    for name, mock in mocks.items():
        if name != "capture":
            monkeypatch.setattr(cv2, name, mock)
    return mocks


def test_preview_quit_and_metadata(desktop: dict[str, Mock]) -> None:
    preview(AppConfig(PipelineConfig("test"), CameraConfig(3)))
    desktop["VideoCapture"].assert_called_once_with(3)
    desktop["capture"].release.assert_called_once_with()
    desktop["destroyWindow"].assert_called_once()
    desktop["imshow"].assert_called_once()
    overlay = desktop["putText"].call_args.args[1]
    assert "seq 0" in overlay and "FPS" in overlay and "80x40" in overlay


@pytest.mark.parametrize("failure_point", ["namedWindow", "imshow", "waitKey"])
def test_display_error_releases_camera_and_window(
    desktop: dict[str, Mock], failure_point: str
) -> None:
    desktop[failure_point].side_effect = cv2.error("display failed")
    with pytest.raises(cv2.error, match="display failed"):
        preview(AppConfig(PipelineConfig("test")))
    desktop["capture"].release.assert_called_once_with()
    desktop["destroyWindow"].assert_called_once()


def test_cli_ctrl_c_cleans_up(desktop: dict[str, Mock], tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    desktop["waitKey"].side_effect = KeyboardInterrupt
    assert main(["--config", str(path)]) == 0
    desktop["capture"].release.assert_called_once_with()
    desktop["destroyWindow"].assert_called_once()


def test_cli_camera_failure_is_clear(
    desktop: dict[str, Mock], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    desktop["capture"].isOpened.return_value = False
    assert main(["--config", str(path)]) == 1
    assert "Cannot open local camera index 0" in capsys.readouterr().err
    desktop["capture"].release.assert_called_once_with()


def test_cli_missing_configuration_does_not_open_camera(
    desktop: dict[str, Mock], tmp_path: Path
) -> None:
    assert main(["--config", str(tmp_path / "missing.toml")]) == 2
    desktop["VideoCapture"].assert_not_called()


def test_tracking_preview_keeps_ids_and_reports_retained_tracks(
    desktop: dict[str, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "rectangle", Mock())
    desktop["waitKey"].side_effect = [-1, -1, ord("Q")]
    detector = Mock()
    person = PersonDetection(BoundingBox(0.1, 0.5, 0.9, 1), 0.942)
    detector.detect.side_effect = [(person,), (), (person,)]
    tracker = IoUPersonTracker(TrackingConfig())
    preview(AppConfig(PipelineConfig("test")), detector, tracker)
    labels = [call.args[1] for call in desktop["putText"].call_args_list]
    assert labels.count("ID 1 | PERSON 94.2%") == 3
    assert labels.count("active tracks 1 (includes missed)") == 3
    assert any("people 0" in text and "inference" in text for text in labels)
    assert not any("ID 2" in text for text in labels)
    desktop["capture"].release.assert_called_once_with()


def test_tracking_requires_detector_before_camera_open(desktop: dict[str, Mock]) -> None:
    with pytest.raises(ValueError, match="requires a detector"):
        preview(AppConfig(PipelineConfig("test")), tracker=IoUPersonTracker(TrackingConfig()))
    desktop["VideoCapture"].assert_not_called()


def test_tracker_failure_cleans_up_preview(desktop: dict[str, Mock]) -> None:
    detector, tracker = Mock(), Mock()
    detector.detect.return_value = ()
    tracker.update.side_effect = TrackerError("invalid session")
    with pytest.raises(TrackerError):
        preview(AppConfig(PipelineConfig("test")), detector, tracker)
    desktop["capture"].release.assert_called_once_with()
    desktop["destroyWindow"].assert_called_once()


def test_cli_track_implies_detect_and_passes_tracker_configuration(
    desktop: dict[str, Mock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[pipeline]\ncamera_id = "test"\n[tracking]\nmin_iou = 0.8\nmax_missed_frames = 3',
        encoding="utf-8",
    )
    detector_factory, tracker_factory = Mock(), Mock()
    detector_factory.return_value.detect.return_value = ()
    tracker_factory.return_value.update.return_value = ()
    monkeypatch.setattr("jake.adapters.yolo_detector.YoloPersonDetector", detector_factory)
    monkeypatch.setattr("jake.adapters.iou_tracker.IoUPersonTracker", tracker_factory)
    assert main(["--config", str(config), "--detect"]) == 0
    tracker_factory.assert_not_called()
    detector_factory.reset_mock()
    assert main(["--config", str(config), "--track"]) == 0
    detector_factory.assert_called_once()
    tracker_factory.assert_called_once_with(TrackingConfig(0.8, 3))
    assert tracker_factory.return_value.update.call_args.args[0].camera_id == "test"


def test_cli_reports_tracker_error(
    desktop: dict[str, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    detector_factory, tracker_factory = Mock(), Mock()
    detector_factory.return_value.detect.return_value = ()
    tracker_factory.return_value.update.side_effect = TrackerError("invalid tracker frame")
    monkeypatch.setattr("jake.adapters.yolo_detector.YoloPersonDetector", detector_factory)
    monkeypatch.setattr("jake.adapters.iou_tracker.IoUPersonTracker", tracker_factory)
    assert main(["--config", str(config), "--track"]) == 1
    assert "invalid tracker frame" in capsys.readouterr().err
    desktop["capture"].release.assert_called_once_with()


@pytest.mark.parametrize("key", ["q", "Q"])
def test_both_quit_keys_release_camera(desktop: dict[str, Mock], key: str) -> None:
    desktop["waitKey"].return_value = ord(key)
    preview(AppConfig(PipelineConfig("test")))
    desktop["capture"].read.assert_called_once()
    desktop["capture"].release.assert_called_once_with()


def test_detection_preview_draws_structured_results(
    desktop: dict[str, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    rectangle = Mock()
    monkeypatch.setattr(cv2, "rectangle", rectangle)
    monkeypatch.setattr("jake.diagnostics.perf_counter", Mock(side_effect=[10.0, 10.025]))
    detector = Mock()
    detector.detect.return_value = (PersonDetection(BoundingBox(0.1, 0.5, 0.9, 1), 0.974),)
    preview(AppConfig(PipelineConfig("test")), detector)
    detector.detect.assert_called_once()
    assert rectangle.call_args.args[1:3] == ((8, 20), (72, 39))
    labels = [call.args[1] for call in desktop["putText"].call_args_list]
    assert "PERSON 97.4%" in labels
    assert any("people 1" in text and "25.0 ms" in text and "40.0 FPS" in text for text in labels)
    desktop["capture"].release.assert_called_once_with()


def test_detector_failure_releases_preview(desktop: dict[str, Mock]) -> None:
    detector = Mock()
    detector.detect.side_effect = DetectorError("inference failed")
    with pytest.raises(DetectorError):
        preview(AppConfig(PipelineConfig("test")), detector)
    desktop["capture"].release.assert_called_once_with()
    desktop["destroyWindow"].assert_called_once()


def test_cli_detection_is_opt_in_and_uses_configuration(
    desktop: dict[str, Mock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[pipeline]\ncamera_id = "test"\nmin_person_confidence = 0.7', encoding="utf-8"
    )
    model_factory = Mock()
    model_factory.return_value.detect.return_value = ()
    monkeypatch.setattr("jake.adapters.yolo_detector.YoloPersonDetector", model_factory)
    assert main(["--config", str(config)]) == 0
    model_factory.assert_not_called()
    assert main(["--config", str(config), "--detect"]) == 0
    assert model_factory.call_args.args[1] == 0.7


def test_missing_model_fails_before_camera_open(desktop: dict[str, Mock], tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[pipeline]\ncamera_id = "test"\n[detector]\nmodel = "missing-model.pt"', encoding="utf-8"
    )
    assert main(["--config", str(config), "--detect"]) == 1
    desktop["VideoCapture"].assert_not_called()


@pytest.mark.parametrize("debug", [False, True])
def test_kalman_cli_and_debug_diagnostics(
    desktop: dict[str, Mock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, debug: bool
) -> None:
    rectangles = Mock()
    monkeypatch.setattr(cv2, "rectangle", rectangles)
    desktop["waitKey"].side_effect = [-1, ord("Q")]
    factory = Mock()
    person = PersonDetection(BoundingBox(0.1, 0.5, 0.9, 1), 0.9)
    factory.return_value.detect.side_effect = [(person,), ()]
    monkeypatch.setattr("jake.adapters.yolo_detector.YoloPersonDetector", factory)
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    args = ["--config", str(path), "--track", "--tracker", "kalman"]
    if debug:
        args.append("--debug-tracks")
    assert main(args) == 0
    labels = [call.args[1] for call in desktop["putText"].call_args_list]
    assert labels.count("ID 1 | PERSON 90.0%") == 2
    assert ("ID 1 | missed 1" in labels) is debug
    assert any(call.args[3] == (0, 165, 255) for call in rectangles.call_args_list) is debug
    desktop["capture"].release.assert_called_once_with()


@pytest.mark.parametrize(
    "args", [["--tracker", "kalman"], ["--debug-tracks"], ["--track", "--debug-tracks"]]
)
def test_invalid_tracker_cli_options_do_not_open_camera(
    desktop: dict[str, Mock], args: list[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args)
    assert raised.value.code == 2
    desktop["VideoCapture"].assert_not_called()


@pytest.mark.parametrize("override", [None, "greedy"])
def test_assignment_cli_config_precedence_and_debug_label(
    desktop: dict[str, Mock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    factory = Mock()
    factory.return_value.detect.return_value = ()
    monkeypatch.setattr("jake.adapters.yolo_detector.YoloPersonDetector", factory)
    path = tmp_path / "config.toml"
    path.write_text(
        '[pipeline]\ncamera_id = "test"\n[tracking]\nassignment = "hungarian"', encoding="utf-8"
    )
    args = ["--config", str(path), "--track", "--tracker", "kalman", "--debug-tracks"]
    if override:
        args.extend(["--assignment", override])
    assert main(args) == 0
    labels = [call.args[1] for call in desktop["putText"].call_args_list]
    assert f"assignment {override or 'hungarian'}" in labels
    assert 'assignment = "hungarian"' in path.read_text(encoding="utf-8")


def test_assignment_flag_requires_tracking(desktop: dict[str, Mock]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--assignment", "hungarian"])
    assert raised.value.code == 2
    desktop["VideoCapture"].assert_not_called()
