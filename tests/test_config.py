from pathlib import Path

import pytest

from jake.config import (
    CameraConfig,
    DetectorConfig,
    KalmanConfig,
    PipelineConfig,
    TrackingConfig,
    load_app_config,
    load_config,
)


def test_example_configuration() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "jake.example.toml"
    assert load_config(path) == PipelineConfig("front-door", 0.5)


def test_default_threshold(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"\n', encoding="utf-8")
    assert load_config(path) == PipelineConfig("test")


@pytest.mark.parametrize(
    "contents",
    [
        "",
        '[other]\ncamera_id = "test"',
        '[pipeline]\ncamera_id = "test"\nunknown = 1',
        '[pipeline]\ncamera_id = "test"\nmin_person_confidence = true',
        '[pipeline]\ncamera_id = "test"\nmin_person_confidence = "high"',
        '[pipeline]\ncamera_id = "test"\nmin_person_confidence = 1.1',
        '[pipeline]\ncamera_id = "test"\nmin_person_confidence = nan',
        '[pipeline]\ncamera_id = " "',
        "[pipeline]\ncamera_id = 123",
        "[pipeline]",
    ],
)
def test_invalid_configuration(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_camera_settings_and_legacy_loading(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"\n[camera]\ndevice = 2', encoding="utf-8")
    config = load_app_config(path)
    assert config.camera == CameraConfig(2)
    assert config.pipeline == load_config(path) == PipelineConfig("test")


def test_legacy_configuration_defaults_camera(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    assert load_app_config(path).camera == CameraConfig(0)


@pytest.mark.parametrize(
    "camera",
    [
        "device = -1",
        "device = true",
        "device = 1.5",
        'device = "0"',
        'device = "https://example.com/camera"',
        'device = "video.mp4"',
        "unknown = 0",
    ],
)
def test_invalid_camera_settings(tmp_path: Path, camera: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[pipeline]\ncamera_id = "test"\n[camera]\n{camera}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_app_config(path)


@pytest.mark.parametrize(
    "settings",
    [
        'model = ""',
        'model = "https://example.com/model.pt"',
        'model = "model.onnx"',
        "model = 3",
        'device = "cuda:auto"',
        "device = 0",
        "device = true",
        "image_size = 0",
        "image_size = 33",
        "image_size = true",
        "image_size = 640.5",
        "unknown = 1",
    ],
)
def test_invalid_detector_configuration(tmp_path: Path, settings: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[pipeline]\ncamera_id = "test"\n[detector]\n{settings}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_app_config(path)


def test_detector_configuration_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    assert load_app_config(path).detector == DetectorConfig()
    path.write_text(
        '[pipeline]\ncamera_id = "test"\n[detector]\n'
        'model = "models/custom.pt"\ndevice = "0"\nimage_size = 320',
        encoding="utf-8",
    )
    assert load_app_config(path).detector == DetectorConfig("models/custom.pt", "0", 320)


@pytest.mark.parametrize(
    "settings",
    [
        "min_iou = -0.1",
        "min_iou = 0",
        "min_iou = 1.1",
        "min_iou = nan",
        "min_iou = inf",
        "min_iou = true",
        'min_iou = "0.3"',
        "max_missed_frames = -1",
        "max_missed_frames = 1.5",
        "max_missed_frames = true",
        'max_missed_frames = "10"',
        "unknown = 1",
    ],
)
def test_invalid_tracking_config(tmp_path: Path, settings: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[pipeline]\ncamera_id = "test"\n[tracking]\n{settings}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_app_config(path)


def test_tracking_defaults_and_valid_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    assert load_app_config(path).tracking == TrackingConfig(0.3, 10)
    path.write_text(
        '[pipeline]\ncamera_id = "test"\n[tracking]\nmin_iou = 1\nmax_missed_frames = 0',
        encoding="utf-8",
    )
    assert load_app_config(path).tracking == TrackingConfig(1, 0)
    assert load_config(path) == PipelineConfig("test")


@pytest.mark.parametrize("name", tuple(KalmanConfig.__dataclass_fields__))
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "true", '"0.1"'])
def test_invalid_kalman_noise(tmp_path: Path, name: str, value: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        f'[pipeline]\ncamera_id = "test"\n[tracking.kalman]\n{name} = {value}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="finite positive"):
        load_app_config(path)


def test_nested_kalman_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[pipeline]\ncamera_id = "test"\n[tracking.kalman]\nmeasurement_noise = 0.02',
        encoding="utf-8",
    )
    assert load_app_config(path).tracking.kalman == KalmanConfig(measurement_noise=0.02)
    path.write_text(
        '[pipeline]\ncamera_id = "test"\n[tracking.kalman]\nunknown = 1', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown setting"):
        load_app_config(path)


@pytest.mark.parametrize("value", ['"bad"', '"HUNGARIAN"', "true", "1", "[]"])
def test_invalid_assignment_setting(tmp_path: Path, value: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        f'[pipeline]\ncamera_id = "test"\n[tracking]\nassignment = {value}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="tracking.assignment"):
        load_app_config(path)


def test_assignment_configuration_and_backwards_default(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\ncamera_id = "test"', encoding="utf-8")
    assert load_app_config(path).tracking.assignment == "greedy"
    path.write_text(
        '[pipeline]\ncamera_id = "test"\n[tracking]\nassignment = "hungarian"', encoding="utf-8"
    )
    assert load_app_config(path).tracking.assignment == "hungarian"
