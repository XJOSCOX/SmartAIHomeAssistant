from pathlib import Path

import pytest

from jake.config import CameraConfig, PipelineConfig, load_app_config, load_config


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
