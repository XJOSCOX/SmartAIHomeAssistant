from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from jake.cli import main
from jake.config import AppConfig, CameraConfig, PipelineConfig
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
