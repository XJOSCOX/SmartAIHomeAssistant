"""Release checks exercise adapter inference without real models or private data."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from jake.application.model_smoke import check_models
from jake.domain import Frame


@pytest.fixture
def adapters(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    replacements = {}
    for module, name in (
        ("yolo_detector", "YoloPersonDetector"),
        ("openvino_appearance", "OpenVINOAppearanceEncoder"),
        ("opencv_faces", "YuNetFaceDetector"),
        ("opencv_faces", "SFaceEncoder"),
    ):
        replacement = Mock()
        monkeypatch.setattr(f"jake.adapters.{module}.{name}", replacement)
        replacements[name] = replacement
    return replacements


def test_model_check_exercises_synthetic_inference(
    tmp_path: Path, adapters: dict[str, Mock]
) -> None:
    assert check_models(tmp_path) == dict.fromkeys(("detector", "appearance", "faces"), "OK")
    detector = adapters["YoloPersonDetector"]
    assert detector.call_args.args[0].model == str(tmp_path / "yolo11n.pt")
    frame = detector.return_value.detect.call_args.args[0]
    assert isinstance(frame, Frame)
    assert frame.pixels == bytes(frame.width * frame.height * 3)
    adapters["OpenVINOAppearanceEncoder"].return_value.encode.assert_called_once()
    adapters["YuNetFaceDetector"].return_value.detect.assert_called_once()
    adapters["SFaceEncoder"].assert_called_once()
    assert list(tmp_path.iterdir()) == []


def test_model_check_isolates_failure_and_redacts_payload(
    tmp_path: Path, adapters: dict[str, Mock]
) -> None:
    error = RuntimeError("private vector=[1,2]")
    error.__cause__ = ImportError("secret dependency detail")
    adapters["YoloPersonDetector"].side_effect = error
    result = check_models(tmp_path)
    assert result == {
        "detector": "FAILED: RuntimeError -> ImportError",
        "appearance": "OK",
        "faces": "OK",
    }
    assert "private" not in str(result)
    assert "secret" not in str(result)


def test_report_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    from jake.desktop.main import main

    report = tmp_path / "existing.json"
    report.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(["--model-smoke-test", str(tmp_path), "--model-smoke-report", str(report)])
    assert report.read_text(encoding="utf-8") == "unchanged"


def test_report_exit_code(tmp_path: Path, adapters: dict[str, Mock]) -> None:
    from jake.desktop.main import main

    report = tmp_path / "report.json"
    assert main(["--model-smoke-test", str(tmp_path), "--model-smoke-report", str(report)]) == 0
    assert '"appearance": "OK"' in report.read_text(encoding="utf-8")


def test_report_requires_both_options() -> None:
    from jake.desktop.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--model-smoke-test", "models"])
    assert exc.value.code == 2


def test_appearance_failure_identifies_component_without_payload() -> None:
    from jake.appearance import AppearanceError
    from jake.application.errors import ui_error

    result = ui_error(AppearanceError("private vector=[1,2]"))
    assert result.title == "Appearance model unavailable"
    assert "OpenVINO CPU runtime" in result.message
    assert "private" not in str(result)
