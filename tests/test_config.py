from pathlib import Path

import pytest

from jake.config import PipelineConfig, load_config


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
