"""Explicit TOML loading; importing Jake never reads configuration or secrets."""

import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    camera_id: str
    min_person_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        value = self.min_person_confidence
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("min_person_confidence must be a number")
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("min_person_confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class CameraConfig:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("camera.device must be a non-negative local camera index")


@dataclass(frozen=True, slots=True)
class AppConfig:
    pipeline: PipelineConfig
    camera: CameraConfig = CameraConfig()


def load_config(path: Path) -> PipelineConfig:
    """Backward-compatible access to pipeline settings; validate the whole file."""
    return load_app_config(path).pipeline


def load_app_config(path: Path) -> AppConfig:
    """Load a strict schema so misspelled settings fail at startup.

    File access errors and TOMLDecodeError propagate to the composition layer;
    semantic errors raise ValueError. No silent fallback is performed.
    """
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) - {"pipeline", "camera"} or not isinstance(data.get("pipeline"), dict):
        raise ValueError("configuration requires [pipeline] and permits only [camera] alongside it")
    pipeline = data["pipeline"]
    if set(pipeline) - {"camera_id", "min_person_confidence"}:
        raise ValueError("unknown pipeline configuration key")
    if "camera_id" not in pipeline:
        raise ValueError("pipeline.camera_id is required")
    camera = data.get("camera", {})
    if not isinstance(camera, dict) or set(camera) - {"device"}:
        raise ValueError("[camera] permits only the device setting")
    pipeline_config = PipelineConfig(
        camera_id=pipeline["camera_id"],
        min_person_confidence=pipeline.get("min_person_confidence", 0.5),
    )
    return AppConfig(pipeline_config, CameraConfig(device=camera.get("device", 0)))
