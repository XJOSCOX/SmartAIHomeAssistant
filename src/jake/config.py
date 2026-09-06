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
class DetectorConfig:
    model: str = "models/yolo11n.pt"
    device: str = "cpu"
    image_size: int = 640

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("detector.model must be a local weight path or filename")
        if "://" in self.model or Path(self.model).suffix.lower() != ".pt":
            raise ValueError("detector.model must name local .pt weights, not a URL")
        if not isinstance(self.device, str) or not (
            self.device in {"cpu", "mps"} or self.device.isdecimal()
        ):
            raise ValueError('detector.device must be "cpu", "mps", or a GPU index string like "0"')
        if (
            isinstance(self.image_size, bool)
            or not isinstance(self.image_size, int)
            or self.image_size < 32
            or self.image_size % 32
        ):
            raise ValueError("detector.image_size must be a positive multiple of 32")


@dataclass(frozen=True, slots=True)
class KalmanConfig:
    process_noise_position: float = 0.0001
    process_noise_velocity: float = 0.001
    measurement_noise: float = 0.001
    initial_position_variance: float = 0.01
    initial_velocity_variance: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "process_noise_position",
            "process_noise_velocity",
            "measurement_noise",
            "initial_position_variance",
            "initial_velocity_variance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"tracking.kalman.{name} must be a finite positive number")


@dataclass(frozen=True, slots=True)
class AppearanceConfig:
    enabled: bool = False
    model: str = "models/person-reidentification-retail-0287.xml"
    weight: float = 0.35
    min_similarity: float = 0.45
    ema_alpha: float = 0.8
    max_embedding_age_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("appearance.enabled must be a boolean")
        if (
            not isinstance(self.model, str)
            or "://" in self.model
            or Path(self.model).suffix.lower() != ".xml"
        ):
            raise ValueError("appearance.model must be a local OpenVINO .xml weight path")
        for name in ("weight", "min_similarity", "ema_alpha", "max_embedding_age_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"appearance.{name} must be a finite number")
        if (
            not 0 < self.weight <= 1
            or not 0 <= self.min_similarity <= 1
            or not 0 <= self.ema_alpha < 1
            or self.max_embedding_age_seconds <= 0
        ):
            raise ValueError(
                "appearance weight in (0,1], similarity in [0,1], alpha in [0,1), age > 0 required"
            )


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    min_iou: float = 0.3
    max_missed_frames: int = 10
    kalman: KalmanConfig = KalmanConfig()
    assignment: str = "greedy"
    confirmation_hits: int = 3
    max_missed_seconds: float = 1.5
    max_center_distance: float = 0.15
    iou_weight: float = 0.6
    distance_weight: float = 0.4
    appearance: AppearanceConfig = AppearanceConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, str) or self.assignment not in {"greedy", "hungarian"}:
            raise ValueError('tracking.assignment must be "greedy" or "hungarian"')
        if (
            isinstance(self.confirmation_hits, bool)
            or not isinstance(self.confirmation_hits, int)
            or self.confirmation_hits < 1
        ):
            raise ValueError("tracking.confirmation_hits must be a positive integer")
        for name in ("max_missed_seconds", "max_center_distance", "iou_weight", "distance_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"tracking.{name} must be finite and positive")
        if self.max_center_distance > 1 or self.iou_weight > 1 or self.distance_weight > 1:
            raise ValueError("tracking distance gate and weights must be at most 1")
        threshold = self.min_iou
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not isfinite(threshold)
            or not 0 < threshold <= 1
        ):
            raise ValueError("tracking.min_iou must be finite and in (0, 1]")
        missed = self.max_missed_frames
        if isinstance(missed, bool) or not isinstance(missed, int) or missed < 0:
            raise ValueError("tracking.max_missed_frames must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class EventConfig:
    present_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        value = self.present_interval_seconds
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError("events.present_interval_seconds must be a finite positive number")


@dataclass(frozen=True, slots=True)
class AppConfig:
    pipeline: PipelineConfig
    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    tracking: TrackingConfig = TrackingConfig()
    events: EventConfig = EventConfig()


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
    if set(data) - {"pipeline", "camera", "detector", "tracking", "events"} or not isinstance(
        data.get("pipeline"), dict
    ):
        raise ValueError(
            "configuration requires [pipeline] and permits "
            "[camera], [detector], [tracking], [events]"
        )
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
    detector = data.get("detector", {})
    if not isinstance(detector, dict) or set(detector) - {"model", "device", "image_size"}:
        raise ValueError("[detector] permits only model, device, and image_size")
    defaults = DetectorConfig()
    tracking = data.get("tracking", {})
    if not isinstance(tracking, dict) or set(tracking) - {
        "min_iou",
        "max_missed_frames",
        "kalman",
        "appearance",
        "assignment",
        "confirmation_hits",
        "max_missed_seconds",
        "max_center_distance",
        "iou_weight",
        "distance_weight",
    }:
        raise ValueError(
            "unknown tracking setting; see config/jake.example.toml for the tracking schema"
        )
    kalman = tracking.get("kalman", {})
    noise_defaults = KalmanConfig()
    noise_fields = tuple(KalmanConfig.__dataclass_fields__)
    if not isinstance(kalman, dict) or set(kalman) - set(noise_fields):
        raise ValueError("unknown setting or invalid table in [tracking.kalman]")
    noise_config = KalmanConfig(
        **{name: kalman.get(name, getattr(noise_defaults, name)) for name in noise_fields}
    )
    appearance = tracking.get("appearance", {})
    appearance_fields = tuple(AppearanceConfig.__dataclass_fields__)
    if not isinstance(appearance, dict) or set(appearance) - set(appearance_fields):
        raise ValueError("unknown setting or invalid table in [tracking.appearance]")
    appearance_config = AppearanceConfig(**appearance)
    tracking_defaults = TrackingConfig()
    events = data.get("events", {})
    if not isinstance(events, dict) or set(events) - {"present_interval_seconds"}:
        raise ValueError("[events] permits only present_interval_seconds")
    return AppConfig(
        pipeline_config,
        CameraConfig(device=camera.get("device", 0)),
        DetectorConfig(
            model=detector.get("model", defaults.model),
            device=detector.get("device", defaults.device),
            image_size=detector.get("image_size", defaults.image_size),
        ),
        TrackingConfig(
            min_iou=tracking.get("min_iou", tracking_defaults.min_iou),
            max_missed_frames=tracking.get(
                "max_missed_frames", tracking_defaults.max_missed_frames
            ),
            kalman=noise_config,
            appearance=appearance_config,
            assignment=tracking.get("assignment", tracking_defaults.assignment),
            **{
                name: tracking.get(name, getattr(tracking_defaults, name))
                for name in (
                    "confirmation_hits",
                    "max_missed_seconds",
                    "max_center_distance",
                    "iou_weight",
                    "distance_weight",
                )
            },
        ),
        EventConfig(present_interval_seconds=events.get("present_interval_seconds", 5.0)),
    )
