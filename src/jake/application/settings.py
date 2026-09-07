"""Explicit application locations and validated atomic TOML settings."""

import os
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from jake.config import AppConfig, PipelineConfig, load_app_config


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @classmethod
    def default(cls) -> "AppPaths":
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
        else:
            base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
        return cls(base / "Jake")

    @property
    def config(self) -> Path:
        return self.root / "config" / "jake.toml"

    def initialize(self) -> None:
        for directory in ("config", "models", "data", "logs"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def defaults(self) -> AppConfig:
        config = AppConfig(PipelineConfig("home"))
        models = self.root / "models"
        return replace(
            config,
            detector=replace(config.detector, model=str(models / "yolo11n.pt")),
            tracking=replace(
                config.tracking,
                appearance=replace(
                    config.tracking.appearance,
                    model=str(models / "person-reidentification-retail-0287.xml"),
                ),
            ),
            identity=replace(
                config.identity,
                store_path=str(self.root / "data" / "identities"),
                detector_model=str(models / "face_detection_yunet_2023mar.onnx"),
                encoder_model=str(models / "face_recognition_sface_2021dec.onnx"),
            ),
        )


class MissingModelsError(FileNotFoundError):
    """Typed preflight failure containing only configured model names and paths."""

    def __init__(self, missing: tuple[tuple[str, str], ...]) -> None:
        self.missing = missing
        super().__init__("Required local model files are missing")


def validate_models(config: AppConfig, *, enrollment: bool = False) -> None:
    paths = [] if enrollment else [("YOLO", config.detector.model)]
    if not enrollment and config.tracking.appearance.enabled:
        paths.extend(
            [
                ("Appearance XML", config.tracking.appearance.model),
                ("Appearance BIN", str(Path(config.tracking.appearance.model).with_suffix(".bin"))),
            ]
        )
    if enrollment or config.identity.enabled or config.visitors.enabled:
        paths.extend(
            [("YuNet", config.identity.detector_model), ("SFace", config.identity.encoder_model)]
        )
    missing = tuple((name, value) for name, value in paths if not Path(value).is_file())
    if missing:
        raise MissingModelsError(missing)


def save_config(path: Path, config: AppConfig) -> None:
    """Validate before replace. Never move stores or write secrets into config."""
    import tomli_w

    path = path.absolute()
    if path.is_symlink():
        raise ValueError("configuration symlinks cannot be overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = asdict(config)
    values["camera"] = {k: v for k, v in values["camera"].items() if v is not None}
    values["audio"] = {k: v for k, v in values["audio"].items() if v is not None}
    values["conversation_ai"] = {
        k: v for k, v in values["conversation_ai"].items() if v is not None
    }
    raw = tomli_w.dumps(values).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".jake-config-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if load_app_config(temporary) != config:
            raise ValueError("configuration verification failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
