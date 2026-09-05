"""Ultralytics integration confined to a replaceable PersonDetector adapter."""

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from jake.config import DetectorConfig, PipelineConfig
from jake.domain import BoundingBox, Frame, PersonDetection


class DetectorError(RuntimeError):
    """Local detector setup, inference, or output decoding failed."""


class TensorRows(Protocol):
    def cpu(self) -> "TensorRows": ...
    def tolist(self) -> list[list[float]]: ...


class ModelBoxes(Protocol):
    @property
    def data(self) -> TensorRows: ...


class ModelResult(Protocol):
    @property
    def boxes(self) -> ModelBoxes | None: ...


class LocalModel(Protocol):
    """Minimal model boundary; tests supply it without importing Ultralytics."""

    @property
    def names(self) -> Mapping[int, str]: ...

    def predict(self, source: NDArray[np.uint8], **kwargs: object) -> Sequence[ModelResult]: ...


def _load_model(path: str) -> LocalModel:
    # Set these before importing the framework: its import otherwise probes DNS.
    # Deliberately override inherited settings for this local-only application.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"
    try:
        package = importlib.import_module("ultralytics")
        utils = importlib.import_module("ultralytics.utils")
    except ImportError as exc:
        raise DetectorError(
            "Detection dependencies unavailable; run uv sync --extra detection"
        ) from exc
    utils.__dict__["ONLINE"] = False
    utils.__dict__["AUTOINSTALL"] = False
    utils.SETTINGS.update({"sync": False})
    return cast(LocalModel, package.YOLO(path, task="detect"))


class YoloPersonDetector:
    """Serial, frame-independent local inference with no identity or tracking.

    Only existing .pt files reach Ultralytics, preventing implicit weight downloads
    or remote inference. Pixel arrays are copies; the Frame remains immutable.
    """

    def __init__(
        self,
        config: DetectorConfig,
        confidence: float,
        *,
        model_factory: Callable[[str], LocalModel] = _load_model,
    ) -> None:
        # Reuse the existing confidence validation without changing the domain.
        PipelineConfig("detector", confidence)
        self._config = config
        self._confidence = confidence
        path = Path(config.model).expanduser()
        if not path.is_file():
            raise DetectorError(
                f"Local model weights not found: {path}. Download weights explicitly "
                "as described in README, or configure detector.model with an existing .pt file."
            )
        try:
            self._model = model_factory(str(path.resolve()))
            self._person_ids = [
                index for index, name in self._model.names.items() if name.casefold() == "person"
            ]
            if not self._person_ids:
                raise DetectorError("Model has no class named 'person'")
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"Could not initialize local person detector from {path}") from exc

    def detect(self, frame: Frame) -> tuple[PersonDetection, ...]:
        try:
            rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            # Ultralytics accepts NumPy images in BGR, not RGB. Give it an owned,
            # contiguous copy so model-side mutation cannot affect frame bytes.
            bgr = rgb[:, :, ::-1].copy()
            results = self._model.predict(
                source=bgr,
                conf=self._confidence,
                classes=self._person_ids,
                device=self._config.device,
                imgsz=self._config.image_size,
                stream=False,
                save=False,
                save_txt=False,
                save_crop=False,
                show=False,
                visualize=False,
                verbose=False,
            )
            if len(results) != 1 or results[0].boxes is None:
                raise DetectorError("Expected one detection result with boxes from the local model")
            # CPU transfer synchronizes GPU results before detect() returns, so
            # caller timing includes completed inference rather than queued work.
            rows = results[0].boxes.data.cpu().tolist()
            detections = []
            for row in rows:
                detection = self._convert(row, frame.width, frame.height)
                if detection is not None:
                    detections.append(detection)
            return tuple(detections)
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(
                "Local person detector inference or result decoding failed"
            ) from exc

    def _convert(self, row: list[float], width: int, height: int) -> PersonDetection | None:
        if len(row) != 6:
            raise DetectorError("Expected untracked xyxy/confidence/class model output")
        if not all(isfinite(value) for value in row):
            return None
        left, top, right, bottom, confidence, class_id = row
        if class_id not in self._person_ids or not self._confidence <= confidence <= 1:
            return None
        # Clip partial off-screen boxes, then reject empty, reversed, or wholly
        # off-screen geometry rather than constructing an invalid domain object.
        left, right = (min(1.0, max(0.0, x / width)) for x in (left, right))
        top, bottom = (min(1.0, max(0.0, y / height)) for y in (top, bottom))
        if left >= right or top >= bottom:
            return None
        return PersonDetection(BoundingBox(left, top, right, bottom), float(confidence))
