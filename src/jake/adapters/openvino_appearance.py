"""Local person-reidentification-retail-0287 adapter; no model download or crop saving."""

import sys
from collections.abc import Callable
from importlib import import_module
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from jake.appearance import AppearanceError, normalize
from jake.config import AppearanceConfig
from jake.domain import AppearanceEmbedding, BoundingBox, Frame

ModelRunner = Callable[[NDArray[np.float32]], NDArray[np.float32]]


def load_model(path: Path) -> ModelRunner:
    """Compile only local IR weights on CPU; disable compiled-model disk caching."""
    if not path.is_file() or not path.with_suffix(".bin").is_file():
        raise AppearanceError(
            "Appearance model missing: explicitly install local .xml and .bin weights"
        )
    try:
        # OpenVINO imports its conversion package, which can send import telemetry.
        # Block that optional package BEFORE importing OpenVINO; it uses its own stub.
        # This is process-wide by design; run Jake in a dedicated process.
        if "openvino" in sys.modules and sys.modules.get("openvino_telemetry") is not None:
            raise AppearanceError(
                "Start a fresh Jake process to disable OpenVINO telemetry before import"
            )
        sys.modules["openvino_telemetry"] = None  # type: ignore[assignment]
        core = import_module("openvino").Core()
        model = core.read_model(str(path), str(path.with_suffix(".bin")))
        if (
            len(model.inputs) != 1
            or len(model.outputs) != 1
            or list(model.input(0).shape) != [1, 3, 256, 128]
            or list(model.output(0).shape) != [1, 256]
        ):
            raise AppearanceError("Expected retail-0287 input [1,3,256,128] and output [1,256]")
        compiled = core.compile_model(model, "CPU", {"CACHE_DIR": ""})
    except AppearanceError:
        raise
    except Exception as exc:
        raise AppearanceError(
            "Cannot load local appearance model; "
            "install the appearance extra and retail-0287 weights"
        ) from exc

    def run(tensor: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.asarray(compiled([tensor])[compiled.output(0)], dtype=np.float32)

    return run


class OpenVINOAppearanceEncoder:
    """Model-specific BGR 0..255 NCHW preprocessing and 256-dimensional output.

    The full frame is only viewed; the detected region becomes an owned resized
    tensor. The adapter retains the compiled model, never frames/crops/embeddings.
    """

    def __init__(self, config: AppearanceConfig) -> None:
        self._run = load_model(Path(config.model))

    def encode(self, frame: Frame, box: BoundingBox) -> AppearanceEmbedding:
        try:
            rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            left = max(0, min(frame.width - 1, floor(box.left * frame.width)))
            top = max(0, min(frame.height - 1, floor(box.top * frame.height)))
            right = max(left + 1, min(frame.width, ceil(box.right * frame.width)))
            bottom = max(top + 1, min(frame.height, ceil(box.bottom * frame.height)))
            crop = rgb[top:bottom, left:right, ::-1]
            resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
            tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
            output = np.asarray(self._run(tensor), dtype=np.float32)
            if output.shape != (1, 256):
                raise AppearanceError("Appearance model output must have shape [1,256]")
            return normalize(tuple(float(v) for v in output[0]))
        except AppearanceError:
            raise
        except Exception as exc:
            raise AppearanceError("Local appearance inference failed") from exc
