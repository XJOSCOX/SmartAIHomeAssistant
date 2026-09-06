"""Explicit release check using local weights and synthetic pixels only."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from jake.application.settings import AppPaths
from jake.domain import BoundingBox, Frame


def check_models(directory: Path) -> dict[str, str]:
    """Never open cameras, profiles or credential vaults; discard model outputs.

    Results contain stage names and exception class names only, not third-party
    payloads or vectors. Each stage runs even when an earlier stage fails.
    """
    config = AppPaths(directory.parent).defaults()
    frame = Frame(
        "model-smoke", 0, datetime(2026, 1, 1, tzinfo=UTC), 320, 320, bytes(320 * 320 * 3)
    )
    box = BoundingBox(0, 0, 1, 1)

    def detector() -> None:
        from jake.adapters.yolo_detector import YoloPersonDetector

        YoloPersonDetector(
            replace(config.detector, model=str(directory / "yolo11n.pt")),
            config.pipeline.min_person_confidence,
        ).detect(frame)

    def appearance() -> None:
        from jake.adapters.openvino_appearance import OpenVINOAppearanceEncoder

        OpenVINOAppearanceEncoder(
            replace(
                config.tracking.appearance,
                model=str(directory / "person-reidentification-retail-0287.xml"),
            )
        ).encode(frame, box)

    def faces() -> None:
        from jake.adapters.opencv_faces import SFaceEncoder, YuNetFaceDetector

        identity = replace(
            config.identity,
            detector_model=str(directory / "face_detection_yunet_2023mar.onnx"),
            encoder_model=str(directory / "face_recognition_sface_2021dec.onnx"),
        )
        YuNetFaceDetector(identity).detect(frame, box)
        SFaceEncoder(identity)

    results = {}
    for name, check in (("detector", detector), ("appearance", appearance), ("faces", faces)):
        try:
            check()
        except Exception as exc:
            chain: list[str] = []
            cause: BaseException | None = exc
            while cause is not None and len(chain) < 8:
                chain.append(type(cause).__name__)
                cause = cause.__cause__
            results[name] = "FAILED: " + " -> ".join(chain)
        else:
            results[name] = "OK"
    return results
