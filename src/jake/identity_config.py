"""Explicit face-identity settings; no model initialization on import."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    detector_model: str = "models/face_detection_yunet_2023mar.onnx"
    encoder_model: str = "models/face_recognition_sface_2021dec.onnx"
    store_path: str = ".jake-identities"
    candidate_similarity: float = 0.45
    resident_similarity: float = 0.60
    ambiguity_margin: float = 0.08
    required_confirmations: int = 3
    confirmation_window_seconds: float = 3.0
    observation_interval_seconds: float = 0.3
    carry_seconds: float = 2.0
    min_face_pixels: int = 64
    min_sharpness: float = 80.0
    min_detector_confidence: float = 0.9
    enrollment_samples: int = 10
    duplicate_similarity: float = 0.995

    def __post_init__(self) -> None:
        for name in ("detector_model", "encoder_model"):
            path = getattr(self, name)
            if not isinstance(path, str) or "://" in path or Path(path).suffix != ".onnx":
                raise ValueError(f"identity.{name} must be a local ONNX path")
        if (
            not isinstance(self.store_path, str)
            or not self.store_path.strip()
            or "://" in self.store_path
        ):
            raise ValueError("identity.store_path must be a dedicated local directory")
        for name in ("required_confirmations", "min_face_pixels", "enrollment_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"identity.{name} must be a positive integer")
        if not 2 <= self.required_confirmations <= 20 or not 3 <= self.enrollment_samples <= 100:
            raise ValueError("identity confirmation/enrollment sample counts out of range")
        for name in (
            "candidate_similarity",
            "resident_similarity",
            "ambiguity_margin",
            "confirmation_window_seconds",
            "observation_interval_seconds",
            "carry_seconds",
            "min_sharpness",
            "min_detector_confidence",
            "duplicate_similarity",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"identity.{name} must be finite and positive")
        if (
            not self.candidate_similarity < self.resident_similarity <= 1
            or not self.resident_similarity < self.duplicate_similarity <= 1
            or self.min_detector_confidence > 1
            or self.ambiguity_margin > 1
        ):
            raise ValueError("invalid identity similarity/confidence thresholds")
        if (
            self.observation_interval_seconds * (self.required_confirmations - 1)
            > self.confirmation_window_seconds
        ):
            raise ValueError("identity confirmation window cannot accommodate observations")
