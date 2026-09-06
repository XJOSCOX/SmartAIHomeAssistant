"""Model-independent appearance mathematics and frame-stage composition."""

from dataclasses import replace
from math import fsum, hypot

from jake.domain import AppearanceEmbedding, Frame, PersonDetection
from jake.ports import AppearanceEncoder


class AppearanceError(ValueError):
    """Invalid appearance data or local encoder setup/inference failure."""


def normalize(values: tuple[float, ...]) -> AppearanceEmbedding:
    norm = hypot(*values)
    if not norm > 0:
        raise AppearanceError("appearance vector must have nonzero finite norm")
    try:
        return AppearanceEmbedding(tuple(v / norm for v in values))
    except ValueError as exc:
        raise AppearanceError("appearance vector must have nonzero finite norm") from exc


def cosine_similarity(first: AppearanceEmbedding, second: AppearanceEmbedding) -> float:
    if len(first.values) != len(second.values):
        raise AppearanceError("appearance dimensions differ; do not mix encoder models")
    return max(
        -1.0, min(1.0, fsum(a * b for a, b in zip(first.values, second.values, strict=True)))
    )


def update_embedding(
    old: AppearanceEmbedding | None, new: AppearanceEmbedding, alpha: float
) -> AppearanceEmbedding:
    if old is None:
        return new
    cosine_similarity(old, new)  # Validate dimensions before zip.
    return normalize(
        tuple(alpha * a + (1 - alpha) * b for a, b in zip(old.values, new.values, strict=True))
    )


def encode_detections(
    frame: Frame, detections: tuple[PersonDetection, ...], encoder: AppearanceEncoder
) -> tuple[PersonDetection, ...]:
    """Only this composition stage sees pixels; trackers continue receiving metadata."""
    return tuple(replace(d, appearance=encoder.encode(frame, d.box)) for d in detections)
