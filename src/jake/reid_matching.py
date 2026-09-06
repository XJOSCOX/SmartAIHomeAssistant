"""Second-stage assignment of unmatched detections to recently-lost candidates."""

from math import hypot

from jake.appearance import cosine_similarity
from jake.config import ReidConfig
from jake.domain import AppearanceEmbedding, BoundingBox, PersonDetection
from jake.motion_matching import assign_candidates


def center_distance(a: BoundingBox, b: BoundingBox) -> float:
    return hypot(
        (a.left + a.right - b.left - b.right) / 2, (a.top + a.bottom - b.top - b.bottom) / 2
    )


def assign_recent(
    predicted: tuple[BoundingBox, ...],
    observed: tuple[BoundingBox, ...],
    embeddings: tuple[AppearanceEmbedding | None, ...],
    detections: tuple[PersonDetection, ...],
    config: ReidConfig,
    strategy: str,
) -> tuple[tuple[int, int], ...]:
    costs = {}
    for i, embedding in enumerate(embeddings):
        if embedding is None:
            continue
        for j, detection in enumerate(detections):
            if detection.appearance is None:
                continue
            similarity = cosine_similarity(embedding, detection.appearance)
            distance = min(
                center_distance(predicted[i], detection.box),
                center_distance(observed[i], detection.box),
            )
            if similarity >= config.min_similarity and distance <= config.max_center_distance:
                costs[i, j] = (
                    config.appearance_weight * (1 - similarity)
                    + config.motion_weight * distance / config.max_center_distance
                ) / (config.appearance_weight + config.motion_weight)
    return assign_candidates(len(predicted), len(detections), costs, strategy)
