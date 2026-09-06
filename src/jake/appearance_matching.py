"""Appearance augments plausible geometry; it never bypasses the outer spatial gate."""

from collections.abc import Sequence
from math import hypot

from jake.appearance import cosine_similarity
from jake.config import TrackingConfig
from jake.domain import AppearanceEmbedding, BoundingBox
from jake.matching import intersection_over_union
from jake.motion_matching import assign_candidates, motion_cost


def appearance_cost(
    track: BoundingBox,
    detection: BoundingBox,
    old: AppearanceEmbedding | None,
    new: AppearanceEmbedding | None,
    config: TrackingConfig,
) -> float | None:
    if old is None or new is None:
        return motion_cost(track, detection, config)
    distance = hypot(
        (track.left + track.right - detection.left - detection.right) / 2,
        (track.top + track.bottom - detection.top - detection.bottom) / 2,
    )
    if distance > config.max_center_distance:
        return None
    similarity = cosine_similarity(old, new)
    if similarity < config.appearance.min_similarity:
        return None
    weight = config.appearance.weight
    return (
        config.iou_weight * (1 - intersection_over_union(track, detection))
        + config.distance_weight * distance / config.max_center_distance
        + weight * (1 - similarity)
    ) / (config.iou_weight + config.distance_weight + weight)


def assign_appearance_boxes(
    tracks: Sequence[BoundingBox],
    detections: Sequence[BoundingBox],
    old: Sequence[AppearanceEmbedding | None],
    new: Sequence[AppearanceEmbedding | None],
    config: TrackingConfig,
) -> tuple[tuple[int, int], ...]:
    candidates = {
        (i, j): cost
        for i, track in enumerate(tracks)
        for j, detection in enumerate(detections)
        if (cost := appearance_cost(track, detection, old[i], new[j], config)) is not None
    }
    return assign_candidates(len(tracks), len(detections), candidates, config.assignment)
