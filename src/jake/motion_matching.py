"""Normalized spatial cost policy, independent of the generic Hungarian solver."""

from collections.abc import Sequence
from math import hypot

from jake.config import TrackingConfig
from jake.domain import BoundingBox
from jake.hungarian import linear_assignment
from jake.matching import intersection_over_union


def motion_cost(first: BoundingBox, second: BoundingBox, config: TrackingConfig) -> float | None:
    distance = hypot(
        (first.left + first.right - second.left - second.right) / 2,
        (first.top + first.bottom - second.top - second.bottom) / 2,
    )
    # Hard center gate applies even to overlapping, very large boxes.
    if distance > config.max_center_distance:
        return None
    iou = intersection_over_union(first, second)
    # Low-overlap recovery requires the tighter inner half of the spatial gate.
    if iou < config.min_iou and distance > config.max_center_distance / 2:
        return None
    return (
        config.iou_weight * (1 - iou)
        + config.distance_weight * distance / config.max_center_distance
    ) / (config.iou_weight + config.distance_weight)


def assign_motion_boxes(
    tracks: Sequence[BoundingBox], detections: Sequence[BoundingBox], config: TrackingConfig
) -> tuple[tuple[int, int], ...]:
    if not tracks or not detections:
        return ()
    candidates = {
        (i, j): cost
        for i, track in enumerate(tracks)
        for j, detection in enumerate(detections)
        if (cost := motion_cost(track, detection, config)) is not None
    }
    return assign_candidates(len(tracks), len(detections), candidates, config.assignment)


def assign_candidates(
    track_count: int, detection_count: int, candidates: dict[tuple[int, int], float], strategy: str
) -> tuple[tuple[int, int], ...]:
    """Assign pre-gated costs in [0,1]; maximize cardinality, then minimize cost."""
    if not track_count or not detection_count:
        return ()
    if strategy == "greedy":
        matches = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for (i, j), _ in sorted(candidates.items(), key=lambda item: (item[1], item[0])):
            if i not in used_tracks and j not in used_detections:
                matches.append((i, j))
                used_tracks.add(i)
                used_detections.add(j)
        return tuple(matches)
    # Same cardinality-first objective as the baseline, with explicit unmatched nodes.
    size = track_count + detection_count
    penalty = float(min(track_count, detection_count) + 1)
    forbidden = (size + 1) * 2 * penalty
    costs = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i < track_count and j < detection_count:
                costs[i][j] = candidates.get((i, j), forbidden)
            elif i < track_count or j < detection_count:
                costs[i][j] = penalty
    return tuple(pair for pair in linear_assignment(costs) if pair in candidates)
