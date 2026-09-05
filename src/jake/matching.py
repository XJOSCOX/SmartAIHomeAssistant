"""Pure box association, isolated from tracker lifecycle and model frameworks."""

from collections.abc import Sequence

from jake.domain import BoundingBox
from jake.hungarian import AssignmentError, linear_assignment


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Intersection area / union area for normalized positive-area boxes."""
    if first == second:
        return 1.0
    width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    first_area = (first.right - first.left) * (first.bottom - first.top)
    second_area = (second.right - second.left) * (second.bottom - second.top)
    union = first_area + second_area - intersection
    # Protect against floating-point underflow for extremely tiny valid boxes.
    return min(1.0, intersection / union) if union > 0 else 0.0


def greedy_iou_assignment(
    tracks: Sequence[BoundingBox], detections: Sequence[BoundingBox], min_iou: float
) -> tuple[tuple[int, int], ...]:
    """Return one-to-one (track index, detection index) assignments.

    Caller supplies a validated threshold in (0, 1]. Highest IoU wins; equal
    scores prefer lower track index, then lower detection index. The tracker
    provides tracks in creation order, so this is deterministic for ordered input.
    O(T*D) candidates, O((T*D) log(T*D)) worst-case sorting, O(T*D) space.
    This baseline remains independent of the Hungarian strategy below.
    """
    candidates = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            score = intersection_over_union(track, detection)
            if score >= min_iou:
                candidates.append((-score, track_index, detection_index))
    candidates.sort()
    assigned_tracks: set[int] = set()
    assigned_detections: set[int] = set()
    matches = []
    for _, track_index, detection_index in candidates:
        if track_index not in assigned_tracks and detection_index not in assigned_detections:
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)
            matches.append((track_index, detection_index))
    return tuple(matches)


def hungarian_iou_assignment(
    tracks: Sequence[BoundingBox], detections: Sequence[BoundingBox], min_iou: float
) -> tuple[tuple[int, int], ...]:
    """Maximize valid match count, then minimize total 1-IoU cost.

    T+D square nodes include explicit unmatched choices for both sides. Valid
    costs are <=1; unmatched penalty min(T,D)+1 prioritizes cardinality above
    any possible change in total valid cost. Forbidden pairs cost more than the
    entire all-unmatched solution. O((T+D)^3) time, O((T+D)^2) space.
    """
    if not tracks or not detections:
        return ()
    track_count, detection_count = len(tracks), len(detections)
    size = track_count + detection_count
    penalty = float(min(track_count, detection_count) + 1)
    forbidden = (size + 1) * 2 * penalty
    scores = [[intersection_over_union(t, d) for d in detections] for t in tracks]
    costs = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i < track_count and j < detection_count:
                costs[i][j] = 1 - scores[i][j] if scores[i][j] >= min_iou else forbidden
            elif i < track_count or j < detection_count:
                costs[i][j] = penalty
    return tuple(
        (i, j)
        for i, j in linear_assignment(costs)
        if i < track_count and j < detection_count and scores[i][j] >= min_iou
    )


def assign_boxes(
    tracks: Sequence[BoundingBox],
    detections: Sequence[BoundingBox],
    min_iou: float,
    strategy: str,
) -> tuple[tuple[int, int], ...]:
    if strategy == "greedy":
        return greedy_iou_assignment(tracks, detections, min_iou)
    if strategy == "hungarian":
        return hungarian_iou_assignment(tracks, detections, min_iou)
    raise AssignmentError(f"unknown assignment strategy: {strategy}")
