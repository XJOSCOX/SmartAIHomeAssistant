"""Pure box association, isolated from tracker lifecycle and model frameworks."""

from collections.abc import Sequence

from jake.domain import BoundingBox


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
    This function can be replaced independently with a future Hungarian solver.
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
