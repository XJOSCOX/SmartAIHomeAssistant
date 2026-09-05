"""Deterministic trajectory comparison: python -m jake.benchmarks. No camera/model."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from jake.adapters.iou_tracker import IoUPersonTracker
from jake.adapters.kalman_tracker import KalmanPersonTracker
from jake.config import TrackingConfig
from jake.domain import BoundingBox, FrameContext, PersonDetection
from jake.matching import greedy_iou_assignment
from jake.ports import PersonTracker


@dataclass(frozen=True, slots=True)
class Comparison:
    births: int
    id_changes: int
    observed_detections: int
    unmatched_detections: int


def compare_trajectory(tracker: PersonTracker, *, occlusion: bool) -> Comparison:
    """Two people in separate image bands; one optionally disappears for 3 frames.

    Changes compare matched output IDs to the last visible ID of each synthetic
    ground-truth person, including across occlusion. Not a MOT accuracy benchmark.
    """
    seen_ids: set[str] = set()
    previous_ids: dict[int, str] = {}
    changes = observed = unmatched = 0
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for sequence in range(20):
        identities, detections = [], []
        for identity in range(2):
            if occlusion and identity == 0 and 8 <= sequence <= 10:
                continue
            cx = 0.15 + sequence * 0.03 if identity == 0 else 0.85 - sequence * 0.02
            top = 0.1 if identity == 0 else 0.6
            identities.append(identity)
            detections.append(
                PersonDetection(BoundingBox(cx - 0.08, top, cx + 0.08, top + 0.2), 0.9)
            )
        context = FrameContext("synthetic", sequence, start + timedelta(seconds=sequence * 0.1))
        tracks = tracker.update(context, tuple(detections))
        seen_ids.update(track.track_id for track in tracks)
        matches = greedy_iou_assignment(
            tuple(t.box for t in tracks), tuple(d.box for d in detections), 0.1
        )
        observed += len(detections)
        unmatched += len(detections) - len(matches)
        for track_index, detection_index in matches:
            identity = identities[detection_index]
            track_id = tracks[track_index].track_id
            if identity in previous_ids and previous_ids[identity] != track_id:
                changes += 1
            previous_ids[identity] = track_id
    return Comparison(len(seen_ids), changes, observed, unmatched)


def main() -> None:
    for occlusion in (False, True):
        for name, tracker in (
            ("iou", IoUPersonTracker(TrackingConfig())),
            ("kalman", KalmanPersonTracker(TrackingConfig())),
        ):
            result = compare_trajectory(tracker, occlusion=occlusion)
            print(f"{name:6} occlusion={occlusion}: {result}")


if __name__ == "__main__":
    main()
