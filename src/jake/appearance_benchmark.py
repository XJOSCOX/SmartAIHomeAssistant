"""Synthetic short-term continuity comparison; no real images/model/identity database."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import sin

from jake.adapters.kalman_tracker import (
    AppearanceKalmanPersonTracker,
    StabilizedKalmanPersonTracker,
)
from jake.appearance import normalize
from jake.config import TrackingConfig
from jake.domain import BoundingBox, FrameContext, PersonDetection

SCENARIOS = ("occlusion", "edge", "low-iou", "crossing", "appearance-noise", "similar-outfits")


@dataclass(frozen=True, slots=True)
class AppearanceComparison:
    ids_created: int
    id_switches: int
    fragmentation: int
    false_reassociations: int
    assignment_ms: float = field(compare=False)


def compare(scenario: str, appearance: bool) -> AppearanceComparison:
    """Ground truth is used only to score exact measured-box associations.

    Fragmentation counts distinct IDs per ground-truth person beyond their first.
    False re-associations count observations assigned to an ID first owned by a
    different fixture person. No second geometric matcher approximates associations.
    """
    if scenario not in SCENARIOS:
        raise ValueError("unknown appearance benchmark scenario")
    config = TrackingConfig(assignment="hungarian")
    tracker = (
        AppearanceKalmanPersonTracker(config)
        if appearance
        else StabilizedKalmanPersonTracker(config)
    )
    all_ids: set[str] = set()
    identities: dict[int, set[str]] = {}
    last: dict[int, str] = {}
    owner: dict[str, int] = {}
    switches = false = 0
    elapsed = 0.0
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for sequence in range(12):
        detections = []
        truth = []
        crossing = scenario in {"crossing", "appearance-noise", "similar-outfits"}
        for identity in range(2 if crossing else 1):
            if scenario == "occlusion" and 3 <= sequence <= 6:
                continue
            if scenario == "low-iou" and sequence in (3, 4):
                continue
            width = 0.04
            if crossing:
                # Two paths cross between samples; nearest geometry favors the wrong person.
                center = (
                    (0.4 if sequence < 3 else 0.54)
                    if identity == 0
                    else (0.6 if sequence < 3 else 0.46)
                )
                width = 0.12
            elif scenario == "edge":
                center = 0.97 if sequence < 3 else 0.88
                width = 0.08 if sequence < 3 else 0.025
            else:
                center = 0.3 if sequence < 3 else 0.4
            box = BoundingBox(max(0, center - width / 2), 0.2, min(1, center + width / 2), 0.8)
            vector = (1.0, 0.0, 0.0) if identity == 0 else (0.0, 1.0, 0.0)
            if scenario == "similar-outfits" and identity == 1:
                vector = (1.0, 0.05, 0.0)
            if scenario == "appearance-noise":
                vector = (vector[0], vector[1], 0.1 * sin(sequence + identity))
            detections.append(PersonDetection(box, 0.9, normalize(vector + (0.0,) * 253)))
            truth.append(identity)
        ctx = FrameContext(
            "appearance-benchmark", sequence, start + timedelta(seconds=sequence * 0.1)
        )
        tracks = tracker.update(ctx, tuple(detections))
        all_ids.update(t.track_id for t in tracks)
        elapsed += tracker.last_assignment_ms
        for diagnostic in tracker.diagnostics():
            if diagnostic.measured_box is None:
                continue
            index = next(
                i
                for i, detection in enumerate(detections)
                if detection.box == diagnostic.measured_box
            )
            identity = truth[index]
            track_id = diagnostic.track_id
            if identity in last and last[identity] != track_id:
                switches += 1
            last[identity] = track_id
            identities.setdefault(identity, set()).add(track_id)
            owner.setdefault(track_id, identity)
            false += owner[track_id] != identity
    return AppearanceComparison(
        len(all_ids), switches, sum(len(ids) - 1 for ids in identities.values()), false, elapsed
    )


def main() -> None:
    for scenario in SCENARIOS:
        for appearance in (False, True):
            print(
                f"{scenario:18} {'APPEARANCE' if appearance else 'GEOMETRY':10} "
                f"{compare(scenario, appearance)}"
            )


if __name__ == "__main__":
    main()
