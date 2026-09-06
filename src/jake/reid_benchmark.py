"""Deterministic recently-lost comparison: python -m jake.reid_benchmark."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from jake.adapters.kalman_tracker import AppearanceKalmanPersonTracker, RecentlyLostPersonTracker
from jake.adapters.person_events import PersonEventGenerator
from jake.appearance import normalize
from jake.config import EventConfig, ReidConfig, TrackingConfig
from jake.domain import BoundingBox, EventKind, FrameContext, PersonDetection

SCENARIOS = ("walk-out-in", "full-occlusion", "doorway", "edge", "two-person", "similar-clothing")


@dataclass(frozen=True, slots=True)
class Comparison:
    ids_created: int
    switches: int
    fragmentation: int
    false_reactivations: int
    entered: int
    left: int
    reid_ms: float = field(compare=False)


def compare(scenario: str, reid: bool) -> Comparison:
    if scenario not in SCENARIOS:
        raise ValueError("unknown ReID benchmark scenario")
    config = TrackingConfig(assignment="hungarian", reid=ReidConfig(enabled=reid))
    tracker = RecentlyLostPersonTracker(config) if reid else AppearanceKalmanPersonTracker(config)
    events = PersonEventGenerator(EventConfig())
    seen: set[str] = set()
    owner: dict[str, int] = {}
    last: dict[int, str] = {}
    labels: dict[int, set[str]] = {}
    switches = false = entered = left = 0
    latency = 0.0
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for sequence, seconds in enumerate((0.0, 0.1, 0.2, 2.0, 2.5, 2.6, 2.7, 8.0, 9.0)):
        detections, truth = [], []
        if seconds <= 0.2 or 2.5 <= seconds <= 2.7:
            returning = seconds >= 2.5
            for person in range(2 if scenario == "two-person" else 1):
                center = 0.2 + 0.4 * person
                if scenario == "edge":
                    center = 0.94
                if returning:
                    center += {
                        "walk-out-in": 0.1,
                        "full-occlusion": 0.0,
                        "doorway": 0.2,
                        "edge": -0.15,
                    }.get(scenario, 0.08)
                identity = 1 if scenario == "similar-clothing" and returning else person
                vector = normalize(((1.0, 0.0) if person == 0 else (0.0, 1.0)) + (0.0,) * 254)
                detections.append(
                    PersonDetection(
                        BoundingBox(center - 0.04, 0.2, center + 0.04, 0.8), 0.9, vector
                    )
                )
                truth.append(identity)
        context = FrameContext("reid-benchmark", sequence, start + timedelta(seconds=seconds))
        tracks = tracker.update(context, tuple(detections))
        latency += tracker.last_reid_ms
        seen.update(t.track_id for t in tracks)
        for diagnostic in tracker.diagnostics():
            if diagnostic.measured_box is None:
                continue
            index = next(i for i, d in enumerate(detections) if d.box == diagnostic.measured_box)
            identity = truth[index]
            track_id = diagnostic.track_id
            switches += identity in last and last[identity] != track_id
            last[identity] = track_id
            labels.setdefault(identity, set()).add(track_id)
            owner.setdefault(track_id, identity)
            false += diagnostic.lifecycle == "REACTIVATED" and owner[track_id] != identity
        for event in events.generate(context, tracks):
            entered += event.kind == EventKind.PERSON_ENTERED
            left += event.kind == EventKind.PERSON_LEFT
    assert tracker.recently_lost_count == 0 and not tracker.diagnostics()
    return Comparison(
        len(seen),
        switches,
        sum(len(ids) - 1 for ids in labels.values()),
        false,
        entered,
        left,
        latency,
    )


def main() -> None:
    for scenario in SCENARIOS:
        for reid in (False, True):
            print(f"{scenario:18} {'REID' if reid else 'BASELINE':8} {compare(scenario, reid)}")


if __name__ == "__main__":
    main()
