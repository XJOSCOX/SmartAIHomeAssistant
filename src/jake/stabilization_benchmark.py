"""Deterministic event-quality fixture; python -m jake.stabilization_benchmark."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import sin

from jake.adapters.kalman_tracker import KalmanPersonTracker, StabilizedKalmanPersonTracker
from jake.adapters.person_events import PersonEventGenerator
from jake.config import EventConfig, TrackingConfig
from jake.domain import BoundingBox, EventKind, FrameContext, PersonDetection


@dataclass(frozen=True, slots=True)
class Quality:
    total_ids: int
    confirmed_ids: int
    id_switches: int
    entered_events: int
    left_events: int
    false_short_events: int


def compare(stabilized: bool) -> Quality:
    """Two crossing people with jitter, dropout, fast motion, width occlusion, and clutter.

    Exact measurement associations map to fixture truth; no truth enters tracking.
    A false short pair is a LEFT within two seconds of entry for a track that
    observed only synthetic clutter. Real fragmented tracks are counted in switches.
    """
    config = TrackingConfig(assignment="hungarian")
    tracker = StabilizedKalmanPersonTracker(config) if stabilized else KalmanPersonTracker(config)
    generator = PersonEventGenerator(EventConfig())
    seen: set[str] = set()
    confirmed: set[str] = set()
    truth_by_track: dict[str, set[int]] = {}
    last_id: dict[int, str] = {}
    switches = entered = left = short = 0
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for sequence in range(100):
        detections: list[PersonDetection] = []
        identities: list[int] = []
        if sequence < 70:
            for identity in range(2):
                if identity == 0 and 22 <= sequence <= 27:
                    continue
                # Accelerated lateral motion then reverse; both paths cross.
                cx = 0.5 + (0.30 if identity == 0 else -0.30) * sin(sequence * 0.075 - 1)
                cx += 0.008 * sin(sequence * 2 + identity)
                width = 0.025 if identity == 0 and 38 <= sequence <= 41 else 0.10
                top = 0.25 + identity * 0.025
                detections.append(
                    PersonDetection(
                        BoundingBox(cx - width / 2, top, cx + width / 2, top + 0.4), 0.9
                    )
                )
                identities.append(identity)
            if sequence % 10 == 5:
                # Isolated one-frame false detector candidate, far from real paths.
                cx = 0.1 + 0.1 * (sequence // 10)
                detections.append(PersonDetection(BoundingBox(cx, 0.85, cx + 0.025, 0.95), 0.6))
                identities.append(-1)
        ctx = FrameContext("quality", sequence, start + timedelta(seconds=sequence / 10))
        tracks = tracker.update(ctx, tuple(detections))
        seen.update(t.track_id for t in tracks)
        confirmed.update(t.track_id for t in tracks if t.confirmed)
        for diagnostic in tracker.diagnostics():
            if diagnostic.measured_box is None:
                continue
            index = next(i for i, d in enumerate(detections) if d.box == diagnostic.measured_box)
            identity = identities[index]
            truth_by_track.setdefault(diagnostic.track_id, set()).add(identity)
            if identity >= 0:
                if identity in last_id and last_id[identity] != diagnostic.track_id:
                    switches += 1
                last_id[identity] = diagnostic.track_id
        for event in generator.generate(ctx, tracks):
            entered += event.kind == EventKind.PERSON_ENTERED
            left += event.kind == EventKind.PERSON_LEFT
            if event.kind == EventKind.PERSON_LEFT and event.duration_seconds is not None:
                short += event.duration_seconds < 2 and truth_by_track[event.track_id] == {-1}
    return Quality(len(seen), len(confirmed), switches, entered, left, short)


def main() -> None:
    for stabilized in (False, True):
        print(f"{'STABILIZED' if stabilized else 'BASELINE'}: {compare(stabilized)}")


if __name__ == "__main__":
    main()
