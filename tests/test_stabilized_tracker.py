from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jake.adapters.iou_tracker import TrackerError
from jake.adapters.kalman_tracker import StabilizedKalmanPersonTracker
from jake.adapters.person_events import PersonEventGenerator
from jake.config import EventConfig, TrackingConfig, load_app_config
from jake.domain import BoundingBox, EventKind, FrameContext, PersonDetection, PersonEvent
from jake.motion_matching import assign_motion_boxes, motion_cost

START = datetime(2026, 1, 1, tzinfo=UTC)
PERSON = PersonDetection(BoundingBox(0.1, 0.2, 0.2, 0.7), 0.9)


def context(sequence: int, seconds: float) -> FrameContext:
    return FrameContext("test", sequence, START + timedelta(seconds=seconds))


def test_tentative_noise_expires_without_events() -> None:
    tracker = StabilizedKalmanPersonTracker(TrackingConfig())
    events = PersonEventGenerator(EventConfig())
    for sequence, (seconds, detections) in enumerate(
        [(0, (PERSON,)), (0.5, ()), (1.5, ()), (1.501, ())]
    ):
        ctx = context(sequence, seconds)
        tracks = tracker.update(ctx, detections)
        assert events.generate(ctx, tracks) == ()
        assert len(tracks) == (0 if seconds > 1.5 else 1)


def test_intermittent_confirmation_loss_reacquisition_and_duration() -> None:
    tracker = StabilizedKalmanPersonTracker(TrackingConfig())
    events = PersonEventGenerator(EventConfig(0.5))
    output: list[PersonEvent] = []
    timeline = [
        (0, True),
        (0.1, False),
        (0.2, True),
        (0.3, True),
        (0.8, False),
        (0.9, True),
        (2.4, False),
        (2.401, False),
        (2.5, False),
    ]
    for sequence, (seconds, visible) in enumerate(timeline):
        ctx = context(sequence, seconds)
        tracks = tracker.update(ctx, (PERSON,) if visible else ())
        output.extend(events.generate(ctx, tracks))
        if seconds == 0.2:
            assert not tracks[0].confirmed
            assert tracker.diagnostics()[0].visible_hits == 2
        if seconds in (0.3, 0.8, 0.9, 2.4):
            assert tracks[0].track_id == "1" and tracks[0].confirmed
        if seconds == 0.8:
            assert tracker.diagnostics()[0].lifecycle == "LOST"
            assert tracker.diagnostics()[0].missed_seconds == pytest.approx(0.5)
    assert [e.kind for e in output] == [
        EventKind.PERSON_ENTERED,
        EventKind.PERSON_PRESENT,
        EventKind.PERSON_LEFT,
    ]
    assert output[0].context.captured_at == START + timedelta(seconds=0.3)
    assert output[-1].duration_seconds == pytest.approx(2.101)


@pytest.mark.parametrize("fps", [10, 30])
def test_time_retention_independent_of_fps(fps: int) -> None:
    tracker = StabilizedKalmanPersonTracker(TrackingConfig())
    events = PersonEventGenerator(EventConfig())
    output: list[PersonEvent] = []
    for sequence in range(4 * fps + 1):
        seconds = sequence / fps
        # Same continuous trajectory with a half-second dropout, then departure.
        visible = seconds <= 0.5 or 1 <= seconds <= 1.5
        ctx = context(sequence, seconds)
        tracks = tracker.update(ctx, (PERSON,) if visible else ())
        output.extend(events.generate(ctx, tracks))
        if seconds == 1:
            assert tracks[0].track_id == "1"
        if seconds == 3:
            assert tracks[0].confirmed
    assert [e.kind for e in output] == [EventKind.PERSON_ENTERED, EventKind.PERSON_LEFT]
    left = (output[-1].context.captured_at - START).total_seconds()
    assert 3 < left <= 3 + 1 / fps + 1e-6


def test_late_reacquisition_is_new_tentative_and_backward_clock_atomic() -> None:
    tracker = StabilizedKalmanPersonTracker(TrackingConfig(confirmation_hits=1))
    tracker.update(context(0, 1), (PERSON,))
    with pytest.raises(TrackerError, match="backwards"):
        tracker.update(context(1, 0), ())
    assert tracker.update(context(1, 1.5), (PERSON,))[0].track_id == "1"
    assert tracker.update(context(2, 3.001), (PERSON,))[0].track_id == "2"


@pytest.mark.parametrize("strategy", ["greedy", "hungarian"])
def test_motion_matching_gates_and_one_to_one(strategy: str) -> None:
    config = TrackingConfig(assignment=strategy)
    box = BoundingBox(0.1, 0.2, 0.13, 0.7)
    close = BoundingBox(0.14, 0.2, 0.17, 0.7)  # Zero IoU, close center.
    far = BoundingBox(0.7, 0.2, 0.73, 0.7)
    marginal = BoundingBox(0.2, 0.2, 0.23, 0.7)
    assert motion_cost(box, box, config) == 0
    assert motion_cost(box, close, config) is not None
    assert motion_cost(box, far, config) is None
    assert motion_cost(box, marginal, config) is None
    assert assign_motion_boxes((box,), (close,), config) == ((0, 0),)
    assert assign_motion_boxes((box,), (far,), config) == ()
    assert assign_motion_boxes((), (box,), config) == ()
    matches = assign_motion_boxes((box, box), (close, close), config)
    assert len(matches) == 2 and len({i for i, _ in matches}) == len({j for _, j in matches}) == 2
    assert matches == assign_motion_boxes((box, box), (close, close), config)


def test_low_iou_recovery_through_tracker() -> None:
    tracker = StabilizedKalmanPersonTracker(TrackingConfig())
    first = replace(PERSON, box=BoundingBox(0.1, 0.2, 0.13, 0.7))
    second = replace(PERSON, box=BoundingBox(0.14, 0.2, 0.17, 0.7))
    tracker.update(context(0, 0), (first,))
    assert tracker.update(context(1, 0.1), (second,))[0].track_id == "1"
    assert len(tracker.diagnostics()) == 1


@pytest.mark.parametrize(
    "name", ["max_missed_seconds", "max_center_distance", "iou_weight", "distance_weight"]
)
@pytest.mark.parametrize("value", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_noise_policy_values(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(TrackingConfig(), **{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_confirmation(value: int) -> None:
    with pytest.raises(ValueError):
        TrackingConfig(confirmation_hits=value)


@pytest.mark.parametrize("name", ["max_center_distance", "iou_weight", "distance_weight"])
def test_excessive_gate_weight(name: str) -> None:
    with pytest.raises(ValueError):
        replace(TrackingConfig(), **{name: 1.1})  # type: ignore[arg-type]


def test_load_stabilization_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[tracking]\nconfirmation_hits=4\nmax_missed_seconds=2\nmax_center_distance=0.1\niou_weight=0.7\ndistance_weight=0.3',
        encoding="utf-8",
    )
    config = load_app_config(path).tracking
    assert (config.confirmation_hits, config.max_missed_seconds, config.max_center_distance) == (
        4,
        2,
        0.1,
    )
    assert (config.iou_weight, config.distance_weight) == (0.7, 0.3)


def test_quality_benchmark_improves_events_without_hiding_switches() -> None:
    from jake.stabilization_benchmark import compare

    baseline, stabilized = compare(False), compare(True)
    assert baseline == compare(False)
    assert stabilized == compare(True)
    assert baseline.false_short_events == 7
    assert stabilized.false_short_events == 0
    assert stabilized.entered_events == stabilized.left_events == stabilized.confirmed_ids == 2
    assert stabilized.total_ids < baseline.total_ids
    assert stabilized.id_switches <= baseline.id_switches
