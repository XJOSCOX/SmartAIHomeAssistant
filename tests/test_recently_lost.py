from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from jake.adapters.iou_tracker import TrackerError
from jake.adapters.kalman_tracker import RecentlyLostPersonTracker
from jake.adapters.person_events import PersonEventGenerator
from jake.appearance import normalize, update_embedding
from jake.config import AppearanceConfig, EventConfig, ReidConfig, TrackingConfig, load_app_config
from jake.domain import BoundingBox, EventKind, FrameContext, PersonDetection, PersonEvent
from jake.reid_matching import assign_recent

START = datetime(2026, 1, 1, tzinfo=UTC)
A, B = normalize((1.0, 0.0)), normalize((0.0, 1.0))
BOX = BoundingBox(0.1, 0.2, 0.2, 0.7)
PERSON = PersonDetection(BOX, 0.9, A)
CONFIG = TrackingConfig(assignment="hungarian", reid=ReidConfig(enabled=True))


def ctx(seq: int, seconds: float) -> FrameContext:
    return FrameContext("test", seq, START + timedelta(seconds=seconds))


def test_confirmation_pool_reactivation_continuous_events_and_cleanup() -> None:
    tracker = RecentlyLostPersonTracker(CONFIG)
    events = PersonEventGenerator(EventConfig(1))
    output: list[PersonEvent] = []
    timeline = [
        (0, True),
        (0.1, True),
        (0.2, True),
        (1.7, False),
        (1.701, False),
        (2.5, True),
        (4.1, False),
        (7.5, False),
        (7.501, False),
        (8, False),
    ]
    for sequence, (seconds, visible) in enumerate(timeline):
        context = ctx(sequence, seconds)
        tracks = tracker.update(context, (PERSON,) if visible else ())
        output.extend(events.generate(context, tracks))
        if seconds == 1.7:
            assert not tracks[0].recently_lost
        if seconds in (1.701, 4.1, 7.5):
            assert tracks[0].recently_lost and tracks[0].confirmed and not tracks[0].visible
            assert tracker.recently_lost_count == 1
        if seconds == 2.5:
            assert tracks[0].track_id == "1" and tracks[0].visible
            assert tracker.diagnostics()[0].lifecycle == "REACTIVATED"
            assert tracker._tracks[0].confirmed_at == START + timedelta(seconds=0.2)
        if seconds >= 7.501:
            assert tracks == () and tracker._tracks == () and tracker.recently_lost_count == 0
    assert [e.kind for e in output] == [
        EventKind.PERSON_ENTERED,
        EventKind.PERSON_PRESENT,
        EventKind.PERSON_LEFT,
    ]
    assert output[-1].duration_seconds == pytest.approx(7.301)
    assert tracker.last_reid_ms >= 0


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (PersonDetection(BoundingBox(0.25, 0.2, 0.35, 0.7), 0.9, A), "1"),
        (PersonDetection(BoundingBox(0.8, 0.2, 0.9, 0.7), 0.9, A), "2"),
        (PersonDetection(BOX, 0.9, B), "2"),
    ],
)
def test_reactivation_gates(candidate: PersonDetection, expected: str) -> None:
    tracker = RecentlyLostPersonTracker(replace(CONFIG, confirmation_hits=1))
    tracker.update(ctx(0, 0), (PERSON,))
    tracker.update(ctx(1, 2), ())
    tracks = tracker.update(ctx(2, 3), (candidate,))
    assert [t.track_id for t in tracks if t.visible] == [expected]


def test_reactivation_updates_ema_and_kalman() -> None:
    tracker = RecentlyLostPersonTracker(replace(CONFIG, confirmation_hits=1))
    tracker.update(ctx(0, 0), (PERSON,))
    tracker.update(ctx(1, 2), ())
    before = tracker._tracks[0].motion.state.copy()
    new = normalize((1.0, 0.2))
    tracker.update(ctx(2, 3), (PersonDetection(BoundingBox(0.25, 0.2, 0.35, 0.7), 0.9, new),))
    track = tracker._tracks[0]
    assert track.appearance == update_embedding(A, new, 0.8)
    assert track.motion.state[0] > before[0]
    assert track.appearance_at == track.last_seen_at == START + timedelta(seconds=3)


@pytest.mark.parametrize("strategy", ["greedy", "hungarian"])
def test_two_recent_tracks_one_to_one_and_deterministic(strategy: str) -> None:
    def replay() -> tuple[str, ...]:
        tracker = RecentlyLostPersonTracker(
            replace(CONFIG, confirmation_hits=1, assignment=strategy)
        )
        tracker.update(ctx(0, 0), (PERSON, PERSON))
        tracker.update(ctx(1, 2), ())
        tracks = tracker.update(ctx(2, 3), (PERSON, PERSON))
        assert tracker.recently_lost_count == 0
        return tuple(t.track_id for t in tracks if t.visible)

    assert replay() == replay() == ("1", "2")


def test_active_matches_take_priority_over_pool() -> None:
    tracker = RecentlyLostPersonTracker(replace(CONFIG, confirmation_hits=1))
    tracker.update(ctx(0, 0), (PERSON, PersonDetection(BoundingBox(0.3, 0.2, 0.4, 0.7), 0.9, A)))
    tracker.update(ctx(1, 1), (PERSON,))
    tracker.update(ctx(2, 2), (PERSON,))
    tracks = tracker.update(ctx(3, 2.1), (PERSON,))
    assert [t.track_id for t in tracks if t.visible] == ["1"]
    assert tracker.recently_lost_count == 1


def test_tentative_not_pooled_and_reentry_after_timeout_new_id() -> None:
    tracker = RecentlyLostPersonTracker(CONFIG)
    tracker.update(ctx(0, 0), (PERSON,))
    assert tracker.update(ctx(1, 2), ()) == ()
    for seq in range(2, 5):
        tracker.update(ctx(seq, seq), (PERSON,))
    assert tracker.update(ctx(5, 9.001), (PERSON,))[0].track_id == "3"


def test_expired_embedding_cannot_reactivate_but_left_waits_for_window() -> None:
    tracker = RecentlyLostPersonTracker(
        replace(
            CONFIG, confirmation_hits=1, appearance=AppearanceConfig(max_embedding_age_seconds=2)
        )
    )
    tracker.update(ctx(0, 0), (PERSON,))
    tracker.update(ctx(1, 2.1), ())
    assert tracker._tracks[0].appearance is None
    tracks = tracker.update(ctx(2, 3), (PERSON,))
    assert [t.track_id for t in tracks if t.visible] == ["2"]
    assert tracker.recently_lost_count == 1


def test_reid_failure_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.hungarian import AssignmentError

    tracker = RecentlyLostPersonTracker(replace(CONFIG, confirmation_hits=1))
    tracker.update(ctx(0, 0), (PERSON,))
    tracker.update(ctx(1, 2), ())
    before = tracker._tracks
    monkeypatch.setattr(
        "jake.adapters.kalman_tracker.assign_recent", Mock(side_effect=AssignmentError("bad"))
    )
    with pytest.raises(TrackerError):
        tracker.update(ctx(2, 3), (PERSON,))
    assert tracker._tracks is before


def test_missing_descriptor_not_reidentified() -> None:
    assert (
        assign_recent((BOX,), (BOX,), (A,), (PersonDetection(BOX, 0.9),), ReidConfig(), "hungarian")
        == ()
    )


@pytest.mark.parametrize(
    "name",
    [
        "window_seconds",
        "min_similarity",
        "max_center_distance",
        "appearance_weight",
        "motion_weight",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "1"])
def test_invalid_config(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        ReidConfig(**{name: value})  # type: ignore[arg-type]


def test_config_loading(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[tracking.reid]\nenabled=true\nwindow_seconds=6',
        encoding="utf-8",
    )
    assert load_app_config(path).tracking.reid == ReidConfig(enabled=True, window_seconds=6)
    for table in (
        "[tracking.reid]\nenabled=1",
        "[tracking.reid]\nmin_similarity=1.1",
        "[tracking.reid]\ntypo=true",
    ):
        path.write_text('[pipeline]\ncamera_id="test"\n' + table, encoding="utf-8")
        with pytest.raises(ValueError):
            load_app_config(path)


def test_window_must_extend_retention() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        replace(CONFIG, reid=ReidConfig(enabled=True, window_seconds=1))
    with pytest.raises(ValueError, match="window"):
        RecentlyLostPersonTracker(TrackingConfig(reid=ReidConfig(window_seconds=1)))


@pytest.mark.parametrize(
    "scenario",
    ["walk-out-in", "full-occlusion", "doorway", "edge", "two-person", "similar-clothing"],
)
def test_benchmark_continuity_and_ambiguity(scenario: str) -> None:
    from jake.reid_benchmark import compare

    baseline, reid = compare(scenario, False), compare(scenario, True)
    assert reid == compare(scenario, True)
    assert reid.ids_created < baseline.ids_created
    assert reid.entered == reid.left < baseline.entered
    assert reid.switches == reid.fragmentation == 0
    assert reid.false_reactivations == (1 if scenario == "similar-clothing" else 0)
