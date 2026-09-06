from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from jake.adapters.iou_tracker import IoUPersonTracker
from jake.adapters.kalman_tracker import KalmanPersonTracker
from jake.adapters.person_events import EventError, PersonEventGenerator
from jake.config import EventConfig, TrackingConfig, load_app_config
from jake.domain import BoundingBox, EventKind, FrameContext, PersonEvent, PersonTrack
from jake.event_console import log_event
from jake.ports import EventGenerator

START = datetime(2026, 1, 1, 18, tzinfo=UTC)
TRACK = PersonTrack("1", BoundingBox(0, 0, 0.4, 1), 0.9)


def context(sequence: int, seconds: float | None = None) -> FrameContext:
    return FrameContext(
        "test", sequence, START + timedelta(seconds=sequence if seconds is None else seconds)
    )


def test_enter_present_throttle_and_leave_duration() -> None:
    generator: EventGenerator = PersonEventGenerator(EventConfig())
    (entered,) = generator.generate(context(0), (TRACK,))
    assert entered.kind == EventKind.PERSON_ENTERED
    assert entered.entered_at == START and entered.duration_seconds == 0
    assert entered.left_at is None
    for sequence in range(1, 500):
        assert generator.generate(context(sequence, sequence / 100), (TRACK,)) == ()
    (present,) = generator.generate(context(500, 5), (TRACK,))
    assert present.kind == EventKind.PERSON_PRESENT and present.duration_seconds == 5
    # A large gap produces one event, never a catch-up burst.
    (present,) = generator.generate(context(501, 50), (TRACK,))
    assert present.kind == EventKind.PERSON_PRESENT
    (left,) = generator.generate(context(502, 53.5), ())
    assert left.kind == EventKind.PERSON_LEFT
    assert left.left_at == START + timedelta(seconds=53.5)
    assert left.entered_at == START and left.duration_seconds == 53.5
    assert generator.generate(context(503, 54), ()) == ()
    assert len({entered.event_id, present.event_id, left.event_id}) == 3


def test_missed_and_tentative_tracks_do_not_announce_presence() -> None:
    generator = PersonEventGenerator(EventConfig(2))
    assert generator.generate(context(0), (replace(TRACK, confirmed=False),)) == ()
    assert generator.generate(context(1), (replace(TRACK, missed_frames=1),)) == ()
    assert generator.generate(context(2), (TRACK,))[0].kind == EventKind.PERSON_ENTERED
    assert generator.generate(context(5), (replace(TRACK, missed_frames=1),)) == ()
    assert generator.generate(context(6), (TRACK,))[0].kind == EventKind.PERSON_PRESENT
    assert generator.generate(context(7), (TRACK,)) == ()


def test_unannounced_tracks_expire_silently_and_empty_sets_work() -> None:
    generator = PersonEventGenerator(EventConfig())
    assert generator.generate(context(0), ()) == ()
    assert generator.generate(context(1), (replace(TRACK, confirmed=False),)) == ()
    assert generator.generate(context(2), ()) == ()


def test_independent_tracks_and_deterministic_timeline() -> None:
    def replay(reverse: bool) -> list[PersonEvent]:
        generator = PersonEventGenerator(EventConfig(), session_id=UUID(int=42))
        other = replace(TRACK, track_id="2")
        timeline = [(0, (TRACK,)), (3, (TRACK, other)), (5, (TRACK, other)), (8, (other,)), (9, ())]
        events: list[PersonEvent] = []
        for second, tracks in timeline:
            events.extend(
                generator.generate(context(second), tuple(reversed(tracks)) if reverse else tracks)
            )
        return events

    events = replay(False)
    assert events == replay(True)
    assert [(e.kind.name, e.track_id, e.duration_seconds) for e in events] == [
        ("PERSON_ENTERED", "1", 0),
        ("PERSON_ENTERED", "2", 0),
        ("PERSON_PRESENT", "1", 5),
        ("PERSON_LEFT", "1", 8),
        ("PERSON_PRESENT", "2", 5),
        ("PERSON_LEFT", "2", 6),
    ]


@pytest.mark.parametrize("strategy", ["greedy", "hungarian"])
@pytest.mark.parametrize("tracker_type", [IoUPersonTracker, KalmanPersonTracker])
def test_real_tracker_lifecycle(
    tracker_type: type[IoUPersonTracker] | type[KalmanPersonTracker], strategy: str
) -> None:
    from jake.domain import PersonDetection

    tracker = tracker_type(TrackingConfig(max_missed_frames=2, assignment=strategy))
    generator = PersonEventGenerator(EventConfig(2))
    detection = (PersonDetection(TRACK.box, TRACK.confidence),)
    events: list[PersonEvent] = []
    for sequence, detections in enumerate(
        (detection, (), (), detection, (), (), (), (), detection)
    ):
        ctx = context(sequence)
        tracks = tracker.update(ctx, detections)
        if sequence in (1, 2, 4, 5):
            assert not tracks[0].visible and tracks[0].missed_frames in (1, 2)
        events.extend(generator.generate(ctx, tracks))
    assert [(e.kind, e.track_id, e.duration_seconds) for e in events] == [
        (EventKind.PERSON_ENTERED, "1", 0),
        (EventKind.PERSON_PRESENT, "1", 3),
        (EventKind.PERSON_LEFT, "1", 6),
        (EventKind.PERSON_ENTERED, "2", 0),
    ]


@pytest.mark.parametrize(
    "invalid", [context(0), context(2, -1), replace(context(2), camera_id="other")]
)
def test_invalid_context_is_atomic(invalid: FrameContext) -> None:
    generator = PersonEventGenerator(EventConfig())
    generator.generate(context(0), (TRACK,))
    with pytest.raises(EventError):
        generator.generate(invalid, ())
    assert generator.generate(context(5), (TRACK,))[0].kind == EventKind.PERSON_PRESENT


def test_duplicate_and_reused_track_ids_rejected() -> None:
    generator = PersonEventGenerator(EventConfig())
    with pytest.raises(EventError, match="duplicate"):
        generator.generate(context(0), (TRACK, TRACK))
    assert generator.generate(context(0), (TRACK,))[0].kind == EventKind.PERSON_ENTERED
    generator.generate(context(1), ())
    with pytest.raises(EventError, match="reused"):
        generator.generate(context(2), (TRACK,))
    assert generator.generate(context(2), ()) == ()


def test_equal_times_and_timezone_offsets() -> None:
    generator = PersonEventGenerator(EventConfig())
    generator.generate(context(0), (TRACK,))
    assert generator.generate(context(1, 0), (TRACK,)) == ()
    ctx = replace(
        context(2, 5),
        captured_at=(START + timedelta(seconds=5)).astimezone(timezone(timedelta(hours=-6))),
    )
    (event,) = generator.generate(ctx, ())
    assert event.duration_seconds == 5 and event.context == ctx


@pytest.mark.parametrize("invalid", [0, -1, True, "5", float("nan"), float("inf")])
def test_invalid_event_config(invalid: object) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        EventConfig(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "table",
    [
        "[events]\npresent_interval_seconds = 2.5",
        "",
        "[events]\nunknown = 2",
        "events = 4",
        "[events]\npresent_interval_seconds = 0",
    ],
)
def test_event_config_loading(tmp_path: Path, table: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(table + '\n[pipeline]\ncamera_id = "test"', encoding="utf-8")
    if "unknown" in table or "events = 4" in table or "= 0" in table:
        with pytest.raises(ValueError):
            load_app_config(path)
    else:
        assert load_app_config(path).events.present_interval_seconds == (2.5 if table else 5)


def test_local_ids_and_metadata_only_payload(capsys: pytest.CaptureFixture[str]) -> None:
    first = PersonEventGenerator(EventConfig()).generate(context(0), (TRACK,))[0]
    second = PersonEventGenerator(EventConfig()).generate(context(0), (TRACK,))[0]
    assert first.event_id != second.event_id
    assert {field.name for field in fields(first)} == {
        "event_id",
        "kind",
        "context",
        "track_id",
        "entered_at",
    }
    log_event(first)
    log_event(replace(first, kind=EventKind.PERSON_LEFT, context=context(5)))
    assert (
        capsys.readouterr().out
        == "[18:00:00.000] ENTERED track=1\n[18:00:05.000] LEFT track=1 duration=5.0s\n"
    )


@pytest.mark.parametrize("entered_at", [START.replace(tzinfo=None), START + timedelta(seconds=1)])
def test_event_entry_timestamp_validation(entered_at: datetime) -> None:
    with pytest.raises(ValueError):
        PersonEvent("e", EventKind.PERSON_ENTERED, context(0), "1", entered_at)


def test_legacy_event_duration_optional() -> None:
    assert PersonEvent("e", EventKind.PERSON_UPDATED, context(0), "1").duration_seconds is None


@pytest.mark.parametrize("missed", [-1, True, 0.5])
def test_invalid_public_missed_count(missed: int) -> None:
    with pytest.raises(ValueError, match="missed_frames"):
        replace(TRACK, missed_frames=missed)


def test_invalid_confirmation() -> None:
    with pytest.raises(ValueError, match="confirmed"):
        replace(TRACK, confirmed=1)  # type: ignore[arg-type]
