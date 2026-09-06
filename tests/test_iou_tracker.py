from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from jake.adapters.iou_tracker import IoUPersonTracker, TrackerError
from jake.config import TrackingConfig
from jake.domain import BoundingBox, FrameContext, PersonDetection
from jake.ports import PersonTracker

START = datetime(2026, 1, 1, tzinfo=UTC)
LEFT = PersonDetection(BoundingBox(0, 0, 0.4, 1), 0.9)
RIGHT = PersonDetection(BoundingBox(0.6, 0, 1, 1), 0.8)


def context(sequence: int, camera_id: str = "test") -> FrameContext:
    return FrameContext(camera_id, sequence, START + timedelta(seconds=sequence))


def test_one_person_retains_id_and_updates_box_confidence() -> None:
    tracker: PersonTracker = IoUPersonTracker(TrackingConfig())
    first = tracker.update(context(0), (LEFT,))
    moved = PersonDetection(BoundingBox(0.05, 0, 0.45, 1), 0.95)
    second = tracker.update(context(1), (moved,))
    assert first[0].track_id == second[0].track_id == "1"
    assert second[0].box == moved.box
    assert second[0].confidence == moved.confidence
    assert first[0].box == LEFT.box  # Prior public result remains an immutable snapshot.


def test_two_people_keep_separate_ids_when_detection_order_changes() -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    first = tracker.update(context(0), (LEFT, RIGHT))
    assert [track.track_id for track in first] == ["1", "2"]
    assert tracker.update(context(1), (RIGHT, LEFT)) == first


def test_new_person_receives_incremental_id() -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    tracker.update(context(0), (LEFT,))
    tracks = tracker.update(context(1), (LEFT, RIGHT))
    assert [track.track_id for track in tracks] == ["1", "2"]


def test_track_survives_limit_then_expires() -> None:
    tracker = IoUPersonTracker(TrackingConfig(max_missed_frames=2))
    first = tracker.update(context(0), (LEFT,))
    assert tracker.update(context(1), ()) == (replace(first[0], missed_frames=1),)
    assert tracker.update(context(2), ()) == (replace(first[0], missed_frames=2),)
    assert tracker.update(context(3), ()) == ()
    assert tracker.update(context(4), ()) == ()


def test_returning_before_expiration_keeps_id_and_resets_misses() -> None:
    tracker = IoUPersonTracker(TrackingConfig(max_missed_frames=2))
    first = tracker.update(context(0), (LEFT,))
    tracker.update(context(1), ())
    tracker.update(context(2), ())
    assert tracker.update(context(3), (LEFT,)) == first
    assert tracker.update(context(4), ()) == (replace(first[0], missed_frames=1),)
    assert tracker.update(context(5), ()) == (replace(first[0], missed_frames=2),)
    assert tracker.update(context(6), ()) == ()


def test_returning_after_expiration_gets_new_id() -> None:
    tracker = IoUPersonTracker(TrackingConfig(max_missed_frames=1))
    tracker.update(context(0), (LEFT,))
    tracker.update(context(1), ())
    assert tracker.update(context(2), ()) == ()
    assert tracker.update(context(3), (LEFT,))[0].track_id == "2"


def test_zero_miss_allowance_expires_on_first_unmatched_update() -> None:
    tracker = IoUPersonTracker(TrackingConfig(max_missed_frames=0))
    tracker.update(context(0), (LEFT,))
    assert tracker.update(context(1), ()) == ()


def test_one_detection_cannot_update_two_tracks() -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    tracker.update(context(0), (LEFT, LEFT))
    changed = PersonDetection(LEFT.box, 0.99)
    tracks = tracker.update(context(1), (changed,))
    assert [(track.track_id, track.confidence) for track in tracks] == [("1", 0.99), ("2", 0.9)]
    assert tracker._tracks[0].missed_frames == 0
    assert tracker._tracks[1].missed_frames == 1


def test_one_track_cannot_consume_two_detections() -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    tracker.update(context(0), (LEFT,))
    changed = PersonDetection(LEFT.box, 0.99)
    tracks = tracker.update(context(1), (LEFT, changed))
    assert [(track.track_id, track.confidence) for track in tracks] == [("1", 0.9), ("2", 0.99)]


def test_empty_initial_frame_and_no_overlap_create_only_observed_tracks() -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    assert tracker.update(context(0), ()) == ()
    assert tracker.update(context(1), (LEFT,))[0].track_id == "1"
    tracks = tracker.update(context(2), (RIGHT,))
    assert [track.track_id for track in tracks] == ["1", "2"]


def test_internal_lifecycle_metadata() -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    tracker.update(context(0), (LEFT,))
    state = tracker._tracks[0]
    assert (state.created_at, state.last_seen_at) == (START, START)
    assert (state.age_frames, state.visible_frames, state.missed_frames) == (1, 1, 0)
    tracker.update(context(1), ())
    state = tracker._tracks[0]
    assert (state.age_frames, state.visible_frames, state.missed_frames) == (2, 1, 1)
    assert state.last_seen_at == START
    tracker.update(context(2), (LEFT,))
    state = tracker._tracks[0]
    assert (state.age_frames, state.visible_frames, state.missed_frames) == (3, 2, 0)
    assert state.created_at == START
    assert state.last_seen_at == context(2).captured_at


@pytest.mark.parametrize("invalid", [context(0), context(1), context(2, "other")])
def test_rejected_context_does_not_mutate_state(invalid: FrameContext) -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    tracker.update(context(1), (LEFT,))
    before = tracker._tracks
    with pytest.raises(TrackerError):
        tracker.update(invalid, (RIGHT,))
    assert tracker._tracks == before
    assert tracker.update(context(2), (LEFT,))[0].track_id == "1"


def test_sequence_gaps_count_only_processed_updates() -> None:
    tracker = IoUPersonTracker(TrackingConfig(max_missed_frames=1))
    first = tracker.update(context(0), (LEFT,))
    assert tracker.update(context(100), ()) == (replace(first[0], missed_frames=1),)
    assert tracker._tracks[0].age_frames == 2
    assert tracker.update(context(200), ()) == ()


def test_ids_are_session_local_and_deterministic() -> None:
    first, second = IoUPersonTracker(TrackingConfig()), IoUPersonTracker(TrackingConfig())
    for sequence, detections in enumerate(((LEFT, LEFT), (LEFT,), (), (RIGHT, LEFT))):
        assert first.update(context(sequence), detections) == second.update(
            context(sequence), detections
        )


def test_public_tracks_remain_in_numeric_creation_order() -> None:
    tracker = IoUPersonTracker(TrackingConfig())
    tracks = tracker.update(context(0), (LEFT,) * 12)
    assert [track.track_id for track in tracks] == [str(i) for i in range(1, 13)]
    assert tracker.update(context(1), (LEFT,) * 12) == tracks
