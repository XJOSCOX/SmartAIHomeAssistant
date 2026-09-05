from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from jake.adapters.iou_tracker import IoUPersonTracker, TrackerError
from jake.adapters.kalman_tracker import KalmanPersonTracker
from jake.benchmarks import compare_trajectory
from jake.config import TrackingConfig
from jake.domain import BoundingBox, FrameContext, PersonDetection
from jake.kalman import KalmanError
from jake.ports import PersonTracker


def context(sequence: int) -> FrameContext:
    return FrameContext(
        "test", sequence, datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=sequence * 0.1)
    )


def person(sequence: int) -> PersonDetection:
    center = 0.15 + sequence * 0.03
    return PersonDetection(BoundingBox(center - 0.08, 0.1, center + 0.08, 0.3), 0.9)


def warmed_tracker() -> KalmanPersonTracker:
    tracker = KalmanPersonTracker(TrackingConfig(max_missed_frames=3))
    for sequence in range(8):
        tracks = tracker.update(context(sequence), (person(sequence),))
        assert len(tracks) == 1 and tracks[0].track_id == "1"
    return tracker


def test_moving_person_short_occlusion_reconnects_to_prediction() -> None:
    tracker = warmed_tracker()
    before = tracker._tracks[0]
    for sequence in (8, 9, 10):
        tracks = tracker.update(context(sequence), ())
        assert tracks[0].track_id == "1"
        assert tracks[0].box.left > before.box.left
        assert tracker.diagnostics()[0].missed_frames == sequence - 7
        assert tracker.diagnostics()[0].measured_box is None
    assert tracker.update(context(11), (person(11),))[0].track_id == "1"
    state = tracker._tracks[0]
    assert state.age_frames == 12 and state.visible_frames == 9 and state.missed_frames == 0
    assert state.created_at == context(0).captured_at
    assert state.last_seen_at == context(11).captured_at


def test_missed_prediction_moves_each_frame_and_expiration_is_unchanged() -> None:
    tracker = warmed_tracker()
    last_x = tracker._tracks[0].box.left
    for sequence in (8, 9, 10):
        track = tracker.update(context(sequence), ())[0]
        assert track.box.left > last_x
        last_x = track.box.left
    assert tracker.update(context(11), ()) == ()
    assert tracker.update(context(12), (person(12),))[0].track_id == "2"


def test_empty_initial_and_zero_miss_allowance() -> None:
    tracker: PersonTracker = KalmanPersonTracker(TrackingConfig(max_missed_frames=0))
    assert tracker.update(context(0), ()) == ()
    assert tracker.update(context(1), (person(1),))[0].track_id == "1"
    assert tracker.update(context(2), ()) == ()


@pytest.mark.parametrize("occlusion", [False, True])
def test_deterministic_comparison_and_two_people(occlusion: bool) -> None:
    baseline = compare_trajectory(IoUPersonTracker(TrackingConfig()), occlusion=occlusion)
    motion = compare_trajectory(KalmanPersonTracker(TrackingConfig()), occlusion=occlusion)
    assert motion.births == 2 and motion.id_changes == 0
    assert motion.unmatched_detections == baseline.unmatched_detections == 0
    assert baseline.births == (3 if occlusion else 2)
    assert baseline.id_changes == (1 if occlusion else 0)
    assert motion == compare_trajectory(KalmanPersonTracker(TrackingConfig()), occlusion=occlusion)


def test_invalid_context_and_math_failure_preserve_session(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = warmed_tracker()
    before = tracker._tracks
    with pytest.raises(TrackerError):
        tracker.update(context(7), ())
    with pytest.raises(TrackerError):
        tracker.update(FrameContext("other", 8, context(8).captured_at), ())
    monkeypatch.setattr(
        "jake.kalman.KalmanFilter.correct", Mock(side_effect=KalmanError("bad solve"))
    )
    with pytest.raises(TrackerError, match="state was preserved"):
        tracker.update(context(8), (person(8),))
    assert tracker._tracks is before
    assert tracker._last_context == context(7)
    assert tracker._next_id == 2


def test_clock_reversal_uses_fallback_and_predictions_remain_bounded() -> None:
    tracker = warmed_tracker()
    reversed_clock = FrameContext("test", 8, context(6).captured_at)
    tracks = tracker.update(reversed_clock, ())
    assert tracks[0].box.left > person(7).box.left
    assert np.all(np.isfinite(tracker._tracks[0].motion.state))
