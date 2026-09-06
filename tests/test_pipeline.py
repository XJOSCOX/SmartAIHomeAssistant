from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from jake.config import PipelineConfig
from jake.domain import (
    BoundingBox,
    EventKind,
    Frame,
    FrameContext,
    PersonDetection,
    PersonEvent,
    PersonTrack,
)
from jake.pipeline import PerceptionPipeline

BOX = BoundingBox(0.1, 0.1, 0.9, 0.9)
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def frame(sequence: int = 0, camera_id: str = "test") -> Frame:
    return Frame(camera_id, sequence, NOW, 1, 1, b"abc")


class Detector:
    def detect(self, frame: Frame) -> tuple[PersonDetection, ...]:
        return (PersonDetection(BOX, 0.49), PersonDetection(BOX, 0.5))


class Tracker:
    def __init__(self) -> None:
        self.received: list[tuple[PersonDetection, ...]] = []

    def update(
        self, context: FrameContext, detections: tuple[PersonDetection, ...]
    ) -> tuple[PersonTrack, ...]:
        self.received.append(detections)
        return tuple(PersonTrack(str(i), d.box, d.confidence) for i, d in enumerate(detections))


class Events:
    def __init__(self) -> None:
        self.received: list[tuple[PersonTrack, ...]] = []

    def generate(
        self, context: FrameContext, tracks: tuple[PersonTrack, ...]
    ) -> tuple[PersonEvent, ...]:
        self.received.append(tracks)
        return tuple(
            PersonEvent(
                f"{context.sequence}-{t.track_id}", EventKind.PERSON_UPDATED, context, t.track_id
            )
            for t in tracks
        )


def test_injected_adapters_filter_and_propagate_context() -> None:
    tracker, events = Tracker(), Events()
    pipeline = PerceptionPipeline(PipelineConfig("test"), Detector(), tracker, events)
    result = pipeline.process(frame())
    assert tracker.received == [(PersonDetection(BOX, 0.5),)]
    assert events.received == [(PersonTrack("0", BOX, 0.5),)]
    assert result == (
        PersonEvent("0-0", EventKind.PERSON_UPDATED, FrameContext("test", 0, NOW), "0"),
    )


def test_empty_updates_reach_tracker_and_event_generator() -> None:
    tracker, events = Tracker(), Events()
    pipeline = PerceptionPipeline(PipelineConfig("test", 0.9), Detector(), tracker, events)
    assert pipeline.process(frame()) == ()
    assert tracker.received == [()]
    assert events.received == [()]


def test_camera_isolation_happens_before_adapter_calls() -> None:
    tracker = Tracker()
    pipeline = PerceptionPipeline(PipelineConfig("test"), Detector(), tracker, Events())
    with pytest.raises(ValueError, match="camera_id"):
        pipeline.process(frame(camera_id="other"))
    assert tracker.received == []
    assert len(pipeline.process(frame())) == 1


@pytest.mark.parametrize("sequence", [0, 1])
def test_duplicate_and_out_of_order_frames_are_rejected(sequence: int) -> None:
    pipeline = PerceptionPipeline(PipelineConfig("test"), Detector(), Tracker(), Events())
    pipeline.process(frame(1))
    with pytest.raises(ValueError, match="strictly increasing"):
        pipeline.process(frame(sequence))


def test_run_pulls_lazily_and_preserves_order() -> None:
    seen: list[int] = []

    def source() -> Iterator[Frame]:
        for sequence in range(3):
            seen.append(sequence)
            yield frame(sequence)

    pipeline = PerceptionPipeline(PipelineConfig("test"), Detector(), Tracker(), Events())
    result = pipeline.run(source())
    assert seen == []
    assert next(result).context.sequence == 0
    assert seen == [0]
    assert [event.context.sequence for event in result] == [1, 2]


def test_adapter_failure_propagates_without_downstream_calls() -> None:
    class FailingDetector:
        def detect(self, frame: Frame) -> tuple[PersonDetection, ...]:
            raise RuntimeError("detector unavailable")

    tracker, events = Tracker(), Events()
    pipeline = PerceptionPipeline(PipelineConfig("test"), FailingDetector(), tracker, events)
    with pytest.raises(RuntimeError, match="detector unavailable"):
        pipeline.process(frame())
    assert tracker.received == []
    assert events.received == []


def test_semantic_events_through_existing_pipeline() -> None:
    from jake.adapters.iou_tracker import IoUPersonTracker
    from jake.adapters.person_events import PersonEventGenerator
    from jake.config import EventConfig, TrackingConfig

    class TimelineDetector:
        def detect(self, frame: Frame) -> tuple[PersonDetection, ...]:
            confidence = 0.9 if frame.sequence < 2 else 0.1
            return (PersonDetection(BOX, confidence),)

    pipeline = PerceptionPipeline(
        PipelineConfig("test"),
        TimelineDetector(),
        IoUPersonTracker(TrackingConfig(max_missed_frames=1)),
        PersonEventGenerator(EventConfig(5)),
    )
    source = (
        Frame("test", sequence, NOW + timedelta(seconds=sequence * 5), 1, 1, b"abc")
        for sequence in range(5)
    )
    events = tuple(pipeline.run(source))
    assert [(event.kind, event.duration_seconds) for event in events] == [
        (EventKind.PERSON_ENTERED, 0),
        (EventKind.PERSON_PRESENT, 5),
        (EventKind.PERSON_LEFT, 15),
    ]
