"""Synchronous, bounded orchestration with explicitly injected components."""

from collections.abc import Iterator

from jake.appearance import encode_detections
from jake.config import PipelineConfig
from jake.domain import Frame, FrameContext, PersonEvent
from jake.face_identity import FaceIdentityService
from jake.identity_domain import IdentityEvent, IdentityMatch
from jake.ports import AppearanceEncoder, EventGenerator, FrameSource, PersonDetector, PersonTracker


class PerceptionPipeline:
    """One pipeline and fresh stateful adapters per camera session.

    Calls are serial, with no background queue or frame persistence. Exceptions
    propagate to the caller; retries and supervision belong outside this layer.
    Do not share an instance between threads or reuse it after an adapter failure.
    """

    def __init__(
        self,
        config: PipelineConfig,
        detector: PersonDetector,
        tracker: PersonTracker,
        events: EventGenerator,
        *,
        encoder: AppearanceEncoder | None = None,
        identity: FaceIdentityService | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._tracker = tracker
        self._events = events
        self._encoder = encoder
        self._identity = identity
        self.identity_matches: dict[str, IdentityMatch] = {}
        self.identity_events: tuple[IdentityEvent, ...] = ()
        self._last_sequence: int | None = None

    def process(self, frame: Frame) -> tuple[PersonEvent, ...]:
        """Process one ordered frame; return events without retaining its pixels."""
        if frame.camera_id != self._config.camera_id:
            raise ValueError("frame camera_id does not match pipeline configuration")
        if self._last_sequence is not None and frame.sequence <= self._last_sequence:
            raise ValueError("frame sequences must be strictly increasing within a session")
        self._last_sequence = frame.sequence
        context = FrameContext(frame.camera_id, frame.sequence, frame.captured_at)
        detections = tuple(
            detection
            for detection in self._detector.detect(frame)
            if detection.confidence >= self._config.min_person_confidence
        )
        if self._encoder is not None:
            detections = encode_detections(frame, detections, self._encoder)
        tracks = self._tracker.update(context, detections)
        if self._identity is not None:
            self.identity_matches, self.identity_events = self._identity.process(frame, tracks)
        return self._events.generate(context, tracks)

    def run(self, source: FrameSource) -> Iterator[PersonEvent]:
        """Pull frames on demand. The caller manages source and adapter resources."""
        for frame in source:
            yield from self.process(frame)
