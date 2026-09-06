"""Synchronous, bounded orchestration with explicitly injected components."""

from collections.abc import Iterator
from time import perf_counter

from jake.appearance import encode_detections
from jake.config import PipelineConfig
from jake.diagnostics import measure_detection
from jake.domain import Frame, FrameContext, PersonEvent, PersonTrack
from jake.face_identity import FaceIdentityService
from jake.identity_domain import IdentityEvent, IdentityMatch
from jake.ports import AppearanceEncoder, EventGenerator, FrameSource, PersonDetector, PersonTracker
from jake.visitor_domain import VisitorEvent, VisitorMatch
from jake.visitors import VisitorMemory


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
        visitors: VisitorMemory | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._tracker = tracker
        self._events = events
        self._encoder = encoder
        self._identity = identity
        if visitors is not None and identity is None:
            raise ValueError("visitor pipeline requires face identity")
        self._visitors = visitors
        self.visitor_matches: dict[str, VisitorMatch] = {}
        self.visitor_events: tuple[VisitorEvent, ...] = ()
        self.identity_matches: dict[str, IdentityMatch] = {}
        self.identity_events: tuple[IdentityEvent, ...] = ()
        self.tracks: tuple[PersonTrack, ...] = ()
        self.inference_ms = 0.0
        self.appearance_ms = 0.0
        self.face_ms = 0.0
        self.people_detected = 0
        self._last_sequence: int | None = None

    def process(self, frame: Frame) -> tuple[PersonEvent, ...]:
        """Process one ordered frame; return events without retaining its pixels."""
        if frame.camera_id != self._config.camera_id:
            raise ValueError("frame camera_id does not match pipeline configuration")
        if self._last_sequence is not None and frame.sequence <= self._last_sequence:
            raise ValueError("frame sequences must be strictly increasing within a session")
        self._last_sequence = frame.sequence
        context = FrameContext(frame.camera_id, frame.sequence, frame.captured_at)
        measured = measure_detection(self._detector, frame)
        self.inference_ms = measured.inference_ms
        detections = tuple(
            detection
            for detection in measured.detections
            if detection.confidence >= self._config.min_person_confidence
        )
        self.appearance_ms = 0.0
        self.face_ms = 0.0
        if self._encoder is not None:
            started = perf_counter()
            detections = encode_detections(frame, detections, self._encoder)
            self.appearance_ms = (perf_counter() - started) * 1000
        self.people_detected = len(detections)
        tracks = self._tracker.update(context, detections)
        self.tracks = tracks
        events = self._events.generate(context, tracks)
        if self._identity is not None:
            started = perf_counter()
            self.identity_matches, self.identity_events = self._identity.process(frame, tracks)
            self.face_ms = (perf_counter() - started) * 1000
            if self._visitors is not None:
                self.visitor_matches, self.visitor_events = self._visitors.process(
                    context,
                    tracks,
                    events,
                    self._identity.visitor_observations,
                    self._identity.visitor_blocked,
                    self.identity_matches,
                    resident_evidence=self._identity.visitor_resident_evidence,
                    face_diagnostics=self._identity.visitor_diagnostics,
                )
                self._identity.visitor_observations.clear()
        return events

    def run(self, source: FrameSource) -> Iterator[PersonEvent]:
        """Pull frames on demand. The caller manages source and adapter resources."""
        for frame in source:
            yield from self.process(frame)
