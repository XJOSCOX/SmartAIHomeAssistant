"""Session services reuse core pipeline/enrollment and expose only preview metadata."""

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter

from jake.adapters.person_events import PersonEventGenerator
from jake.application.composition import compose
from jake.application.settings import validate_models
from jake.camera_info import CameraInfo
from jake.config import AppConfig
from jake.domain import BoundingBox, Frame
from jake.identity import Enrollment, IdentityError
from jake.identity_domain import IdentityState
from jake.pipeline import PerceptionPipeline
from jake.visitor_domain import VisitorState


@dataclass(frozen=True)
class EventRow:
    at: datetime
    kind: str
    track: str
    display: str
    summary: str


@dataclass(frozen=True)
class Overlay:
    box: BoundingBox
    label: str


@dataclass(frozen=True)
class Update:
    frame: Frame = field(repr=False)
    overlays: tuple[Overlay, ...] = ()
    events: tuple[EventRow, ...] = ()
    fps: float = 0
    inference_ms: float = 0
    people: int = 0
    residents: int = 0
    visitors: int = 0
    visitor_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    progress: str = ""
    complete: bool = False
    camera_info: CameraInfo | None = None
    appearance_ms: float = 0
    face_ms: float = 0


class LiveSession:
    def __init__(self, config: AppConfig) -> None:
        validate_models(config)
        self.components = compose(
            config, identify=config.identity.enabled or config.visitors.enabled, debug_tracks=True
        )
        c = self.components
        assert c.detector is not None and c.tracker is not None
        self.pipeline = PerceptionPipeline(
            config.pipeline,
            c.detector,
            c.tracker,
            PersonEventGenerator(config.events),
            encoder=c.encoder,
            identity=c.identity,
            visitors=c.visitors,
        )

    def process(self, frame: Frame) -> Update:
        started = perf_counter()
        p = self.pipeline
        person_events = p.process(frame)
        now = perf_counter()
        fps = 1 / max(now - started, 1e-9)
        overlays = []
        residents: set[str] = set()
        visitors: set[str] = set()
        for track in p.tracks:
            if not track.visible:
                continue
            label = f"ID {track.track_id} | PERSON {track.confidence:.1%}"
            identity = p.identity_matches.get(track.track_id)
            visitor = p.visitor_matches.get(track.track_id)
            if identity and identity.state == IdentityState.RESIDENT:
                label = f"ID {track.track_id} | {identity.display_name} | RESIDENT"
                residents.add(identity.resident_id or track.track_id)
            elif (
                visitor
                and visitor.visitor_id
                and visitor.state not in (VisitorState.UNKNOWN, VisitorState.VISITOR_CANDIDATE)
            ):
                label = (
                    f"ID {track.track_id} | "
                    f"{visitor.display_name or 'VISITOR ' + visitor.visitor_id[:4].upper()} "
                    f"| {visitor.state}"
                )
                visitors.add(visitor.visitor_id)
            overlays.append(Overlay(track.box, label))
        rows = [
            EventRow(e.context.captured_at, str(e.kind), e.track_id, "—", "Person lifecycle")
            for e in person_events
        ]
        rows.extend(
            EventRow(
                e.context.captured_at,
                e.kind,
                e.track_id,
                e.match.display_name or "—",
                str(e.match.state),
            )
            for e in p.identity_events
        )
        rows.extend(
            EventRow(
                e.context.captured_at,
                e.kind,
                e.track_id,
                e.match.display_name or (e.match.visitor_id or "")[:8],
                f"{e.match.state} | sessions={e.match.session_count} "
                f"days={e.match.distinct_visit_days}",
            )
            for e in p.visitor_events
        )
        diagnostics: list[str] = []
        if self.components.diagnostics:
            diagnostics.extend(
                f"ID {d.track_id}: {d.lifecycle or 'active'}, missed={d.missed_frames}"
                for d in self.components.diagnostics()
            )
        if self.components.identity:
            diagnostics.extend(
                f"FACE {k}: {v.summary}" for k, v in self.components.identity.face_qualities.items()
            )
        if self.components.visitors:
            diagnostics.extend(
                f"VISITOR {k}: {v}" for k, v in self.components.visitors.diagnostics.items()
            )
        return Update(
            frame,
            tuple(overlays),
            tuple(rows),
            fps,
            p.inference_ms,
            p.people_detected,
            len(residents),
            len(visitors),
            tuple(sorted(visitors)),
            tuple(diagnostics),
            appearance_ms=p.appearance_ms,
            face_ms=p.face_ms,
        )


class EnrollmentSession:
    def __init__(self, config: AppConfig, name: str, consent: bool) -> None:
        from pathlib import Path

        from jake.adapters.local_identity_store import LocalIdentityStore
        from jake.adapters.opencv_faces import SFaceEncoder, YuNetFaceDetector

        if not consent:
            raise IdentityError("explicit resident consent is required")
        name = name.strip()
        if not name or len(name) > 80 or not name.isprintable():
            raise IdentityError("resident name must be printable and 1-80 characters")
        self.store = LocalIdentityStore(Path(config.identity.store_path))
        if any(p.display_name.casefold() == name.casefold() for p in self.store.profiles()):
            raise IdentityError("resident name already enrolled")
        validate_models(config, enrollment=True)
        self.detector = YuNetFaceDetector(config.identity)
        self.encoder = SFaceEncoder(config.identity)
        self.enrollment = Enrollment(config.identity)
        self.config, self.name = config, name
        self.saved = False
        self._last_attempt: datetime | None = None
        self._progress = "Waiting for face observation"

    def process(self, frame: Frame) -> Update:
        from jake.adapters.opencv_faces import face_quality

        if self._last_attempt is not None and (
            frame.captured_at - self._last_attempt
        ).total_seconds() < max(0.5, self.config.identity.observation_interval_seconds):
            return Update(frame, progress=self._progress, complete=self.enrollment.ready)
        self._last_attempt = frame.captured_at
        started = perf_counter()
        faces = self.detector.detect(frame, BoundingBox(0, 0, 1, 1))
        reason = "Exactly one consenting face required; look forward then turn slightly"
        if len(faces) == 1:
            quality = face_quality(frame, faces[0], self.config.identity)
            embedding = self.encoder.encode(frame, faces[0]) if quality.accepted else None
            reason = self.enrollment.accept(quality, embedding, frame.captured_at)
            if quality.accepted:
                reason += f" · {quality.summary}"
        self._progress = (
            f"{len(self.enrollment.samples)} / "
            f"{self.config.identity.enrollment_samples} accepted · {reason}"
        )
        return Update(
            frame,
            progress=self._progress,
            complete=self.enrollment.ready,
            face_ms=(perf_counter() - started) * 1000,
        )

    def commit(self) -> None:
        if not self.saved:
            self.store.add(self.enrollment.profile(self.name))
            self.saved = True
