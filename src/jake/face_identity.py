"""Optional face stage, separate from perception events and body tracking."""

from collections.abc import Callable
from datetime import UTC, datetime

from jake.domain import Frame, FrameContext, PersonTrack
from jake.identity import CosineIdentityMatcher, TemporalIdentity, face_cosine
from jake.identity_config import IdentityConfig
from jake.identity_domain import (
    FaceDetection,
    FaceEmbedding,
    FaceQuality,
    IdentityEvent,
    IdentityMatch,
)
from jake.identity_ports import FaceDetector, FaceEncoder, IdentityMatcher, IdentityStore
from jake.matching import intersection_over_union
from jake.visitor_domain import ResidentEvidence, VisitorObservation


class FaceIdentityService:
    def confidently_nonresident(self, embedding: FaceEmbedding) -> bool:
        return not any(
            face_cosine(embedding, template) >= self.config.candidate_similarity
            for profile in self.profiles
            for template in profile.templates
        )

    def __init__(
        self,
        config: IdentityConfig,
        detector: FaceDetector,
        encoder: FaceEncoder,
        store: IdentityStore,
        quality: Callable[[Frame, FaceDetection, IdentityConfig], FaceQuality],
        matcher: IdentityMatcher | None = None,
        *,
        collect_visitors: bool = False,
    ) -> None:
        self.config = config
        self.detector = detector
        self.encoder = encoder
        self.profiles = store.profiles()
        self.quality = quality
        self.matcher = matcher or CosineIdentityMatcher(config)
        self.temporal = TemporalIdentity(config)
        self._attempts: dict[str, datetime] = {}
        self.collect_visitors = collect_visitors
        self.visitor_observations: dict[str, VisitorObservation] = {}
        self.visitor_blocked: set[str] = set()
        self.visitor_resident_evidence: dict[str, ResidentEvidence] = {}
        self.visitor_diagnostics: dict[str, str] = {}

    def process(
        self, frame: Frame, tracks: tuple[PersonTrack, ...]
    ) -> tuple[dict[str, IdentityMatch], tuple[IdentityEvent, ...]]:
        now = frame.captured_at.astimezone(UTC)
        self.visitor_observations.clear()
        self.visitor_blocked.clear()
        self.visitor_resident_evidence.clear()
        self.visitor_diagnostics = {t.track_id: "waiting for face observation" for t in tracks}
        self._attempts = {
            k: v for k, v in self._attempts.items() if any(t.track_id == k for t in tracks)
        }
        faces = []
        for track in tracks:
            if not track.confirmed or not track.visible:
                self.visitor_diagnostics[track.track_id] = "waiting for visible confirmed track"
                continue
            last = self._attempts.get(track.track_id)
            if last and (now - last).total_seconds() < self.config.observation_interval_seconds:
                continue
            self._attempts[track.track_id] = now
            detected = self.detector.detect(frame, track.box)
            if len(detected) == 1:
                faces.append((track.track_id, detected[0]))
            else:
                self.visitor_diagnostics[track.track_id] = (
                    f"rejected face count {len(detected)} (requires one)"
                )
        observations = {}
        for track_id, face in faces:
            if any(
                other_id != track_id and intersection_over_union(face.box, other.box) > 0.5
                for other_id, other in faces
            ):
                self.visitor_diagnostics[track_id] = "rejected overlapping person face crops"
                continue  # A face seen in overlapping person crops cannot identify both tracks.
            quality = self.quality(frame, face, self.config)
            if not quality.accepted:
                self.visitor_diagnostics[track_id] = (
                    f"rejected detector confidence {face.confidence:.2f} "
                    f"< {self.config.min_detector_confidence:.2f}"
                    if face.confidence < self.config.min_detector_confidence
                    else f"rejected {quality.reason}"
                )
            if quality.accepted:
                embedding = self.encoder.encode(frame, face)
                observations[track_id] = self.matcher.match(embedding, self.profiles)
                if self.collect_visitors:
                    # UNKNOWN from an ambiguous resident match is NOT non-resident evidence.
                    if not self.confidently_nonresident(embedding):
                        self.visitor_blocked.add(track_id)
                        score, resident_id = max(
                            (face_cosine(embedding, template), profile.resident_id)
                            for profile in self.profiles
                            for template in profile.templates
                        )
                        self.visitor_resident_evidence[track_id] = ResidentEvidence(
                            resident_id, score, score >= self.config.resident_similarity
                        )
                        self.visitor_diagnostics[track_id] = (
                            f"paused possible resident similarity {score:.2f}"
                        )
                    else:
                        self.visitor_diagnostics[track_id] = (
                            "quality-approved nonresident observation"
                        )
                        self.visitor_observations[track_id] = VisitorObservation(
                            embedding, face.confidence
                        )
        return self.temporal.update(
            FrameContext(frame.camera_id, frame.sequence, frame.captured_at), tracks, observations
        )
