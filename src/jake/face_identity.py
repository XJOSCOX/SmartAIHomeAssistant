"""Optional face stage, separate from perception events and body tracking."""

from collections.abc import Callable
from datetime import UTC, datetime

from jake.domain import Frame, FrameContext, PersonTrack
from jake.identity import CosineIdentityMatcher, TemporalIdentity
from jake.identity_config import IdentityConfig
from jake.identity_domain import FaceDetection, FaceQuality, IdentityEvent, IdentityMatch
from jake.identity_ports import FaceDetector, FaceEncoder, IdentityMatcher, IdentityStore
from jake.matching import intersection_over_union


class FaceIdentityService:
    def __init__(
        self,
        config: IdentityConfig,
        detector: FaceDetector,
        encoder: FaceEncoder,
        store: IdentityStore,
        quality: Callable[[Frame, FaceDetection, IdentityConfig], FaceQuality],
        matcher: IdentityMatcher | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.encoder = encoder
        self.profiles = store.profiles()
        self.quality = quality
        self.matcher = matcher or CosineIdentityMatcher(config)
        self.temporal = TemporalIdentity(config)
        self._attempts: dict[str, datetime] = {}

    def process(
        self, frame: Frame, tracks: tuple[PersonTrack, ...]
    ) -> tuple[dict[str, IdentityMatch], tuple[IdentityEvent, ...]]:
        now = frame.captured_at.astimezone(UTC)
        self._attempts = {
            k: v for k, v in self._attempts.items() if any(t.track_id == k for t in tracks)
        }
        faces = []
        for track in tracks:
            if not track.confirmed or not track.visible:
                continue
            last = self._attempts.get(track.track_id)
            if last and (now - last).total_seconds() < self.config.observation_interval_seconds:
                continue
            self._attempts[track.track_id] = now
            detected = self.detector.detect(frame, track.box)
            if len(detected) == 1:
                faces.append((track.track_id, detected[0]))
        observations = {}
        for track_id, face in faces:
            if any(
                other_id != track_id and intersection_over_union(face.box, other.box) > 0.5
                for other_id, other in faces
            ):
                continue  # A face seen in overlapping person crops cannot identify both tracks.
            if self.quality(frame, face, self.config).accepted:
                observations[track_id] = self.matcher.match(
                    self.encoder.encode(frame, face), self.profiles
                )
        return self.temporal.update(
            FrameContext(frame.camera_id, frame.sequence, frame.captured_at), tracks, observations
        )
