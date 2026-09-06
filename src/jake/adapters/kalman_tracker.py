"""Jake-owned motion prediction and correction around a selectable matcher."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter

import numpy as np

from jake.adapters.iou_tracker import TrackerError
from jake.appearance import AppearanceError, cosine_similarity, update_embedding
from jake.appearance_matching import assign_appearance_boxes
from jake.config import TrackingConfig
from jake.diagnostics import TrackDiagnostic
from jake.domain import AppearanceEmbedding, BoundingBox, FrameContext, PersonDetection, PersonTrack
from jake.hungarian import AssignmentError
from jake.kalman import KalmanError, KalmanFilter
from jake.matching import assign_boxes
from jake.motion import (
    bounded_box,
    frame_dt,
    initialize,
    measurement,
    observation_matrix,
    process_noise,
    transition_matrix,
)
from jake.motion_matching import assign_motion_boxes
from jake.reid_matching import assign_recent


class TrackState(StrEnum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    EXPIRED = "EXPIRED"
    RECENTLY_LOST = "RECENTLY_LOST"


@dataclass(frozen=True, slots=True)
class _MotionTrack:
    track_id: str
    motion: KalmanFilter
    box: BoundingBox
    predicted_box: BoundingBox
    confidence: float
    created_at: datetime
    last_seen_at: datetime
    measured_box: BoundingBox | None
    age_frames: int = 1
    visible_frames: int = 1
    missed_frames: int = 0
    lifecycle: TrackState = TrackState.CONFIRMED
    appearance: AppearanceEmbedding | None = field(default=None, repr=False)
    appearance_at: datetime | None = None
    appearance_similarity: float | None = None
    confirmed_at: datetime | None = None
    last_observed_box: BoundingBox | None = None
    reactivated: bool = False
    continuity_epoch: int = 0


class KalmanPersonTracker:
    """Session-local PersonTracker: predict → assignment → correct → lifecycle.

    Functional filter updates allow failed updates to leave all session state
    unchanged. The baseline counts missed updates; stabilized mode uses elapsed
    capture time and confirmation hits. EXPIRED states are removed before matching.
    """

    def __init__(
        self,
        config: TrackingConfig,
        *,
        stabilized: bool = False,
        appearance: bool = False,
        reid: bool = False,
    ) -> None:
        self._config = config
        self._stabilized = stabilized
        self._use_appearance = appearance
        self._use_reid = reid
        self._last_reid_ms = 0.0
        self._embedding_dimension: int | None = None
        self._tracks: tuple[_MotionTrack, ...] = ()
        self._last_context: FrameContext | None = None
        self._next_id = 1
        self._last_assignment_ms = 0.0

    @property
    def recently_lost_count(self) -> int:
        return sum(t.lifecycle == TrackState.RECENTLY_LOST for t in self._tracks)

    @property
    def last_reid_ms(self) -> float:
        return self._last_reid_ms

    @property
    def last_assignment_ms(self) -> float:
        """Last successful cost construction + gating + assignment; excludes filter work."""
        return self._last_assignment_ms

    def diagnostics(self) -> tuple[TrackDiagnostic, ...]:
        return tuple(
            TrackDiagnostic(
                t.track_id,
                t.missed_frames,
                t.predicted_box,
                t.measured_box,
                lifecycle=("REACTIVATED" if t.reactivated else t.lifecycle.value)
                if self._stabilized
                else None,
                visible_hits=t.visible_frames,
                confirmation_hits=self._config.confirmation_hits,
                missed_seconds=(
                    self._last_context.captured_at.astimezone(UTC) - t.last_seen_at.astimezone(UTC)
                ).total_seconds()
                if self._last_context
                else 0.0,
                appearance_similarity=t.appearance_similarity,
            )
            for t in self._tracks
        )

    def update(
        self, context: FrameContext, detections: tuple[PersonDetection, ...]
    ) -> tuple[PersonTrack, ...]:
        if self._last_context is not None:
            if context.camera_id != self._last_context.camera_id:
                raise TrackerError("tracker cannot mix camera sessions")
            if context.sequence <= self._last_context.sequence:
                raise TrackerError("tracker frame sequences must be strictly increasing")
            if self._stabilized and context.captured_at.astimezone(
                UTC
            ) < self._last_context.captured_at.astimezone(UTC):
                raise TrackerError("stabilized capture timestamps must not move backwards")
        dimension = self._embedding_dimension
        if self._use_appearance:
            for detection in detections:
                if detection.appearance is None:
                    raise TrackerError("appearance tracking requires encoded detections")
                length = len(detection.appearance.values)
                if dimension is not None and dimension != length:
                    raise TrackerError("appearance embedding dimension changed within session")
                dimension = length
        try:
            tracks, next_id, assignment_ms, reid_ms = self._advance(context, detections)
        except (KalmanError, AssignmentError, AppearanceError, np.linalg.LinAlgError) as exc:
            raise TrackerError(
                "Kalman tracking update failed; session state was preserved"
            ) from exc
        self._tracks, self._next_id, self._last_context = tracks, next_id, context
        self._last_assignment_ms = assignment_ms
        self._last_reid_ms = reid_ms
        self._embedding_dimension = dimension
        return tuple(
            PersonTrack(
                t.track_id,
                t.box,
                t.confidence,
                t.missed_frames,
                t.lifecycle != TrackState.TENTATIVE,
                t.lifecycle == TrackState.RECENTLY_LOST,
                t.continuity_epoch,
            )
            for t in tracks
        )

    def _advance(
        self, context: FrameContext, detections: tuple[PersonDetection, ...]
    ) -> tuple[tuple[_MotionTrack, ...], int, float, float]:
        noise = self._config.kalman
        dt = frame_dt(self._last_context, context)
        transition, process = transition_matrix(dt), process_noise(noise, dt)
        predicted = []
        for track in self._tracks:
            if (
                self._stabilized
                and (
                    context.captured_at.astimezone(UTC) - track.last_seen_at.astimezone(UTC)
                ).total_seconds()
                > self._config.max_missed_seconds
            ):
                # Expire before association: late detections cannot resurrect an old ID.
                gap = (
                    context.captured_at.astimezone(UTC) - track.last_seen_at.astimezone(UTC)
                ).total_seconds()
                if (
                    self._use_reid
                    and track.lifecycle != TrackState.TENTATIVE
                    and gap <= self._config.reid.window_seconds
                ):
                    track = replace(track, lifecycle=TrackState.RECENTLY_LOST)
                else:
                    continue
            if (
                self._use_appearance
                and track.appearance_at is not None
                and (
                    context.captured_at.astimezone(UTC) - track.appearance_at.astimezone(UTC)
                ).total_seconds()
                > self._config.appearance.max_embedding_age_seconds
            ):
                track = replace(track, appearance=None, appearance_at=None)
            motion = track.motion.predict(transition, process)
            box = bounded_box(motion.state)
            predicted.append(
                replace(
                    track,
                    motion=motion,
                    box=box,
                    predicted_box=box,
                    measured_box=None,
                    appearance_similarity=None,
                    reactivated=False,
                    age_frames=track.age_frames + 1,
                )
            )
        assignment_start = perf_counter()
        active_indices = [
            i for i, t in enumerate(predicted) if t.lifecycle != TrackState.RECENTLY_LOST
        ]
        active = [predicted[i] for i in active_indices]
        track_boxes = tuple(t.predicted_box for t in active)
        detection_boxes = tuple(d.box for d in detections)
        matches = dict(
            assign_appearance_boxes(
                track_boxes,
                detection_boxes,
                tuple(t.appearance for t in active),
                tuple(d.appearance for d in detections),
                self._config,
            )
            if self._use_appearance
            else assign_motion_boxes(track_boxes, detection_boxes, self._config)
            if self._stabilized
            else assign_boxes(
                track_boxes, detection_boxes, self._config.min_iou, self._config.assignment
            )
        )
        matches = {active_indices[i]: j for i, j in matches.items()}
        assignment_ms = (perf_counter() - assignment_start) * 1000
        reid_ms = 0.0
        if self._use_reid:
            reid_start = perf_counter()
            recent_indices = [
                i for i, t in enumerate(predicted) if t.lifecycle == TrackState.RECENTLY_LOST
            ]
            available = [j for j in range(len(detections)) if j not in matches.values()]
            recent = [predicted[i] for i in recent_indices]
            for i, j in assign_recent(
                tuple(t.predicted_box for t in recent),
                tuple(t.last_observed_box or t.box for t in recent),
                tuple(t.appearance for t in recent),
                tuple(detections[j] for j in available),
                self._config.reid,
                self._config.assignment,
            ):
                matches[recent_indices[i]] = available[j]
            reid_ms = (perf_counter() - reid_start) * 1000
        updated = []
        for index, track in enumerate(predicted):
            if index in matches:
                detection = detections[matches[index]]
                motion = track.motion.correct(
                    measurement(detection.box),
                    observation_matrix(),
                    np.eye(4) * noise.measurement_noise,
                )
                appearance = track.appearance
                similarity = None
                if self._use_appearance and detection.appearance is not None:
                    if appearance is not None:
                        similarity = cosine_similarity(appearance, detection.appearance)
                    appearance = update_embedding(
                        appearance, detection.appearance, self._config.appearance.ema_alpha
                    )
                updated.append(
                    replace(
                        track,
                        appearance=appearance,
                        last_observed_box=detection.box,
                        confirmed_at=track.confirmed_at
                        or (
                            context.captured_at
                            if track.visible_frames + 1 >= self._config.confirmation_hits
                            else None
                        ),
                        reactivated=track.lifecycle == TrackState.RECENTLY_LOST,
                        continuity_epoch=track.continuity_epoch
                        + int(track.lifecycle == TrackState.RECENTLY_LOST),
                        appearance_at=context.captured_at if self._use_appearance else None,
                        appearance_similarity=similarity,
                        motion=motion,
                        box=bounded_box(motion.state),
                        confidence=detection.confidence,
                        measured_box=detection.box,
                        last_seen_at=context.captured_at,
                        visible_frames=track.visible_frames + 1,
                        missed_frames=0,
                        lifecycle=(
                            TrackState.CONFIRMED
                            if not self._stabilized
                            or track.visible_frames + 1 >= self._config.confirmation_hits
                            else TrackState.TENTATIVE
                        ),
                    )
                )
            elif self._stabilized or track.missed_frames < self._config.max_missed_frames:
                updated.append(
                    replace(
                        track,
                        missed_frames=track.missed_frames + 1,
                        lifecycle=TrackState.RECENTLY_LOST
                        if track.lifecycle == TrackState.RECENTLY_LOST
                        else TrackState.LOST
                        if track.lifecycle != TrackState.TENTATIVE
                        else TrackState.TENTATIVE,
                    )
                )
        used = set(matches.values())
        next_id = self._next_id
        for index, detection in enumerate(detections):
            if index not in used:
                updated.append(
                    _MotionTrack(
                        str(next_id),
                        initialize(detection.box, noise),
                        detection.box,
                        detection.box,
                        detection.confidence,
                        context.captured_at,
                        context.captured_at,
                        detection.box,
                        lifecycle=TrackState.TENTATIVE
                        if self._stabilized and self._config.confirmation_hits > 1
                        else TrackState.CONFIRMED,
                        last_observed_box=detection.box,
                        confirmed_at=context.captured_at
                        if not self._stabilized or self._config.confirmation_hits == 1
                        else None,
                        appearance=detection.appearance if self._use_appearance else None,
                        appearance_at=context.captured_at if self._use_appearance else None,
                    )
                )
                next_id += 1
        return tuple(updated), next_id, assignment_ms, reid_ms


class StabilizedKalmanPersonTracker(KalmanPersonTracker):
    """Time-retained, confirmation-gated Kalman tracking with motion association."""

    def __init__(self, config: TrackingConfig) -> None:
        super().__init__(config, stabilized=True)


class AppearanceKalmanPersonTracker(KalmanPersonTracker):
    """Stabilized tracking with session-local, expiring appearance vectors."""

    def __init__(self, config: TrackingConfig) -> None:
        super().__init__(config, stabilized=True, appearance=True)


class RecentlyLostPersonTracker(KalmanPersonTracker):
    """Appearance tracker retaining confirmed IDs through a bounded ReID window."""

    def __init__(self, config: TrackingConfig) -> None:
        if config.reid.window_seconds <= config.max_missed_seconds:
            raise ValueError("ReID window must exceed active retention")
        super().__init__(config, stabilized=True, appearance=True, reid=True)
