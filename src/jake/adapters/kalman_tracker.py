"""Jake-owned motion prediction and correction around a selectable matcher."""

from dataclasses import dataclass, replace
from datetime import datetime
from time import perf_counter

import numpy as np

from jake.adapters.iou_tracker import TrackerError
from jake.config import TrackingConfig
from jake.diagnostics import TrackDiagnostic
from jake.domain import BoundingBox, FrameContext, PersonDetection, PersonTrack
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


class KalmanPersonTracker:
    """Session-local PersonTracker: predict → assignment → correct → lifecycle.

    Functional filter updates allow failed updates to leave all session state
    unchanged. Misses count processed updates as in Phase 1C, not elapsed time.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._tracks: tuple[_MotionTrack, ...] = ()
        self._last_context: FrameContext | None = None
        self._next_id = 1
        self._last_assignment_ms = 0.0

    @property
    def last_assignment_ms(self) -> float:
        """Last successful cost construction + gating + assignment; excludes filter work."""
        return self._last_assignment_ms

    def diagnostics(self) -> tuple[TrackDiagnostic, ...]:
        return tuple(
            TrackDiagnostic(t.track_id, t.missed_frames, t.predicted_box, t.measured_box)
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
        try:
            tracks, next_id, assignment_ms = self._advance(context, detections)
        except (KalmanError, AssignmentError, np.linalg.LinAlgError) as exc:
            raise TrackerError(
                "Kalman tracking update failed; session state was preserved"
            ) from exc
        self._tracks, self._next_id, self._last_context = tracks, next_id, context
        self._last_assignment_ms = assignment_ms
        return tuple(PersonTrack(t.track_id, t.box, t.confidence, t.missed_frames) for t in tracks)

    def _advance(
        self, context: FrameContext, detections: tuple[PersonDetection, ...]
    ) -> tuple[tuple[_MotionTrack, ...], int, float]:
        noise = self._config.kalman
        dt = frame_dt(self._last_context, context)
        transition, process = transition_matrix(dt), process_noise(noise, dt)
        predicted = []
        for track in self._tracks:
            motion = track.motion.predict(transition, process)
            box = bounded_box(motion.state)
            predicted.append(
                replace(
                    track,
                    motion=motion,
                    box=box,
                    predicted_box=box,
                    measured_box=None,
                    age_frames=track.age_frames + 1,
                )
            )
        assignment_start = perf_counter()
        matches = dict(
            assign_boxes(
                tuple(t.predicted_box for t in predicted),
                tuple(d.box for d in detections),
                self._config.min_iou,
                self._config.assignment,
            )
        )
        assignment_ms = (perf_counter() - assignment_start) * 1000
        updated = []
        for index, track in enumerate(predicted):
            if index in matches:
                detection = detections[matches[index]]
                motion = track.motion.correct(
                    measurement(detection.box),
                    observation_matrix(),
                    np.eye(4) * noise.measurement_noise,
                )
                updated.append(
                    replace(
                        track,
                        motion=motion,
                        box=bounded_box(motion.state),
                        confidence=detection.confidence,
                        measured_box=detection.box,
                        last_seen_at=context.captured_at,
                        visible_frames=track.visible_frames + 1,
                        missed_frames=0,
                    )
                )
            elif track.missed_frames < self._config.max_missed_frames:
                updated.append(replace(track, missed_frames=track.missed_frames + 1))
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
                    )
                )
                next_id += 1
        return tuple(updated), next_id, assignment_ms
