"""Jake-owned, metadata-only tracking; no model tracking API is involved."""

from dataclasses import dataclass, replace
from datetime import datetime

from jake.config import TrackingConfig
from jake.domain import BoundingBox, FrameContext, PersonDetection, PersonTrack
from jake.matching import assign_boxes


class TrackerError(ValueError):
    """A frame violates the tracker's single-camera, ordered-session contract."""


@dataclass(frozen=True, slots=True)
class _TrackState:
    track_id: str
    box: BoundingBox
    confidence: float
    created_at: datetime
    last_seen_at: datetime
    age_frames: int = 1
    visible_frames: int = 1
    missed_frames: int = 0

    def public_track(self) -> PersonTrack:
        return PersonTrack(self.track_id, self.box, self.confidence)


class IoUPersonTracker:
    """One instance per camera session, updated serially, with no persistence.

    Ages/misses count calls to update(), not gaps in capture sequence numbers.
    Retained unmatched tracks expose their last-seen box/confidence. IDs begin at
    '1' and are never reused within an instance; they are not resident identities.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._tracks: tuple[_TrackState, ...] = ()
        self._next_id = 1
        self._last_context: FrameContext | None = None

    def update(
        self, context: FrameContext, detections: tuple[PersonDetection, ...]
    ) -> tuple[PersonTrack, ...]:
        if self._last_context is not None:
            if context.camera_id != self._last_context.camera_id:
                raise TrackerError("tracker cannot mix camera sessions; create a separate instance")
            if context.sequence <= self._last_context.sequence:
                raise TrackerError("tracker frame sequences must be strictly increasing")

        matches = dict(
            assign_boxes(
                tuple(track.box for track in self._tracks),
                tuple(detection.box for detection in detections),
                self._config.min_iou,
                self._config.assignment,
            )
        )
        used_detections = set(matches.values())
        updated = []
        for index, track in enumerate(self._tracks):
            if index in matches:
                detection = detections[matches[index]]
                updated.append(
                    replace(
                        track,
                        box=detection.box,
                        confidence=detection.confidence,
                        last_seen_at=context.captured_at,
                        age_frames=track.age_frames + 1,
                        visible_frames=track.visible_frames + 1,
                        missed_frames=0,
                    )
                )
            else:
                missed = track.missed_frames + 1
                if missed <= self._config.max_missed_frames:
                    updated.append(
                        replace(track, age_frames=track.age_frames + 1, missed_frames=missed)
                    )

        for index, detection in enumerate(detections):
            if index not in used_detections:
                updated.append(
                    _TrackState(
                        str(self._next_id),
                        detection.box,
                        detection.confidence,
                        context.captured_at,
                        context.captured_at,
                    )
                )
                self._next_id += 1

        self._tracks = tuple(updated)
        self._last_context = context
        return tuple(track.public_track() for track in self._tracks)
