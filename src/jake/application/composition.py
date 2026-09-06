"""Shared CLI/desktop composition. Heavy adapters load only on session start."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jake.config import AppConfig
from jake.diagnostics import TrackDiagnostic
from jake.face_identity import FaceIdentityService
from jake.ports import AppearanceEncoder, PersonDetector, PersonTracker
from jake.visitors import VisitorMemory


@dataclass
class Components:
    detector: PersonDetector | None
    tracker: PersonTracker | None
    encoder: AppearanceEncoder | None
    identity: FaceIdentityService | None
    visitors: VisitorMemory | None
    diagnostics: Callable[[], tuple[TrackDiagnostic, ...]] | None


def compose(
    config: AppConfig,
    *,
    detect: bool = True,
    track: bool = True,
    tracker_mode: str = "kalman",
    identify: bool = False,
    debug_tracks: bool = False,
) -> Components:
    from jake.adapters.iou_tracker import IoUPersonTracker
    from jake.adapters.kalman_tracker import KalmanPersonTracker, StabilizedKalmanPersonTracker
    from jake.adapters.yolo_detector import YoloPersonDetector

    detector = (
        YoloPersonDetector(config.detector, config.pipeline.min_person_confidence)
        if detect or track
        else None
    )
    encoder = None
    tracker: PersonTracker | None = None
    diagnostics = None
    if track:
        if tracker_mode in {"kalman", "kalman-baseline"}:
            motion_tracker = (
                StabilizedKalmanPersonTracker(config.tracking)
                if tracker_mode == "kalman"
                else KalmanPersonTracker(config.tracking)
            )
            if tracker_mode == "kalman" and config.tracking.appearance.enabled:
                from jake.adapters.kalman_tracker import AppearanceKalmanPersonTracker
                from jake.adapters.openvino_appearance import OpenVINOAppearanceEncoder

                encoder = OpenVINOAppearanceEncoder(config.tracking.appearance)
                motion_tracker = AppearanceKalmanPersonTracker(config.tracking)
                if config.tracking.reid.enabled:
                    from jake.adapters.kalman_tracker import RecentlyLostPersonTracker

                    motion_tracker = RecentlyLostPersonTracker(config.tracking)
            tracker = motion_tracker
            if debug_tracks:
                diagnostics = motion_tracker.diagnostics
        else:
            tracker = IoUPersonTracker(config.tracking)
    identity = None
    visitors = None
    if identify:
        from jake.adapters.local_identity_store import LocalIdentityStore
        from jake.adapters.opencv_faces import SFaceEncoder, YuNetFaceDetector, face_quality
        from jake.face_identity import FaceIdentityService

        identity = FaceIdentityService(
            config.identity,
            YuNetFaceDetector(config.identity),
            SFaceEncoder(config.identity),
            LocalIdentityStore(Path(config.identity.store_path)),
            face_quality,
            collect_visitors=config.visitors.enabled,
        )
        if config.visitors.enabled:
            from jake.adapters.visitor_store import EncryptedVisitorStore
            from jake.visitors import VisitorMemory

            visitors = VisitorMemory(
                config.visitors,
                EncryptedVisitorStore(
                    Path(config.identity.store_path), timezone=config.home.timezone
                ),
                timezone=config.home.timezone,
                is_nonresident=identity.confidently_nonresident,
            )
    return Components(detector, tracker, encoder, identity, visitors, diagnostics)
