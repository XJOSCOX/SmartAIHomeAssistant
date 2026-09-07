"""Optional camera worker feeding voice metadata without desktop dependencies."""

from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import perf_counter

from jake.application.voice import VoiceService
from jake.application.voice_binding import VisualVoiceBridge
from jake.config import AppConfig
from jake.domain import Frame, PersonTrack
from jake.voice_domain import VisualContext


@dataclass(frozen=True, slots=True)
class VoicePreviewSnapshot:
    frame: Frame
    tracks: tuple[PersonTrack, ...]
    context: VisualContext
    processing_fps: float


class VoiceCamera:
    def __init__(self, config: AppConfig, voice: VoiceService, *, preview: bool = False) -> None:
        self.config, self.voice = config, voice
        self._cancel = Event()
        self.finished = Event()
        self._preview = preview
        self._snapshot_lock = Lock()
        self._snapshot: VoicePreviewSnapshot | None = None
        self.error: str | None = None
        self._thread = Thread(target=self._run, name="jake-voice-camera")

    def preview_snapshot(self) -> VoicePreviewSnapshot | None:
        """Return the latest immutable frame; never hold the lock while rendering."""
        with self._snapshot_lock:
            return self._snapshot

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self.request_stop()
        if self._thread.ident is not None:
            self._thread.join()
        with self._snapshot_lock:
            self._snapshot = None

    def _run(self) -> None:
        try:
            from jake.adapters.opencv_camera import OpenCVCamera
            from jake.adapters.person_events import PersonEventGenerator
            from jake.application.composition import compose
            from jake.pipeline import PerceptionPipeline

            config = self.config
            components = compose(
                config, identify=config.identity.enabled or config.visitors.enabled
            )
            assert components.detector is not None and components.tracker is not None
            pipeline = PerceptionPipeline(
                config.pipeline,
                components.detector,
                components.tracker,
                PersonEventGenerator(config.events),
                encoder=components.encoder,
                identity=components.identity,
                visitors=components.visitors,
            )
            bridge = VisualVoiceBridge()
            if self._cancel.is_set():
                return
            with OpenCVCamera(config.pipeline.camera_id, config.camera) as source:
                for frame in source:
                    if self._cancel.is_set():
                        break
                    started = perf_counter()
                    events = pipeline.process(frame)
                    processing_fps = 1 / max(perf_counter() - started, 1e-9)
                    context = bridge.observe(
                        frame.captured_at,
                        pipeline.tracks,
                        events,
                        pipeline.identity_matches,
                        pipeline.visitor_matches,
                    )
                    self.voice.publish(context)
                    if self._preview:
                        snapshot = VoicePreviewSnapshot(
                            frame, pipeline.tracks, context, processing_fps
                        )
                        with self._snapshot_lock:
                            self._snapshot = snapshot
        except Exception as exc:
            self.error = type(exc).__name__
        finally:
            self.finished.set()
