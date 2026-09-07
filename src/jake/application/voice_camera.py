"""Optional camera worker feeding voice metadata without desktop dependencies."""

from threading import Event, Thread

from jake.application.voice import VoiceService
from jake.application.voice_binding import VisualVoiceBridge
from jake.config import AppConfig


class VoiceCamera:
    def __init__(self, config: AppConfig, voice: VoiceService) -> None:
        self.config, self.voice = config, voice
        self._cancel = Event()
        self.error: str | None = None
        self._thread = Thread(target=self._run, name="jake-voice-camera")

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self.request_stop()
        if self._thread.ident is not None:
            self._thread.join()

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
                    events = pipeline.process(frame)
                    self.voice.publish(
                        bridge.observe(
                            frame.captured_at,
                            pipeline.tracks,
                            events,
                            pipeline.identity_matches,
                            pipeline.visitor_matches,
                        )
                    )
        except Exception as exc:
            self.error = type(exc).__name__
