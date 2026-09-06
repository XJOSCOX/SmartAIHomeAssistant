"""Cooperative worker lifecycle and bounded preview handoff; no GUI inference."""

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from threading import Event, Lock
from typing import Protocol

from PySide6.QtCore import QObject, QThread, Signal

from jake.application.errors import ui_error
from jake.application.sessions import EnrollmentSession, EventRow, LiveSession, Update
from jake.config import AppConfig
from jake.domain import Frame


class Session(Protocol):
    def process(self, frame: Frame) -> Update: ...


class Mailbox:
    def __init__(self) -> None:
        self.lock = Lock()
        self.latest: Update | None = None
        self.events: deque[EventRow] = deque(maxlen=1000)

    def publish(self, update: Update) -> None:
        with self.lock:
            self.latest = update
            self.events.extend(update.events)

    def take(self) -> tuple[Update | None, tuple[EventRow, ...]]:
        with self.lock:
            latest, events = self.latest, tuple(self.events)
            self.latest = None
            self.events.clear()
            return latest, events


def camera_source(config: AppConfig) -> AbstractContextManager[Iterator[Frame]]:
    from jake.adapters.opencv_camera import OpenCVCamera

    return OpenCVCamera(config.pipeline.camera_id, config.camera)


class CameraWorker(QThread):
    failed = Signal(object)
    running = Signal()
    enrolled = Signal()

    def __init__(
        self,
        config: AppConfig,
        factory: Callable[[], Session],
        source: Callable[[AppConfig], AbstractContextManager[Iterator[Frame]]] = camera_source,
    ) -> None:
        super().__init__()
        self.config, self.factory, self.source = config, factory, source
        self.cancel = Event()
        self.mailbox = Mailbox()

    def run(self) -> None:
        session = None
        try:
            if self.cancel.is_set():
                return
            session = self.factory()  # Models and identity cache belong to this worker.
            if self.cancel.is_set():
                return
            with self.source(self.config) as frames:
                self.running.emit()
                while not self.cancel.is_set():
                    try:
                        frame = next(frames)
                    except StopIteration:
                        break
                    if self.cancel.is_set():
                        break
                    update = session.process(frame)
                    if self.cancel.is_set():
                        break
                    self.mailbox.publish(update)
                    if update.complete:
                        if isinstance(session, EnrollmentSession):
                            session.commit()
                            self.enrolled.emit()
                        break
        except Exception as exc:
            self.failed.emit(ui_error(exc))
        finally:
            session = None  # Drop model/cache/template ownership before thread finishes.


class CameraController(QObject):
    drained = Signal(object, object)
    state_changed = Signal(str)
    failed = Signal(object)
    enrolled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.worker: CameraWorker | None = None
        self.state = "Stopped"

    @property
    def busy(self) -> bool:
        return self.worker is not None

    def _state(self, state: str) -> None:
        self.state = state
        self.state_changed.emit(state)

    def start(
        self,
        config: AppConfig,
        factory: Callable[[], Session] | None = None,
        source: Callable[[AppConfig], AbstractContextManager[Iterator[Frame]]] = camera_source,
    ) -> None:
        if self.busy:
            raise ValueError("camera is already running or stopping")
        worker = CameraWorker(config, factory or (lambda: LiveSession(config)), source)
        self.worker = worker
        worker.running.connect(self._running)
        worker.failed.connect(self.failed.emit)
        worker.enrolled.connect(self.enrolled.emit)
        worker.finished.connect(self._finished)
        self._state("Starting")
        worker.start()

    def _running(self) -> None:
        if self.state == "Starting":
            self._state("Running")

    def stop(self) -> None:
        if self.worker:
            self._state("Stopping")
            self.worker.cancel.set()  # Does not depend on the worker's Qt event loop.

    def _finished(self) -> None:
        if self.worker:
            self.drained.emit(*self.worker.mailbox.take())
            self.worker.deleteLater()
            self.worker = None
        self._state("Stopped")

    def take(self) -> tuple[Update | None, tuple[EventRow, ...]]:
        return self.worker.mailbox.take() if self.worker else (None, ())


class TaskWorker(QThread):
    result = Signal(object)
    failed = Signal(object)

    def __init__(self, operation: Callable[[], object]) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            self.result.emit(self.operation())
        except Exception as exc:
            self.failed.emit(ui_error(exc))
