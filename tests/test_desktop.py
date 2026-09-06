"""Offscreen desktop contracts. No real camera, models, vault or display required."""

# ruff: noqa: E402
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, get_ident
from time import monotonic, sleep
from typing import cast
from unittest.mock import Mock
from uuid import UUID

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("tomli_w")
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from jake.application.administration import Administration, ResidentRow, VisitorRow
from jake.application.errors import UIError, ui_error
from jake.application.sessions import EventRow, Update
from jake.application.settings import AppPaths, save_config, validate_models
from jake.config import AppConfig, load_app_config
from jake.desktop.controller import CameraController, CameraWorker, Mailbox
from jake.desktop.style import apply_theme
from jake.desktop.widgets import CameraPanel, SettingsPanel
from jake.desktop.window import PAGES, MainWindow
from jake.domain import Frame
from jake.identity import IdentityError, face_normalize
from jake.identity_domain import ResidentProfile
from jake.visitor_domain import VisitorProfile

NOW = datetime(2026, 9, 6, tzinfo=UTC)
FRAME = Frame("home", 0, NOW, 2, 1, bytes([255, 0, 0, 0, 255, 0]))


@pytest.fixture(scope="module")
def app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppPaths(tmp_path).defaults()


def until(app: QApplication, predicate: Callable[[], bool]) -> None:
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        app.processEvents()
        sleep(0.001)
    assert predicate()


@pytest.fixture
def window(
    app: QApplication, config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MainWindow]:
    monkeypatch.setattr(QMessageBox, "warning", Mock())
    monkeypatch.setattr(QMessageBox, "information", Mock())
    item = MainWindow(config, tmp_path / "config.toml")
    yield item
    item.controller.stop()
    until(app, lambda: not item.controller.busy and item.task is None)
    item.close()
    app.processEvents()


def test_navigation_and_no_automatic_store_access(window: MainWindow, tmp_path: Path) -> None:
    assert window.stack.count() == len(PAGES) == 7
    for index in range(7):
        window.navigation[index].click()
        assert window.stack.currentIndex() == index
        assert window.navigation[index].isChecked()
    assert not list(tmp_path.rglob("residents.json"))
    assert not list(tmp_path.rglob("visitors.json"))
    assert not window.controller.busy


@pytest.mark.parametrize("mode", ["Dark", "Light", "System"])
def test_theme_and_preview_render(app: QApplication, mode: str) -> None:
    apply_theme(app, mode)
    panel = CameraPanel()
    panel.present(Update(FRAME))
    assert panel.image.pixelColor(0, 0).red() == 255
    assert panel.image.pixelColor(1, 0).green() == 255
    panel.resize(600, 300)
    assert not panel.grab().isNull()
    panel.clear()
    assert panel.image.isNull() and panel.update_data is None
    panel.close()


def test_settings_atomic_roundtrip_and_failure(
    config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.toml"
    save_config(path, config)
    assert load_app_config(path) == config
    before = path.read_bytes()
    monkeypatch.setattr("jake.application.settings.os.replace", Mock(side_effect=OSError("disk")))
    with pytest.raises(OSError):
        save_config(path, replace(config, camera=replace(config.camera, device=2)))
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".jake-config-*")) == []


def test_settings_validation_without_writing(window: MainWindow) -> None:
    window.settings.timezone.setText("Invalid/Zone")
    with pytest.raises(ValueError):
        window.settings.value()
    window.save_settings()
    assert not window.path.exists()
    window.settings.timezone.setText("UTC")
    window.settings.recurring.setValue(7)
    window.settings.frequent.setValue(5)
    with pytest.raises(ValueError):
        window.settings.value()


def test_settings_retains_unedited_algorithms(window: MainWindow) -> None:
    window.settings.device.setValue(3)
    window.settings.identity.setChecked(True)
    window.settings.visitors.setChecked(True)
    window.save_settings()
    updated = load_app_config(window.path)
    assert updated.camera.device == 3
    assert updated.identity.enabled and updated.visitors.enabled
    assert updated.tracking == window.config.tracking


def test_app_locations_and_model_preflight(
    config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("jake.application.settings.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    paths = AppPaths.default()
    paths.initialize()
    assert paths.root == tmp_path / "Jake"
    assert all((paths.root / name).is_dir() for name in ("models", "config", "data", "logs"))
    with pytest.raises(FileNotFoundError):
        validate_models(config)
    Path(config.detector.model).parent.mkdir(exist_ok=True)
    Path(config.detector.model).write_bytes(b"fake weights, never loaded")
    validate_models(config)
    with pytest.raises(FileNotFoundError):
        validate_models(config, enrollment=True)


def test_mailbox_bounds_and_drains() -> None:
    mailbox = Mailbox()
    for index in range(1100):
        mailbox.publish(
            Update(
                replace(FRAME, sequence=index),
                events=(EventRow(NOW, "TEST", str(index), "—", "metadata"),),
            )
        )
    latest, events = mailbox.take()
    assert latest and latest.frame.sequence == 1099
    assert len(events) == 1000 and events[0].track == "100"
    assert mailbox.take() == (None, ())


def test_controller_start_stop_restart_releases(app: QApplication, config: AppConfig) -> None:
    controller = CameraController()
    worker_threads = []
    releases = []

    def process(frame: Frame) -> Update:
        worker_threads.append(get_ident())
        sleep(0.001)
        return Update(frame)

    @contextmanager
    def source(_: AppConfig) -> Iterator[Iterator[Frame]]:
        def frames() -> Iterator[Frame]:
            while True:
                yield FRAME

        try:
            yield frames()
        finally:
            releases.append(True)

    for _ in range(2):
        controller.start(config, lambda: Mock(process=process), source)
        until(app, lambda: controller.state == "Running")
        with pytest.raises(ValueError):
            controller.start(config)
        controller.stop()
        until(app, lambda: not controller.busy)
        assert controller.state == "Stopped"
    assert len(releases) == 2
    assert worker_threads and all(t != get_ident() for t in worker_threads)


def test_worker_exception_and_cancel_release(app: QApplication, config: AppConfig) -> None:
    released = Event()
    errors: list[UIError] = []

    @contextmanager
    def source(_: AppConfig) -> Iterator[Iterator[Frame]]:
        try:
            yield iter((FRAME,))
        finally:
            released.set()

    worker = CameraWorker(
        config, lambda: Mock(process=Mock(side_effect=RuntimeError("camera failed"))), source
    )
    worker.failed.connect(errors.append)
    worker.start()
    until(app, lambda: not worker.isRunning())
    app.processEvents()
    assert released.is_set() and errors[0].title == "Camera unavailable"
    factory = Mock()
    worker = CameraWorker(config, factory, source)
    worker.cancel.set()
    worker.start()
    until(app, lambda: not worker.isRunning())
    factory.assert_not_called()


def test_close_defers_until_worker_releases(
    window: MainWindow, app: QApplication, config: AppConfig
) -> None:
    @contextmanager
    def source(_: AppConfig) -> Iterator[Iterator[Frame]]:
        yield iter((FRAME,))

    entered = Event()
    finish = Event()

    def process(frame: Frame) -> Update:
        entered.set()
        finish.wait(3)
        return Update(frame)

    window.controller.start(config, lambda: Mock(process=process), source)
    until(app, entered.is_set)
    window.show()
    assert not window.close()
    assert window.controller.busy
    finish.set()
    until(app, lambda: not window.controller.busy)
    assert not window.isVisible()


@pytest.mark.parametrize(
    ("error", "title"),
    [
        (IdentityError("legacy plaintext; migrate"), "Store migration required"),
        (
            IdentityError("authentication failure secret vector=[1,2]"),
            "Store cannot be authenticated",
        ),
        (IdentityError("key unavailable"), "Secure storage unavailable"),
        (ValueError("bad setting"), "Invalid configuration or input"),
        (ImportError("missing dependency"), "Local model or dependency unavailable"),
        (RuntimeError("private vector=[1,2]"), "Operation failed"),
    ],
)
def test_ui_error_categories_never_expose_payload(error: Exception, title: str) -> None:
    result = ui_error(error)
    assert result.title == title
    assert "[1,2]" not in repr(result)


def test_administration_metadata_and_mutations(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("jake.adapters.local_identity_store.private_permissions", Mock())
    admin = Administration(config)
    embedding = face_normalize("test", (1.0, 0.0))
    uuid = str(UUID(int=1))
    admin.residents.add(ResidentProfile(uuid, "Resident", (embedding,), 10, NOW))
    admin.visitors.add(
        VisitorProfile(uuid, embedding, NOW, NOW, NOW, last_visit_local_date=NOW.date())
    )
    assert admin.resident_rows() == (ResidentRow(uuid, "Resident", 10, NOW),)
    assert admin.visitor_rows()[0].sessions == 1
    admin.label_visitor(uuid, "Daniel")
    assert admin.visitor_rows()[0].state == "KNOWN_VISITOR"
    admin.label_visitor(uuid, None)
    assert admin.visitor_rows()[0].name == "Anonymous"
    assert "template" not in repr(admin.resident_rows()) + repr(admin.visitor_rows())
    assert all("template" not in f.name for f in fields(ResidentRow))
    admin.delete_resident(uuid)
    admin.delete_visitor(uuid)
    admin.delete_all_visitors()
    assert admin.resident_rows() == ()
    assert admin.visitor_rows() == ()


def test_no_delete_without_confirmation(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    uuid = str(UUID(int=3))
    window.show_residents((ResidentRow(uuid, "Resident", 10, NOW),))
    window.residents_table.selectRow(0)
    monkeypatch.setattr(QMessageBox, "question", Mock(return_value=QMessageBox.StandardButton.No))
    run = Mock()
    monkeypatch.setattr(window, "run_task", run)
    window.delete_resident()
    run.assert_not_called()


def test_timeline_and_dashboard_metadata(window: MainWindow) -> None:
    event = EventRow(NOW, "PERSON_ENTERED", "1", "—", "Metadata")
    update = Update(
        FRAME, fps=19, inference_ms=12, people=2, residents=1, visitors=1, visitor_ids=("abc",)
    )
    window.consume(update, (event,))
    assert window.events_table.rowCount() == 1
    assert window.cards["Active people"].value.text() == "2"
    assert window.cards["Detector latency"].value.text() == "12.0 ms"
    window.toggle_annotations(False)
    assert not window.camera.annotated
    window.clear_events()
    assert window.recent.rowCount() == 0


def test_task_worker_uses_metadata_result(window: MainWindow, app: QApplication) -> None:
    window.run_task(lambda: (ResidentRow("uuid", "Name", 10, NOW),), window.show_residents)
    until(app, lambda: window.task is None)
    assert window.residents_table.rowCount() == 1


def test_first_run_cancel_does_not_write(window: MainWindow, app: QApplication) -> None:
    from PySide6.QtCore import QTimer

    def cancel() -> None:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, QDialog)
        dialog.reject()

    QTimer.singleShot(0, cancel)
    window.first_run()
    assert not window.path.exists()


def test_live_service_uses_pipeline_metadata(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jake.application.composition import Components
    from jake.application.sessions import LiveSession
    from jake.domain import BoundingBox, PersonDetection, PersonTrack
    from jake.identity_domain import IdentityMatch, IdentityState
    from jake.visitor_domain import VisitorMatch, VisitorState

    box = BoundingBox(0, 0, 1, 1)
    detector = Mock(detect=Mock(return_value=(PersonDetection(box, 0.9),)))
    tracker = Mock(update=Mock(return_value=(PersonTrack("1", box, 0.9),)))
    identity = Mock(
        process=Mock(
            return_value=({"1": IdentityMatch(IdentityState.RESIDENT, "r", "Name", 0.9)}, ())
        )
    )
    components = Components(detector, tracker, None, identity, None, None)
    monkeypatch.setattr("jake.application.sessions.validate_models", Mock())
    monkeypatch.setattr("jake.application.sessions.compose", Mock(return_value=components))
    session = LiveSession(config)
    update = session.process(FRAME)
    assert update.people == 1 and update.residents == 1
    assert "Name | RESIDENT" in update.overlays[0].label
    assert update.events[0].kind == "person_entered"
    assert update.frame.pixels == FRAME.pixels
    assert "pixels" not in repr(update) and "embedding" not in repr(update)
    session.pipeline.visitor_matches = {"1": VisitorMatch(VisitorState.FREQUENT_VISITOR, "abcde")}
    identity.process.return_value = ({}, ())
    update = session.process(replace(FRAME, sequence=1))
    assert update.visitors == 1 and "VISITOR ABCD" in update.overlays[0].label


def test_enrollment_service_preserves_gates_and_commits_once(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jake.application.sessions import EnrollmentSession
    from jake.identity_domain import FaceDetection, FaceQuality

    monkeypatch.setattr("jake.application.sessions.validate_models", Mock())
    store = Mock(profiles=Mock(return_value=()))
    monkeypatch.setattr(
        "jake.adapters.local_identity_store.LocalIdentityStore", Mock(return_value=store)
    )
    detector = Mock(detect=Mock(return_value=()))
    encoder = Mock()
    monkeypatch.setattr("jake.adapters.opencv_faces.YuNetFaceDetector", Mock(return_value=detector))
    monkeypatch.setattr("jake.adapters.opencv_faces.SFaceEncoder", Mock(return_value=encoder))
    session = EnrollmentSession(config, "Resident", True)
    assert "Exactly one" in session.process(FRAME).progress
    with pytest.raises(IdentityError):
        session.commit()
    store.add.assert_not_called()
    detector.detect.return_value = (Mock(spec=FaceDetection),)
    monkeypatch.setattr(
        "jake.adapters.opencv_faces.face_quality",
        Mock(return_value=FaceQuality(False, "blurred face", "front")),
    )
    assert "blurred face" in session.process(FRAME).progress
    encoder.encode.assert_not_called()
    ready = Mock(ready=True, samples=[Mock()] * 10)
    monkeypatch.setattr(session, "enrollment", ready)
    session.commit()
    session.commit()
    store.add.assert_called_once()


@pytest.mark.parametrize(("name", "consent"), [("", True), ("Name", False), ("a\nb", True)])
def test_enrollment_rejects_missing_consent_or_bad_name(
    config: AppConfig, name: str, consent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jake.application.sessions import EnrollmentSession

    store = Mock()
    monkeypatch.setattr("jake.adapters.local_identity_store.LocalIdentityStore", store)
    with pytest.raises(IdentityError):
        EnrollmentSession(config, name, consent)
    store.assert_not_called()


def test_last_frame_events_are_delivered_before_worker_cleanup(
    app: QApplication, config: AppConfig
) -> None:
    controller = CameraController()
    drained: list[object] = []
    controller.drained.connect(lambda update, events: drained.extend(events))

    @contextmanager
    def source(_: AppConfig) -> Iterator[Iterator[Frame]]:
        yield iter((FRAME,))

    row = EventRow(NOW, "PERSON_LEFT", "1", "", "")
    controller.start(
        config, lambda: Mock(process=Mock(return_value=Update(FRAME, events=(row,)))), source
    )
    until(app, lambda: not controller.busy)
    assert drained == [row]


def test_visitor_ui_label_delete_and_explicit_migration(
    window: MainWindow, app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QInputDialog

    admin = Mock(visitor_rows=Mock(return_value=()))
    monkeypatch.setattr("jake.desktop.window.Administration", Mock(return_value=admin))
    monkeypatch.setattr(QMessageBox, "question", Mock(return_value=QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QInputDialog, "getText", Mock(return_value=("Daniel", True)))
    row = VisitorRow("uuid", "Anonymous", "FIRST_TIME_VISITOR", 1, 1, NOW, "—")
    window.show_visitors((row,))
    window.visitors_table.selectRow(0)
    window.label_visitor()
    until(app, lambda: window.task is None)
    admin.label_visitor.assert_called_once_with("uuid", "Daniel")
    monkeypatch.setattr(QInputDialog, "getText", Mock(return_value=("DELETE", True)))
    window.delete_all()
    admin.delete_all_visitors.assert_not_called()
    monkeypatch.setattr(QInputDialog, "getText", Mock(return_value=("DELETE ALL", True)))
    window.delete_all()
    until(app, lambda: window.task is None)
    admin.delete_all_visitors.assert_called_once()
    window.migrate("visitors")
    until(app, lambda: window.task is None)
    admin.migrate.assert_called_once_with("visitors")


def test_resident_migration_noop_is_clear(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    admin = Administration(config)
    monkeypatch.setattr(
        admin.residents,
        "migrate",
        Mock(
            side_effect=IdentityError(
                "Identity store is already encrypted; migration is not required."
            )
        ),
    )
    assert "already encrypted" in admin.migrate("residents")
    monkeypatch.setattr(admin.residents, "migrate", Mock(side_effect=IdentityError("corrupt")))
    with pytest.raises(IdentityError):
        admin.migrate("residents")


def test_first_run_saves_and_refreshes_editor(
    window: MainWindow, app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialogButtonBox

    monkeypatch.setattr("jake.desktop.window.validate_models", Mock())

    def accept() -> None:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, QDialog)
        panel = dialog.findChild(SettingsPanel)
        assert panel is not None
        panel.timezone.setText("America/Chicago")
        buttons = dialog.findChild(QDialogButtonBox)
        assert buttons is not None
        buttons.button(QDialogButtonBox.StandardButton.Save).click()

    QTimer.singleShot(0, accept)
    window.first_run()
    assert load_app_config(window.path).home.timezone == "America/Chicago"
    assert window.settings.timezone.text() == "America/Chicago"
