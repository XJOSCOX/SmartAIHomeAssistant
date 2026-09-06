"""Native operator interface. Controllers own services; widgets render metadata."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jake.application.administration import Administration, ResidentRow, VisitorRow
from jake.application.errors import UIError, ui_error
from jake.application.sessions import EnrollmentSession, EventRow, Update
from jake.application.settings import save_config, validate_models
from jake.config import AppConfig
from jake.desktop.controller import CameraController, TaskWorker
from jake.desktop.style import apply_theme
from jake.desktop.widgets import CameraPanel, Card, SettingsPanel, label

PAGES = (
    "Dashboard",
    "Live Camera",
    "Residents",
    "Visitors",
    "Events",
    "Settings",
    "Developer / Diagnostics",
)


def table(headers: tuple[str, ...]) -> QTableWidget:
    item = QTableWidget(0, len(headers))
    item.setHorizontalHeaderLabels(headers)
    item.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    item.verticalHeader().hide()
    item.setAlternatingRowColors(True)
    item.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    item.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    item.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    return item


def fill(item: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    item.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            item.setItem(row, column, QTableWidgetItem(value))


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, path: Path, *, first_run: bool = False) -> None:
        super().__init__()
        self.config, self.path = config, path
        self.configured = not first_run
        self.controller = CameraController()
        self.task: TaskWorker | None = None
        self.on_result: Callable[[object], None] = lambda _: None
        self.closing = False
        self.enrolling = False
        self.resident_rows: tuple[ResidentRow, ...] = ()
        self.visitor_rows: tuple[VisitorRow, ...] = ()
        self.today: date | None = None
        self.seen_today: set[str] = set()
        self.setWindowTitle("Jake · Desktop Control Center")
        self.resize(1280, 840)
        shell = QWidget()
        self.setCentralWidget(shell)
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(22)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(215)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(16, 24, 16, 20)
        nav.addWidget(label("jake", "brand"))
        nav.addWidget(label("HOME INTELLIGENCE\nLocal. Private. Yours.", "muted"))
        nav.addSpacing(28)
        self.stack = QStackedWidget()
        self.navigation: list[QPushButton] = []
        for index, name in enumerate(PAGES):
            button = QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, index=index: self.navigate(index))
            nav.addWidget(button)
            self.navigation.append(button)
        nav.addStretch()
        nav.addWidget(label("●  LOCAL PROCESSING", "badge"))
        self.theme = QComboBox()
        self.theme.addItems(["Dark", "Light", "System"])
        self.theme.currentTextChanged.connect(self.set_theme)
        nav.addWidget(self.theme)
        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)
        self.cards: dict[str, Card] = {}
        self.build_dashboard()
        self.build_camera()
        self.build_residents()
        self.build_visitors()
        self.build_events()
        self.build_settings()
        self.build_diagnostics()
        self.navigate(0)
        self.statusBar().showMessage(
            "First-run setup required" if first_run else "Ready · Camera is off"
        )
        self.controller.state_changed.connect(self.state_changed)
        self.controller.failed.connect(self.show_error)
        self.controller.drained.connect(self.consume)
        self.controller.enrolled.connect(self.enrollment_done)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.state_changed("Stopped")

    def set_theme(self, name: str) -> None:
        app = cast(QApplication, QApplication.instance())
        apply_theme(app, name)

    def page(self, title: str, subtitle: str) -> QVBoxLayout:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(18)
        layout.addWidget(label(title, "title"))
        layout.addWidget(label(subtitle, "muted"))
        self.stack.addWidget(page)
        return layout

    def navigate(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, button in enumerate(self.navigation):
            button.setChecked(i == index)

    def button(self, text: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def build_dashboard(self) -> None:
        layout = self.page(
            "Your home, at a glance", "Live metadata from this desktop session · No video recording"
        )
        grid = QGridLayout()
        for index, title in enumerate(
            (
                "System",
                "Camera",
                "Processing FPS",
                "Detector latency",
                "Active people",
                "Residents now",
                "Visitors now",
                "Visitors today · observed here",
            )
        ):
            self.cards[title] = Card(title)
            grid.addWidget(self.cards[title], index // 4, index % 4)
        layout.addLayout(grid)
        layout.addWidget(self.button("Open first-run setup", self.first_run))
        layout.addWidget(label("Recent semantic events", "muted"))
        self.recent = table(("Time", "Event", "Track", "Display"))
        layout.addWidget(self.recent, 1)
        layout.addWidget(
            label(
                "Visitor totals cover this app session only. Jake has no historical "
                "daily analytics or long-term event database.",
                "muted",
            )
        )

    def build_camera(self) -> None:
        layout = self.page("Live Camera", "Detection → Jake tracking → identity → semantic events")
        controls = QHBoxLayout()
        self.start_button = self.button("Start camera", self.start_camera)
        self.start_button.setObjectName("primary")
        self.stop_button = self.button("Stop camera", self.controller.stop)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        self.annotated = QCheckBox("Annotated preview")
        self.annotated.setChecked(True)
        self.annotated.toggled.connect(self.toggle_annotations)
        controls.addWidget(self.annotated)
        controls.addStretch()
        self.live_stats = label("Camera off", "badge")
        controls.addWidget(self.live_stats)
        layout.addLayout(controls)
        self.camera = CameraPanel()
        layout.addWidget(self.camera, 1)
        self.debug_toggle = QCheckBox("Show live tracker / visitor diagnostics")
        self.live_debug = label("", "muted")
        self.debug_toggle.toggled.connect(lambda enabled: self.live_debug.setVisible(enabled))
        layout.addWidget(self.debug_toggle)
        self.live_debug.hide()
        layout.addWidget(self.live_debug)

    def build_residents(self) -> None:
        layout = self.page(
            "Residents", "Deliberate enrollment · Encrypted templates · No automatic naming"
        )
        buttons = QHBoxLayout()
        buttons.addWidget(self.button("Refresh residents", self.refresh_residents))
        buttons.addWidget(self.button("Delete selected resident…", self.delete_resident))
        buttons.addWidget(self.button("Migrate resident store…", lambda: self.migrate("residents")))
        layout.addLayout(buttons)
        self.residents_table = table(("Name", "Resident UUID", "Samples", "Enrolled"))
        layout.addWidget(self.residents_table)
        form = QHBoxLayout()
        self.resident_name = QLineEdit()
        self.resident_name.setPlaceholderText("Resident display name")
        form.addWidget(self.resident_name)
        self.consent = QCheckBox("This resident knowingly consents to local biometric enrollment")
        form.addWidget(self.consent)
        layout.addLayout(form)
        actions = QHBoxLayout()
        actions.addWidget(self.button("Begin enrollment", self.enroll))
        actions.addWidget(self.button("Cancel enrollment", self.controller.stop))
        layout.addLayout(actions)
        self.enrollment_progress = label(
            "Enrollment uses one consenting face and varied poses. No images are saved.", "muted"
        )
        layout.addWidget(self.enrollment_progress)
        self.enroll_camera = CameraPanel()
        self.enroll_camera.setMaximumHeight(280)
        layout.addWidget(self.enroll_camera)

    def build_visitors(self) -> None:
        layout = self.page(
            "Visitors",
            "Anonymous frequency metadata · Frequency never implies trust or resident identity",
        )
        buttons = QHBoxLayout()
        for title, callback in (
            ("Refresh", self.refresh_visitors),
            ("Label / rename…", self.label_visitor),
            ("Remove label", self.remove_label),
            ("Delete…", self.delete_visitor),
            ("Delete all…", self.delete_all),
            ("Migrate store…", lambda: self.migrate("visitors")),
        ):
            buttons.addWidget(self.button(title, callback))
        layout.addLayout(buttons)
        self.visitors_table = table(
            ("Visitor", "Name", "State", "Sessions", "Visit days", "Last seen", "Mean session")
        )
        layout.addWidget(self.visitors_table, 1)
        layout.addWidget(
            label(
                "Profiles are loaded only on explicit Refresh. Templates and encryption keys "
                "never enter these tables. Stop the camera before management.",
                "muted",
            )
        )

    def build_events(self) -> None:
        layout = self.page(
            "Events", "Session-only timeline · Most recent 1,000 metadata events · No database"
        )
        layout.addWidget(self.button("Clear session timeline", self.clear_events))
        self.events_table = table(("Time", "Event", "Track", "Identity / visitor", "Metadata"))
        layout.addWidget(self.events_table, 1)

    def build_settings(self) -> None:
        layout = self.page(
            "Settings", "Validated configuration · Changes take effect on the next camera start"
        )
        self.settings = SettingsPanel(self.config)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.settings_scroll = scroll
        scroll.setWidget(self.settings)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.button("Validate and save settings", self.save_settings))
        layout.addWidget(label(f"Configuration: {self.path}", "muted"))

    def build_diagnostics(self) -> None:
        layout = self.page(
            "Developer / Diagnostics",
            "Safe error categories and live metadata · No vectors, keys or persisted logs",
        )
        self.security_status = label(
            "Secure storage: not checked · Access is checked on explicit profile refresh", "badge"
        )
        layout.addWidget(self.security_status)
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMaximumBlockCount(200)
        layout.addWidget(self.diagnostics, 1)
        layout.addWidget(
            label(
                "Stop is cooperative. If a native camera driver is blocked, Jake waits "
                "for that call to return rather than terminating a thread unsafely.",
                "muted",
            )
        )

    def idle(self) -> bool:
        if self.controller.busy or self.task:
            QMessageBox.information(
                self,
                "Operation in progress",
                "Stop the camera and wait for the current operation to finish.",
            )
            return False
        return True

    def start_camera(self) -> None:
        if not self.idle():
            return
        if not self.configured:
            self.first_run()
            return
        self.enrolling = False
        self.controller.start(self.config)

    def state_changed(self, state: str) -> None:
        self.cards["System"].value.setText(state)
        self.cards["Camera"].value.setText(state)
        self.start_button.setEnabled(state == "Stopped" and self.task is None)
        self.stop_button.setEnabled(state in ("Starting", "Running"))
        self.statusBar().showMessage(state + " · Local processing")
        if state == "Stopped":
            self.camera.clear()
            self.enroll_camera.clear()
            self.live_stats.setText("Camera off")
            for key in (
                "Processing FPS",
                "Detector latency",
                "Active people",
                "Residents now",
                "Visitors now",
            ):
                self.cards[key].value.setText("—")
            if self.closing:
                self.close()

    def toggle_annotations(self, enabled: bool) -> None:
        self.camera.annotated = enabled
        self.camera.update()

    def poll(self) -> None:
        update, events = self.controller.take()
        self.consume(update, events)

    def consume(self, update: Update | None, events: tuple[EventRow, ...]) -> None:
        for event in events:
            values = (
                event.at.astimezone(ZoneInfo(self.config.home.timezone)).strftime("%H:%M:%S"),
                event.kind,
                event.track,
                event.display,
                event.summary,
            )
            for widget, row_values, limit in (
                (self.events_table, values, 1000),
                (self.recent, values[:4], 12),
            ):
                widget.insertRow(0)
                for col, value in enumerate(row_values):
                    widget.setItem(0, col, QTableWidgetItem(value))
                if widget.rowCount() > limit:
                    widget.removeRow(limit)
        if update is None:
            return
        if self.enrolling:
            self.enroll_camera.present(update)
            self.enrollment_progress.setText(update.progress)
            return
        self.camera.present(update)
        self.live_stats.setText(
            f"{update.fps:.1f} processing FPS · {update.frame.width} × {update.frame.height} "
            f"· #{update.frame.sequence}"
        )
        metrics = {
            "Processing FPS": f"{update.fps:.1f}",
            "Detector latency": f"{update.inference_ms:.1f} ms",
            "Active people": str(update.people),
            "Residents now": str(update.residents),
            "Visitors now": str(update.visitors),
        }
        day = update.frame.captured_at.astimezone(ZoneInfo(self.config.home.timezone)).date()
        if self.today != day:
            self.today, self.seen_today = day, set()
        self.seen_today.update(update.visitor_ids)
        metrics["Visitors today · observed here"] = (
            str(len(self.seen_today)) if self.config.visitors.enabled else "Disabled"
        )
        for key, value in metrics.items():
            self.cards[key].value.setText(value)
        self.live_debug.setText("\n".join(update.diagnostics[:20]))

    def clear_events(self) -> None:
        self.events_table.setRowCount(0)
        self.recent.setRowCount(0)

    def show_error(self, error: UIError) -> None:
        self.diagnostics.appendPlainText(error.diagnostic)
        self.security_status.setText(error.title)
        self.statusBar().showMessage(error.title)
        if not self.closing:
            QMessageBox.warning(self, error.title, error.message)

    def run_task(self, operation: Callable[[], object], result: Callable[[object], None]) -> None:
        if not self.idle():
            return
        self.on_result = result
        self.task = TaskWorker(operation)
        self.task.result.connect(self.task_result)
        self.task.failed.connect(self.show_error)
        self.task.finished.connect(self.task_finished)
        self.start_button.setEnabled(False)
        self.task.start()

    def task_result(self, value: object) -> None:
        self.on_result(value)
        self.security_status.setText(
            "Operation completed · Existing encrypted stores authenticated when present"
        )

    def task_finished(self) -> None:
        if self.task:
            self.task.deleteLater()
            self.task = None
        self.start_button.setEnabled(True)
        if self.closing:
            self.close()

    def refresh_residents(self) -> None:
        self.run_task(lambda: Administration(self.config).resident_rows(), self.show_residents)

    def show_residents(self, result: object) -> None:
        self.resident_rows = cast(tuple[ResidentRow, ...], result)
        fill(
            self.residents_table,
            [(r.name, r.uuid, str(r.samples), r.enrolled.isoformat()) for r in self.resident_rows],
        )

    def refresh_visitors(self) -> None:
        self.run_task(lambda: Administration(self.config).visitor_rows(), self.show_visitors)

    def show_visitors(self, result: object) -> None:
        self.visitor_rows = cast(tuple[VisitorRow, ...], result)
        fill(
            self.visitors_table,
            [
                (
                    r.uuid[:8].upper(),
                    r.name,
                    r.state,
                    str(r.sessions),
                    str(r.days),
                    r.last_seen.isoformat(),
                    r.duration,
                )
                for r in self.visitor_rows
            ],
        )

    def selected(self, kind: str) -> str | None:
        widget = self.residents_table if kind == "residents" else self.visitors_table
        rows = self.resident_rows if kind == "residents" else self.visitor_rows
        index = widget.currentRow()
        return rows[index].uuid if 0 <= index < len(rows) else None

    def confirmed(self, title: str, message: str) -> bool:
        return (
            QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def delete_resident(self) -> None:
        uuid = self.selected("residents")
        if (
            uuid
            and self.idle()
            and self.confirmed(
                "Delete resident?",
                "Delete this resident's encrypted templates? This cannot be undone.",
            )
        ):

            def operation() -> object:
                admin = Administration(self.config)
                admin.delete_resident(uuid)
                return admin.resident_rows()

            self.run_task(operation, self.show_residents)

    def visitor_action(self, action: Callable[[Administration, str], None]) -> None:
        uuid = self.selected("visitors")
        if uuid:

            def operation() -> object:
                admin = Administration(self.config)
                action(admin, uuid)
                return admin.visitor_rows()

            self.run_task(operation, self.show_visitors)

    def label_visitor(self) -> None:
        if self.idle() and self.selected("visitors"):
            name, ok = QInputDialog.getText(
                self, "Explicit visitor label", "Name assigned by the operator:"
            )
            if ok:
                self.visitor_action(lambda admin, uuid: admin.label_visitor(uuid, name.strip()))

    def remove_label(self) -> None:
        self.visitor_action(lambda admin, uuid: admin.label_visitor(uuid, None))

    def delete_visitor(self) -> None:
        if (
            self.idle()
            and self.selected("visitors")
            and self.confirmed(
                "Delete visitor?", "Delete this encrypted visitor profile? This cannot be undone."
            )
        ):
            self.visitor_action(lambda admin, uuid: admin.delete_visitor(uuid))

    def delete_all(self) -> None:
        if not self.idle():
            return
        text, ok = QInputDialog.getText(
            self,
            "Delete ALL visitor profiles",
            "Type DELETE ALL to permanently remove every visitor profile:",
        )
        if ok and text == "DELETE ALL":

            def operation() -> object:
                admin = Administration(self.config)
                admin.delete_all_visitors()
                return admin.visitor_rows()

            self.run_task(operation, self.show_visitors)

    def migrate(self, kind: str) -> None:
        if self.idle() and self.confirmed(
            "Explicit encrypted-store migration",
            (
                "Authenticate and upgrade visitor metadata using its existing key. "
                "Preserve UUIDs, templates and sessions; "
                "initialize distinct days to one. No retention cleanup runs."
                if kind == "visitors"
                else "Encrypt the existing plaintext resident store using a new OS-vault key. "
                "Preserve resident profiles. An already encrypted store will not be rewritten."
            )
            + " The original survives a failed migration. Continue?",
        ):
            self.run_task(
                lambda: Administration(self.config).migrate(kind),
                self.migration_done,
            )

    def migration_done(self, value: object) -> None:
        QMessageBox.information(self, "Migration", str(value))

    def enroll(self) -> None:
        if not self.idle():
            return
        if not self.consent.isChecked() or not self.resident_name.text().strip():
            QMessageBox.information(
                self,
                "Enrollment needs consent",
                "Enter a name and confirm the resident's explicit consent.",
            )
            return
        name = self.resident_name.text()
        self.enrolling = True
        self.controller.start(self.config, lambda: EnrollmentSession(self.config, name, True))

    def enrollment_done(self) -> None:
        self.enrollment_progress.setText(
            "Enrollment complete · Encrypted profile saved. "
            "Refresh residents after the camera stops."
        )
        self.consent.setChecked(False)

    def save_settings(self) -> None:
        if not self.idle():
            return
        try:
            config = self.settings.value()
            save_config(self.path, config)
            self.config, self.configured = config, True
            self.resident_rows, self.visitor_rows = (), ()
            self.residents_table.setRowCount(0)
            self.visitors_table.setRowCount(0)
            self.statusBar().showMessage(
                "Settings saved atomically · Models are checked at camera start"
            )
        except Exception as exc:
            self.show_error(ui_error(exc))

    def first_run(self) -> None:
        if not self.idle():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Welcome to Jake · Local setup")
        dialog.resize(820, 740)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Make Jake at home", "title"))
        layout.addWidget(
            label(
                "Choose your timezone, camera and existing model files. "
                "Visitor persistence is optional. No downloads or store migrations run here.",
                "muted",
            )
        )
        panel = SettingsPanel(self.config)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.rejected.connect(dialog.reject)

        def accept() -> None:
            try:
                config = panel.value()
                validate_models(config)
                save_config(self.path, config)
                self.config, self.configured = config, True
                # Rebuild only the settings editor, preserving other page state.
                self.settings = SettingsPanel(config)
                self.settings_scroll.setWidget(self.settings)
                dialog.accept()
                self.statusBar().showMessage("Setup saved · Camera remains off")
            except Exception as exc:
                self.show_error(ui_error(exc))

        buttons.accepted.connect(accept)
        layout.addWidget(buttons)
        dialog.exec()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.controller.busy or self.task:
            self.closing = True
            self.controller.stop()
            self.statusBar().showMessage(
                "Closing safely · Waiting for the worker to release resources…"
            )
            event.ignore()
            return
        self.timer.stop()
        self.camera.clear()
        self.enroll_camera.clear()
        event.accept()
