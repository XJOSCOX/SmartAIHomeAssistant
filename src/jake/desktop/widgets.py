"""Reusable Qt presentation components; no perception algorithms."""

import os
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from jake.application.errors import ui_error
from jake.application.sessions import Update
from jake.application.settings import validate_models
from jake.config import AppConfig
from jake.home_config import HomeConfig


def label(text: str, name: str = "") -> QLabel:
    item = QLabel(text)
    item.setTextFormat(Qt.TextFormat.PlainText)
    item.setObjectName(name)
    item.setWordWrap(True)
    return item


class Card(QFrame):
    def __init__(self, title: str, value: str = "—") -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.addWidget(label(title, "muted"))
        self.value = label(value, "metric")
        layout.addWidget(self.value)


class CameraPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(420, 280)
        self.image = QImage()
        self.update_data: Update | None = None
        self.annotated = True

    def present(self, update: Update) -> None:
        frame = update.frame
        self.image = QImage(
            frame.pixels, frame.width, frame.height, frame.width * 3, QImage.Format.Format_RGB888
        ).copy()
        self.update_data = update
        self.update()

    def clear(self) -> None:
        self.image = QImage()
        self.update_data = None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#090f19"))
        if self.image.isNull():
            painter.setPen(QColor("#8797ac"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Camera is off\nFrames stay on this device",
            )
            return
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
        from PySide6.QtCore import QRect

        target = QRect(x, y, size.width(), size.height())
        painter.drawImage(target, self.image)
        if self.annotated and self.update_data:
            painter.setPen(QPen(QColor("#5ee0bc"), 2))
            for overlay in self.update_data.overlays:
                box = overlay.box
                left, top = x + int(box.left * size.width()), y + int(box.top * size.height())
                painter.drawRect(
                    left,
                    top,
                    int((box.right - box.left) * size.width()),
                    int((box.bottom - box.top) * size.height()),
                )
                painter.drawText(left + 4, max(y + 18, top - 6), overlay.label)


class PathInput(QWidget):
    def __init__(self, value: str, *, directory: bool = False) -> None:
        super().__init__()
        self.directory = directory
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(value)
        layout.addWidget(self.edit)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        layout.addWidget(browse)

    def browse(self) -> None:
        value = (
            QFileDialog.getExistingDirectory(self, "Choose local directory")
            if self.directory
            else QFileDialog.getOpenFileName(self, "Choose local model")[0]
        )
        if value:
            self.edit.setText(value)

    def value(self) -> str:
        text = self.edit.text().strip()
        if not text:
            raise ValueError("local path must not be blank")
        return str(Path(text).absolute())


class SettingsPanel(QWidget):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setObjectName("settingsPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.config = config
        form = QFormLayout(self)
        form.setVerticalSpacing(14)
        self.timezone = QLineEdit(config.home.timezone)
        form.addRow("Household timezone (IANA)", self.timezone)
        self.device = QSpinBox()
        self.device.setRange(0, 99)
        self.device.setValue(config.camera.device)
        form.addRow("OpenCV camera index", self.device)
        self.cameras = QComboBox()
        self.cameras.addItem("Select a discovered camera…", -1)
        for index, camera in enumerate(
            [] if os.environ.get("QT_QPA_PLATFORM") == "offscreen" else QMediaDevices.videoInputs()
        ):
            self.cameras.addItem(f"{index} · {camera.description()}", index)
        self.cameras.currentIndexChanged.connect(self.select_camera)
        form.addRow("Windows / Qt camera discovery", self.cameras)
        form.addRow(
            label(
                "Camera ordering can differ between Qt and OpenCV. The explicit index "
                "remains editable. Resolution and FPS are negotiated by the existing "
                "camera adapter; this configuration has no requested-size/FPS fields.",
                "muted",
            )
        )
        self.identity = QCheckBox("Enable deliberately enrolled resident recognition")
        self.identity.setChecked(config.identity.enabled)
        self.visitors = QCheckBox("Allow encrypted anonymous visitor memory on this device")
        self.visitors.setChecked(config.visitors.enabled)
        self.appearance = QCheckBox("Enable existing appearance encoder / configured body ReID")
        self.appearance.setChecked(config.tracking.appearance.enabled)
        form.addRow("Recognition", self.identity)
        form.addRow("Visitor persistence (opt-in)", self.visitors)
        form.addRow("Tracking", self.appearance)
        self.recurring = QSpinBox()
        self.recurring.setRange(2, 100)
        self.recurring.setValue(config.visitors.recurring_distinct_days)
        self.frequent = QSpinBox()
        self.frequent.setRange(3, 365)
        self.frequent.setValue(config.visitors.frequent_distinct_days)
        self.retention = QSpinBox()
        self.retention.setRange(1, 365)
        self.retention.setValue(config.visitors.retention_days)
        form.addRow("Recurring · distinct local days", self.recurring)
        form.addRow("Frequent · distinct local days", self.frequent)
        form.addRow("Visitor retention · days", self.retention)
        model_folder = QPushButton("Use existing model folder…")
        model_folder.clicked.connect(self.choose_model_folder)
        form.addRow("Already have Jake models?", model_folder)
        form.addRow(
            label(
                "Select your existing repository models folder to update all four "
                "model paths. Files and biometric stores stay where they are.",
                "muted",
            )
        )
        self.yolo = PathInput(config.detector.model)
        self.body = PathInput(config.tracking.appearance.model)
        self.yunet = PathInput(config.identity.detector_model)
        self.sface = PathInput(config.identity.encoder_model)
        self.store = PathInput(config.identity.store_path, directory=True)
        for title, widget in (
            ("YOLO weights (.pt)", self.yolo),
            ("Appearance model (.xml)", self.body),
            ("YuNet (.onnx)", self.yunet),
            ("SFace (.onnx)", self.sface),
            ("Encrypted identity directory", self.store),
        ):
            form.addRow(title, widget)
        form.addRow(
            label(
                "Residents: residents.json · Visitors: visitors.json\n"
                "AES-256-GCM, separate keys in the OS credential vault. "
                "Changing the directory does not copy or migrate profiles. "
                "Keys are checked only when a store is accessed. Models are never "
                "downloaded automatically.",
                "muted",
            )
        )

    def choose_model_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose existing Jake models folder")
        if not directory:
            return
        try:
            self.use_model_folder(Path(directory))
        except Exception as exc:
            error = ui_error(exc)
            QMessageBox.warning(self, error.title, error.message)

    def use_model_folder(self, directory: Path) -> None:
        """Explicit selection only; validate required files before changing any fields."""
        inputs = (self.yolo, self.body, self.yunet, self.sface)
        previous = tuple(item.edit.text() for item in inputs)
        try:
            for item in inputs:
                item.edit.setText(str(directory.absolute() / Path(item.edit.text()).name))
            validate_models(self.value())
        except Exception:
            for item, value in zip(inputs, previous, strict=True):
                item.edit.setText(value)
            raise

    def select_camera(self, index: int) -> None:
        value = self.cameras.itemData(index)
        if isinstance(value, int) and value >= 0:
            self.device.setValue(value)

    def value(self) -> AppConfig:
        c = self.config
        return replace(
            c,
            home=HomeConfig(self.timezone.text().strip()),
            camera=replace(c.camera, device=self.device.value()),
            detector=replace(c.detector, model=self.yolo.value()),
            tracking=replace(
                c.tracking,
                appearance=replace(
                    c.tracking.appearance,
                    model=self.body.value(),
                    enabled=self.appearance.isChecked(),
                ),
            ),
            identity=replace(
                c.identity,
                enabled=self.identity.isChecked(),
                detector_model=self.yunet.value(),
                encoder_model=self.sface.value(),
                store_path=self.store.value(),
            ),
            visitors=replace(
                c.visitors,
                enabled=self.visitors.isChecked(),
                recurring_distinct_days=self.recurring.value(),
                frequent_distinct_days=self.frequent.value(),
                retention_days=self.retention.value(),
            ),
        )
