"""Small, centralized palette for native Qt widgets."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication, mode: str) -> None:
    dark = mode == "Dark" or (
        mode == "System" and app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    )
    background, panel, text, muted, edge = (
        ("#101722", "#182231", "#edf3fa", "#96a9c0", "#29394e")
        if dark
        else ("#f1f5fa", "#ffffff", "#16263b", "#536881", "#d7e1ed")
    )
    app.setStyleSheet(f"""
        QWidget {{ color: {text}; font-family: 'Segoe UI'; font-size: 10pt; }}
        QMainWindow, QDialog, QStackedWidget, QWidget#settingsPanel {{ background: {background}; }}
        QFrame#sidebar, QFrame#card {{ background: {panel}; border: 1px solid {edge};
                                     border-radius: 12px; }}
        QLabel#title {{ font-size: 23pt; font-weight: 600; }}
        QLabel#brand {{ font-size: 28pt; font-weight: 700; color: #35bda0; }}
        QLabel#metric {{ font-size: 22pt; font-weight: 600; }}
        QLabel#muted {{ color: {muted}; }}
        QLabel#badge {{ background: {panel}; border: 1px solid {edge}; padding: 8px;
                       border-radius: 8px; color: #35bda0; }}
        QPushButton {{ background: {panel}; border: 1px solid {edge}; border-radius: 7px;
                       padding: 9px 14px; }}
        QPushButton:hover {{ border-color: #35bda0; }}
        QPushButton:checked, QPushButton#primary {{ background: #147e6c; color: white; }}
        QPushButton:disabled {{ color: {muted}; }}
        QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {{ background: {panel};
            border: 1px solid {edge}; padding: 7px; border-radius: 5px; }}
        QTableWidget {{ background: {panel}; alternate-background-color: {background};
            border: 1px solid {edge}; gridline-color: {edge}; border-radius: 8px; }}
        QHeaderView::section {{ background: {background}; padding: 10px; border: none; }}
        QTableWidget::item {{ padding: 8px; }}
        QTableWidget::item:selected {{ background: #147e6c; color: white; }}
        QScrollArea {{ border: none; background: {background}; }}
        QToolTip {{ background: {panel}; color: {text}; border: 1px solid {edge}; }}
    """)
