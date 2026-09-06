"""Windowed desktop entry point. No camera/store opens at launch."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jake Desktop Control Center")
    parser.add_argument(
        "--config",
        type=Path,
        help="Existing development config; relative paths use the working directory",
    )
    parser.add_argument(
        "--model-smoke-test",
        type=Path,
        help="Check local models in this folder using synthetic pixels; no camera or stores",
    )
    parser.add_argument("--model-smoke-report", type=Path, help="Write model check status JSON")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Offscreen startup/exit; no camera or biometric data access",
    )
    args = parser.parse_args(argv)
    if args.model_smoke_test is not None or args.model_smoke_report is not None:
        if args.model_smoke_test is None or args.model_smoke_report is None:
            parser.error("--model-smoke-test and --model-smoke-report must be used together")
        from jake.application.model_smoke import check_models

        # Exclusive creation prevents overwriting any existing file.
        with args.model_smoke_report.open("x", encoding="utf-8") as report:
            results = check_models(args.model_smoke_test.absolute())
            json.dump(results, report, indent=2)
        return 0 if all(value == "OK" for value in results.values()) else 1
    if args.smoke_test:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    from jake.application.errors import ui_error
    from jake.application.settings import AppPaths
    from jake.config import load_app_config
    from jake.desktop.style import apply_theme
    from jake.desktop.window import MainWindow

    app = QApplication(["Jake"])
    app.setApplicationName("Jake")
    app.setOrganizationName("Jake")
    apply_theme(app, "Dark")
    # Smoke mode cannot open a caller's config or discover real camera devices.
    with TemporaryDirectory(prefix="jake-smoke-") as temporary:
        paths = AppPaths(Path(temporary)) if args.smoke_test else AppPaths.default()
        path = paths.config if args.smoke_test else (args.config or paths.config).absolute()
        config = paths.defaults()
        first_run = not path.exists()
        if path.exists():
            try:
                config = load_app_config(path)
            except Exception as exc:
                error = ui_error(exc)
                QMessageBox.critical(
                    None,
                    error.title,
                    error.message
                    + "\nChoose a valid --config file; existing file was not changed.",
                )
                return 2
        if args.config is None and not args.smoke_test:
            paths.initialize()
        window = MainWindow(config, path, first_run=first_run)
        window.show()
        if args.smoke_test:
            QTimer.singleShot(150, window.close)
        elif first_run:
            QTimer.singleShot(0, window.first_run)
        return app.exec()


if __name__ == "__main__":
    sys.exit(main())
