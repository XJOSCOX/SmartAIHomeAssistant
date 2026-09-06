"""PyInstaller entry: imports do not load weights or open stores."""

from jake.desktop.main import main

if __name__ == "__main__":
    raise SystemExit(main())
