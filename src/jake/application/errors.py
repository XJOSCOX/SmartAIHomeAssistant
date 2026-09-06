"""Safe error categories: no exception payloads, keys, vectors or tracebacks in UI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UIError:
    title: str
    message: str
    diagnostic: str
    migration: bool = False


def ui_error(error: Exception) -> UIError:
    text = str(error).lower()  # Match known categories only; never display raw third-party text.
    diagnostic = type(error).__name__
    if "migrat" in text or "plaintext" in text:
        return UIError(
            "Store migration required",
            "Use the explicit migration action on the "
            "Residents or Visitors page. Existing data has not been changed.",
            diagnostic + ": store schema requires migration",
            True,
        )
    if "key" in text or "credential" in text or "secret service" in text:
        return UIError(
            "Secure storage unavailable",
            "Unlock the Windows credential vault and "
            "use the same OS account that created this store. No key was displayed.",
            diagnostic + ": key-provider access failed",
        )
    if "corrupt" in text or "authentication" in text or "unreadable" in text:
        return UIError(
            "Store cannot be authenticated",
            "Check the store and its original key. "
            "Jake will not overwrite unreadable biometric data.",
            diagnostic + ": encrypted-store validation failed",
        )
    if "camera" in text:
        return UIError(
            "Camera unavailable",
            "Check the selected index, Windows camera permission "
            "and whether another application is using the camera.",
            diagnostic + ": camera acquisition failed",
        )
    if "model" in text or "weight" in text or isinstance(error, (ImportError, FileNotFoundError)):
        return UIError(
            "Local model or dependency unavailable",
            "Check model paths in Settings "
            "and install the required runtime extras. No model is downloaded automatically.",
            diagnostic + ": model/dependency setup failed",
        )
    if isinstance(error, ValueError):
        return UIError(
            "Invalid configuration or input",
            "Check timezone, thresholds and input values. No settings were saved.",
            diagnostic + ": validation failed",
        )
    return UIError(
        "Operation failed",
        "Jake stopped this operation safely. Check local file "
        "permissions and the selected configuration.",
        diagnostic + ": operation failed",
    )
