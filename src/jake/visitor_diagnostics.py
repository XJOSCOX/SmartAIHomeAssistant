"""Metadata-only preview diagnostics with encoder-idle gap suppression."""


class VisitorDiagnostics:
    def __init__(self) -> None:
        self._previous: dict[str, str] = {}

    def changes(self, reasons: dict[str, str]) -> tuple[str, ...]:
        lines = []
        for track_id, reason in reasons.items():
            if reason == "waiting for face observation" and track_id in self._previous:
                continue
            if self._previous.get(track_id) != reason:
                lines.append(f"VISITOR track={track_id} {reason}")
            self._previous[track_id] = reason
        self._previous = {k: v for k, v in self._previous.items() if k in reasons}
        return tuple(lines)
