"""Opt-in development formatting, separate from event generation."""

from jake.domain import EventKind, PersonEvent


def log_event(event: PersonEvent) -> None:
    timestamp = event.context.captured_at.strftime("%H:%M:%S.%f")[:-3]
    label = event.kind.name.removeprefix("PERSON_")
    duration = event.duration_seconds
    suffix = (
        f" duration={duration:.1f}s"
        if event.kind != EventKind.PERSON_ENTERED and duration is not None
        else ""
    )
    print(f"[{timestamp}] {label} track={event.track_id}{suffix}")
