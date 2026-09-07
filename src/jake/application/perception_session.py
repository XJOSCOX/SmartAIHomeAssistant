"""Shared perception runtime settings, independent of CLI parsing and hardware."""

from dataclasses import dataclass, replace

from jake.adapters.person_events import PersonEventGenerator
from jake.application.composition import Components
from jake.config import AppConfig


@dataclass(frozen=True, slots=True)
class PerceptionOverrides:
    tracker: str = "kalman"
    assignment: str | None = None
    appearance: bool | None = None
    reid: bool | None = None
    identity: bool | None = None
    visitors: bool | None = None


@dataclass(frozen=True, slots=True)
class PerceptionSession:
    config: AppConfig
    tracker: str
    detect: bool
    track: bool
    events: bool
    debug_tracks: bool

    def compose(self) -> Components:
        from jake.application.composition import compose

        return compose(
            self.config,
            detect=self.detect,
            track=self.track,
            tracker_mode=self.tracker,
            identify=self.config.identity.enabled or self.config.visitors.enabled,
            debug_tracks=self.debug_tracks,
        )

    def event_generator(self) -> PersonEventGenerator | None:
        return PersonEventGenerator(self.config.events) if self.events else None


def configure_perception(
    config: AppConfig,
    overrides: PerceptionOverrides | None = None,
    *,
    detect: bool = True,
    track: bool = True,
    events: bool = True,
    debug_tracks: bool = False,
) -> PerceptionSession:
    """Resolve explicit overrides over TOML; never edit files or initialize models.

    Camera-only/detection-only and geometry baselines remain selectable. Appearance
    and ReID config are dormant for geometry baselines, as in the original CLI.
    Explicit incompatible feature requests fail instead of silently being ignored.
    """
    overrides = overrides or PerceptionOverrides()
    if overrides.tracker not in {"kalman", "kalman-baseline", "iou"}:
        raise ValueError("unsupported tracker mode")
    for name in ("appearance", "reid", "identity", "visitors"):
        value = getattr(overrides, name)
        if value is not None and type(value) is not bool:
            raise ValueError(f"{name} override must be boolean")
    appearance = overrides.appearance
    if overrides.reid is True:
        if not track or overrides.tracker != "kalman" or appearance is False:
            raise ValueError("--reid requires --track --tracker kalman and appearance")
        if appearance is None:
            appearance = True
    if appearance is True and not (track and overrides.tracker == "kalman"):
        raise ValueError("--appearance requires --track --tracker kalman")
    tracking = config.tracking
    if overrides.reid is not None:
        tracking = replace(tracking, reid=replace(tracking.reid, enabled=overrides.reid))
    if appearance is not None:
        tracking = replace(tracking, appearance=replace(tracking.appearance, enabled=appearance))
    if overrides.assignment is not None:
        if not track:
            raise ValueError("--assignment requires --track")
        tracking = replace(tracking, assignment=overrides.assignment)
    identity = config.identity
    visitors = config.visitors
    if overrides.identity is not None:
        identity = replace(identity, enabled=overrides.identity)
    if overrides.visitors is not None:
        visitors = replace(visitors, enabled=overrides.visitors)
    # Visitor learning always needs resident-first screening, even --no-identity.
    if visitors.enabled:
        identity = replace(identity, enabled=True)
    if identity.enabled and not track:
        raise ValueError("identity/visitor memory requires --track")
    if events and not track:
        raise ValueError("--events requires --track")
    if overrides.tracker != "iou" and not track:
        raise ValueError("--tracker kalman requires --track")
    return PerceptionSession(
        replace(config, tracking=tracking, identity=identity, visitors=visitors),
        overrides.tracker,
        detect,
        track,
        events or visitors.enabled,
        debug_tracks,
    )
