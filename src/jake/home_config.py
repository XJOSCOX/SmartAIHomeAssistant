"""Household calendar configuration, independent of machine local time."""

from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True, slots=True)
class HomeConfig:
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.timezone, str):
                raise ValueError("invalid timezone")
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("home.timezone must be an available IANA timezone") from None
