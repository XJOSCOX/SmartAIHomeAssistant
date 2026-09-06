from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from jake.visitor_domain import VisitorProfile


class VisitorStore(Protocol):
    def profiles(self) -> tuple[VisitorProfile, ...]: ...
    def add(self, profile: VisitorProfile) -> None: ...
    def update(
        self, visitor_id: str, transform: Callable[[VisitorProfile], VisitorProfile]
    ) -> None: ...
    def expire(self, now: datetime, retention_days: int) -> None: ...
