"""Metadata-only administration boundary. Templates never enter widget state."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jake.config import AppConfig
from jake.identity import IdentityError
from jake.visitors import visitor_match


@dataclass(frozen=True)
class ResidentRow:
    uuid: str
    name: str
    samples: int
    enrolled: datetime


@dataclass(frozen=True)
class VisitorRow:
    uuid: str
    name: str
    state: str
    sessions: int
    days: int
    last_seen: datetime
    duration: str


class Administration:
    def __init__(self, config: AppConfig) -> None:
        from jake.adapters.local_identity_store import LocalIdentityStore
        from jake.adapters.visitor_store import EncryptedVisitorStore

        self.config = config
        self.residents = LocalIdentityStore(Path(config.identity.store_path))
        self.visitors = EncryptedVisitorStore(
            Path(config.identity.store_path), timezone=config.home.timezone
        )

    def resident_rows(self) -> tuple[ResidentRow, ...]:
        return tuple(
            ResidentRow(p.resident_id, p.display_name, p.sample_count, p.enrolled_at)
            for p in self.residents.profiles()
        )

    def visitor_rows(self) -> tuple[VisitorRow, ...]:
        # Listing is read-only; existing retention remains in live visitor processing.
        return tuple(
            VisitorRow(
                p.visitor_id,
                p.display_name or "Anonymous",
                visitor_match(p, self.config.visitors).state,
                p.session_count,
                p.distinct_visit_days,
                p.last_seen_at,
                f"{p.statistics.mean_seconds:.1f}s" if p.statistics.completed else "—",
            )
            for p in self.visitors.profiles()
        )

    def delete_resident(self, uuid: str) -> None:
        self.residents.delete(uuid)

    def label_visitor(self, uuid: str, name: str | None) -> None:
        self.visitors.label(uuid, name)

    def delete_visitor(self, uuid: str) -> None:
        self.visitors.delete(uuid)

    def delete_all_visitors(self) -> None:
        self.visitors.delete_all()

    def migrate(self, kind: str) -> str:
        if kind == "residents":
            try:
                self.residents.migrate()
            except IdentityError as exc:
                if str(exc) == "Identity store is already encrypted; migration is not required.":
                    return "Identity store is already encrypted; migration is not required."
                raise
            return "Resident store encrypted successfully."
        if kind == "visitors":
            return self.visitors.migrate()
        raise ValueError("unknown store")
