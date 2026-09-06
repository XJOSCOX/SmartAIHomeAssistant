"""Separate encrypted visitor payload using Phase 2C file/key primitives."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from jake.adapters.local_identity_store import LocalIdentityStore
from jake.identity import IdentityError
from jake.identity_domain import FaceEmbedding
from jake.identity_encryption import decrypt
from jake.identity_ports import KeyProvider
from jake.visitor_domain import VisitorProfile, VisitStatistics

VISITOR_DOMAIN = "Jake/visitor-store"


class EncryptedVisitorStore:
    def __init__(
        self, root: Path, key_provider: KeyProvider | None = None, *, timezone: str = "UTC"
    ) -> None:
        self.timezone = ZoneInfo(timezone)
        # Reuse locking, permissions, encryption verification and atomic replacement,
        # not the resident schema or resident CRUD/migration operations.
        self._file = LocalIdentityStore(root, key_provider)
        self._file.path = self._file.root / "visitors.json"

    def parse(self, payload: object) -> tuple[VisitorProfile, ...]:
        try:
            if (
                not isinstance(payload, dict)
                or set(payload) != {"version", "visitors", "timezone"}
                or type(payload["version"]) is not int
                or payload["version"] != 2
            ):
                raise ValueError("invalid visitor schema")
            if payload["timezone"] != self.timezone.key:
                raise IdentityError("visitor store household timezone differs from configuration")
            rows = payload["visitors"]
            if not isinstance(rows, list) or len(rows) > 100:
                raise ValueError("invalid visitor count")
            profiles = []
            for row in rows:
                if set(row) != {
                    "visitor_id",
                    "template",
                    "created_at",
                    "last_seen_at",
                    "last_visit_at",
                    "session_count",
                    "distinct_visit_days",
                    "last_visit_local_date",
                    "statistics",
                    "display_name",
                    "explicitly_labeled",
                }:
                    raise ValueError("invalid visitor record")
                template = row["template"]
                if not isinstance(row["statistics"], dict) or set(row["statistics"]) != {
                    "completed",
                    "mean_seconds",
                    "m2_seconds",
                }:
                    raise ValueError("invalid visitor statistics schema")
                if (
                    set(template) != {"model_id", "values"}
                    or not isinstance(template["values"], list)
                    or len(template["values"]) > 4096
                ):
                    raise ValueError("invalid visitor template")
                profiles.append(
                    VisitorProfile(
                        row["visitor_id"],
                        FaceEmbedding(template["model_id"], tuple(template["values"])),
                        datetime.fromisoformat(row["created_at"]),
                        datetime.fromisoformat(row["last_seen_at"]),
                        datetime.fromisoformat(row["last_visit_at"]),
                        row["session_count"],
                        VisitStatistics(**row["statistics"]),
                        row["display_name"],
                        row["explicitly_labeled"],
                        row["distinct_visit_days"],
                        last_visit_local_date=date.fromisoformat(row["last_visit_local_date"]),
                    )
                )
            if len({p.visitor_id for p in profiles}) != len(profiles):
                raise ValueError("duplicate visitor IDs")
            return tuple(profiles)
        except IdentityError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError):
            raise IdentityError("visitor store corrupt; no profiles replaced") from None

    def _load(self) -> tuple[tuple[VisitorProfile, ...], str | None]:
        self._file._check_paths()
        if not self._file.path.exists():
            return (), None
        payload, key_id = decrypt(self._file._document(), self._file.provider, VISITOR_DOMAIN)
        if isinstance(payload, dict) and payload.get("version") == 1:
            self._upgrade(payload)  # Validate authenticated legacy data before giving guidance.
            raise IdentityError(
                "visitor store requires explicit migration: jake-visitors --migrate-store"
            )
        return self.parse(payload), key_id

    def profiles(self) -> tuple[VisitorProfile, ...]:
        return self._load()[0]

    def payload(self, profiles: tuple[VisitorProfile, ...]) -> object:
        return {
            "version": 2,
            "timezone": self.timezone.key,
            "visitors": [
                {
                    "visitor_id": p.visitor_id,
                    "template": {
                        "model_id": p.template.model_id,
                        "values": list(p.template.values),
                    },
                    "created_at": p.created_at.isoformat(),
                    "last_seen_at": p.last_seen_at.isoformat(),
                    "last_visit_at": p.last_visit_at.isoformat(),
                    "session_count": p.session_count,
                    "distinct_visit_days": p.distinct_visit_days,
                    "last_visit_local_date": p.last_visit_local_date.isoformat(),
                    "statistics": {
                        "completed": p.statistics.completed,
                        "mean_seconds": p.statistics.mean_seconds,
                        "m2_seconds": p.statistics.m2_seconds,
                    },
                    "display_name": p.display_name,
                    "explicitly_labeled": p.explicitly_labeled,
                }
                for p in profiles
            ],
        }

    def _upgrade(self, payload: object) -> object:
        """Transform authenticated v1 in memory; never infer historic visit days."""
        try:
            if (
                not isinstance(payload, dict)
                or set(payload) != {"version", "visitors"}
                or type(payload["version"]) is not int
                or payload["version"] != 1
                or not isinstance(payload["visitors"], list)
            ):
                raise ValueError("invalid legacy schema")
            updated = deepcopy(payload)
            for row in updated["visitors"]:
                if any(
                    k in row
                    for k in ("session_count", "distinct_visit_days", "last_visit_local_date")
                ):
                    raise ValueError("invalid legacy record")
                row["session_count"] = row.pop("visit_count")
                row["distinct_visit_days"] = 1
                timestamp = datetime.fromisoformat(row["last_visit_at"])
                if timestamp.utcoffset() is None:
                    raise ValueError("naive timestamp")
                row["last_visit_local_date"] = (
                    timestamp.astimezone(self.timezone).date().isoformat()
                )
            updated["version"] = 2
            updated["timezone"] = self.timezone.key
            self.parse(updated)
            return updated
        except IdentityError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError):
            raise IdentityError("visitor store corrupt; no profiles replaced") from None

    def migrate(self) -> str:
        """Reuse the existing key and verified atomic encrypted writer."""
        with self._file._locked():
            if not self._file.path.exists():
                raise IdentityError("visitor store does not exist; nothing migrated")
            payload, key_id = decrypt(self._file._document(), self._file.provider, VISITOR_DOMAIN)
            if isinstance(payload, dict) and payload.get("version") == 2:
                self.parse(payload)
                return "Visitor store already uses distinct visit days; migration is not required."
            updated = self._upgrade(payload)
            self._file.write_document(updated, key_id, VISITOR_DOMAIN, self.parse)
            return (
                "Visitor store migrated; sessions preserved, "
                "distinct visit days initialized to one."
            )

    def _change(
        self, operation: Callable[[tuple[VisitorProfile, ...]], tuple[VisitorProfile, ...]]
    ) -> None:
        with self._file._locked():
            old, key_id = self._load()
            new = operation(old)
            if new == old:
                return
            payload = self.payload(new)
            self.parse(payload)
            if key_id is None:
                key_id, _ = self._file.provider.create()
            self._file.write_document(payload, key_id, VISITOR_DOMAIN, self.parse)

    def add(self, profile: VisitorProfile) -> None:
        def add(profiles: tuple[VisitorProfile, ...]) -> tuple[VisitorProfile, ...]:
            if any(p.visitor_id == profile.visitor_id for p in profiles):
                raise IdentityError("duplicate visitor ID")
            return (*profiles, profile)

        self._change(add)

    def update(
        self, visitor_id: str, transform: Callable[[VisitorProfile], VisitorProfile]
    ) -> None:
        def update(profiles: tuple[VisitorProfile, ...]) -> tuple[VisitorProfile, ...]:
            if not any(p.visitor_id == visitor_id for p in profiles):
                raise IdentityError("visitor no longer exists")
            return tuple(transform(p) if p.visitor_id == visitor_id else p for p in profiles)

        self._change(update)

    def delete(self, visitor_id: str) -> None:
        self._change(lambda profiles: tuple(p for p in profiles if p.visitor_id != visitor_id))

    def delete_all(self) -> None:
        self._change(lambda _: ())

    def label(self, visitor_id: str, name: str | None) -> None:
        self.update(
            visitor_id, lambda p: replace(p, display_name=name, explicitly_labeled=name is not None)
        )

    def expire(self, now: datetime, retention_days: int) -> None:
        if now.utcoffset() is None or type(retention_days) is not int or retention_days < 1:
            raise IdentityError("invalid visitor retention policy")
        cutoff = now - timedelta(days=retention_days)
        # Avoid locks/disk writes when no expiration is due (including absent stores).
        if any(p.last_seen_at < cutoff for p in self.profiles()):
            self._change(lambda profiles: tuple(p for p in profiles if p.last_seen_at >= cutoff))
