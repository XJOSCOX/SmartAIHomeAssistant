"""Separate encrypted visitor payload using Phase 2C file/key primitives."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from jake.adapters.local_identity_store import LocalIdentityStore
from jake.identity import IdentityError
from jake.identity_domain import FaceEmbedding
from jake.identity_encryption import decrypt
from jake.identity_ports import KeyProvider
from jake.visitor_domain import VisitorProfile, VisitStatistics

VISITOR_DOMAIN = "Jake/visitor-store"


class EncryptedVisitorStore:
    def __init__(self, root: Path, key_provider: KeyProvider | None = None) -> None:
        # Reuse locking, permissions, encryption verification and atomic replacement,
        # not the resident schema or resident CRUD/migration operations.
        self._file = LocalIdentityStore(root, key_provider)
        self._file.path = self._file.root / "visitors.json"

    @staticmethod
    def parse(payload: object) -> tuple[VisitorProfile, ...]:
        try:
            if (
                not isinstance(payload, dict)
                or set(payload) != {"version", "visitors"}
                or type(payload["version"]) is not int
                or payload["version"] != 1
            ):
                raise ValueError("invalid visitor schema")
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
                    "visit_count",
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
                        row["visit_count"],
                        VisitStatistics(**row["statistics"]),
                        row["display_name"],
                        row["explicitly_labeled"],
                    )
                )
            if len({p.visitor_id for p in profiles}) != len(profiles):
                raise ValueError("duplicate visitor IDs")
            return tuple(profiles)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise IdentityError("visitor store corrupt; no profiles replaced") from None

    def _load(self) -> tuple[tuple[VisitorProfile, ...], str | None]:
        self._file._check_paths()
        if not self._file.path.exists():
            return (), None
        payload, key_id = decrypt(self._file._document(), self._file.provider, VISITOR_DOMAIN)
        return self.parse(payload), key_id

    def profiles(self) -> tuple[VisitorProfile, ...]:
        return self._load()[0]

    @staticmethod
    def payload(profiles: tuple[VisitorProfile, ...]) -> object:
        return {
            "version": 1,
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
                    "visit_count": p.visit_count,
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
