"""Versioned encrypted biometric templates in a private, dedicated local directory."""

import csv
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from jake.adapters.identity_keys import default_key_provider
from jake.identity import IdentityError
from jake.identity_domain import FaceEmbedding, ResidentProfile
from jake.identity_encryption import DOMAIN, decrypt, encrypt
from jake.identity_ports import KeyProvider


def private_permissions(path: Path, directory: bool = False) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"], check=True, capture_output=True, text=True
        )
        sid = next(csv.reader(result.stdout.splitlines()))[1]
        if not sid.startswith("S-1-") or any(c not in "S0123456789-" for c in sid):
            raise IdentityError("cannot resolve current Windows account SID")
        rights = "(OI)(CI)F" if directory else "F"
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:{rights}"],
            check=True,
            capture_output=True,
        )
    else:
        path.chmod(0o700 if directory else 0o600)


class LocalIdentityStore:
    """Single-writer lock + atomic replacement. No images, histories, or credentials."""

    def __init__(self, root: Path, key_provider: KeyProvider | None = None) -> None:
        self.root = root.absolute()
        self.path = self.root / "residents.json"
        self._provider = key_provider

    @property
    def provider(self) -> KeyProvider:
        if self._provider is None:
            self._provider = default_key_provider()
        return self._provider

    def _check_paths(self) -> None:
        if any(p.is_symlink() for p in (self.path, self.root, *self.root.parents)):
            raise IdentityError("identity storage cannot use symlinks")

    def _document(self) -> object:
        self._check_paths()
        try:
            if self.path.stat().st_size > 3_000_000:
                raise ValueError("store too large")
            return json.loads(self.path.read_bytes())
        except (OSError, ValueError):
            raise IdentityError("identity store unreadable or corrupt") from None

    def _load(self) -> tuple[tuple[ResidentProfile, ...], str | None]:
        self._check_paths()
        if not self.path.exists():
            return (), None
        data = self._document()
        if isinstance(data, dict) and type(data.get("version")) is int and data["version"] == 1:
            raise IdentityError(
                "legacy plaintext v1 identity store; explicit --migrate-store required"
            )
        payload, key_id = decrypt(data, self.provider)
        return self._parse(payload), key_id

    def profiles(self) -> tuple[ResidentProfile, ...]:
        return self._load()[0]

    @staticmethod
    def _parse(data: object) -> tuple[ResidentProfile, ...]:
        try:
            if (
                not isinstance(data, dict)
                or set(data) != {"version", "residents"}
                or type(data["version"]) is not int
                or data["version"] != 1
                or not isinstance(data["residents"], list)
                or len(data["residents"]) > 100
            ):
                raise ValueError("invalid schema")
            profiles = []
            for item in data["residents"]:
                if (
                    set(item)
                    != {"resident_id", "display_name", "templates", "sample_count", "enrolled_at"}
                    or type(item["sample_count"]) is not int
                ):
                    raise ValueError("invalid resident schema")
                templates = []
                for template in item["templates"]:
                    if (
                        set(template) != {"model_id", "values"}
                        or not isinstance(template["values"], list)
                        or len(template["values"]) > 4096
                    ):
                        raise ValueError("invalid face template schema")
                    templates.append(FaceEmbedding(template["model_id"], tuple(template["values"])))
                profiles.append(
                    ResidentProfile(
                        item["resident_id"],
                        item["display_name"],
                        tuple(templates),
                        item["sample_count"],
                        datetime.fromisoformat(item["enrolled_at"]),
                    )
                )
            if len({p.resident_id for p in profiles}) != len(profiles) or len(
                {p.display_name.strip().casefold() for p in profiles}
            ) != len(profiles):
                raise ValueError("duplicate resident records")
            return tuple(profiles)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise IdentityError(
                "identity store is unreadable or corrupt; no records replaced"
            ) from exc

    @staticmethod
    def _payload(profiles: tuple[ResidentProfile, ...]) -> dict[str, object]:
        data = {
            "version": 1,
            "residents": [
                {
                    "resident_id": p.resident_id,
                    "display_name": p.display_name,
                    "sample_count": p.sample_count,
                    "enrolled_at": p.enrolled_at.isoformat(),
                    "templates": [
                        {"model_id": t.model_id, "values": list(t.values)} for t in p.templates
                    ],
                }
                for p in profiles
            ],
        }
        return data

    def _write(self, profiles: tuple[ResidentProfile, ...], key_id: str) -> None:
        self.write_document(self._payload(profiles), key_id, DOMAIN, self._parse)

    def write_document(
        self, document: object, key_id: str, domain: str, validate: Callable[[object], object]
    ) -> None:
        """Shared encrypted-file primitive. Caller holds the writer lock."""
        expected = validate(document)
        encrypted = encrypt(document, key_id, self.provider, domain)
        if len(encrypted) > 3_000_000:
            raise IdentityError("encrypted identity store too large")
        descriptor, name = tempfile.mkstemp(prefix=".residents-", dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                private_permissions(temporary)
                stream.write(encrypted)
                stream.flush()
                os.fsync(stream.fileno())
            # Verify the actual encrypted temporary file before replacing any original.
            payload, verified_id = decrypt(
                json.loads(temporary.read_bytes()), self.provider, domain
            )
            if verified_id != key_id or validate(payload) != expected:
                raise IdentityError("encrypted identity verification failed")
            self._check_paths()
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._check_paths()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        private_permissions(self.root, True)
        lock = self.root / ".writer.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise IdentityError(
                "identity store is locked; inspect stale lock after a crash"
            ) from exc
        try:
            os.close(descriptor)
            yield
        finally:
            lock.unlink(missing_ok=True)

    def migrate(self) -> None:
        """Explicit v1 conversion only; original survives any failure before replacement."""
        with self._locked():
            document = self._document()
            if isinstance(document, dict) and document.get("version") == 2:
                # Authenticate and validate before describing the store as encrypted.
                payload, _ = decrypt(document, self.provider)
                self._parse(payload)
                raise IdentityError(
                    "Identity store is already encrypted; migration is not required."
                )
            profiles = self._parse(document)
            key_id, _ = self.provider.create()
            self._write(profiles, key_id)

    def _change(self, profile: ResidentProfile | None, resident_id: str | None = None) -> None:
        with self._locked():
            profiles, key_id = self._load()
            if profile is not None:
                if any(
                    p.resident_id == profile.resident_id
                    or p.display_name.casefold() == profile.display_name.casefold()
                    for p in profiles
                ):
                    raise IdentityError(
                        "resident ID or name already enrolled; "
                        "delete explicitly before re-enrollment"
                    )
                if len(profiles) >= 100:
                    raise IdentityError("identity store resident limit reached")
                profiles = (*profiles, profile)
            else:
                if not any(p.resident_id == resident_id for p in profiles):
                    raise IdentityError("resident ID not found")
                profiles = tuple(p for p in profiles if p.resident_id != resident_id)
            if key_id is None:
                key_id, _ = self.provider.create()
            self._write(profiles, key_id)

    def add(self, profile: ResidentProfile) -> None:
        self._change(profile)

    def delete(self, resident_id: str) -> None:
        self._change(None, resident_id)
