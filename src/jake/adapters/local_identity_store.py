"""Versioned plaintext biometric templates in a private, dedicated local directory."""

import csv
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from jake.identity import IdentityError
from jake.identity_domain import FaceEmbedding, ResidentProfile


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

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.path = self.root / "residents.json"

    def profiles(self) -> tuple[ResidentProfile, ...]:
        if self.root.is_symlink() or self.path.is_symlink():
            raise IdentityError("identity storage cannot use symlinks")
        if not self.path.exists():
            return ()
        try:
            if self.path.stat().st_size > 2_000_000:
                raise ValueError("store too large")
            data = json.loads(self.path.read_text(encoding="utf-8"))
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

    def _write(self, profiles: tuple[ResidentProfile, ...]) -> None:
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
        descriptor, name = tempfile.mkstemp(prefix=".residents-", dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                private_permissions(temporary)
                json.dump(data, stream, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _change(self, profile: ResidentProfile | None, resident_id: str | None = None) -> None:
        if self.root.is_symlink():
            raise IdentityError("identity directory cannot be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        private_permissions(self.root, True)
        lock = self.root / ".writer.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise IdentityError(
                "identity store is locked by another writer; inspect stale lock after a crash"
            ) from exc
        try:
            os.close(descriptor)
            profiles = self.profiles()
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
            self._write(profiles)
        finally:
            lock.unlink(missing_ok=True)

    def add(self, profile: ResidentProfile) -> None:
        self._change(profile)

    def delete(self, resident_id: str) -> None:
        self._change(None, resident_id)
