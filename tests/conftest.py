"""CI uses ephemeral keys; it must never read or write an operating-system vault."""

from uuid import uuid4

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from jake.identity import IdentityError


class MemoryKeys:
    def __init__(self) -> None:
        self.keys: dict[str, bytes] = {}

    def create(self) -> tuple[str, bytes]:
        key_id, key = str(uuid4()), AESGCM.generate_key(bit_length=256)
        self.keys[key_id] = key
        return key_id, key

    def get(self, key_id: str) -> bytes:
        if key_id not in self.keys:
            raise IdentityError("identity key unavailable")
        return self.keys[key_id]


@pytest.fixture(autouse=True)
def memory_keys(monkeypatch: pytest.MonkeyPatch) -> MemoryKeys:
    keys = MemoryKeys()
    monkeypatch.setattr("jake.adapters.local_identity_store.default_key_provider", lambda: keys)
    return keys
