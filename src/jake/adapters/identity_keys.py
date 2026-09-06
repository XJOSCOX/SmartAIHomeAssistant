"""Explicit OS credential backends; never auto-select third-party/file keyrings."""

import base64
import sys
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from keyring.backend import KeyringBackend

from jake.identity import IdentityError
from jake.identity_ports import KeyProvider

SERVICE = "Jake/biometric-store/v2"


class SystemKeyringProvider:
    """Linux Secret Service, with no plaintext or environment fallback."""

    def __init__(self) -> None:
        try:
            from keyring.backends.SecretService import Keyring

            self._backend: KeyringBackend = Keyring()  # type: ignore[no-untyped-call]
            # Force availability checks now, rather than creating an empty store.
            if self._backend.priority <= 0:
                raise RuntimeError("unavailable")
        except Exception:
            raise IdentityError(
                "OS Secret Service unavailable; unlock a secure login keyring"
            ) from None

    def get(self, key_id: str) -> bytes:
        try:
            if str(UUID(key_id)) != key_id:
                raise ValueError("invalid key ID")
            encoded = self._backend.get_password(SERVICE, key_id)
            if encoded is None:
                raise ValueError("missing key")
            key = base64.b64decode(encoded, validate=True)
            if len(key) != 32:
                raise ValueError("invalid key")
            return key
        except Exception:
            raise IdentityError(
                "identity key unavailable or invalid in OS credential store"
            ) from None

    def create(self) -> tuple[str, bytes]:
        key_id, key = str(uuid4()), AESGCM.generate_key(bit_length=256)
        try:
            if self._backend.get_password(SERVICE, key_id) is not None:
                raise ValueError("key ID collision")
            self._backend.set_password(SERVICE, key_id, base64.b64encode(key).decode("ascii"))
            if self.get(key_id) != key:
                raise ValueError("credential verification failed")
            return key_id, key
        except Exception:
            raise IdentityError(
                "cannot create and verify identity key in OS credential store"
            ) from None


class WindowsKeyProvider(SystemKeyringProvider):
    """Windows Credential Manager generic credential, protected by the user OS vault."""

    def __init__(self) -> None:
        try:
            from keyring.backends.Windows import WinVaultKeyring

            self._backend = WinVaultKeyring()  # type: ignore[no-untyped-call]
            if self._backend.priority <= 0:
                raise RuntimeError("unavailable")
        except Exception:
            raise IdentityError("Windows Credential Manager unavailable for this account") from None


def default_key_provider() -> KeyProvider:
    if sys.platform == "win32":
        return WindowsKeyProvider()
    if sys.platform.startswith("linux"):
        return SystemKeyringProvider()
    raise IdentityError("no supported OS key provider on this platform")
