"""AES-256-GCM envelope: no cryptographic primitives implemented by Jake."""

import base64
import json
import secrets
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from jake.identity import IdentityError
from jake.identity_ports import KeyProvider

DOMAIN = "Jake/biometric-store"


def serialize(data: object) -> bytes:
    return json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def metadata(key_id: str, domain: str = DOMAIN) -> dict[str, object]:
    return {"version": 2, "algorithm": "AES-256-GCM", "domain": domain, "key_id": key_id}


def encrypt(payload: object, key_id: str, provider: KeyProvider, domain: str = DOMAIN) -> bytes:
    header = metadata(key_id, domain)
    key = provider.get(key_id)
    if len(key) != 32:
        raise IdentityError("identity encryption requires a 256-bit key")
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, serialize(payload), serialize(header))
    return serialize(
        {
            **header,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
    )


def decrypt(data: object, provider: KeyProvider, domain: str = DOMAIN) -> tuple[object, str]:
    try:
        if not isinstance(data, dict) or set(data) != {
            "version",
            "algorithm",
            "domain",
            "key_id",
            "nonce",
            "ciphertext",
        }:
            raise ValueError("invalid envelope")
        key_id = data["key_id"]
        if not isinstance(key_id, str) or str(UUID(key_id)) != key_id:
            raise ValueError("invalid key ID")
        header = metadata(key_id, domain)
        if type(data["version"]) is not int or any(data[k] != v for k, v in header.items()):
            raise ValueError("unsupported envelope")
        nonce = base64.b64decode(data["nonce"], validate=True)
        ciphertext = base64.b64decode(data["ciphertext"], validate=True)
        if len(nonce) != 12 or len(ciphertext) < 16:
            raise ValueError("invalid envelope size")
        key = provider.get(key_id)
        if len(key) != 32:
            raise ValueError("invalid key")
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, serialize(header))
        return json.loads(plaintext), key_id
    except IdentityError:
        raise
    except Exception:
        raise IdentityError(
            "encrypted identity store corrupt, unsupported, or authentication failed"
        ) from None
