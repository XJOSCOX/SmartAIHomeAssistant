import base64
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest
from conftest import MemoryKeys

from jake.adapters.identity_keys import SERVICE, SystemKeyringProvider
from jake.adapters.local_identity_store import LocalIdentityStore
from jake.identity import IdentityError
from jake.identity_domain import FaceEmbedding, ResidentProfile
from jake.identity_encryption import decrypt, encrypt, serialize

PROFILE = ResidentProfile(
    str(UUID(int=42)),
    "Synthetic Resident",
    (FaceEmbedding("test-model", (0.6, 0.8)),),
    10,
    datetime(2026, 1, 1, tzinfo=UTC),
)


@pytest.fixture
def store(
    tmp_path: Path, memory_keys: MemoryKeys, monkeypatch: pytest.MonkeyPatch
) -> LocalIdentityStore:
    monkeypatch.setattr("jake.adapters.local_identity_store.private_permissions", Mock())
    return LocalIdentityStore(tmp_path, memory_keys)


def test_encrypted_round_trip_and_unique_nonces(store: LocalIdentityStore) -> None:
    nonces = set()
    for _ in range(12):
        store.add(PROFILE)
        document = json.loads(store.path.read_bytes())
        assert document["version"] == 2 and document["algorithm"] == "AES-256-GCM"
        nonces.add(document["nonce"])
        raw = store.path.read_bytes()
        for secret in (
            b"Synthetic Resident",
            b"test-model",
            b"[0.6,0.8]",
            PROFILE.resident_id.encode(),
        ):
            assert secret not in raw
        assert LocalIdentityStore(store.root, store.provider).profiles() == (PROFILE,)
        store.delete(PROFILE.resident_id)
        assert store.profiles() == ()
    assert len(nonces) == 12


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 3),
        ("version", True),
        ("algorithm", "AES-128-GCM"),
        ("domain", "other-app"),
        ("nonce", "bad"),
        ("ciphertext", "bad"),
        ("key_id", "bad-id"),
        ("extra", "unexpected"),
    ],
)
def test_bad_envelope_fails_without_overwrite(
    field: str,
    value: object,
    store: LocalIdentityStore,
) -> None:
    store.add(PROFILE)
    data = json.loads(store.path.read_bytes())
    data[field] = value
    store.path.write_bytes(serialize(data))
    original = store.path.read_bytes()
    for operation in (
        store.profiles,
        lambda: store.add(PROFILE),
        lambda: store.delete(PROFILE.resident_id),
    ):
        with pytest.raises(IdentityError):
            operation()
        assert store.path.read_bytes() == original


@pytest.mark.parametrize("field", ["nonce", "ciphertext"])
def test_authentication_rejects_bit_flip(field: str, store: LocalIdentityStore) -> None:
    store.add(PROFILE)
    data = json.loads(store.path.read_bytes())
    raw = bytearray(base64.b64decode(data[field]))
    raw[0] ^= 1
    data[field] = base64.b64encode(raw).decode()
    store.path.write_bytes(serialize(data))
    with pytest.raises(IdentityError, match="authentication"):
        store.profiles()


def test_key_id_is_authenticated_even_if_key_material_same(
    store: LocalIdentityStore,
    memory_keys: MemoryKeys,
) -> None:
    store.add(PROFILE)
    data = json.loads(store.path.read_bytes())
    alternate, _ = memory_keys.create()
    memory_keys.keys[alternate] = memory_keys.get(data["key_id"])
    data["key_id"] = alternate
    store.path.write_bytes(serialize(data))
    with pytest.raises(IdentityError, match="authentication"):
        store.profiles()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_wrong_key_never_recreated(
    missing: bool,
    store: LocalIdentityStore,
    memory_keys: MemoryKeys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store.add(PROFILE)
    key_id = json.loads(store.path.read_bytes())["key_id"]
    if missing:
        del memory_keys.keys[key_id]
    else:
        memory_keys.keys[key_id] = b"x" * 32
    create = Mock(side_effect=AssertionError("must not recreate key"))
    monkeypatch.setattr(memory_keys, "create", create)
    original = store.path.read_bytes()
    with pytest.raises(IdentityError):
        store.add(PROFILE)
    assert original == store.path.read_bytes()
    create.assert_not_called()


def legacy(store: LocalIdentityStore) -> bytes:
    profiles = (PROFILE, replace(PROFILE, resident_id=str(UUID(int=43)), display_name="Second"))
    plaintext = serialize(store._payload(profiles))
    store.path.write_bytes(plaintext)
    return plaintext


def test_explicit_migration_preserves_every_field(store: LocalIdentityStore) -> None:
    original = legacy(store)
    expected = store._parse(json.loads(original))
    with pytest.raises(IdentityError, match="migrate-store"):
        store.profiles()
    assert store.path.read_bytes() == original
    store.migrate()
    assert store.profiles() == expected
    assert json.loads(store.path.read_bytes())["version"] == 2
    assert sorted(p.name for p in store.root.iterdir()) == ["residents.json"]
    with pytest.raises(IdentityError, match="already encrypted; migration is not required"):
        store.migrate()
    assert store.profiles() == expected
    store.delete(PROFILE.resident_id)
    store.add(PROFILE)
    assert set(store.profiles()) == set(expected)


def test_second_migration_does_not_create_key_or_rewrite(
    store: LocalIdentityStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy(store)
    store.migrate()
    original = store.path.read_bytes()
    profiles = store.profiles()
    create = Mock(side_effect=AssertionError("must not create another key"))
    write = Mock(side_effect=AssertionError("must not rewrite encrypted store"))
    monkeypatch.setattr(store.provider, "create", create)
    monkeypatch.setattr(store, "_write", write)
    with pytest.raises(IdentityError, match="already encrypted; migration is not required"):
        store.migrate()
    create.assert_not_called()
    write.assert_not_called()
    assert store.path.read_bytes() == original
    assert store.profiles() == profiles


@pytest.mark.parametrize("corruption", ["ciphertext", "envelope", "payload", "missing_key"])
def test_migration_does_not_mislabel_corrupt_v2_as_already_encrypted(
    corruption: str,
    store: LocalIdentityStore,
    memory_keys: MemoryKeys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store.add(PROFILE)
    document = json.loads(store.path.read_bytes())
    if corruption == "ciphertext":
        raw = bytearray(base64.b64decode(document["ciphertext"]))
        raw[0] ^= 1
        document["ciphertext"] = base64.b64encode(raw).decode()
    elif corruption == "envelope":
        del document["nonce"]
    elif corruption == "payload":
        document = json.loads(
            encrypt({"version": 1, "residents": [{}]}, document["key_id"], store.provider)
        )
    else:
        del memory_keys.keys[document["key_id"]]
    store.path.write_bytes(serialize(document))
    original = store.path.read_bytes()
    create = Mock(side_effect=AssertionError("must not create key"))
    monkeypatch.setattr(store.provider, "create", create)
    with pytest.raises(IdentityError, match="corrupt|authentication|key unavailable") as error:
        store.migrate()
    assert "already encrypted" not in str(error.value)
    assert store.path.read_bytes() == original
    create.assert_not_called()


def test_migration_cli_reports_already_encrypted(
    store: LocalIdentityStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jake.enroll_cli import main

    store.add(PROFILE)
    original = store.path.read_bytes()
    config = tmp_path / "config.toml"
    config.write_text('[pipeline]\ncamera_id="test"', encoding="utf-8")
    monkeypatch.setattr(
        "jake.adapters.local_identity_store.LocalIdentityStore", Mock(return_value=store)
    )
    assert main(["--config", str(config), "--migrate-store"]) == 1
    output = capsys.readouterr()
    assert "Identity store is already encrypted; migration is not required." in output.err
    assert "corrupt" not in output.err
    assert store.path.read_bytes() == original


@pytest.mark.parametrize("failure", ["key", "verify", "replace", "permission", "fsync"])
def test_migration_failure_keeps_original_and_cleans_encrypted_temporary(
    failure: str,
    store: LocalIdentityStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = legacy(store)
    if failure == "key":
        monkeypatch.setattr(
            store.provider, "create", Mock(side_effect=IdentityError("key unavailable"))
        )
    elif failure == "verify":
        monkeypatch.setattr(
            "jake.adapters.local_identity_store.decrypt", Mock(side_effect=IdentityError("verify"))
        )
    elif failure == "replace":

        def replacement(source: Path, destination: Path) -> None:
            # The only temporary file contains authenticated ciphertext, never plaintext.
            data = json.loads(source.read_bytes())
            assert data["version"] == 2 and "residents" not in data
            assert destination.read_bytes() == original
            raise OSError("replacement failed")

        monkeypatch.setattr("jake.adapters.local_identity_store.os.replace", replacement)
    elif failure == "permission":
        monkeypatch.setattr(
            "jake.adapters.local_identity_store.private_permissions",
            Mock(side_effect=[None, OSError("permissions")]),
        )
    else:
        monkeypatch.setattr(
            "jake.adapters.local_identity_store.os.fsync", Mock(side_effect=OSError("fsync"))
        )
    with pytest.raises((IdentityError, OSError)):
        store.migrate()
    assert store.path.read_bytes() == original
    assert sorted(p.name for p in store.root.iterdir()) == ["residents.json"]


def test_migration_respects_lock(store: LocalIdentityStore) -> None:
    original = legacy(store)
    (store.root / ".writer.lock").touch()
    with pytest.raises(IdentityError, match="locked"):
        store.migrate()
    assert store.path.read_bytes() == original


def test_migration_cli_is_explicit_and_camera_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jake.enroll_cli import main

    config = tmp_path / "config.toml"
    config.write_text('[pipeline]\ncamera_id="test"', encoding="utf-8")
    mocked = Mock()
    monkeypatch.setattr(
        "jake.adapters.local_identity_store.LocalIdentityStore", Mock(return_value=mocked)
    )
    assert main(["--config", str(config), "--migrate-store"]) == 0
    mocked.migrate.assert_called_once_with()
    mocked.profiles.assert_not_called()


def test_key_provider_uses_vault_and_never_logs_secret(capsys: pytest.CaptureFixture[str]) -> None:
    provider = object.__new__(SystemKeyringProvider)
    backend = Mock()
    vault: dict[str, str] = {}
    backend.get_password.side_effect = lambda service, key_id: vault.get(key_id)
    backend.set_password.side_effect = lambda service, key_id, value: vault.update({key_id: value})
    provider._backend = backend
    key_id, key = provider.create()
    assert provider.get(key_id) == key and len(key) == 32
    assert backend.set_password.call_args.args[0] == SERVICE
    backend.get_password.side_effect = RuntimeError(base64.b64encode(key).decode())
    with pytest.raises(IdentityError) as error:
        provider.get(key_id)
    assert base64.b64encode(key).decode() not in str(error.value)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("value", [None, "invalid base64!", "eA=="])
def test_provider_missing_or_corrupt_key(value: str | None) -> None:
    provider = object.__new__(SystemKeyringProvider)
    provider._backend = Mock(get_password=Mock(return_value=value))
    with pytest.raises(IdentityError, match="key unavailable"):
        provider.get(str(UUID(int=1)))


def test_invalid_key_size_rejected(memory_keys: MemoryKeys) -> None:
    key_id, _ = memory_keys.create()
    memory_keys.keys[key_id] = b"x" * 16
    with pytest.raises(IdentityError):
        encrypt({}, key_id, memory_keys)
    with pytest.raises(IdentityError):
        decrypt({}, memory_keys)


@pytest.mark.parametrize(
    "platform,module,class_name",
    [
        ("win32", "keyring.backends.Windows", "WinVaultKeyring"),
        ("linux", "keyring.backends.SecretService", "Keyring"),
    ],
)
@pytest.mark.parametrize("available", [True, False])
def test_explicit_platform_backend_selection(
    platform: str,
    module: str,
    class_name: str,
    available: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    from jake.adapters.identity_keys import default_key_provider

    backend = Mock(priority=5 if available else 0)
    factory = Mock(return_value=backend)
    fake_module = ModuleType(module)
    setattr(fake_module, class_name, factory)
    monkeypatch.setitem(sys.modules, module, fake_module)
    monkeypatch.setattr(sys, "platform", platform)
    if available:
        assert default_key_provider()._backend is backend  # type: ignore[attr-defined]
    else:
        with pytest.raises(IdentityError, match="unavailable"):
            default_key_provider()
    factory.assert_called_once_with()


def test_unsupported_platform_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from jake.adapters.identity_keys import default_key_provider

    monkeypatch.setattr("sys.platform", "unsupported")
    with pytest.raises(IdentityError, match="no supported"):
        default_key_provider()


@pytest.mark.parametrize("collision", [True, False])
def test_key_creation_never_overwrites_collision_or_accepts_failed_readback(
    collision: bool,
) -> None:
    provider = object.__new__(SystemKeyringProvider)
    backend = Mock()
    backend.get_password.side_effect = ["existing"] if collision else [None, None]
    provider._backend = backend
    with pytest.raises(IdentityError, match="create and verify"):
        provider.create()
    if collision:
        backend.set_password.assert_not_called()


def test_store_rejects_symlink_before_key_access(
    store: LocalIdentityStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == store.root)
    with pytest.raises(IdentityError, match="symlinks"):
        store.profiles()
    with pytest.raises(IdentityError, match="symlinks"):
        store.migrate()
