# Phase 2C: encrypted local biometric storage

Phase 2B enrollment/recognition remains unchanged after migration. `IdentityStore`
still exposes `profiles`, `add`, and `delete`; `LocalIdentityStore` adds an explicit
`migrate` operation. No visitor gallery, automatic enrollment, or new identity
inference is introduced. The `identity` optional extra pins `cryptography==50.0.1`
and `keyring==25.7.0`; the lockfile pins their transitive dependencies. Core tracking
does not require these extras. Development dependencies include them for fake-key CI.

## Encryption and envelope

Jake uses PyCA cryptography's [AESGCM](https://cryptography.io/en/latest/hazmat/primitives/aead/)
with a 256-bit random key. No cipher, MAC, or key derivation is implemented by Jake.
Serialization happens in memory. The entire version-1 resident payload is encrypted,
including names, UUIDs, timestamps, sample counts, model identifiers and vectors.

The `residents.json` envelope has exactly these fields:

```json
{
  "version": 2,
  "algorithm": "AES-256-GCM",
  "domain": "Jake/biometric-store",
  "key_id": "<random local UUID>",
  "nonce": "<base64 of 12 random bytes>",
  "ciphertext": "<base64 of encrypted payload plus 16-byte authentication tag>"
}
```

Associated data is canonical UTF-8 JSON of version, algorithm, domain, and key ID:
sorted keys, compact separators. These fields are non-secret but authenticated.
Changing a key ID fails authentication even if both vault entries contain the same
key. Envelope fields/types and supported identifiers are validated before decryption.
The nonce comes from `secrets.token_bytes(12)` on every write; it is never derived
from timestamps or reset counters. Random 96-bit nonces follow the standard GCM
construction; collisions are negligibly probable for this low-volume resident store,
not mathematically impossible. This is not a high-volume event database.

The envelope reveals file size, version, algorithm, and key identifier; it conceals
resident payload contents. Authentication failure, missing/wrong keys, unsupported
versions/algorithms, corrupt JSON/base64, and invalid decrypted schemas raise a clear
`IdentityError`. There is no fallback, silent reset, or replacement key on reads.

## Key provider architecture

`KeyProvider.create()` creates a fresh UUID/key in protected OS storage and verifies
readback. `get(key_id)` retrieves an existing key and never creates/replaces it.
Existing stores retain their key across updates, while each write has a new nonce.
Keys are never saved beside ciphertext, placed in config, derived from names, or
printed. The vault service label is `Jake/biometric-store/v2`, with the random key UUID
as credential username. Base64 is only the vault API's string encoding, not encryption.

Jake explicitly instantiates an approved backend rather than using keyring's automatic
discovery. Environment-selected, third-party, chained, null, and plaintext-file
backends are not selected. Tests inject in-memory providers and never contact an OS
vault. The implementations use the maintained [Python keyring package](https://keyring.readthedocs.io/en/stable/).

### Windows

`WindowsKeyProvider` uses `keyring.backends.Windows.WinVaultKeyring`: a generic
credential in the current Windows user's Credential Manager, protected by Windows'
user credential vault/DPAPI facilities. It is not a raw key file in Jake's directory.
The interactive Windows account must have a working user profile and Credential
Manager. Running under another account/service context will not automatically have
the same credentials. There is no application-specific vault prompt on every read.

Actual Windows validation on this development machine exercised create/read, an
encrypted disk round trip, independent-process decryption, deletion, and restrictive
file writes. It used a synthetic profile in a disposable directory and removed its
test credential afterward. No enrolled residents were accessed or migrated. Linux
backend integration has not been physically validated here.

### Linux, Jetson Orin, and Raspberry Pi

`SystemKeyringProvider` explicitly uses `keyring.backends.SecretService.Keyring`.
It requires a session D-Bus service and an unlocked, **password-protected** Secret
Service collection (for example GNOME Keyring). The key is held in that service's
user keyring and protected at rest by its keyring password. Jake cannot ensure that
an administrator has not configured a passwordless collection; do not use one.

On Ubuntu/Debian-based desktop installs, a secure setup path is:

```sh
sudo apt install gnome-keyring dbus-user-session seahorse
```

Log into the intended local account and use Passwords and Keys (Seahorse) to create
or verify a password-protected login keyring. Unlock it through the desktop login
integration or its password prompt. Run Jake as that same user/session, not with
`sudo`. Jetson Ubuntu and Raspberry Pi OS desktop follow this model when those
packages/services are available; board hardware alone does not provide a keyring.

For headless development, provision a protected collection first, then use a D-Bus
session and unlock it interactively. Python's password prompt avoids shell history
and command-line password exposure:

```sh
dbus-run-session -- bash
python3 -c 'import getpass, subprocess; subprocess.run(["gnome-keyring-daemon", "--unlock", "--components=secrets"], input=(getpass.getpass("Keyring password: ")+"\n").encode(), check=True)'
```

Subsequent Jake commands must run inside that same D-Bus session. Availability depends
on the installed Secret Service implementation. No service, locked collection, denied
access, or missing key results in a clear failure. Jake does not fall back to a file.
Unattended boot needs separately designed secure unlock/provisioning (for example an
OS/TPM-managed deployment); there is no password-in-config workaround. macOS and other
platforms are not yet supported by the default provider.

## Explicit migration, after repository review

Stop enrollment writers and identity sessions before eventual migration. Existing
v1 stores are detected on read and rejected with an explicit migration message.
Jake does not enroll, migrate, or open a webcam automatically.

The exact migration command, documented for later execution, is:

```sh
uv run --extra identity jake-enroll-resident --config config/local.toml --migrate-store
```

Migration acquires the existing exclusive writer lock, validates the v1 schema,
creates and verifies an OS-protected key, then serializes/encrypts in memory. Only
encrypted bytes go into a same-directory restricted temporary file. After flush and
fsync, Jake reads that actual temporary file, authenticates/decrypts it, validates its
schema, and compares every profile against the original before atomic replacement.
Resident UUIDs, names, templates, timestamp values, and sample counts are preserved.
Any failure before replacement leaves the original file untouched. Repeated migration
of a v2 store is rejected rather than generating another key or resetting the store.

There is no automatic plaintext backup. Migration replaces the plaintext file but
cannot securely erase old blocks, snapshots, manually created backups, or Phase 2B
crash leftovers. Normal Phase 2C temporary files contain only ciphertext and are
cleaned on ordinary exceptions. A crash may leave an encrypted temp file or writer
lock; inspect/remove only after confirming all writers have stopped. A failed first
write/migration may leave an unused OS credential; it contains no resident payload
and is not automatically deleted in case a successful replacement was interrupted.

After migration, the documented enrollment, list, delete and live commands use
`--extra identity` along with their existing vision/detection/appearance extras.
Deleting a resident retains the store key so remaining profiles and older encrypted
backups remain readable. Deletion does not automatically destroy backups or purge
profiles cached in existing sessions; restart those sessions.

## Atomicity, recovery, and threat model

Writer locking, same-directory replacement, restrictive permissions, symlink rejection
(including ancestors), and file fsync are preserved. Reads reject envelopes over 3 MB;
decrypted profiles retain schema/count limits. POSIX uses 0700/0600; Windows removes
inherited ACL entries and grants the current SID access before writing. Use a fresh
dedicated private directory: existing explicit Windows grants are not sanitized.
Network filesystems, concurrent distributed writers, and hostile same-account races
are outside this local store design. Atomic replacement is not a universal guarantee
against filesystem/device power-loss failures.

Back up ciphertext only, through an appropriate local backup process. A copied
`residents.json` alone is **not recoverable without its matching vault key**. File
backups and OS credential/profile recovery need a coordinated strategy using the OS's
supported protected backup mechanisms. There is no Jake raw-key export, escrow,
rotation, cross-device transfer, or recovery password in this phase. Do not assume
an OS reinstall/account change or keyring reset preserves access. If credentials or
the key are permanently lost, the encrypted profiles are unrecoverable; deliberate
re-enrollment into a new store is required. Jake never makes that decision automatically.

Encryption protects a copied encrypted file when the attacker lacks the OS-protected
key. It does not protect an unlocked account from malware, administrators able to
compromise that account, live process inspection, malicious models, or camera access.
Keys/plaintext necessarily exist in RAM; Python cannot guarantee zeroization, and OS
swap/crash dumps require OS-level protection. No key, vector, plaintext template, or
full ciphertext is logged by storage errors. Existing deliberate `--list` output and
identity events still contain authorized resident metadata. GCM provides integrity,
not rollback prevention: a valid older encrypted file can be replayed.

CI covers cryptographic round trips, nonce variation, tampering, missing/wrong keys,
explicit migration fidelity/failure, atomicity, locking, encrypted-only temp files,
existing duplicate/deletion behavior, and safe provider errors without real vaults,
model weights, cameras, GPUs, or network.
