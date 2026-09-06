# Phase 2F · Jake Desktop Control Center

Jake provides a native PySide6/Qt Widgets operator interface. It calls shared Python
services directly; it never spawns the camera, enrollment or visitor CLIs. Existing
CLI modes, algorithms, benchmarks and encrypted schemas remain available.

## Development launch after repository review

From the repository root, with existing local models and development config:

```powershell
uv run --extra desktop --extra detection --extra appearance --extra identity jake-desktop --config config/local.toml
```

Without `--config`, Windows uses `%LOCALAPPDATA%/Jake/config/jake.toml` and shows setup
if it is absent. No camera starts automatically. Setup collects an IANA timezone,
camera selection, model paths and visitor persistence preference. Selected runtime
model files must exist before setup saves. The new AppData models folder starts empty.
Use **Use existing model folder…** to select an existing repository `models` directory;
this updates model paths only, without copying files or changing the biometric-store
location. Missing-file messages list every required model path, including the appearance
BIN companion, instead of suggesting a Python dependency failure. Cancelling leaves setup unsaved. Resident
enrollment is optional. No downloads, biometric imports or migrations run in setup.

Qt is optional: the `desktop` extra locks PySide6 6.11.2 and includes `tomli-w` for
TOML serialization. Core/headless installations have no Qt requirement. Desktop tests
skip when that extra is absent; the full suite still needs its usual vision extras.

## Architecture and threading

```text
Qt pages/widgets → camera controller / administration actions
                         ↓
               jake.application services
               composition / LiveSession / EnrollmentSession
                         ↓
               existing PerceptionPipeline and adapters
               detection → tracking → identity/visitors → events
```

`application.composition.compose` creates fresh adapters for each start and is shared
with the camera CLI. `EnrollmentSession` is shared with the enrollment CLI, preserving
consent, quality, pose, consistency and sample gates. Widgets contain no recognition
or tracking mathematics. The pipeline exposes its latest tracks and detector-call
measurement while preserving its `process()` event return contract.

`CameraWorker` owns capture, model initialization and inference on a QThread. A locked
mailbox retains one latest frame and at most 1,000 pending metadata events. Qt polls
every 50 ms; the displayed FPS measures worker throughput, not UI paint frequency; display frames may be dropped under load, avoiding unbounded image-signal
queues. Final events drain before cleanup. Qt copies RGB8 into QImage and draws boxes;
it uses no OpenCV preview window. Stop/close clears displayed images.

Stop sets a threading Event, including during model initialization. The camera context
releases on stop, exhaustion or exceptions. Restart waits for the old worker. Close
defers destruction until camera and administration workers finish; no thread is
terminated unsafely or abandoned. A blocked native driver/model call cannot be forcibly
interrupted: Jake remains Stopping/Closing until it returns. Diagnostics explain this.

Administration also runs off the UI thread and returns immutable metadata rows, not
profiles/templates. Management/settings require stopped capture, so the next start
rebuilds identity caches. Templates stay in existing service/storage scopes; no key
or vector is copied into widget state or logged.

## Pages

- **Dashboard:** system/camera state, processing FPS, detector-call latency, detected
  people, verified residents/visitors, visitors observed today in this app session,
  recent events. No historical daily analytics are fabricated.
- **Live Camera:** Start/Stop, raw/annotated toggle, resolution/FPS/sequence, person
  boxes, track IDs and identity labels. Optional diagnostic text is metadata-only.
- **Residents:** explicit refresh, name/UUID/sample count/date, consent and name input,
  enrollment preview/progress/rejection reasons, confirmed deletion and migration.
- **Visitors:** short UUID/name/frequency state, session/day counts, last seen, mean
  completed session duration; label/rename/remove/delete. Delete-all requires typing
  `DELETE ALL`. Profiles are loaded only on explicit refresh.
- **Events:** newest 1,000 in-memory person/identity/visitor events. Clear affects only
  the UI timeline; there is no event database.
- **Settings:** timezone, camera discovery/index, resident/visitor opt-in, existing
  appearance toggle, visitor thresholds/retention and local model/store paths.
- **Developer / Diagnostics:** safe error categories, operation and storage-access
  status, lifecycle limits. Exception payloads/tracebacks are excluded because external
  errors may contain biometric data. Paths remain visible in Settings. Logs are bounded
  and memory-only.

Dark/light/system palettes share reusable sidebar, cards, badges, tables, forms and
camera panels. System selects the current native palette; automatic OS-theme-change
tracking and persistent theme/window-position preferences are not implemented.

## Configuration and locations

```text
%LOCALAPPDATA%/Jake/
  config/jake.toml
  models/
  data/identities/    residents.json and visitors.json, when explicitly used
  logs/              reserved; no default log persistence
```

Other platforms use `$XDG_DATA_HOME/Jake` or `~/.local/share/Jake`. Packaged builds do
not require a writable installation directory. Development relative paths retain CLI
working-directory semantics. UI-saved paths become absolute. TOML comments/formatting
are not retained, but supported unedited values round-trip through the strict parser.

Settings use a same-directory temporary file, flush/fsync, strict reload/comparison
and atomic replacement. Failure leaves the original intact. New `identity.enabled`
defaults false and works in CLI and GUI; visitor opt-in still enables resident-first
processing. Existing `--identity` remains supported and requires tracking. Settings
may save model paths before installation; runtime checks them before starting capture.

Camera config currently supports only a numeric device index. Requested width/height/
FPS controls are intentionally absent; actual values are displayed. Qt Multimedia
discovers descriptions without opening captures. Qt/OpenCV ordering can differ, so
the numeric index stays editable. Offscreen tests/smoke skip discovery entirely.

## Existing data and migration

No operator config or biometric store is silently moved. Continue using the existing
development config explicitly. Production can select the existing store directory
under the same OS account. A separately planned backup/import must preserve encrypted
files and original vault keys; ciphertext alone is not portable to another account.

Legacy stores report migration-required. Each management page offers an explicit
explained confirmation action reusing existing migration implementations. Resident
migration encrypts plaintext v1 with an OS-vault key. Visitor migration preserves UUID,
template/session metadata, initializes days to one and reuses its key. Repeating is
non-destructive; authentication failures remain fail-closed. Migration skips retention.
Management lists are read-only; live visitor retention behavior is unchanged.
No actual household profiles were accessed during implementation.

## Windows packaging

The first target is **PyInstaller onedir**, avoiding onefile extraction of large ML
runtimes. Hooks collect Qt/OpenCV/Torch dependencies; the spec includes dynamic adapters,
OpenVINO, Windows keyring and package metadata. See the official
[Qt deployment guide](https://doc.qt.io/qtforpython-6/deployment/deployment-pyinstaller.html),
[PyInstaller hooks](https://pyinstaller.org/en/latest/hooks.html) and
[windowed option](https://pyinstaller.org/en/stable/man/pyinstaller.html).

```powershell
.\packaging\build-windows.ps1 -SmokeTest
```

The script creates isolated `.venv-build`, syncs `uv.lock` with runtime extras, and
builds `dist/Jake/Jake.exe` with a windowed bootloader. Its DLL search PATH is isolated
from unrelated installed tools to prevent native-library collisions. Keep the entire `dist/Jake`
directory together. The spec includes no model weights or private repository data.
Artifacts are Git-ignored. Smoke mode starts hidden/offscreen with temporary settings,
constructs all pages, opens no camera/store, and exits. It does not validate physical
camera, GPU inference, recognition accuracy or vault access. Review before live use.

For release verification with existing local weights, the build script also accepts
`-ModelDirectory C:\path\to\models`. This runs the packaged executable's
`--model-smoke-test C:\path\to\models --model-smoke-report C:\path\to\new-report.json`
check. The folder must contain the standard YOLO11n, retail-0287 XML/BIN, YuNet and
SFace filenames used by setup. It runs YOLO, appearance encoding and face detection
on generated blank pixels and loads SFace; it never opens a camera, profile store or
credential vault. The report contains only stage status and exception class names,
and refuses to overwrite an existing file. No weights are downloaded.

OpenVINO's IR reader and CPU plugin are dynamically loaded DLLs, so the packaging
spec explicitly collects OpenVINO runtime libraries. An import-only check does not
detect missing plugins. Resident enrollment can succeed while Start camera fails
because enrollment uses OpenCV face models; live tracking additionally uses YOLO
and OpenVINO appearance inference.

The spec also collects Torchvision's native extensions explicitly. Torchvision 0.29
loads `_C_stable.pyd` dynamically for NMS; older PyInstaller hooks still look for
`_C`, producing an `operator torchvision::nms does not exist` failure at the first
YOLO inference. The synthetic release check exercises this first inference too.

The installer definition targets **Inno Setup 6**, a separate external tool:

```powershell
ISCC.exe packaging\Jake.iss
```

Output is `dist/installer/JakeSetup-0.1.0.exe`, with per-user installation/uninstallation,
Start Menu and optional Desktop shortcuts. Installation does not launch Jake. Uninstall
leaves application data and vault keys intact. No private config, biometric profiles
or user models are included, and no installer compiler is vendored.

Dependencies are locked, but builds are not byte-for-byte signed releases. Bundles are
large; GPU availability remains driver-dependent and CPU is the default. Target-PC
driver/plugin validation remains necessary. Sign release binaries and review Qt and
Ultralytics redistribution licenses before external distribution. Qt's transitive
network DLL can be present for Multimedia; Jake exposes no network/cloud functionality.


## Implementation validation

- 630 tests passed; 93% total coverage.
- Ruff lint, format check, strict mypy, Python wheel and source distribution passed.
- Windows PyInstaller windowed EXE build and offscreen startup/exit smoke passed.
- Packaged YOLO and OpenVINO inference on synthetic pixels, YuNet detection and
  SFace loading passed with existing local weights; no camera or profiles were accessed.
- PE subsystem is checked as Windows GUI; no console bootloader is used in release mode.
- Inno Setup configuration is supplied; its compiler is not installed in this environment,
  so installer compilation has not been performed.
- Physical camera, recognition accuracy, real biometric store and production key-vault
  checks are separate from the synthetic model release check.
