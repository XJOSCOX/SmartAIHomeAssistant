# Jake — SmartAIHomeAssistant

Jake is the foundation for a privacy-first, local-first smart home AI system.
The intended system will understand household events locally and eventually
support natural conversation. **Phase 2D adds opt-in anonymous recurring visitor memory,
separate from enrolled residents, with encrypted templates and retention controls.
All tracker modes remain available; track IDs are session-local, not resident identities.**

## Phase 2F Desktop Control Center

Jake now provides a native Windows PySide6 dashboard, embedded camera, consent-based
resident enrollment, encrypted visitor management, session events, settings and diagnostics.
The GUI and CLIs share application services; perception algorithms remain intact.
Camera/model work stays off the UI thread. No camera or biometric store opens at launch.

After repository review:

```powershell
uv run --extra desktop --extra detection --extra appearance --extra identity jake-desktop --config config/local.toml
```

Without `--config`, first-run setup uses `%LOCALAPPDATA%/Jake/`. Existing configs/stores
are never silently moved or migrated. See the [Desktop Control Center guide](docs/desktop-control-center.md)
for architecture, data handling, threading limits, EXE and Inno Setup build instructions.

## Phase 1 scope

The initial boundary is:

```text
Camera adapter → Frame → PersonDetector → PersonTracker → EventGenerator → caller
                    RGB      detections       tracks          metadata
```

The production package defines immutable data contracts, structural interfaces,
validated TOML configuration, synchronous pipeline orchestration, and a concrete
OpenCV camera adapter, replaceable YOLO person detector, and Jake-owned tracker.
The preview optionally routes tracks through a metadata-only event generator
and logs semantic events locally with `--events`. Optional `--identity` adds a
separate metadata-only identity stream. Deliberate enrollment persists face templates
in a dedicated local store; there are no recordings or background services.

## Phase 2B resident identity

See [resident identity design and setup](docs/resident-identity.md) for explicit
model downloads, consent, enrollment, thresholds, reactivation safeguards, and deletion.
YuNet detects faces inside person regions; SFace produces normalized 128-dimensional
embeddings. Both run locally on CPU behind replaceable interfaces. A resident match
requires multiple high-quality observations. Clothing similarity cannot grant identity.

**Biometric storage is encrypted and permission-restricted.** Windows Credential Manager
or Linux Secret Service holds the key, separately from the store. Existing plaintext
stores require explicit migration; reads never migrate or fall back to plaintext.
See [encryption, migration, and recovery](docs/encrypted-identity-storage.md).
Enrollment is never automatic. The commands
below are documented for review and subsequent live validation; automated tests use
fake models and cameras, and do not establish real-world recognition accuracy.

After the explicit local model setup in the design guide:

```sh
uv run --extra identity --extra vision jake-enroll-resident --config config/local.toml --name "Joseph" --consent
uv run --extra identity --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --reid --events --identity
```

Migration command for after repository review (no camera required):

```sh
uv run --extra identity jake-enroll-resident --config config/local.toml --migrate-store
```

## Phase 2D.2 visitor frequency

Phase 2D visitor persistence defaults **off**. It requires five quality-approved,
confidently non-resident face observations, never just an UNKNOWN frame. Visits follow
semantic ENTERED/LEFT events, with body ReID preserving the same visit. Resident
ambiguity blocks visitor learning; names require explicit operator labeling.
Transient resident ambiguity now pauses collection rather than blocking the whole
visit; confirmed/repeated strong resident evidence still blocks it. Visitor detector
confidence defaults to `min_detector_confidence=0.90`, while legacy explicit
`min_face_quality` settings retain their values. `--debug-visitors` exposes per-track
rejection reasons, recovery support, and candidate progress without biometric vectors.
See [visitor memory design and management](docs/visitor-memory.md) for thresholds,
retention, consent considerations, statistics, encryption, and deletion limitations.
Existing encrypted resident stores require no migration or rewrite.

Future live command, after repository review and existing local model setup:

```sh
uv run --extra identity --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --reid --events --identity --visitors
```

Frequency counts **distinct household-local visit days**, separately from physical
presence sessions. Same-day returns stay FIRST_TIME; defaults are RECURRING at two
days and FREQUENT at five. Explicit labels take KNOWN precedence without granting
resident status or trust. `[home] timezone` uses an IANA zone (UTC default); the example
configuration uses America/Chicago. Welford duration statistics still count completed
physical sessions. Debug logs suppress repetitive encoder-idle/confirmed alternation.

Existing encrypted visitor payloads require the explicit, authenticated
`jake-visitors --config config/local.toml --migrate-store` upgrade after repository
review and configuration updates. Old session counts/templates are preserved; distinct
days initialize conservatively to one. Resident storage is unchanged. See the
[migration and calendar policy](docs/visitor-memory.md#explicit-migration-of-existing-visitor-data)
before future live execution. No real visitor data was migrated during implementation.

The visitor store is `.jake-identities/visitors.json` by default. It has a separate
key and authenticated domain from residents. Use `--no-visitors` to override an
enabled configuration. Stopping persistence does not delete existing templates;
explicit list, label, delete, and delete-all commands are documented in the guide.

## Development

Requires Python 3.11+ and `uv`. From the repository root:

```sh
uv sync --locked --group dev --extra vision
uv run pytest --cov=jake --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

`.python-version` selects Python 3.11; `uv` can provision it if necessary.
`uv.lock` pins dependencies. The core has no runtime dependencies; the optional
`vision` extra installs `opencv-python` (with GUI support) and NumPy. Tests use real
color conversion with fake camera/GUI boundaries; no physical camera or desktop
is required in CI. For another interpreter, add `--python 3.13` to the sync command.
CI runs checks and packaging on Python 3.11–3.14 with the vision extra installed.
Detector tests mock the model boundary and need neither Ultralytics/PyTorch nor
weights, GPU, network, or webcam. Install `--extra detection` to run real inference;
this includes the vision dependencies and a pinned Ultralytics version. Keep that
pin reviewed when upgrading because privacy controls depend on framework behavior.

## Local camera setup and preview

Connect a USB webcam or use the built-in webcam. Run from the repository root in
a local graphical desktop session:

```sh
uv sync --locked --extra vision
```

Copy the example settings once. On Windows PowerShell:

```powershell
Copy-Item config/jake.example.toml config/local.toml
```

On macOS/Linux:

```sh
cp config/jake.example.toml config/local.toml
```

Edit `config/local.toml` to choose the camera:

```toml
[pipeline]
camera_id = "front-door"
min_person_confidence = 0.5

[camera]
device = 0
```

`camera_id` is Jake's logical source label; `device` is the non-negative local
OpenCV device index. Start with `0`; try `1` or `2` if another camera is selected.
Only integer indices are accepted; stream URLs and video-file paths are rejected.
The confidence setting also controls person detection when `--detect` is supplied.
It has no effect on camera-only preview.

```sh
uv run --extra vision jake-camera --config config/local.toml
```

Equivalent module command:

```sh
uv run --extra vision python -m jake.cli --config config/local.toml
```

The window shows sequence number, dimensions, and average delivered FPS since
preview startup (not the camera's advertised rate). Focus the preview window and
press **q** or **Q** to exit, or press **Ctrl+C** in the terminal. These paths release the
camera and destroy the preview window. No video or images are recorded or saved.
Exit codes are 0 for normal exit/Ctrl+C, 1 for camera/display/detector errors
(including missing detection dependencies), and 2 for configuration or missing
vision dependencies. `jake-camera --help` does not load OpenCV.

If the device cannot open, check the index, close other apps using it, and enable
camera access for desktop apps/your terminal in OS privacy settings. On Linux,
check access permissions for the camera device. A graphical session and OpenCV
GUI libraries are required; `opencv-python-headless` cannot provide this preview.
Avoid installing multiple OpenCV package variants in the same environment.
Package installation may use the network; camera acquisition and preview do not
make network calls, upload data, or emit telemetry.

## Phase 1B model setup and person detection

The default model is the lightweight COCO-pretrained
[YOLO11n detection model](https://docs.ultralytics.com/models/yolo11/).
Weights are **not bundled**. Initial setup may require a download; Jake itself
does not automatically download missing weights. It reports a setup error before
opening the camera. Download once explicitly, or copy an existing trusted local
Ultralytics detection `.pt` file onto the machine.

On Windows PowerShell, from the repository root:

```powershell
uv sync --locked --extra detection
New-Item -ItemType Directory -Force models
Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -OutFile models/yolo11n.pt
```

On macOS/Linux, the equivalent weight setup is:

```sh
uv sync --locked --extra detection
mkdir -p models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o models/yolo11n.pt
```

Keep your existing `config/local.toml` camera settings and add:

```toml
[detector]
model = "models/yolo11n.pt"
device = "cpu"
image_size = 640
```

Set `min_person_confidence = 0.5` under your existing `[pipeline]` section
(do not add a duplicate table). This is the single threshold for inference and
Jake's returned detections. `image_size` is the inference resize target, a positive
multiple of 32; it does not change camera capture resolution.

For existing weights, use `model = "C:/Jake/models/custom.pt"`, or a local filename
such as `model = "yolo11n.pt"` if that file is in the working directory. Relative
paths resolve from the working directory, not the TOML file. Model metadata must
contain a class named `person`; Jake does not assume a fixed numeric class ID.
This adapter supports Ultralytics detection `.pt` checkpoints, not cloud model
IDs, URLs, ONNX, segmentation, or tracking models.

Run Phase 1B:

```sh
uv run --extra detection jake-camera --config config/local.toml --detect
```

After dependency and weight setup, you can also run fully offline without a
dependency sync using `uv run --offline --no-sync jake-camera --config config/local.toml --detect`.
Omit `--detect` to keep the original camera-only preview. Press **q**, **Q**, or
**Ctrl+C** to stop. Each box is labeled `PERSON 97.4%` (confidence varies).
The people count is the number detected in the current frame, not unique people
across time. This `--detect` mode performs no tracking; use `--track` for Phase 1C.

### Timing and CPU/GPU expectations

- `inference ms` measures one complete `detect()` call with `perf_counter()`:
  pixel reconstruction, model preprocessing/inference/postprocessing, and result
  conversion. GPU results are transferred to CPU before the call returns. It
  excludes camera reads, drawing, and GUI waits; it is not kernel-only timing.
- `detection FPS` is `1000 / inference_ms` for that frame: effective detector-only
  throughput. The other `FPS` is average delivered preview throughput since
  startup, including acquisition and processing. Both are development estimates.
- Model initialization occurs before camera opening and outside the measurement.
  The first detection includes lazy initialization/warmup and can be much slower.
- CPU is the explicit default. Your validated camera-only result of about 19 FPS
  at 640×480 on Windows is an acquisition baseline, not a detection benchmark.
  Detection adds work and may reduce preview FPS. Try `image_size = 320` for
  less compute, with a possible small/distant-person accuracy tradeoff.
- For an NVIDIA GPU, install a matching CUDA-enabled PyTorch/torchvision build
  using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
  and set `device = "0"`. A GPU index alone does not install GPU support. When
  maintaining a custom PyTorch build, run with `--no-sync` to retain it. Supported
  Apple hardware can use `device = "mps"`. Unavailable devices report a detector
  error; Jake does not silently switch devices. No hardware FPS is guaranteed.

### Detector privacy and boundary decisions

Only `adapters/yolo_detector.py` knows about Ultralytics. It reconstructs the
RGB8 buffer and makes an owned BGR NumPy copy, as required by the framework's
[NumPy input contract](https://docs.ultralytics.com/modes/predict/). The frame
cannot be changed by the model. Results become immutable `PersonDetection`
objects with normalized boxes. Partial off-screen boxes are clipped; degenerate,
non-finite, wrong-class, and invalid-confidence detections are discarded.
Unexpected result structure and model failures raise `DetectorError`.

The adapter sets `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false`, and disables
Ultralytics `sync` before model loading; prediction explicitly disables image,
text, crop, and visualization saving and model-side display. Normal inference
uses local arrays and weights, with no network calls, telemetry, uploads, or
cloud inference. The framework may create/update its local settings file and
runtime caches; these contain no captured images. It can retain its latest
input/results in process memory; there is no disk image persistence. Weights and
local configuration are ignored by Git. Start Jake in its own process so these
privacy settings are applied before other uses of the framework.

## Phase 1C: Jake-owned multi-person tracking

Detection answers "where are people in this frame?" Tracking associates those
detections across successive frames and assigns a stable session-local label.
**A track ID is NOT a resident identity.** It does not identify a person by face,
appearance, gait, name, or household membership. IDs are stored only in memory
and start again at `1` when a new tracker instance/session starts.

With the Phase 1B dependencies and local weights already set up, run:

```sh
uv run --extra detection jake-camera --config config/local.toml --track
```

`--track` automatically enables detection. `--detect` alone retains independent
per-frame detection. Omit both flags for camera-only preview. For an
already-installed environment, use `uv run --offline --no-sync jake-camera --config config/local.toml --track`.
Both **q/Q** and **Ctrl+C** still close the preview and release the camera.

Optionally add these settings to your existing `config/local.toml`:

```toml
[tracking]
min_iou = 0.3
max_missed_frames = 10
```

These are also the defaults for old configurations. `min_iou` must be finite,
strictly greater than zero, and at most one; zero is rejected so disjoint boxes
cannot be associated. `max_missed_frames` must be a non-negative integer. A value
of zero expires a track on its first unmatched update. Booleans, numeric strings,
fractional missed-frame counts, and unknown settings are rejected.

The preview labels tracks as `ID 1 | PERSON 94.2%`. It shows active tracks,
people detected in the current frame, detector inference milliseconds, preview
FPS, sequence, and resolution. **Active includes temporarily missed tracks**:
their last-seen box and confidence remain displayed during the grace period.
They are not newly observed people or motion predictions. As a result, active
track count can exceed current detections, and stale boxes can remain briefly.
Tracking and rendering run outside the detector-only latency measurement;
their cost is included in preview throughput.

### IoU and deterministic assignment

For two boxes A and B:

```text
IoU(A, B) = area(A ∩ B) / (area(A) + area(B) - area(A ∩ B))
```

Identical boxes score 1; disjoint or merely touching boxes score 0. The tracker
compares every active track's last-seen box to every new detection, keeping pairs
whose score is at least `min_iou`. It sorts candidates by descending IoU, breaking
ties by track creation order and then input detection order. It greedily accepts
a pair only when neither member has been assigned. Thus each detection updates
at most one track, and each track consumes at most one detection. Equal ordered
inputs produce equal results; arbitrary changes in detection order can affect ties.

The pure functions in `matching.py` own geometry and assignment. `IoUPersonTracker`
implements the existing `PersonTracker` protocol and owns lifecycle state.
It imports neither OpenCV nor Ultralytics, never calls YOLO's tracking API, and
receives only metadata and structured detections. The preview contains no
association or lifecycle state, and public tracks expose minimal visibility/confirmation metadata for events.

### Track lifecycle

- Unmatched detections create new tracks with incremental string IDs (`1`, `2`,
  ...), in detection order. IDs are never reused within a session.
- A matched track keeps its ID, updates box/confidence and `last_seen_at`,
  increments `visible_frames`, and resets `missed_frames` to zero.
- Unmatched tracks increment consecutive `missed_frames`. They remain active
  while `missed_frames <= max_missed_frames`; the next unmatched update removes
  them. Reconnection with sufficient IoU before removal keeps the original ID.
  Reappearance after removal creates a new ID.
- Internal state also retains `created_at` and `age_frames`. Age and cumulative
  visible count both start at one. Age advances on every subsequent processed
  update while alive, including empty detection updates. Last-seen time and
  confidence do not change on a miss.

For `max_missed_frames = 2`, two empty updates preserve a track. A matching third
update reconnects it; an empty third update expires it. Misses count processed
`update()` calls, not wall-clock time or gaps in capture sequence numbers. Skipped
camera frames were not evaluated and do not count as observed misses. One tracker
belongs to one camera session and must be called serially. Mixed camera IDs and
duplicate/decreasing sequences raise `TrackerError` without changing state.

For T active tracks and D detections, candidate generation is **O(T × D)**.
Sorting K valid candidates costs **O(K log K)**, at worst approximately
**O((T × D) log(T × D))**. Candidate storage is **O(T × D)** worst case;
lifecycle maintenance costs O(T + D), with O(T) persistent state. No image history,
network calls, identity recognition, or disk persistence is involved.

### Limitations and next step

IoU-only matching cannot predict motion or recover identity from appearance.
Fast movement, camera movement, changed box sizes, and missed detections can
break overlap. IDs can switch when people cross or become heavily occluded.
A new person entering a retained box can inherit its ID. Increasing the grace
period preserves IDs longer but can keep stale tracks alive; raising the IoU
threshold rejects weak matches but can fragment tracks. Greedy assignment is
deterministic, not globally optimal.

Phase 1D adds Kalman prediction/correction behind the same protocol. Phase 1E
adds selectable Hungarian/global assignment; use `--assignment greedy` to retain
the original association baseline.

## Phase 1D: Kalman-assisted tracking

Phase 1C has been physically validated with multiple people. Its ID fragmentation
during crossings and occlusion is the baseline for this phase. Select the new
tracker with your existing camera and local YOLO weight setup:

```sh
uv run --extra detection jake-camera --config config/local.toml --track --tracker kalman
```

`--track --tracker iou` (or just `--track`) keeps the original baseline. Normal
labels and controls are unchanged. Add `--debug-tracks` with the Kalman tracker
to show predicted/measurement boxes plus per-track ID and missed count. No
OpenCV KalmanFilter or YOLO tracking is used. Assignment is now selectable;
use `--assignment greedy` for the Phase 1D baseline.

Each track estimates `[cx, cy, w, h, vx, vy, vw, vh]` in normalized coordinates,
predicts forward before greedy IoU matching, and corrects from matched detections.
Missed tracks continue moving until their configured expiration. Public boxes
are bounded; track IDs remain local to the session. Kalman predictions reduce
some occlusion-related ID losses but cannot guarantee IDs through crossings.

Optional noise overrides (defaults shown):

```toml
[tracking.kalman]
process_noise_position = 0.0001
process_noise_velocity = 0.001
measurement_noise = 0.001
initial_position_variance = 0.01
initial_velocity_variance = 1.0
```

All are finite positive variances/intensities, not standard deviations. Old
configuration files use these defaults. dt uses capture timestamps, clamped to
1 ms–1 s; repeated/backward clocks fall back to 1/30 s. See the
[Kalman design and diagnostics](docs/kalman-tracking.md) for the matrices,
noise model, numerical stability, lifecycle, and limitations.

Run the deterministic comparison without a camera, model, GPU, or network:

```sh
uv run --extra tracking python -m jake.benchmarks
```

On two uninterrupted trajectories both trackers create two IDs with no switches.
With a three-frame occlusion, the IoU baseline creates three IDs with one change;
Kalman keeps two IDs with no changes. This is a synthetic regression comparison,
not proof of real-world crossing accuracy. Phase 1D has now been physically
validated and works on the user's webcam.

## Phase 1E: global assignment

Jake's Hungarian solver assigns tracks and detections globally using gated
`1 - IoU` costs. It supports rectangular and empty inputs and explicit unmatched
choices, so below-threshold pairs cannot be forced together. It uses neither
SciPy's solver nor model-provided tracking. Greedy remains selectable.

```sh
uv run --extra detection jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian
```

Set `assignment = "hungarian"` under your existing `[tracking]` table to select it
without the CLI flag. CLI selection overrides the file without changing it.
Omitted settings preserve `greedy` for backwards compatibility; the example file
explicitly selects Hungarian. Use `--assignment greedy` to reproduce Phase 1C/1D
baselines even when your file selects Hungarian. In `--debug-tracks` mode, the
preview displays the selected strategy. Normal labels and exit controls remain.

The global objective maximizes valid match count, then minimizes total IoU cost.
It uses dummy unmatched nodes and forbidden costs above the all-unmatched solution.
Hungarian complexity is approximately O(n³) time and O(n²) space, with n=T+D for
the gated tracking matrix. It costs more than greedy, and cannot guarantee fewer
ID switches when geometry is ambiguous. Track IDs remain session-local.

`uv run --extra tracking python -m jake.benchmarks` now compares Kalman + greedy
and Kalman + Hungarian on deterministic crossing trajectories, reporting total
IDs, switches, and assignment time separately from filter/inference work. Smooth
crossings yield 2 IDs/0 switches for both; the jittered crossing yields 2 IDs/4
switches for both. A gated-bottleneck regression demonstrates a case where global
matching preserves two connections while greedy creates a third ID.

See [global assignment design](docs/global-assignment.md) for the algorithm,
penalty policy, complexity, evaluation method, and limitations. Phase 1A–1E have been physically validated; no appearance or identity recognition is added.

## Phase 1F: semantic person events

After the local model setup above, run:

```sh
uv run --extra detection jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --events
```

For an already-installed environment without dependency synchronization:

```sh
uv run --offline --no-sync jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --events
```

Add this optional table to your existing local config (do not duplicate a table):

```toml
[events]
present_interval_seconds = 5.0
```

Frames carry pixels; detections describe a single frame; tracks associate boxes
across frames; events describe track presence transitions. `--events` requires
`--track` and works with either tracker and either assignment strategy. The normal
preview remains available with q/Q and Ctrl+C cleanup. Console output is opt-in:

```text
[18:00:00.132] ENTERED track=1
[18:00:05.141] PRESENT track=1 duration=5.0s
[18:00:18.421] LEFT track=1 duration=18.3s
```

ENTERED is emitted once on first confirmed visibility (baseline trackers confirm
on the first detection; Phase 1G Kalman requires confirmation hits). PRESENT is throttled per track, defaults to five seconds,
and pauses while a track is missed. No catch-up burst occurs after a gap. LEFT is
emitted once when the tracker removes the ID after its missed-frame allowance.
Short occlusions preserve event state and do not generate another ENTERED.

Entry, departure, and duration derive exclusively from frame timestamps. LEFT
uses the expiration frame time, so duration includes the grace period and is an
approximation of physical presence. Backward timestamps fail clearly; equal times
are allowed. Shutdown does not fabricate departures. Tracker ID switches and
false detections can affect the event stream; track IDs are not resident identities.

Events contain metadata only and remain in memory; console logging does not
persist video, images, or event files. No network or later-phase functionality is
added. See [event state machine and timing design](docs/person-events.md) for the
public lifecycle extension, local UUIDs, timing rules, and limitations.

## Phase 1G: track stabilization

Live event testing exposed fragmentation. The normal `--tracker kalman` path now
uses three-hit confirmation, 1.5-second retention, and combined IoU/center-distance
matching. Tentative tracks emit no household events. Confirmed tracks become LOST
on misses; LEFT occurs only after timeout. Duration begins at confirmation.

```powershell
uv run --extra detection jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --events --debug-tracks
```

Debug labels show TENTATIVE hit counts, CONFIRMED, and LOST elapsed seconds.
Omit `--debug-tracks` for normal preview. Add the new settings from
`config/jake.example.toml` to your existing `[tracking]` table; old files use defaults.
`max_missed_frames` now applies only to `iou` and `kalman-baseline`.
Use `--tracker kalman-baseline` to reproduce the earlier Kalman behavior described
in Phase 1D–1F above; the Python `KalmanPersonTracker` also retains that baseline.
`StabilizedKalmanPersonTracker` is the new production adapter.

```sh
uv run --extra tracking python -m jake.stabilization_benchmark
```

On the deterministic dropout/occlusion/motion/jitter/crossing fixture, baseline vs
stabilized results are: 13 to 9 IDs, 13 to 2 confirmed IDs, 5 to 4 switches, 13 to 2
ENTERED, 13 to 2 LEFT, and 7 to 0 false short pairs. Four crossing switches remain;
track IDs are not physical identities. These synthetic results await live validation.
See [stabilization design and migration](docs/track-stabilization.md) for lifecycle,
cost/gating, defaults, timing, metric definitions, and limitations.

## Phase 2A: appearance-assisted continuity

Appearance now supplements Kalman prediction and spatial matching within the
current camera session. It does not recognize residents or faces. A separate
`AppearanceEncoder` encodes person crops; the tracking core receives immutable
vectors and stays independent of the model framework. Baselines remain available.

The first adapter uses CPU OpenVINO with the pretrained
`person-reidentification-retail-0287` body-appearance model: 256-dimensional,
L2-normalized embeddings. Weights are installed explicitly as a local XML/BIN pair;
there is no automatic download. Follow the exact
[model setup and privacy instructions](docs/appearance-tracking.md) first, then run:

```powershell
uv run --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --events --debug-tracks
```

Omit debug mode for normal preview; use `--no-appearance` for stabilized geometry.
`[tracking.appearance] enabled = true` also enables it with `--tracker kalman`.
The default remains disabled so existing configurations need no weights or runtime.
The runtime import has a process-wide telemetry block; run Jake in a fresh process.
Encoding time is displayed separately from detector inference. Embeddings remain
in memory, expire by capture-time age, and are removed with tracks. They never
enter semantic events. Appearance does not extend track retention beyond
`max_missed_seconds` or resurrect expired IDs.

```sh
uv run --extra tracking python -m jake.appearance_benchmark
```

Across six synthetic scenarios, geometry versus appearance produced 12/9 IDs,
9/2 switches, 9/2 fragmentation, and 54/18 wrong-association observations. Similar
outfits remain ambiguous. This tests matching with synthetic descriptors, not
real encoder accuracy; live pretrained-model validation remains pending. See the
[appearance design](docs/appearance-tracking.md) for model contract, cosine/EMA,
weights, gating, benchmark definitions, and limitations.

## Phase 2A.1: recently-lost re-identification

Confirmed appearance tracks can now remain in a recently-lost pool after normal
active retention. Strong appearance plus plausible motion can reactivate the same
ID within five seconds of its last observation. Event history stays open: no
premature LEFT or second ENTERED, and duration continues from original confirmation.
Final timeout produces one LEFT. Tentative tracks do not enter the pool.

With existing Phase 2A weights installed:

```powershell
uv run --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --reid --events --debug-tracks
```

Use `--no-reid` for the previous appearance baseline. The new example configuration
contains `[tracking.reid] enabled = true`; existing files without it remain disabled.
Debug text shows RECENTLY_LOST age and REACTIVATED similarity. Normal preview hides
pool boxes. All vectors remain in memory, subject to the existing appearance-age cap.

```sh
uv run --extra tracking python -m jake.reid_benchmark
```

Six synthetic sessions produced 14/7 IDs and 14/7 ENTERED/LEFT counts for baseline
versus ReID, with 6/0 switches and fragmentation. One same-clothing impostor case
falsely reactivated an ID, compared with none in the baseline. This is continuity
assistance, not verified identity. See [recently-lost lifecycle and design](docs/recently-lost.md)
for the exact time window, formula, migration, metric definitions, and limitations.

## Repository layout

```text
src/jake/
  domain.py       Immutable frames, boxes, detections, tracks, and events
  ports.py        Camera, detector, tracker, and event generator protocols
  config.py       Validated settings and explicit TOML loading
  pipeline.py     Dependency-injected Phase 1 orchestration
  adapters/
    opencv_camera.py  Context-managed local FrameSource implementation
    yolo_detector.py  Local Ultralytics PersonDetector implementation
    iou_tracker.py   Jake-owned PersonTracker and internal lifecycle state
    kalman_tracker.py  Motion-aware PersonTracker using Jake's filter
    person_events.py   Metadata-only EventGenerator and presence state machine
    openvino_appearance.py  Local CPU body-appearance encoder
  kalman.py       Reusable NumPy linear Kalman mathematics
  motion.py       Constant-velocity box model and dt policy
  benchmarks.py   Deterministic synthetic tracker comparison
  stabilization_benchmark.py  Synthetic tracking and event-quality comparison
  motion_matching.py  Stabilized spatial cost and gating
  matching.py     IoU geometry, strategy selection, and gated global matching
  hungarian.py    Jake-owned rectangular linear assignment solver
  diagnostics.py  Framework-independent detector timing
  appearance.py    Unit-vector math and encoder orchestration
  appearance_matching.py  Spatial and appearance cost policy
  appearance_benchmark.py  Synthetic continuity comparison
  reid_matching.py  Recently-lost appearance/spatial association
  reid_benchmark.py  Re-entry and event-continuity comparison
  event_console.py  Opt-in semantic event console formatting
  preview.py      Local OpenCV display and development overlay
  cli.py          jake-camera entry point
  py.typed        Package type information marker
config/
  jake.example.toml
tests/            Contract, configuration, and synthetic pipeline tests
.github/workflows/ci.yml
pyproject.toml    Packaging, development dependencies, and check configuration
uv.lock          Reproducible dependency resolution
```

## Architecture and contracts

Dependencies point inward: future camera SDKs and model adapters implement
`ports.py` and translate their native data into `domain.py` contracts. The core
imports no vision framework. Replacing a detector requires a new adapter and
composition change, without modifying the tracker or pipeline.

- `FrameSource` yields ordered frames from one camera session. The caller owns
  resource acquisition and cleanup. `OpenCVCamera` implements this protocol and
  requires a context manager. The pipeline neither opens nor closes a source.
- `Frame` uses immutable packed RGB8 bytes, a camera ID, a non-negative sequence,
  and a timezone-aware capture timestamp. Boxes use normalized coordinates.
- `PersonDetector` produces person detections. The pipeline filters confidence
  scores below the configured threshold before updating the tracker.
- `PersonTracker` receives metadata and detections, including empty updates, and
  returns the current active tracks. Track IDs are scoped to a camera session;
  they are not resident identities. `IoUPersonTracker` implements greedy IoU
  assignment and missed-frame expiry. Image-based tracking would require an
  explicit extension to the current metadata-only tracker contract.
- `EventGenerator` receives metadata and active tracks, including empty updates.
  `PersonEventGenerator` implements entered, periodic present, and left events
  with transition state, deduplication, and local UUID allocation.
- `PerceptionPipeline.process()` returns events for a single frame; `run()` pulls
  a source lazily and yields events. It validates camera isolation and strictly
  increasing frame sequences. Gaps are allowed. It stores only the last sequence.

Use fresh tracker, event generator, and pipeline instances per camera session.
Calls are serial; instances are not thread-safe. There is no internal queue,
automatic retry, or persistence. Adapter errors propagate, and the caller must
close resources and discard that session after a failure because adapter state
may already have changed. Source buffering and future reconnect supervision
belong to adapters and the application layer, respectively.

`OpenCVCamera` converts three-channel uint8 BGR images to packed RGB8 at the
adapter boundary, including non-contiguous image buffers. Capture timestamps are
UTC host times taken immediately after a successful read, not hardware exposure
timestamps. Sequence numbers start at zero and increase per delivered frame.
Adapters are single-use to prevent accidental session/sequence resets. Read or
conversion failures close the camera and raise `CameraError`; there is no retry
or reconnection loop. Use a `with` block even when breaking iteration early.
Reads are synchronous: a stalled native driver can delay Ctrl+C handling until
control returns to Python. Jake adds no frame queue, though the driver may buffer.

Acquisition follows OpenCV's [VideoCapture lifecycle](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html);
display is isolated in `preview.py` using [HighGUI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html).

## Configuration and composition

Copy `config/jake.example.toml` to `config/local.toml` (ignored by Git) for local
settings. Configuration loading is explicit and rejects unknown settings and
invalid values. No environment variables or files are read on package import.

To consume structured events without the development preview (requires the
detection extra and configured local model weights):

```python
from pathlib import Path

from jake.adapters.opencv_camera import OpenCVCamera
from jake.adapters.kalman_tracker import StabilizedKalmanPersonTracker
from jake.adapters.person_events import PersonEventGenerator
from jake.adapters.yolo_detector import YoloPersonDetector
from jake.config import load_app_config
from jake.event_console import log_event
from jake.pipeline import PerceptionPipeline

config = load_app_config(Path("config/local.toml"))
detector = YoloPersonDetector(config.detector, config.pipeline.min_person_confidence)
tracker = StabilizedKalmanPersonTracker(config.tracking)
event_generator = PersonEventGenerator(config.events)
pipeline = PerceptionPipeline(config.pipeline, detector, tracker, event_generator)
with OpenCVCamera(config.pipeline.camera_id, config.camera) as source:
    for event in pipeline.run(source):
        log_event(event)  # Replace with your metadata consumer.
```

The preview CLI uses the source directly; it does not instantiate this perception
pipeline or fake detections. Existing callers of `load_config()` still receive
`PipelineConfig`; `load_app_config()` also exposes `CameraConfig`, `DetectorConfig`,
`TrackingConfig`, and `EventConfig`. Legacy configuration defaults to camera zero and the
documented detector/tracker settings.
Weights are checked only when detection is enabled. Both loaders
validate the entire file and reject unknown sections/settings.

## Privacy boundaries

The current core performs no network access, telemetry, disk writes, or model
downloads. Pixels stay in the detection stage and are omitted from frame repr;
tracking and event generation receive metadata only. Optional tracking appearance
vectors stay out of public tracks and semantic events. Events carry only an ID,
event kind, camera/frame/time context, session-local track ID, and entry timestamp. The core does
not retain frames after processing; callers and adapters control their own memory.
This is data minimization, not a guarantee of secure memory erasure.

Future adapters must be audited for network access, telemetry, logging, buffering,
and retention. Python protocols are not a security sandbox. Even event metadata
can reveal household activity. Persistence, retention/deletion controls, access
control, and explicit consent remain requirements for future integrations. Phase 2B
adds deliberate consent-based enrollment and deletion; its store contains sensitive
face templates, never crops, and its identity events never contain vectors.
Local data, recordings, model files, and secret settings
are ignored by Git; ignore rules are not an access-control mechanism.

## Roadmap

The Phase 1 foundation, 1A acquisition, 1B detection, 1C IoU tracking, 1D
Kalman-assisted tracking, 1E global assignment, 1F semantic events, and 1G stabilization are implemented.
Phase 1A was physically validated on Windows at approximately 19 FPS, 640×480,
with advancing sequences and successful shutdown. Phase 1B was also physically
validated with multiple people and CPU inference fast enough for development.
Phase 1C has been physically validated with multiple people; crossing/occlusion
ID losses establish the baseline. Phase 1D has been physically validated and works.
Phase 1E has also been physically validated. Phase 1F is tested synthetically and
awaits physical validation. The sequence below is a planning outline.

| Phase | Planned capabilities |
| --- | --- |
| 1 — perception | Foundation through 1G track stabilization implemented |
| 2 — recognition | 2A/2A.1 body continuity, 2B resident identity, 2C encrypted storage, and 2D opt-in anonymous recurring visitor memory implemented; no automatic naming or visitor classification |
| 3 — understanding and memory | Activity recognition, event memory, household behavioral learning, anomaly detection, and governed continual learning |
| 4 — voice and interaction | Speech recognition, text-to-speech, basic conversational AI, context-aware resident greetings, and daily/event summaries |
| 5 — multiple hubs | Privacy-preserving context coordination and conversational handoff between household hubs |

Keep new model frameworks behind adapters and add dependencies only with a
concrete integration. Later phases should introduce their own contracts when
requirements are established, rather than expanding the perception core early.
