# Jake — SmartAIHomeAssistant

Jake is the foundation for a privacy-first, local-first smart home AI system.
The intended system will understand household events locally and eventually
support natural conversation. **Phase 1A supports a local USB/webcam and live
development preview. No AI models or perception algorithms run yet.**

## Phase 1 scope

The initial boundary is:

```text
Camera adapter → Frame → PersonDetector → PersonTracker → EventGenerator → caller
                    RGB      detections       tracks          metadata
```

The production package defines immutable data contracts, structural interfaces,
validated TOML configuration, synchronous pipeline orchestration, and a concrete
OpenCV camera adapter. Detector, tracker, and event-generator fakes live only in
tests. There are no model downloads, network clients,
recordings, databases, recognition algorithms, or background services.

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
The confidence setting is reserved for the perception pipeline and has no effect
on this preview.

```sh
uv run --extra vision jake-camera --config config/local.toml
```

Equivalent module command:

```sh
uv run --extra vision python -m jake.cli --config config/local.toml
```

The window shows sequence number, dimensions, and average delivered FPS since
preview startup (not the camera's advertised rate). Focus the preview window and
press **q** to exit, or press **Ctrl+C** in the terminal. Both paths release the
camera and destroy the preview window. No video or images are recorded or saved.
Exit codes are 0 for normal exit/Ctrl+C, 1 for camera/display errors, and 2 for
configuration or missing-dependency errors. `jake-camera --help` does not load OpenCV.

If the device cannot open, check the index, close other apps using it, and enable
camera access for desktop apps/your terminal in OS privacy settings. On Linux,
check access permissions for the camera device. A graphical session and OpenCV
GUI libraries are required; `opencv-python-headless` cannot provide this preview.
Avoid installing multiple OpenCV package variants in the same environment.
Package installation may use the network; camera acquisition and preview do not
make network calls, upload data, or emit telemetry.

## Repository layout

```text
src/jake/
  domain.py       Immutable frames, boxes, detections, tracks, and events
  ports.py        Camera, detector, tracker, and event generator protocols
  config.py       Validated settings and explicit TOML loading
  pipeline.py     Dependency-injected Phase 1 orchestration
  adapters/
    opencv_camera.py  Context-managed local FrameSource implementation
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
  they are not resident identities. Occlusion, expiry, and assignment policies
  belong to a future implementation. Image-based tracking would require an
  explicit extension to the current metadata-only tracker contract.
- `EventGenerator` receives metadata and active tracks, including empty updates.
  The contract supports entered, updated, and left events. Transition logic,
  deduplication, and event ID allocation are not implemented yet.
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

The following is a composition sketch **requiring user-supplied adapters**:

```python
from pathlib import Path

from jake.adapters.opencv_camera import OpenCVCamera
from jake.config import load_app_config
from jake.pipeline import PerceptionPipeline

config = load_app_config(Path("config/local.toml"))
# detector, tracker, and event_generator implement the protocols in jake.ports.
pipeline = PerceptionPipeline(config.pipeline, detector, tracker, event_generator)
with OpenCVCamera(config.pipeline.camera_id, config.camera) as source:
    for event in pipeline.run(source):
        handle_event(event)
```

The preview CLI uses the source directly; it does not instantiate this perception
pipeline or fake detections. Existing callers of `load_config()` still receive
`PipelineConfig`; `load_app_config()` also exposes the separate `CameraConfig`.
Legacy configuration without `[camera]` defaults to device zero. Both loaders
validate the entire file and reject unknown sections/settings.

## Privacy boundaries

The current core performs no network access, telemetry, disk writes, or model
downloads. Pixels stay in the detection stage and are omitted from frame repr;
tracking and event generation receive metadata only. Events carry only an ID,
event kind, camera/frame/time context, and session-local track ID. The core does
not retain frames after processing; callers and adapters control their own memory.
This is data minimization, not a guarantee of secure memory erasure.

Future adapters must be audited for network access, telemetry, logging, buffering,
and retention. Python protocols are not a security sandbox. Even event metadata
can reveal household activity. Persistence, retention/deletion controls, access
control, and explicit consent for recognition must be designed before those
features are enabled. Local data, recordings, model files, and secret settings
are ignored by Git; ignore rules are not an access-control mechanism.

## Roadmap

The Phase 1 foundation and Phase 1A local acquisition are implemented. The sequence below is a planning
outline, not a promise that later phases already exist.

| Phase | Planned capabilities |
| --- | --- |
| 1 — perception | Foundation and 1A local camera acquisition/preview complete; person detection, tracking, and event-generation algorithms remain future work |
| 2 — recognition | Resident recognition, frequent visitor recognition, delivery/visitor classification, with consent and identity-data controls |
| 3 — understanding and memory | Activity recognition, event memory, household behavioral learning, anomaly detection, and governed continual learning |
| 4 — voice and interaction | Speech recognition, text-to-speech, basic conversational AI, context-aware resident greetings, and daily/event summaries |
| 5 — multiple hubs | Privacy-preserving context coordination and conversational handoff between household hubs |

Keep new model frameworks behind adapters and add dependencies only with a
concrete integration. Later phases should introduce their own contracts when
requirements are established, rather than expanding the perception core early.
