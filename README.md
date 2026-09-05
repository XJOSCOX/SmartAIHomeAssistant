# Jake — SmartAIHomeAssistant

Jake is the foundation for a privacy-first, local-first smart home AI system.
The intended system will understand household events locally and eventually
support natural conversation. **This repository currently contains architecture
and orchestration only. It does not connect to cameras or run AI models.**

## Phase 1 scope

The initial boundary is:

```text
Camera adapter → Frame → PersonDetector → PersonTracker → EventGenerator → caller
                    RGB      detections       tracks          metadata
```

The production package defines immutable data contracts, structural interfaces,
validated TOML configuration, and synchronous pipeline orchestration. Synthetic
adapters live only in tests. There are no model downloads, network clients,
recordings, databases, recognition algorithms, or background services.

## Development

Requires Python 3.11+ and `uv`. From the repository root:

```sh
uv sync --locked --group dev
uv run pytest --cov=jake --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

`.python-version` selects Python 3.11; `uv` can provision it if necessary.
`uv.lock` pins development dependencies. Runtime dependencies are currently empty.
For a different supported interpreter, use `uv sync --python 3.13 --locked --group dev`.
CI runs the checks and packaging on Python 3.11–3.14.

## Repository layout

```text
src/jake/
  domain.py       Immutable frames, boxes, detections, tracks, and events
  ports.py        Camera, detector, tracker, and event generator protocols
  config.py       Validated settings and explicit TOML loading
  pipeline.py     Dependency-injected Phase 1 orchestration
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
  resource acquisition and cleanup; a real camera adapter should expose a context
  manager. The pipeline neither opens nor closes a source.
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

## Configuration and composition

Copy `config/jake.example.toml` to `config/local.toml` (ignored by Git) for local
settings. Configuration loading is explicit and rejects unknown settings and
invalid values. No environment variables or files are read on package import.

The following is a composition sketch **requiring user-supplied adapters**:

```python
from pathlib import Path

from jake.config import load_config
from jake.pipeline import PerceptionPipeline

config = load_config(Path("config/local.toml"))
# detector, tracker, and event_generator implement the protocols in jake.ports.
pipeline = PerceptionPipeline(config, detector, tracker, event_generator)
# Within the source adapter's resource context:
for event in pipeline.run(source):
    handle_event(event)
```

No runnable camera CLI is supplied until concrete adapters and lifecycle behavior
are implemented. Camera credentials must remain outside version control and
should never appear in logs or committed configuration.

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

Only the Phase 1 foundation is implemented. The sequence below is a planning
outline, not a promise that later phases already exist.

| Phase | Planned capabilities |
| --- | --- |
| 1 — perception | Local camera acquisition, replaceable person detection, tracking, event generation; next work is concrete adapters, lifecycle handling, and integration validation |
| 2 — recognition | Resident recognition, frequent visitor recognition, delivery/visitor classification, with consent and identity-data controls |
| 3 — understanding and memory | Activity recognition, event memory, household behavioral learning, anomaly detection, and governed continual learning |
| 4 — voice and interaction | Speech recognition, text-to-speech, basic conversational AI, context-aware resident greetings, and daily/event summaries |
| 5 — multiple hubs | Privacy-preserving context coordination and conversational handoff between household hubs |

Keep new model frameworks behind adapters and add dependencies only with a
concrete integration. Later phases should introduce their own contracts when
requirements are established, rather than expanding the perception core early.
