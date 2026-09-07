# Phase 3A: local voice foundation

Status: implemented for repository review. Validation uses fake devices and models;
no physical microphone/speaker test or real speech inference has been performed.

## Architecture

```text
PortAudio callback -> bounded PCM queue -> WebRTC VAD / utterance segmentation
 -> SpeechSegment -> faster-whisper -> SpeechRecognitionResult
 -> ConversationManager -> deterministic response -> bounded async Piper queue
 -> AudioOutput / speaker

Existing camera pipeline -> metadata-only VisualVoiceBridge -> latest visual context
```

`voice_domain.py` defines immutable audio, speech, visual-context and event contracts.
`voice_ports.py` defines AudioInput, VoiceActivityDetector, SpeechRecognizer,
SpeechSynthesizer and AudioOutput protocols. The domain and conversation policy do
not import speech engines. `application/voice.py` owns orchestration and asynchronous
`speak(text)`, `stop()` and `is_speaking`; adapters translate native formats.
Camera adapters contain no speech logic. Resident and visitor stores are unchanged.

The capture callback queues 30 ms mono PCM16 little-endian chunks at 16 kHz.
The voice worker segments and transcribes utterances; a separate worker synthesizes
and plays responses. Optional camera processing runs on its own worker and publishes
only the latest metadata snapshot, never images or embeddings. Native CPU workloads
can still compete for resources even though perception never waits for TTS.

Shutdown stops input, cancels queued output, aborts playback, clears buffers and
session state, and joins owned workers. Native model inference and camera reads are
cooperatively stopped: shutdown can wait for an in-progress native call to return.

## Component choices

- **STT: faster-whisper / CTranslate2**, with a local `base.en` model and CPU INT8 by
  default. Its quantized CPU path and optional CUDA path provide a practical adapter
  foundation without making PyTorch part of the speech domain. See the
  [upstream implementation and hardware requirements](https://github.com/SYSTRAN/faster-whisper)
  and [base.en model](https://huggingface.co/Systran/faster-whisper-base.en).
- **VAD: WebRTC**, using `webrtcvad-wheels`. It needs no separately downloaded neural
  weights and accepts the selected 16 kHz, 30 ms PCM format. Silero remains a possible
  replacement behind the same port; this phase favors a small deployment footprint.
  [Supported formats](https://github.com/wiseman/py-webrtcvad),
  [wheel distribution](https://github.com/daanzu/py-webrtcvad-wheels).
- **TTS: Piper**, local ONNX CPU synthesis, with the selected voice's native output
  sample rate. The adapter uses the maintained streaming Python API but bounds the
  collected output before playback. See [Piper](https://github.com/OHF-Voice/piper1-gpl)
  and its [Python API](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md).
  Review engine and selected voice licenses when distributing a package.
- **Devices: sounddevice / PortAudio**, using raw streams rather than audio files.
  [Raw stream API](https://python-sounddevice.readthedocs.io/en/latest/api/raw-streams.html).

These choices have not been benchmarked on the user's microphone or edge hardware.
The ports allow replacing engines without changing conversation or perception logic.

## Configuration and explicit model setup

The complete opt-in schema is in `config/jake.example.toml`. Existing configurations
remain valid and default to audio and both greeting policies disabled. Unknown fields,
invalid device values, unsupported rates/channel counts, non-finite durations and
out-of-range limits are rejected. WebRTC uses `vad_mode = 0..3`, not a probability
threshold; VAD cannot be disabled in this phase. Language config selects transcription
language, but the current intent vocabulary and response templates are English.

The following commands are documented for **after repository review**, not a request
for physical testing now. Run from the repository root. Install the voice extra:

```powershell
uv sync --extra voice
```

For an environment also used for vision/desktop, retain its extras when syncing:

```powershell
uv sync --extra voice --extra desktop --extra detection --extra appearance --extra identity
```

Explicit setup downloads require network access. These are separate tools, not Jake
runtime behavior:

```powershell
uv run --extra voice python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-base.en', local_dir='models/speech/faster-whisper-base.en')"
uv run --extra voice python -m piper.download_voices en_US-lessac-medium --download-dir models/speech
```

Alternatively provision existing local assets and configure their absolute paths:

- STT: a CTranslate2 model directory containing `model.bin`, `config.json` and
  `tokenizer.json`; retain the other files from the model snapshot too.
- TTS: `en_US-lessac-medium.onnx` and adjacent
  `en_US-lessac-medium.onnx.json` (or another compatible local voice).

The runtime requires these files before loading; it does not interpret a model name
as permission to download. Hugging Face offline mode and telemetry suppression are
set before STT loading, and `local_files_only=True` is passed explicitly. Requiring
`tokenizer.json` prevents a tokenizer fallback download. ONNX telemetry is disabled.
Only bundled eSpeak/text Piper phonemizers are accepted; optional diacritization is
disabled to avoid language-helper downloads. No network setup happens on first speech.

Add the example voice tables to an existing `config/local.toml`, preserving camera,
tracking and identity settings, then explicitly set `[audio] enabled = true` when
ready for reviewed testing. Do not overwrite an existing local configuration. Even
voice-only configurations retain the existing required `[pipeline]` table.

Device discovery does not require a config, model, microphone stream or speaker stream:

```powershell
uv run --extra voice jake-voice --list-devices
```

It reports descriptions, host APIs, channel counts, native indexes and default rates.
Omit device fields for OS defaults, or choose an index or description accepted by
PortAudio. Indexes are not portable across Windows/Linux or device changes;
descriptions are best-effort and may be ambiguous. Phase 3A requires input support
for 16 kHz mono; it does not silently resample incompatible hardware.

Exact standalone command:

```powershell
uv run --extra voice jake-voice --config config/local.toml
```

Integrated camera/voice command (no desktop required):

```powershell
uv run --extra voice --extra detection --extra appearance --extra identity jake-voice --config config/local.toml --with-camera
```

This uses existing configured vision models, tracker, identity/visitor opt-ins and
semantic events on a separate worker. Add `--preview` for an annotated development
window; existing preview commands are preserved. Ctrl+C shuts down the session. Desktop settings preserve
voice config, but desktop audio controls, voice autostart and packaged voice runtimes
are not added in this phase.

## Segmentation, quality and half-duplex behavior

The segmenter transitions IDLE -> LISTENING -> FINALIZING; the service then reports
TRANSCRIBING. Defaults require 240 ms of voiced chunks, retain approximately 180 ms
pre-roll, end after 600 ms silence and cap an utterance at 15 seconds. All boundaries
are quantized to 30 ms chunks. A sequence gap or discontinuous timestamp resets the
utterance rather than joining unrelated audio. Long continuous speech is split at
the cap; brief noises fail the minimum-speech gate.

STT consumes the finalized PCM segment only. It rejects segments with high no-speech
probability (>0.6), low average log probability (<-1), or high compression ratio
(>2.4), and rejects empty/nonprintable/oversized text. These are model adapter quality
heuristics, not a calibrated transcript confidence. `confidence` remains `None`.
Whisper can still hallucinate on noise; these gates are not proof that speech occurred.

Recognition is suppressed while output is queued, synthesized or played and for a
300 ms post-playback guard. Buffered microphone data is discarded, including capture
backlog after transcription. This is intentionally **half-duplex**: talking over Jake
or during STT can lose speech. There is no acoustic echo cancellation or barge-in.
Long room echoes may require a larger `echo_guard_ms`.

## Conversations, identity and greetings

ConversationManager supports GREETING, DELIVERY, WHO_ARE_YOU, IS_ANYONE_HOME,
GOODBYE and UNKNOWN with deterministic templates. The occupancy question does not
disclose who is home. No response operates devices or performs emergency actions.

A session has a local UUID, timestamps, optional visual association and a bounded
turn deque. Defaults retain 20 turn entries (user and Jake each count), with a
60-second inactivity timeout. Goodbye, timeout, association changes and shutdown
clear retained turns. No persistent conversation history is created.

At utterance start, association requires a fresh (at most two seconds old), non-future
snapshot with exactly one visible person, who must be confirmed. Multiple visible
people, including an additional tentative track, leave association unresolved.
This is a visual context assumption, **not evidence of who spoke**: an off-camera
speaker remains possible. No diarization, voiceprint or acoustic localization exists.

Both greeting policies default off. When enabled:

- A visible confirmed semantic arrival plus stable RESIDENT metadata can produce
  `Welcome home, <display name>.` Identity may stabilize after the arrival event.
- A semantic arrival plus confirmed visitor metadata and UNKNOWN resident state
  can produce a generic visitor greeting. Recurring/frequent/known visitor classes
  are not announced, and this policy does not speak visitor names.
- A track is greeted at most once while its semantic lifecycle remains active.
  A stable resident/visitor ID cooldown (60 minutes default) suppresses a new track's
  greeting after flicker or re-entry. Retained lost tracks preserve arrival context.

Cooldowns are process-memory only and reset on restart. Semantic arrivals mean camera
session arrivals, not proof someone crossed a home's entrance. Greetings skipped when
the speech queue is full are not retried indefinitely.

## Privacy, bounds and diagnostics

No raw audio files, transcript files, voiceprints, image copies or network requests
are produced by the voice workflow. Resident/visitor encrypted storage and schemas
are untouched. Domain reprs hide PCM and transcript text. Generic logs/events contain
metadata only; worker failures expose error classes, not exception payloads.
Python/native memory release is not a guarantee of cryptographic memory erasure.

Memory limits include:

- Input queue: 16 chunks / 480 ms default, maximum 64; overflow drops oldest audio.
- Utterance: 15 seconds default, configurable up to 30; bounded pre-roll.
- Speech output: at most four pending requests including active work, 500 characters
  per request and at most 60 seconds PCM per synthesis result.
- Conversation: at most 20 turn entries by default; generic event buffers 128 entries.
- Visual context: latest snapshot only; greeting cooldowns at most 1,024 entries,
  declining new greetings if the live cache is full.

SPEECH_STARTED, SPEECH_RECOGNIZED, CONVERSATION_STARTED, CONVERSATION_ENDED and
JAKE_SPOKE carry timestamps and optional local conversation/track IDs, never audio,
transcripts or biometric vectors. JAKE_SPOKE follows completed output playback.

CLI metrics show input status, VAD state, utterance duration, STT processing latency,
TTS generation latency and real-time factor. STT timing includes consumption of lazy
model segments; TTS timing excludes speaker playback. Both use a high-resolution
monotonic timer. `RTF = STT processing seconds / PCM duration seconds`; RTF < 1 means
transcription completed faster than the supplied audio duration. It is not end-to-end
response latency and excludes capture, silence timeout, queueing and playback.

## Platform expectations

Windows Python 3.11 dependency imports and installed API signatures were checked with
faster-whisper 1.2.1, Piper 1.8.0, sounddevice 0.5.6 and webrtcvad-wheels 2.0.14.
No hardware compatibility or model accuracy claim follows from that import check.
CPU INT8 STT with two threads is the default; Piper uses CPU and may have its own
native thread scheduling. CUDA STT requires a compatible CTranslate2/CUDA/cuDNN stack.

Jetson is a future deployment target, not a validated build: ARM64 wheels/native
builds and GPU libraries must match JetPack, and vision/audio memory contention needs
measurement. Raspberry Pi likewise needs a suitable 64-bit native stack; CPU tiny.en
or base.en can be evaluated behind the same port, but real-time performance is not
promised. Linux may require a system PortAudio package such as `libportaudio2`;
see [sounddevice installation](https://python-sounddevice.readthedocs.io/en/latest/installation.html).

## Changed files and validation

New contracts/configuration: `voice_domain.py`, `voice_ports.py`, `voice_config.py`.
New policy/segmentation: `conversation.py`, `voice_segmentation.py`.
New adapters: `adapters/local_speech.py`, `adapters/sounddevice_audio.py`.
New application services: `application/voice.py`, `voice_binding.py`,
`voice_camera.py`, `voice_composition.py`; new entry point: `voice_cli.py`.
Modified integration: `config.py`, `application/settings.py`, `pyproject.toml`,
`uv.lock`, `config/jake.example.toml`, `README.md`. New documentation: this guide.
New tests: `tests/test_voice.py`, `tests/test_voice_adapters.py`.

Tests fake native devices, VAD and model boundaries, including bounded buffering,
segmentation, transcript rejection, association, greetings, echo suppression,
shutdown, metadata-only camera integration and offline model loading. Adapter tests
reject socket connections. The existing perception/identity/visitor suite remains
part of full validation. Physical microphone/model testing is deferred for review.

Final repository validation: **738 tests passed; 93% total coverage**. Ruff check,
Ruff format check (111 files), mypy (97 source files), and source/wheel package build
all passed. Windows dependency import/API checks passed without loading model weights
or opening devices. These checks establish repository readiness for review, not
physical microphone or edge-device validation.


## Phase 3A.1 integrated annotated preview

For subsequent validation after review (not a request to run hardware now):

```powershell
uv run --extra voice --extra detection --extra appearance --extra identity jake-voice --config config/local.toml --with-camera --preview
```

`--preview` requires `--with-camera`; argparse rejects it otherwise before opening
models or devices. Without `--preview`, integrated voice remains headless.

The existing camera worker publishes one latest immutable frame/track/context snapshot.
The main thread renders it using OpenCV and the existing person-box drawing helper.
There is exactly one camera and one visual pipeline; rendering performs no detection,
tracking or recognition. Slow rendering skips intermediate snapshots instead of
queueing frames. Camera and voice workers never wait for rendering; the snapshot
lock is held only to exchange a reference. No audio, VAD, STT, TTS, greeting or store
behavior changes. Frames are memory-only and released at shutdown; no screenshot or
recording feature is present.

The window shows visible person boxes and track IDs, resident name/state or visitor
short ID/state, actual delivered frame dimensions, sequence and processing FPS.
Processing FPS is the reciprocal of the visual pipeline processing time, excluding
camera acquisition, preview rendering and audio processing; it is not delivered FPS.

The header displays, for example, `VOICE CONTEXT: Joseph | RESIDENT | track=4`.
It uses the exact metadata snapshot sent to the voice bridge and the existing
association function, recalculating freshness on each redraw. Multiple visible
people, tentative-only or stale/future snapshots display `VOICE CONTEXT: unresolved`.
Visitor/unknown context is labeled accordingly. This is available visual context,
not the currently speaking person's verified identity or an active session's binding.
The window explicitly labels this limitation; there is no acoustic identification.

Lowercase `q` or uppercase `Q` stops the complete integrated session, releases the
camera and audio resources and destroys the preview window. Ctrl+C follows the same
cleanup path. Native model calls still finish cooperatively as described above.

Regression tests fake HighGUI, devices and models. They cover argument validation,
one-session snapshot reuse, resident/visitor/unknown/ambiguous/stale labels, both quit
keys, Ctrl+C cleanup, immutable frame rendering and continued VAD/STT/TTS processing
while a renderer is deliberately blocked. No physical camera/audio test was performed.


## Shared perception settings and explicit overrides

Integrated voice and the normal camera CLI now resolve settings through
`application/perception_session.py`: immutable `PerceptionOverrides` -> resolved
`PerceptionSession` -> the existing adapter composer and event-generator factory.
No face check, tracker or identity pipeline is duplicated for the voice preview.

The integration mismatch was that `jake-camera` applied runtime flags before composing
adapters, whereas voice mode read only TOML and hard-coded the default tracker mode.
An enrolled profile does not itself enable recognition. Flags used on another command
are not persisted into TOML. The bridge already propagated resident names correctly;
it cannot produce RESIDENT when the upstream identity stage is disabled.

Both commands now support explicit `--identity` / `--no-identity`, `--visitors` /
`--no-visitors`, `--appearance` / `--no-appearance`, `--reid` / `--no-reid`,
`--tracker {kalman,kalman-baseline,iou}` and `--assignment {greedy,hungarian}`.
Omitted boolean/assignment overrides honor TOML. Overrides affect this session only;
no configuration files or resident/visitor templates are rewritten.

Voice retains its production `kalman` default; the camera CLI retains its historical
`iou` default and requires `--track` for tracking. Tracker mode is an explicit runtime
selection, not a new TOML field. For command parity, specify the same tracker mode.
Appearance and ReID settings remain dormant in geometry baseline modes, preserving
previous baseline behavior. Explicit incompatible appearance/ReID requests are rejected.
Explicit `--reid` implies appearance unless `--no-appearance` conflicts with it.
Visitor memory always enables resident-first screening, even with `--no-identity`;
use `--no-visitors --no-identity` to disable both. Events are always present in the
integrated session and are also enabled when camera visitor memory needs them.

Exact integrated command matching the previously flag-driven stack, for future
validation after repository review:

```powershell
uv run --extra voice --extra detection --extra appearance --extra identity jake-voice --config config/local.toml --with-camera --preview --tracker kalman --assignment hungarian --appearance --reid --identity --visitors
```

Alternatively retain the short `--with-camera --preview` command and explicitly set
`[identity] enabled = true`, `[visitors] enabled = true`,
`[tracking.appearance] enabled = true`, `[tracking.reid] enabled = true`, and
`[tracking] assignment = "hungarian"` in the chosen TOML file. These are opt-ins;
Jake does not enable biometric features merely because profiles exist.
Voice perception flags require `--with-camera` and are validated before opening audio.
The normal camera equivalents still require `--track`.

Tests exercise actual temporal resident confirmation using an in-memory enrolled
profile and fake face-model/device boundaries, then verify the same pipeline result
reaches the bridge and preview. They also cover disabled identity and explicit
overrides, common CLI configuration, resident-first visitor support and all tracker
modes. Existing multi-person unresolved and single-camera tests remain in the suite.
No physical camera, microphone or real biometric store was accessed for this fix.
