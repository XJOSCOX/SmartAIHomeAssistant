# Phase 3B: contextual conversational AI

Status: implemented for repository review. The Windows runtime imports successfully;
no GGUF weights were downloaded, loaded or tested, and no devices or real biometric
stores were opened for this phase.

## Scope and language boundary

Jake now uses a local model to interpret the current message with bounded prior turns
and structured visual context. This is a deliberately constrained conversational
foundation: the model selects an appropriate sentence from a small approved response
set. It can handle greetings, acknowledgments, clarification and descriptions of
configured capabilities, including follow-up wording interpreted using history.
It is **not unrestricted question answering or general-purpose free-form chat**.

A prompt or keyword filter cannot guarantee that arbitrary generated prose avoids
invented residents, occupancy leaks or fictional device actions. This phase therefore
uses a JSON response schema with allowed wording and an independent exact-membership
policy check. Invalid output becomes a generic fallback. Replacing this boundary with
open-ended text requires a separately reviewed policy design; it is not a config toggle.

Greeting personalization is inserted by deterministic code after validation. For a
resolved, still-fresh resident context, a selected `Hello. How can I help you?` may
become `Hello Joseph. How can I help you?` once per conversation. Other replies do
not repeatedly add the name. Visitors receive generic wording; known-visitor names
and visitor classifications are never spoken by this policy.

## Architecture

```text
STT result -> ConversationManager.begin
           -> immutable context builder
           -> high-priority deterministic guards
           -> single bounded ConversationWorker (when no guard applies)
           -> ConversationModel.generate
           -> Response policy / current visual-context check
           -> ConversationManager.finish -> asynchronous TTS
```

- `conversation_domain.py`: immutable ConversationContext, ConversationRequest,
  ConversationResponse, SpeakerContext, ScenePerson and ModelStatus; ConversationModel
  protocol with generate/cancel/status/close. No model vendor appears in these contracts.
- `conversation_context.py`: context assembly, central prompt construction, capability
  catalogue, deterministic precedence, response validation and greeting personalization.
- `conversation_ai_config.py`: strict, independent opt-in configuration.
- `adapters/llama_conversation.py`: lazy embedded GGUF load, token budgeting, schema
  constrained generation, cancellation hook, native metrics and state cleanup.
- `application/conversation_worker.py`: one active generation and bounded handoff.
- Existing ConversationManager retains session IDs, expiry and memory-only turns.
  Existing voice composition/worker/CLI connect the optional path. Vision, identity,
  visitor algorithms, encrypted stores, STT, TTS, VAD and arrival greetings are unchanged.

When disabled, the original deterministic responses and existing Phase 3A flow remain
in use. Merely installing the optional dependency does not load a model or enable AI.

## Context and prompt

The context contains:

- Conversation ID and local timestamp using the configured `[home] timezone` via
  Python ZoneInfo, with no internet time lookup.
- Speaker association status (`visual_only` or `unresolved`), track ID, identity state,
  authorized resident ID/name, and visitor ID/state where appropriate.
- Visible track metadata and resident/visitor counts from the fresh visual snapshot,
  plus the snapshot timestamp. Names of other visible people are not forwarded.
- Bounded prior turns and a list of supported/unsupported capabilities. Standalone
  voice does not claim visual recognition capabilities; integrated CLI overrides
  are resolved before those capabilities are assembled.

The existing exactly-one-visible-confirmed-person rule remains authoritative.
A second visible person, tentative-only person, stale/future context or absent camera
leaves association unresolved. User statements such as `I am Joseph` never create
identity. Candidate names are not treated as resolved resident names. Context is
associated at speech onset, including the actual VAD onset timestamp rather than
pre-roll start. Before speaking a personalized response, fresh visual evidence must
still match the same resident/track. This does not identify speakers acoustically.

Prompts use a fixed Jake persona and explicitly serialized JSON sections for identity,
scene, capabilities, local time, prior conversation and current text. No Python repr,
configuration dump, image, audio, embedding, key or store path is serialized. JSON
escapes control characters and chat-delimiter angle brackets. User text and names
remain data, not trusted instructions; output enforcement does not depend on escaping
alone. Prompts and raw responses are not logged.

The backend counts prompt-section tokens with its own tokenizer and reserves space
for chat-template overhead and output. It drops the oldest history exchanges when
needed. If the current request still cannot fit, generation fails safely. No persistent
summary is created. The backend's returned usage, rather than that budgeting estimate,
is used for reported prompt/output token counts.

## Deterministic precedence and policy

Before generation, occupancy/location questions (including `Is Joseph home?`), emergency
requests, unsupported controls, private system/biometric questions, goodbye and delivery
receive fixed responses. These broad guards deliberately favor refusal over disclosure;
phrasing that mentions `home`, residents or visitors can receive a conservative refusal.
Goodbye closes the session. Delivery instructions never claim a package was received.
These additional guards apply only when conversation AI is enabled, preserving Phase 3A
behavior when disabled.

After generation, only an exact allowed sentence passes. Capability sentences are
available only for enabled integration capabilities. Arbitrary names, visitor labels,
claims about who is present, unlock/lock actions, calls to police, package receipt,
historical events, prompts and secrets are not valid generated responses. A model
cannot invoke tools: no tool execution surface or home-control adapter was introduced.

The approved set is intentionally small. A tiny model may still choose an irrelevant
but allowed reply; the policy limits what it can claim, not whether every selection
is conversationally useful. No model-quality or response-latency claim has been
established without physical model evaluation.

## Bounded execution, expiry and failures

The camera worker, native microphone callback, voice/VAD worker, conversation worker
and TTS worker have separate ownership. LLM load/generation happens only on the
conversation worker. One job, including its pending result, occupies the slot; no
unbounded utterance queue exists. VAD continues during generation, but additional
completed utterances are discarded until the answer is ready. This phase has no
barge-in. Existing half-duplex TTS suppression remains unchanged.

Model load is lazy, on the first non-guarded request. The default 20-second request
budget includes first load. Generation errors and timeouts return `I'm having trouble
answering that right now.` Subsequent unguarded requests use the fallback after a
backend failure/timeout; restart is required to retry the model. Deterministic guards
remain available. Rejected wording also falls back, without necessarily disabling
the otherwise healthy backend.

Cancellation is terminal for an adapter instance, avoiding races that would clear an
in-flight cancellation signal. Shutdown cancels generation and joins the owner before
closing the native model. Cancellation is checked through llama.cpp's supported logits
processor hook and around load/prompt operations. Native loading or prompt evaluation
may finish its current operation before stopping; this is cooperative cancellation,
not a promise of hard preemption. Timeout can deliver a fallback while native work is
finishing, without starting another generation.

The existing conversation timeout clears manager history. If expiry occurs during
inference, active generation is cancelled and its response cannot attach to a new
session. A cancelled/expired request releases its remaining worker references when
the cooperative native call returns. No transcript is saved to disk. Native attention
state is reset between requests; explicit bounded history supplies continuity.
Memory cleanup is not a guarantee of secure erasure from native/OS memory.

## Metrics and safe diagnostics

The CLI prints AI enabled/disabled, worker state, load milliseconds, generation
milliseconds, backend prompt/output token counts, tokens/sec and fallback-used status.
Missing metrics are `None`, never invented zero counts or rates. Generation latency
includes prompt preparation/evaluation and output generation, excluding model load;
a failed request's latency is elapsed request time. Load time is reported separately.
This non-streaming adapter does not expose isolated decode timing, so tokens/sec is
currently unavailable rather than presenting total request throughput as decode speed.

The integrated preview continues to show the source-of-truth speaker context and
identity diagnostics. No system-prompt inspection, transcript logs or biometric-vector
logging was added.

## Backend and model choice

Embedded **llama-cpp-python / llama.cpp** avoids an Ollama service dependency and offers
GGUF CPU execution plus future native hardware backends. Ollama would add a separate
service lifecycle; this phase does not need one. See the
[official Python runtime documentation](https://llama-cpp-python.readthedocs.io/en/latest/).

Recommended initial model: **Qwen2.5-1.5B-Instruct, Q4_K_M GGUF**, configurable as a local
path. The publisher's file is approximately **1.12 GB**. Its small size and existing
instruct GGUF packaging make it a conservative first candidate; this is not a claim
that it outperforms Qwen3, Llama, Gemma or Phi. Those remain replaceable model choices
subject to local chat-template/schema compatibility and later evaluation.
[Publisher model files](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main).

The inspected Windows host has a Ryzen 9 9950X (16 cores/32 threads) and approximately
31.2 GiB physical memory. Four LLM CPU threads leave room for perception and speech.
Budget roughly **2–4 GiB additional RAM** for this model/runtime at 4,096 context tokens
as a planning estimate, not a measured peak; the complete vision/STT/TTS stack needs
additional memory. No tokens/sec estimate is claimed.

Windows CPU wheel 0.3.35 was installed and its import/API signatures checked without
loading weights. Jetson Orin needs an ARM64 build appropriate for its JetPack/native
libraries; the current adapter uses CPU (`n_gpu_layers=0`). GPU offload is not enabled
in this phase. Raspberry Pi may need a smaller model/context and native build; neither
latency nor deployment has been validated there. Optional native dependencies do not
become mandatory for core tests or disabled AI.

## Explicit setup and future commands

These commands are documentation for **after repository review**, not a request to
run a model now. Use the existing voice configuration and model assets described in
[the local voice guide](local-voice.md). No GGUF is bundled or downloaded by Jake.

Windows CPU dependency setup using the upstream wheel index (the tested version):

```powershell
uv sync --extra voice
uv pip install --python .venv/Scripts/python.exe --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python==0.3.35
```

For an integrated environment, retain its extras when syncing:

```powershell
uv sync --extra voice --extra detection --extra appearance --extra identity
uv pip install --python .venv/Scripts/python.exe --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python==0.3.35
```

The optional project extra is named `conversation`. `uv sync --extra conversation`
can build llama-cpp-python from source and requires the platform toolchain. The above
wheel setup avoids requiring a Windows compiler. Subsequent `uv run` commands retain
installed optional packages; an exact `uv sync` without that extra can remove them,
so reinstall the wheel after such a sync. Do not change the whole project's package
index to the wheel repository.

Explicit model download with a separate setup tool (network required only here):

```powershell
uv run --extra voice python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models/llm')"
```

Alternatively copy an already-provisioned GGUF into a local folder and set its absolute
path. Runtime accepts a local file only and never invokes hub download helpers or a
network model server. Review publisher terms when redistributing models.

Add the example table to your existing TOML without replacing its other settings:

```toml
[conversation_ai]
enabled = true
backend = "llama_cpp"
model = "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
context_tokens = 4096
max_output_tokens = 96
temperature = 0.2
top_p = 0.9
threads = 4
timeout_seconds = 20.0
```

The repository example defaults `enabled = false`. `[audio] enabled = true` remains
an independent requirement. Existing configs are valid unchanged. Relative model
paths use the working directory; no configuration or model file is silently migrated.

Exact standalone command after setup:

```powershell
uv run --extra voice jake-voice --config config/local.toml
```

Exact integrated command after setup:

```powershell
uv run --extra voice --extra detection --extra appearance --extra identity jake-voice --config config/local.toml --with-camera --preview --tracker kalman --assignment hungarian --appearance --reid --identity --visitors
```

q/Q in the preview and Ctrl+C stop the session, including pending conversational work.

## Files and validation

Added: `conversation_ai_config.py`, `conversation_domain.py`, `conversation_context.py`,
`adapters/llama_conversation.py`, `application/conversation_worker.py`,
`tests/test_conversation_ai.py`, `tests/test_llama_conversation.py`, and this guide.
Updated: `conversation.py`, `application/voice.py`, `application/voice_composition.py`,
`voice_cli.py`, `config.py`, `config/jake.example.toml`, `pyproject.toml`, `uv.lock`,
`README.md`.

Fake-model tests exercise opt-in configuration, unchanged deterministic mode, context
provenance/timezone, personalization, guarded intents, rejected output, bounded history,
timeout/failure fallback, one active job, independent VAD/camera handoff, shutdown,
GGUF local-only loading, JSON schema handling and token-pressure history dropping.
Adapter tests forbid network connections. Physical LLM testing remains deferred for
repository review.

Final validation: **838 tests passed, 93% total coverage**; Ruff check, format check
(125 files), mypy, and source/wheel package build passed. The existing Phase 3A suite
remains included. Runtime verification was limited to installation/import/API signatures;
no real LLM, microphone, speaker, camera or biometric-store test was performed.
