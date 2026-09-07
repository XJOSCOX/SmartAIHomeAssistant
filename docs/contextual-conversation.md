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
  protocol with load/generate/cancel_current/cancel/status/close. No model vendor appears in these contracts.
- `conversation_context.py`: context assembly, central prompt construction, capability
  catalogue, deterministic precedence, response validation and greeting personalization.
- `conversation_ai_config.py`: strict, independent opt-in configuration.
- `adapters/llama_conversation.py`: explicit worker-owned GGUF load, token budgeting, schema
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

The backend renders the GGUF's own chat template and counts that exact prompt with
its tokenizer, reserving the configured output budget. It drops oldest history
exchanges if needed, and reports CONTEXT_OVERFLOW if the remaining request cannot
fit. The same formatter is installed as the native chat handler; no fallback to an
unrelated template occurs. Returned native usage supplies reported token counts.


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

Model loading starts asynchronously when the worker starts, before any user request.
The default load deadline is 120 seconds; the independent generation deadline is
30 seconds and starts only when the owner begins a request. While loading,
unguarded utterances receive `I'm still starting up.` without failing the backend.
Deterministic privacy/action guards remain active. Audio/camera initialization does
not wait for GGUF loading.

Missing files, import/load/template/schema failures and reset failures make the
backend unavailable until restart. A generation timeout, malformed JSON, rejected
reply, context overflow or generation error is recoverable after the owner finishes
and native reset succeeds. The same generic spoken fallback remains privacy-safe;
metadata codes explain the difference. No new request can overlap an operation
that is still unwinding after timeout.

Per-generation cancellation uses its own event; shutdown uses a separate terminal
signal. Timeout or conversation expiry cancels only the current generation. The
native owner resets state before allowing another request. All native load, inference,
reset and close calls occur on the owner thread. A failed reset marks the backend
unavailable. Native load or prompt evaluation may finish its current operation before
stopping; deadlines are observable, not hard native preemption. Shutdown waits for
that operation to return and closes the model on the same thread.

The existing conversation timeout clears manager history. If expiry occurs during
inference, active generation is cancelled and its response cannot attach to a new
session. A cancelled/expired request releases its remaining worker references when
the cooperative native call returns. No transcript is saved to disk. Native attention
state is reset between requests; explicit bounded history supplies continuity.
Memory cleanup is not a guarantee of secure erasure from native/OS memory.

## Metrics and safe diagnostics

The CLI prints AI enabled/disabled, worker state, load milliseconds, generation
milliseconds, backend prompt/output token counts, tokens/sec, fallback-used status
and a typed `last_error_code`. Exception messages, prompts and transcripts are never logged.
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
load_timeout_seconds = 120.0
generation_timeout_seconds = 30.0
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


## Runtime recovery and model-only health command

```powershell
uv run --extra voice jake-voice --config config/local.toml --test-conversation-model
```

This explicit probe does not require audio.enabled or conversation_ai.enabled. It
loads the configured local model, checks the GGUF template, generates one fixed
internal `Hello.` request, validates JSON/schema and the unchanged allowed-reply
policy, reports token/timing metadata and unloads. It accepts no arbitrary prompt
and never composes microphone, camera, STT, TTS, residents or visitor stores. Exit
status is 0 for READY, 1 for a model failure and 130 for interruption. It prints no
prompt or reply text. The optional native conversation runtime must already be installed.

The normal standalone voice command remains:

```powershell
uv run --extra voice jake-voice --config config/local.toml
```

Integrated flags and policy/identity behavior are unchanged. Full voice testing
remains deferred until repository review.

Legacy `conversation_ai.timeout_seconds` remains accepted as a generation-only alias.
It never limits model loading. If both the legacy alias and the new generation field
are supplied, the legacy alias takes precedence; remove it to use the new field.
Existing config serialization preserves this behavior and omits null aliases. Defaults
are 120 seconds for load and 30 for generation; existing explicit 20-second configs
retain 20 seconds for generation.

Safe error categories:

| Code | Behavior |
| --- | --- |
| MODEL_NOT_FOUND | Configured GGUF is absent; unavailable until restart |
| BACKEND_IMPORT_FAILED | Optional native runtime cannot import; unavailable |
| MODEL_LOAD_FAILED / MODEL_LOAD_TIMEOUT | Initialization failed/exceeded its own budget; unavailable |
| CHAT_TEMPLATE_FAILED | Missing/invalid GGUF template or rendering failure; unavailable |
| SCHEMA_FAILED | Enum grammar compilation failed; unavailable, no generic-JSON fallback |
| CONTEXT_OVERFLOW / TOKEN_BUDGET_FAILED | Request rejected; later requests may retry |
| GENERATION_TIMEOUT / GENERATION_FAILED | Retry after current operation returns and reset succeeds |
| INVALID_JSON / INVALID_RESPONSE_SHAPE / INVALID_REPLY / POLICY_REJECTED | Safe fallback; healthy model remains reusable |
| CANCELLED | Current request cancelled; shutdown cancellation is separately terminal |
| RESET_FAILED | Native reset/teardown failed; unavailable |

States exposed by the worker are disabled, loading, ready, generating, failed and
closed. A recoverable error can remain in last_error_code while state returns to
ready; successful generation clears it. During timeout cleanup state remains
generating, correctly indicating the occupied native slot.

### Structured generation and EOS handling

Physical inference with the installed Qwen2.5-1.5B-Instruct Q4_K_M and
llama-cpp-python 0.3.35 reproduced `INVALID_JSON`: assistant content was an empty
string and finish_reason was `stop`, despite the explicit grammar being supplied.
The cause was EOS detokenization with the default `special=False`, which returns
`b""` for Qwen's EOS. Jinja2ChatFormatter used this as a textual stop, matching at
position zero and truncating the output. Jake now uses `special=True` for BOS/EOS
and rejects an empty EOS during initialization. Qwen's EOS is `<|im_end|>`.

The GGUF's own Jinja template still renders the assistant generation prefix.
`added_special=True` prevents an extra BOS; EOS token-ID stopping and the nonempty
textual EOS stop remain in place. The installed chat handler forwards Jake's explicit
grammar to native `create_completion`. No response_format helper is used because
its schema compiler can silently fall back to generic JSON. The enum of approved
replies and final policy checks are unchanged.

Parsing trims surrounding whitespace only. Malformed JSON, prose around JSON and
Markdown fences remain `INVALID_JSON`. Non-string content, non-object JSON, extra
or missing keys and non-string replies are `INVALID_RESPONSE_SHAPE`. An unapproved
reply is `POLICY_REJECTED`. These request failures remain recoverable after reset;
schema compilation and chat-template failures retain their separate terminal codes.

Only the fixed internal health composition enables additional metadata: response
type and length, opening/closing braces, finish reason, grammar enabled, formatter
type and JSON parse error type. No raw content, prompts, transcripts or identity data
are printed, even in health mode. Normal conversations do not enable this sink.

After this fix the actual local model-only probe returned READY: 480 prompt tokens,
12 output tokens and approximately 3.5 seconds generation on this machine. This is
one health sample, not a performance guarantee. No weights were downloaded, and no
microphone, camera, STT, TTS or biometric stores were accessed. Full voice/camera
validation remains deferred until repository review.
