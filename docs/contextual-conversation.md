# Phase 3B.1: controlled free-form local conversation

## Free language != free authority

Jake can explain cameras, computers, gardening and smart homes, brainstorm, joke and
ask follow-up questions using the local model's pretrained knowledge. General knowledge
is not evidence of live house facts. There are no tools, physical actions, emergency
calling, browsing, cloud inference or persistent conversation memory.

This replaces the Phase 3B exact-sentence enum. Existing deterministic responses remain
for sensitive requests and the non-LLM voice path; they are not a generation catalogue.

## Architecture

STT -> deterministic request routing -> structured context -> one asynchronous model
worker -> strict JSON parsing -> categorized response policy -> fresh visual identity
recheck -> bounded session history -> asynchronous TTS.

- `conversation_policy.py` owns request classification, restricted claim categories,
  response bounds and visual name authorization, independently of the model vendor.
- `conversation_context.py` builds context, persona and prompts, routes fixed replies
  and optionally personalizes greetings.
- `adapters/llama_conversation.py` implements the model port, GGUF loading, shape grammar,
  token budgeting, strict parsing, policy checking, request cancellation and cleanup.
- `application/voice.py` revalidates generated text against current visual association
  before speech. The existing conversation worker keeps a single occupied generation
  slot through timeout cleanup; no overlapping native inference or unbounded queue.

Camera, tracking, identity, microphone callbacks, VAD and UI do not render/generate on
the LLM worker. STT/TTS, recognition algorithms, arrival greeting policies and encrypted
stores are unchanged.

## Output contract

```json
{
  "type": "object",
  "properties": {
    "reply": {"type": "string", "minLength": 1, "maxLength": 800}
  },
  "required": ["reply"],
  "additionalProperties": false
}
```

Jake explicitly compiles and supplies the grammar; no enum or response_format fallback
is used. Parsing trims outer whitespace only, requires exactly the reply key and a
nonblank string, and independently checks length and policy. Prose, fences, malformed
JSON and extra keys remain errors. There is no arbitrary extraction, rewriting into
valid JSON, or mid-sentence truncation. The persona targets 1-3 spoken sentences; token
and 800-character limits are hard bounds. A truncated invalid response falls back safely.

The validated Qwen2.5-1.5B-Instruct Q4_K_M remains the initial model. Its own GGUF Jinja
chat template and assistant prefix are used. BOS/EOS detokenization preserves special
tokens; an empty EOS stop is rejected. This preserves the previous physical INVALID_JSON
fix. Native timing excludes load, and the exact rendered prompt determines token budget.

## Deterministic request guards

Occupancy/location requests such as `Is Joseph home?`, `Who is inside?` and `Where is
Joseph?` return the fixed privacy refusal. Occupancy disclosure remains disabled for
all speakers, including residents, in this phase. General `What is a smart home?` or
`How can I improve my home's Wi-Fi?` reaches conversation.

Requests to unlock doors, switch lights, set thermostats, arm alarms or browse receive
an unavailable-action response. Explanations such as `How do smart locks work?` reach
the model. Emergency calling requests receive the fixed no-calling response; general
emergency information questions can be discussed. Private prompt, key, credential and
biometric-template requests are guarded. Explicit goodbye closes the conversation.
`I have a package` retains deterministic delivery routing; questions about delivery
companies are ordinary conversation.

## Post-generation policy and limitations

A deterministic validator classifies subject/predicate patterns for occupancy/location,
action completion or authority, emergency dispatch, identity assertions/direct address,
visitor status, secret disclosure and unsupported sensor/history facts. Capability
claims are checked against configured perception support. It inspects wording and
context, not membership in a list of responses or model-supplied claims. Novel safe
wording is accepted. Rejected claims produce the existing fallback and POLICY_REJECTED.

These English patterns are defense in depth, not a semantic proof that arbitrary prose
is safe. Novel phrasing, other languages and indirect claims can evade patterns; cautious
patterns can also reject benign wording. Prompt instructions are not an authorization
boundary. No tool execution exists regardless of generated text. General model knowledge
can be inaccurate. Quality and robustness remain model-dependent and require review and
adversarial testing before broader deployment.

As an additional privacy boundary, the model prompt omits household counts, other people,
resident/visitor IDs and visitor classifications. The domain retains scene metadata for
other deterministic consumers. The conversation prompt includes authorized speaker name,
association state, configured capabilities, timestamp and bounded turns, but no credentials,
biometrics, images or store data. Raw prompts and replies are not logged.

## Identity and history

Only an exactly-one-fresh-confirmed-visible-person association in RESIDENT state with an
authorized name permits direct address. A visitor, ambiguous scene or self-declaration
such as `I am Joseph` grants no identity. Visitor classifications are never automatically
spoken. Natural `Hey Joseph` and `Good evening, Joseph` are permitted for the authorized
resident, without requiring one exact greeting string. The prompt discourages repeated
names; optional deterministic name insertion happens once per session. Generated names
are checked again using the latest visual association before TTS, so stale/changed
association cannot authorize a delayed personalized response. This is visual conversation
context, not acoustic speaker identification.

History uses the existing bounded session-local deque. It supplies prior user/Jake turns
for references such as `Low light` after a camera discussion. Oldest exchanges are removed
if the exact token budget requires it. Session timeout or association change clears the
history; goodbye clears it. Nothing is written to disk.

## Configuration and runtime

AI stays opt-in with `[conversation_ai] enabled = true`. Model is a configurable local
GGUF path; Qwen is unchanged and other GGUF models can later implement the same port.
No weights are downloaded automatically. Native conversation dependencies must already
be installed. Recommended initial temperature is **0.6**, now the default/example, to
allow more variation than enum-era 0.2. This is a starting point for review, not a measured
quality optimum or a safety control. Existing explicit local values are untouched. Model
quality evaluation is deferred. The output token budget remains configurable (default 96).

Loading defaults to 120 seconds and generation to 30 seconds. The legacy timeout_seconds
is a generation-only alias and takes precedence if both fields are present. Failures
retain typed codes: SCHEMA_FAILED and CHAT_TEMPLATE_FAILED are distinct from
GENERATION_FAILED, GENERATION_TIMEOUT, INVALID_JSON, INVALID_RESPONSE_SHAPE, INVALID_REPLY
and POLICY_REJECTED. Request errors remain recoverable after native reset; reset failures
make the model unavailable. Shutdown is terminal and separate from request cancellation.

## Commands after repository review

Fixed model-only health check (no microphone, camera, STT, TTS or identity stores):

```powershell
uv run --extra voice jake-voice --config config/local.toml --test-conversation-model
```

READY means actual generation produced a nonempty, structurally valid, policy-safe string;
it no longer requires prewritten wording. Health-only metadata reports grammar, formatter,
content type/length, finish reason and parse-error type without printing raw text. Load,
token counts and generation latency remain visible. The owner thread resets/unloads.

Standalone voice:

```powershell
uv run --extra voice jake-voice --config config/local.toml
```

Integrated voice and annotated camera, using existing local models and configured stores:

```powershell
uv run --extra voice --extra detection --extra appearance --extra identity jake-voice --config config/local.toml --with-camera --preview --tracker kalman --assignment hungarian --appearance --reid --identity --visitors
```

These commands document future validation; full voice/camera testing remains deferred
until repository review. Tests use fake model outputs with no devices, GPU, weights,
network or biometric stores.
