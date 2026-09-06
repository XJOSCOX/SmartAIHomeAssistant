# Semantic person events (Phase 1F)

Frames contain RGB pixels. Detections describe per-frame person boxes. Tracks
associate detections over time using session-local IDs. Events describe changes
in those tracks so downstream Jake components can consume metadata instead of video.
No identity, persistence, speech, or behavioral inference is implemented.

## Boundary and state machine

`adapters/person_events.py:PersonEventGenerator` implements `EventGenerator` using
only standard-library code, `EventConfig`, and domain types. It receives the
**complete active track set**, including retained missed tracks, on every processed
frame. It never inspects tracker internals or decides when a track should expire.
Both IoU and Kalman trackers expose two additional immutable `PersonTrack` fields:

- `missed_frames=0`: consecutive processed updates without a matching detection;
  `visible` is a derived property equivalent to `missed_frames == 0`.
- `confirmed=True`: eligibility for semantic entry. Current trackers confirm on
  the first detection. This is not multi-frame noise rejection or resident recognition;
  future trackers can expose tentative tracks with `confirmed=False`.

Existing three-argument track construction remains valid. Snapshot equality now
includes lifecycle metadata, so a missed snapshot differs from a visible one.
Rich tracker history, covariance, and matching data remain internal to trackers.

| Event state | Input | Result |
| --- | --- | --- |
| Unannounced | Visible, confirmed track | ENTERED once; remember entry and emission time |
| Unannounced | Tentative or missed track | No event |
| Present | Visible, confirmed; interval elapsed | PRESENT once; advance last emission time |
| Present | Missed, tentative, or interval not elapsed | No event; retain entry time |
| Present | ID absent from active set | LEFT once; discard presence state; retain ID tombstone |
| Closed | ID still absent | No event |

Expired IDs must never be reused in the same session; reusing an announced closed
ID raises `EventError`. A new ID after expiration gets a new ENTERED. A tentative
track that disappears before entry produces no LEFT. Short occlusions neither
close event state nor cause another ENTERED when the track reconnects.

LEFT events are emitted first, sorted lexically by track ID, followed by ENTERED
or PRESENT events sorted by ID. Input tuple order does not affect event ordering.
Each generator has a locally generated UUID4 namespace; event IDs are UUID5 values
derived from that namespace and camera, sequence, track, and kind. Tests/replay can
inject `session_id=UUID(...)` for deterministic IDs. Fresh runtime instances get
separate namespaces even when their track IDs restart at `1`.

## Timing and throttling

`[events] present_interval_seconds = 5.0` is optional and defaults to five seconds.
Values must be finite positive numbers; booleans, zero, negatives, strings, and
unknown keys fail configuration loading. ENTERED is immediate. PRESENT is emitted
only on a visible, confirmed update at least one interval after the previous
ENTERED/PRESENT. Missed updates suppress PRESENT. A gap never triggers catch-up
bursts: the first eligible frame emits at most one PRESENT per track and starts
the next interval from that frame. The schedule may therefore drift with frame rate.

All event timestamps come from `FrameContext.captured_at`. `PersonEvent.entered_at`
records first confirmed visibility; `context.captured_at` is the current event time;
`left_at` returns that time for LEFT only. `duration_seconds` is their elapsed UTC
difference, including missed-frame grace time. LEFT is timestamped at **expiration
observation**, not last detection, so duration is approximate, not physical dwell
time. Example: entry at 18:41:10.200 and expiration at 18:42:03.700 gives 53.5 seconds.

Equal timestamps are accepted without flooding. Decreasing timestamps, duplicate
or decreasing sequences, camera changes, and duplicate active IDs raise `EventError`
before changing event state. Timestamp comparisons and subtraction use UTC, including
across offset/DST changes. Sequence gaps are allowed; time gaps affect duration,
while only tracker update calls count toward missed-frame expiration. Unlike the
Kalman motion dt fallback, event mode rejects backward clocks to avoid inventing
elapsed presence time. The camera currently supplies host UTC capture timestamps.

The original `PERSON_UPDATED` enum value and four-argument `PersonEvent` constructor
remain available for compatibility. The semantic generator never emits UPDATED.
Legacy events without `entered_at` have `duration_seconds=None`.

## Composition, logging, and limits

The existing `PerceptionPipeline` already routes tracker output through its
injected `EventGenerator` and returns structured events. The development CLI
creates a fresh generator when `--events` accompanies `--track`; preview forwards
tracks to it and passes results to `event_console.log_event`. Console formatting
and rendering hold no event lifecycle state. Camera-only and detection-only modes,
both trackers, both matchers, diagnostics, and benchmarks remain available.

Quitting with q/Q or Ctrl+C does not fabricate LEFT events. No further frames means
no evidence of expiration: a stopped/stalled camera cannot establish a departure.
There is no retry or delivery guarantee after downstream sink failure; discard the
session after an error. Generators are single-camera, serial, and not thread-safe.

Events contain IDs, kind, frame context, and entry time only. There are no pixels,
embeddings, audio, files, uploads, telemetry, or network calls. Console logging is
opt-in and Jake does not write logs to disk (a user may redirect their terminal).
Presence state uses O(A) memory for announced active tracks and closed-ID tombstones
use O(E) memory over a session's expired announced tracks. Processing costs
O(T log T + A log A) for deterministic ordering, where T is active input count.

Track IDs are **not resident identities**. Detector false positives can produce
ENTERED because confirmation is immediate. Tracker ID fragmentation produces
extra entry/departure pairs; an ID switch can transfer presence history to a
different person. Occlusion handling and global assignment cannot guarantee
physical identity continuity. Events describe tracker observations, not verified
household activity or intent. No persistence or later-phase recognition is added.
