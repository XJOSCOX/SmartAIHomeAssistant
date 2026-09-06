# Phase 2D: privacy-governed anonymous recurring visitor memory

## Scope and opt-in

Visitor memory is persistent biometric processing and defaults **off**. With it off,
unknown faces remain UNKNOWN; no visitor store is opened or templates created by the
perception pipeline. Existing resident commands and encrypted v2 resident files need
no migration. Body tracking, Kalman, Hungarian, appearance/ReID and person events are
preserved. Resident recognition always has priority.

`[visitors] enabled = true` opts in through configuration. `--visitors` explicitly
enables it for a development session; `--no-visitors` overrides enabled configuration.
Tracking is required, and visitor mode enables the resident-first face stage and
person-event lifecycle internally. `--events` additionally prints person events.

Future command for after repository review and existing YOLO, appearance, YuNet and
SFace setup (no new model downloads):

```sh
uv run --extra identity --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --reid --events --identity --visitors
```

No visitor persistence/live camera was run during implementation. Tests use synthetic
images/vectors, mock models and in-memory keys. No recognition accuracy is claimed
from synthetic tests.

Before enabling persistent visitor recognition, the operator should establish the
appropriate notice/consent and access arrangements for the household and visitors,
explain retention/deletion, and choose a suitable camera scope. Local encryption is
not a substitute for those decisions. There is no covert-learning mode, automatic
naming, criminal lookup, threat score, or suspicious-person classification.

## Domain and processing boundaries

| Concept | Meaning | Lifetime |
| --- | --- | --- |
| `PersonTrack` | Physical track, independent of person identity | Camera session |
| `ResidentProfile` | Deliberately enrolled resident | Encrypted resident store |
| `VisitorProfile` | Anonymous non-resident face template and aggregates | Separate encrypted store with retention |
| `VisitorMatch` / `VisitorEvent` | Metadata result/transition | Consumer controlled; Jake does not persist events |

`FaceIdentityService` uses SFace embeddings already produced after the existing
quality gates. It exposes transient `VisitorObservation` values only in opt-in mode;
preview/pipeline consume and clear them after each update. No second encoder pass
or body/clothing vector is used. `VisitorMemory` owns confirmation/session state;
`VisitorStore` abstracts persistence. `EncryptedVisitorStore` implements visitor CRUD
with its own strict schema. It composes the Phase 2C encrypted file primitives;
visitor profiles never subclass or enter resident records.

In programmatic pipelines, construct the face service with `collect_visitors=True`,
and inject `VisitorMemory(..., is_nonresident=identity.confidently_nonresident)` along
with that service. The resident guard is a required callback. Person-event return
values remain unchanged; `pipeline.visitor_matches` and `pipeline.visitor_events`
are separate metadata outputs. Use fresh services for each camera session and one
live writer per store; multi-hub synchronization is not implemented.

## Resident-first and quality policy

Every valid face observation is matched against enrolled residents first. UNKNOWN
alone does not establish non-residency: a tie between residents can produce UNKNOWN.
Visitor learning requires cosine similarity **below the resident candidate threshold
(default 0.45) against every resident template**. Any plausible resident evidence or
resident CANDIDATE/RESIDENT result blocks visitor learning for that entire presence
session, even if subsequent results become UNKNOWN. The service checks the final
visitor centroid against residents too, since averaging can change similarities.

There is no mathematical proof that a face below threshold is a non-resident. These
are conservative model-based gates; an unenrolled resident is not represented in the
resident store. Model fingerprint/dimension mismatches fail clearly, never silently
compare incompatible vectors. Changes to resident enrollment require restarting
live sessions to reload the existing resident cache.

All Phase 2B size, sharpness, clipping, landmark/pose, and confidence checks run before
encoding. Visitor `min_face_quality=0.95` is an **additional detector confidence floor**,
not an aggregate quality probability. Rejected faces never enter the candidate buffer.

## Confirmation and false-merge protections

| Configuration | Default | Purpose |
| --- | --- | --- |
| `enabled` | false | Independent persistence opt-in |
| `required_observations` | 5 | Multiple supporting face observations |
| `observation_window_seconds` | 10.0 | Rolling confirmation window |
| `observation_interval_seconds` | 0.5 | Minimum observation spacing |
| `carry_seconds` | 2.0 | Short display carry without a new quality-approved face |
| `min_face_quality` | 0.95 | Additional detector-confidence floor |
| `match_similarity` | 0.70 | Conservative visitor cosine threshold |
| `ambiguity_margin` | 0.10 | Separation from runner-up and uncertainty band |
| `recurring_visit_count` | 2 | Distinct confirmed visits needed for recurring status |
| `retention_days` | 30 | Time since last accepted face evidence before expiry |

These defaults demand more evidence than resident recognition; they are not calibrated
probabilities or guarantees. Configuration validates types, finite/ranged values,
minimum observation counts and feasibility of the observation window.

Unit vectors use cosine `s = dot(a, b)`. Existing visitor matching requires a score
of at least 0.70 and a gap of at least 0.10 to the runner-up. Scores in the uncertainty
band [0.60, 0.70), or insufficient winner separation, remain UNKNOWN and do not create
a competing profile. Samples must be mutually consistent (pairwise cosine ≥0.70).
Switching the best visitor candidate restarts confirmation, rather than combining
support for different people. Every buffered observation must support the final match.

For a novel candidate, five accepted samples form
`centroid = normalize(mean(e1, ..., en))`. The centroid is checked again against both
residents and existing visitor profiles before creation. If uncertainty appears only
after averaging, the candidate stays unresolved. A stored centroid is **fixed** in
this phase: later visits update metadata/statistics, never continuously average face
observations. No unlimited embedding history or template drift is introduced.

Candidates are evaluated as a batch. If simultaneous tracks claim the same profile,
or two ready centroids are too similar, neither is selected as a winner. An already
associated profile cannot be attached to another live visit. Distinct simultaneous
faces can create distinct UUIDs. Confirmed identities are never silently switched
to a different profile within the same semantic visit. There is no automatic profile
merging; false splits require future/manual review rather than irreversible merging.

Matching costs O(P×E) per observation for P profiles and E embedding dimensions, plus
sorting profile scores and bounded pairwise sample checks. Simultaneous ready-candidate
conflict checks are quadratic in ready tracks. The store caps profiles at 100 and
candidate buffers at the configured observation count (3–20), rather than accumulating
all frames. Storage is synchronous: encrypted reads and occasional writes add CPU/I/O
cost, and the preview FPS includes it. This is a single-camera development stage.

## Visits, states, and event semantics

A visit begins with `PERSON_ENTERED` and ends only with finalized `PERSON_LEFT` from
the existing person-event generator. Frames, missed detections and periodic PRESENT
events are not visits. Profiles are created only after face confirmation, with their
first semantic entry timestamp and `visit_count=1`.

For an existing profile, multiple supporting observations during a **later** presence
session increment `visit_count` once. At the configured count (default two), the result
becomes RECURRING_VISITOR. Deliberate labeling changes it to KNOWN_VISITOR even if only
one visit has occurred. Removing a label returns to the count-based anonymous state.
KNOWN_VISITOR is never RESIDENT; resident enrollment remains separate.

Recently-lost body ReID retains the same visit and count. Its continuity epoch clears
visitor face confirmation; five fresh observations of the same face restore the label
without another visit increment. Face evidence is briefly carried on the same visible
active track; it is hidden after two seconds without a valid face. A consistent face
can restore an already verified active visit, while body reactivation or conflicting
evidence requires renewed multi-observation confirmation. Clothing alone cannot grant
a visitor profile. Evidence buffers disappear on LEFT.

Separate metadata events are `VISITOR_FIRST_SEEN`, `VISITOR_RECOGNIZED`,
`VISITOR_BECAME_RECURRING` (threshold transition only), and `VISITOR_LEFT`. They contain
event/visitor UUIDs, frame timestamps/context, track ID, state, count and an optional
explicit display name. They never contain vectors, crops, audio or movement history.
The preview shows short anonymous IDs or explicit names; full UUIDs are reserved for
debug/management use. Metadata console output can still disclose household presence.

Process restarts do not synthesize LEFT or resume old tracking sessions. A subsequent
confirmed session can count as another visit even if it was physically continuous
across restart. Unfinished visits keep their count but do not add a duration sample.
Uncertain resident evidence after visitor association prevents further learning and
duration updates for that session; existing stored profiles are not automatically
deleted or converted into residents.

## Statistics without visit histories

Profiles store `visit_count` and Welford aggregates: completed-duration sample count
`n`, mean duration and `M2`. Counted visits may exceed completed samples after crashes
or unresolved session endings. On one finalized LEFT, duration is frame-timeline
`left_at - entered_at`, including brief body ReID gaps:

```text
n = n + 1
delta = duration - mean
mean = mean + delta / n
M2 = M2 + delta * (duration - mean)
sample_variance = M2 / (n - 1), if n > 1; otherwise 0
```

This avoids subtracting large squared sums and stores no duration list. Variance is
in seconds squared. Count/statistics update once per finalized visit. Small negative
roundoff in M2 is clamped to zero; invalid/non-finite durations are rejected.

## Storage, retention, and management

The separate file is `<identity.store_path>/visitors.json`, defaulting to
`.jake-identities/visitors.json`. It uses the existing AES-256-GCM v2 envelope with
authenticated domain **`Jake/visitor-store`** and a newly created, distinct key UUID.
The same Windows Credential Manager/Linux Secret Service provider architecture holds
the key; no key is stored beside ciphertext. Resident envelopes retain their original
domain/key and need no migration. Decrypting under the wrong domain fails closed.

Visitor payload version 1 contains only UUID, one face template/model fingerprint,
created/last-seen/last-visit timestamps, visit count, duration aggregates, and optional
explicit label/flag. Raw images, video, audio, clothing vectors, names inferred from
appearance, and full visit/movement history are excluded. Writes reuse Phase 2C lock,
permissions, encrypted-only temporary file, fsync, readback validation and atomic
replacement. Resident and visitor writes share the directory writer lock; they do
not share schemas, keys or data files. Key loss, same-account malware, backup recovery,
rollback and Python memory limitations remain as documented in
[encrypted identity storage](encrypted-identity-storage.md).

Retention uses camera/frame time during live processing. Profiles older than the
retention cutoff are removed from the active encrypted file. Last-seen evidence is
checkpointed at confirmation, at most hourly during long verified visits, and at LEFT;
no per-frame history or disk write is made. A model observation gap can therefore
make retention approximately one checkpoint interval conservative. When visitor mode
is off, there is no background task reading/pruning visitor data. Management operations
also expire overdue profiles using current UTC. Thus retention runs on next enabled
processing/management, not while Jake is stopped.

Stop live visitor sessions before management, labeling or resident enrollment changes.
The dedicated CLI does not open a camera or alter resident records:

```sh
uv run --extra identity jake-visitors --config config/local.toml --list
uv run --extra identity jake-visitors --config config/local.toml --delete <visitor-uuid>
uv run --extra identity jake-visitors --config config/local.toml --delete-all
uv run --extra identity jake-visitors --config config/local.toml --label <visitor-uuid> --name "Daniel"
uv run --extra identity jake-visitors --config config/local.toml --remove-label <visitor-uuid>
```

Repeating `--label` deliberately changes the label. CLI access relies on the trusted
local OS account, not automated facial authorization. Deletion removes active records;
it cannot erase separately retained backups, stale filesystem blocks or snapshots.
Delete-all retains the separate key for encrypted-backup compatibility. Disabling
visitor persistence alone does not delete stored data. The default directory is
ignored by Git; custom identity directories must also be excluded.

## Remaining limitations

Lookalikes, lighting/pose changes, ID switches, model biases and replayed photos/screens
can cause false matches or false splits. There is no liveness/anti-spoofing or reliable
real-world identity guarantee. Conservative refusal reduces some false merges at the
cost of missed recognition; similar people may stay unresolved. Fixed templates avoid
drift but can become stale. Restart and retention boundaries affect visit counts.
No automatic merging, visitor naming, classification, threat scoring, cloud lookup,
speech, LLM, behavior learning or multi-hub coordination is implemented.
