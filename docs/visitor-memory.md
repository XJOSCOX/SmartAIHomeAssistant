# Phase 2D.2: meaningful visitor frequency

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
and inject `VisitorMemory(..., timezone=config.home.timezone,
is_nonresident=identity.confidently_nonresident)` along
with that service. Construct `EncryptedVisitorStore` with the same `timezone` keyword.
The resident guard is a required callback. Person-event return
values remain unchanged; `pipeline.visitor_matches` and `pipeline.visitor_events`
are separate metadata outputs. Use fresh services for each camera session and one
live writer per store; multi-hub synchronization is not implemented.

## Resident-first and quality policy

Every valid face observation is matched against enrolled residents first. UNKNOWN
alone does not establish non-residency: a tie between residents can produce UNKNOWN.
Visitor learning requires cosine similarity **below the resident candidate threshold
(default 0.45) against every resident template**. RESIDENT permanently blocks learning
for that visit. Transient resident ambiguity pauses learning and clears samples; it
does not poison the entire visit. The service checks the final
visitor centroid against residents too, since averaging can change similarities.

There is no mathematical proof that a face below threshold is a non-resident. These
are conservative model-based gates; an unenrolled resident is not represented in the
resident store. Model fingerprint/dimension mismatches fail clearly, never silently
compare incompatible vectors. Changes to resident enrollment require restarting
live sessions to reload the existing resident cache.

All Phase 2B size, sharpness, clipping, landmark/pose, and confidence checks run before
encoding. Visitor `min_detector_confidence=0.90` is an **additional detector confidence
floor**, not an aggregate quality probability. Rejected faces never enter the buffer.
The new default matches Jake's existing identity confidence floor and the 0.90 starting
point in the [official YuNet demo](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/demo.py).
It removes the extra rejection of otherwise quality-approved observations between
0.90 and 0.95. Resolution alone does not establish a calibrated threshold: there is
no claim that 0.95 is universally unsuitable for 720p/1080p, or that this change has
been physically validated. Size, sharpness, pose, clipping, identity similarity,
temporal confirmation and final-centroid checks remain unchanged.

The legacy TOML key `min_face_quality` remains an alias. Its explicit value is preserved,
so an existing file with `min_face_quality=0.95` still uses 0.95. Configuring both names
is rejected. No local settings or biometric store is silently rewritten. The effective
detector floor is the higher of the identity and visitor settings; diagnostics show
the actual gate responsible for rejection.

## Confirmation and false-merge protections

| Configuration | Default | Purpose |
| --- | --- | --- |
| `enabled` | false | Independent persistence opt-in |
| `required_observations` | 5 | Multiple supporting face observations |
| `observation_window_seconds` | 10.0 | Rolling confirmation window |
| `observation_interval_seconds` | 0.5 | Minimum observation spacing |
| `carry_seconds` | 2.0 | Short display carry without a new quality-approved face |
| `min_detector_confidence` | 0.90 | Additional detector-confidence floor |
| `resident_candidate_confirmations` | 3 | Fresh strong comparisons to one resident before blocking |
| `resident_candidate_window_seconds` | 3.0 | Rolling strong-evidence window |
| `nonresident_recovery_observations` | 3 | Spaced nonresident observations needed after a pause |
| `match_similarity` | 0.70 | Conservative visitor cosine threshold |
| `ambiguity_margin` | 0.10 | Separation from runner-up and uncertainty band |
| `recurrence_policy` | distinct_day | Only supported frequency policy |
| `recurring_distinct_days` | 2 | Local visit days needed for recurring status |
| `frequent_distinct_days` | 5 | Local visit days needed for frequent status |
| `retention_days` | 30 | Time since last accepted face evidence before expiry |

These defaults demand more evidence than resident recognition; they are not calibrated
probabilities or guarantees. Configuration validates types, finite/ranged values,
minimum observation counts and feasibility of the observation window.

Resident evidence is tracked explicitly: verified-resident status, a same-resident
strong-candidate count/timestamps, pause state, and nonresident recovery support.
Only fresh quality-approved comparisons at or above `identity.resident_similarity`
(default 0.60) count as strong; they must be spaced by the visitor observation interval.
Three within three seconds permanently block that visit. Carried CANDIDATE display
states never count as new face observations. Changing the resident candidate or
receiving weak fresh resident evidence clears the strong sequence.

A single/weak/ambiguous resident comparison pauses collection. Recovery requires three
quality-approved, confidently nonresident observations spaced at least 0.5 seconds
apart; a gap beyond the visitor observation window resets recovery progress. Recovery
samples are discarded, then five new visitor observations are required. Reappearing
resident ambiguity resets recovery. Confirmed or repeatedly strong resident evidence
cannot be cleared this way. A plausibly resident final centroid still prevents
persistence and returns the visit to a paused state.

## Visitor diagnostics

`--debug-visitors` prints changed per-track reasons in the development console and
works with every tracker mode. `--debug-tracks` also enables those diagnostics when
visitor memory is enabled. Normal preview output remains unchanged. Example lines:

```text
VISITOR track=3 rejected face too small
VISITOR track=3 rejected detector confidence 0.87 < 0.90
VISITOR track=3 paused possible resident similarity 0.48
VISITOR track=3 recovering nonresident 2/3
VISITOR track=3 candidate 2/5
VISITOR track=3 blocked resident confirmed
VISITOR track=3 confirmed FIRST_TIME_VISITOR
```

Encoder-idle "waiting for face observation" gaps do not replace the last meaningful
reason. Identical reasons are suppressed across these gaps; changed progress, rejection,
and confirmation reasons still print. Removed tracks release their diagnostic state.

Diagnostics include scalar confidence/similarity, progress, and explicit quality or
matching rejection reasons, never biometric vectors or crops. They are session-local,
not part of encrypted storage. For future validation after review, append
`--debug-visitors` to the live command above. No camera execution is part of this fix.

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

A physical presence session begins with `PERSON_ENTERED` and ends only with finalized
`PERSON_LEFT`. Frames, missed detections and periodic PRESENT events are not sessions.
After the first successful face confirmation, a new profile has `session_count=1`,
`distinct_visit_days=1`, and `last_visit_local_date` set to the confirmation frame's
household date. A later confirmed presence session increments `session_count` once.
It increments `distinct_visit_days` only if its local confirmation date is later than
the stored date. Five separate sessions on Monday therefore mean five sessions and
one day. A Tuesday return means six sessions and two days.

| Presentation precedence | Condition |
| --- | --- |
| KNOWN_VISITOR | Explicit operator label |
| FREQUENT_VISITOR | At least `frequent_distinct_days` (default 5) |
| RECURRING_VISITOR | At least `recurring_distinct_days` (default 2) |
| FIRST_TIME_VISITOR | Otherwise, a confirmed profile |

UNKNOWN and VISITOR_CANDIDATE remain observation/confirmation states. A label does
not reset aggregates; removing it reveals the current count-derived state. Frequent
and recurring mean observation frequency, never trust, authorization, safety, intent,
or household membership. KNOWN means explicitly named, never automatically RESIDENT.
Resident enrollment remains separate. Recognition and resident-first gates are unchanged.

### Household calendar

```toml
[home]
timezone = "America/Chicago" # Choose the household's IANA zone.

[visitors]
recurrence_policy = "distinct_day"
recurring_distinct_days = 2
frequent_distinct_days = 5
```

Configuration requires integer `frequent_distinct_days > recurring_distinct_days >= 2`.
Only `distinct_day` is implemented; `minimum_gap` is reserved for future design and
currently rejected. The old `recurring_visit_count` key produces an actionable error:
replace it with the explicit day thresholds instead of silently changing its units.
The ignored operator `config/local.toml` is not rewritten by this change.

`home.timezone` defaults to UTC for configurations without a household zone. Set it
explicitly before creating or migrating visitor memory. Frame timestamps remain aware
instants; `ZoneInfo` converts them to household dates, independent of OS timezone.
The declared `tzdata` dependency supplies IANA data on Windows as well as hosts without
system data ([Python zoneinfo documentation](https://docs.python.org/3/library/zoneinfo.html)).
UTC midnight alone does not count a new day. Local midnight does, on the next confirmed
session; spring-forward and repeated fall-back hours remain one local date.

A continuous overnight session counts only its first confirmation day. Reconfirmation
within that same semantic session does not add days or sessions. No date list is stored.
The last counted date is a high-water mark: backdated sessions after a clock rollback
can add sessions but cannot recount an earlier date. This conservatively undercounts
rather than inventing distinct days. Backward timestamps within a running pipeline
remain rejected. The encrypted payload binds the configured timezone name; a changed
zone fails clearly without rewriting historical aggregates. Automatic timezone rebasing
is not implemented.

Recently-lost body ReID retains the same visit and count. Its continuity epoch clears
visitor face confirmation; five fresh observations of the same face restore the label
without another visit increment. Face evidence is briefly carried on the same visible
active track; it is hidden after two seconds without a valid face. A consistent face
can restore an already verified active visit, while body reactivation or conflicting
evidence requires renewed multi-observation confirmation. Clothing alone cannot grant
a visitor profile. Evidence buffers disappear on LEFT.

Separate metadata events are `VISITOR_FIRST_SEEN`, `VISITOR_RECOGNIZED`,
`VISITOR_BECAME_RECURRING`, `VISITOR_BECAME_FREQUENT` (day-threshold crossings only),
and `VISITOR_LEFT`. Each threshold event occurs once when the day count crosses it;
same-day sessions never repeat it. This assumes stable configured thresholds; changing
thresholds recalculates presentation but does not synthesize historical events. They contain
event/visitor UUIDs, frame timestamps/context, track ID, state, session/day counts and an optional
explicit display name. They never contain vectors, crops, audio or movement history.
The preview shows short anonymous IDs or explicit names; full UUIDs are reserved for
debug/management use. Metadata console output can still disclose household presence.

Process restarts do not synthesize LEFT or resume old tracking sessions. A subsequent
confirmed session can count as another visit even if it was physically continuous
across restart. Unfinished visits keep their count but do not add a duration sample.
Unresolved resident evidence after visitor association pauses further learning and
duration updates until recovery; permanently blocked visits omit duration updates.
Existing stored profiles are not automatically
deleted or converted into residents.

## Statistics without visit histories

Profiles store `session_count`, `distinct_visit_days`, `last_visit_local_date`, and
Welford aggregates over **completed physical sessions**: completed-duration sample count
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
in seconds squared. Duration statistics update once per finalized session, independently of distinct days. Small negative
roundoff in M2 is clamped to zero; invalid/non-finite durations are rejected.

## Storage, retention, and management

The separate file is `<identity.store_path>/visitors.json`, defaulting to
`.jake-identities/visitors.json`. It uses the existing AES-256-GCM v2 envelope with
authenticated domain **`Jake/visitor-store`** and a newly created, distinct key UUID.
The same Windows Credential Manager/Linux Secret Service provider architecture holds
the key; no key is stored beside ciphertext. Resident envelopes retain their original
domain/key and need no migration. Decrypting under the wrong domain fails closed.

Visitor payload version 2 contains the household timezone and profiles with UUID,
one face template/model fingerprint, created/last-seen/last-visit timestamps,
`session_count`, `distinct_visit_days`, `last_visit_local_date`, duration aggregates,
and optional explicit label/flag. The outer encryption envelope remains version 2. Raw images, video, audio, clothing vectors, names inferred from
appearance, and full visit/movement history are excluded. Writes reuse Phase 2C lock,
permissions, encrypted-only temporary file, fsync, readback validation and atomic
replacement. Resident and visitor writes share the directory writer lock; they do
not share schemas, keys or data files. Key loss, same-account malware, backup recovery,
rollback and Python memory limitations remain as documented in
[encrypted identity storage](encrypted-identity-storage.md).

### Explicit migration of existing visitor data

Older encrypted payload v1 profiles are preserved, not discarded or silently upgraded.
Authentication and strict legacy-schema validation precede a clear migration-required
error on ordinary access. After repository review, with live sessions stopped and the
household timezone/threshold configuration chosen, the migration command is:

```sh
uv run --extra identity jake-visitors --config config/local.toml --migrate-store
```

Migration preserves UUID, template values/model fingerprint exactly, timestamps,
explicit label and all Welford aggregates. Old `visit_count` becomes `session_count`;
every existing profile starts with `distinct_visit_days=1`. Its initial local date is
derived from old `last_visit_at` in the configured household timezone. Historic sessions
cannot establish historic distinct days. An anonymous old RECURRING profile can safely
become FIRST_TIME until another local-day return; explicit labels retain precedence.

The operation reuses the existing visitor key and authenticated domain. It transforms
only authenticated data in memory, writes an encrypted temporary file, verifies its
decryption/schema/content, and atomically replaces the original. Failure before replace
leaves the original intact and cleans the temporary file. Repeating migration validates
the current payload and reports that no migration is needed, without rewriting or
creating another key. Migration does **not** run retention or touch residents. No real
enrolled store was accessed or migrated during this implementation; tests use fake keys.

Retention uses camera/frame time during live processing. Profiles older than the
retention cutoff are removed from the active encrypted file. Last-seen evidence is
checkpointed at confirmation, at most hourly during long verified visits, and at LEFT;
no per-frame history or disk write is made. A model observation gap can therefore
make retention approximately one checkpoint interval conservative. When visitor mode
is off, there is no background task reading/pruning visitor data. Management operations
also expire overdue profiles using current UTC. Thus retention runs on next enabled
processing/management, not while Jake is stopped.

Stop live visitor sessions before management, labeling or resident enrollment changes.
The dedicated CLI does not open a camera or alter resident records. `--list` displays
UUID/name, state using loaded thresholds, Sessions, Visit Days, completed session
samples, mean session duration and sample variance, never vectors:

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
