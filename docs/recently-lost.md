# Phase 2A.1: recently-lost track re-identification

The appearance baseline expired confirmed IDs after 1.5 seconds without a match.
This mode preserves confirmed identity state longer without introducing a resident
identity, face recognition, or persistent appearance database.

## Lifecycle and event continuity

`RecentlyLostPersonTracker` reuses Jake's appearance tracker and Kalman state.
Tentative tracks still expire at normal active retention. Confirmed tracks move
from CONFIRMED to LOST on misses, then to RECENTLY_LOST after normal retention.
The recently-lost pool is an explicitly partitioned subset of the immutable internal
track tuple. Keeping both partitions in one snapshot makes failed updates atomic;
pool tracks are excluded from first-stage active assignment. They retain the ID,
last observed box, predicted box, Kalman state/covariance, last seen time, original
confirmation timestamp, appearance EMA, and lifecycle counters.

Each successful update returns the entire **unexpired lifecycle set**, including
recently-lost snapshots with `confirmed=True`, positive `missed_frames`, and
`recently_lost=True`. The event generator therefore keeps its existing entry and
emission history open. It owns that history; the tracker neither duplicates it nor
stores event objects. `PersonTrack.recently_lost` is a small model-independent domain
extension defaulting to false. The normal preview hides pool boxes and excludes them
from its active count. Debug text still shows pool membership and age.

Active matching runs first. Only unmatched detections enter second-stage pool
assignment, before new IDs are created. A successful match restores the original ID,
corrects the predicted Kalman state with the new observation, updates the appearance
EMA, resets misses, and restores CONFIRMED. A one-update debug marker reports
REACTIVATED. Original confirmation time and event entry time stay unchanged.
PRESENT resumes according to its existing throttle; ENTERED is not repeated.

Final expiration removes a track from the returned lifecycle set. The unchanged
event generator emits one LEFT and removes its presence state. Its existing
closed-ID tombstones retain only IDs for duplicate protection, not embeddings.
All track/pool state and vectors are removed at timeout; there is no expired gallery.
Updates must continue (including empty detections) for timeouts to be observed.
Shutdown does not fabricate departures. Source stalls delay observation of expiration.

## Time boundaries

`window_seconds` is the **total gap since last successful observation**, not extra
time added after entering the pool. With active retention 1.5 seconds and a five-second
window, a track is LOST through 1.5 seconds, RECENTLY_LOST for gaps >1.5 through 5.0,
and finally expires when the gap exceeds 5.0. Exact equality is eligible. Expiration
runs before association, so late detections receive new IDs. The enabled window must
exceed active retention. All age checks use UTC `FrameContext` timestamps; no wall-clock
timer decides identity or event duration. Backward times and duplicate sequences are
rejected before state changes. Kalman prediction retains its existing dt clamp.

Appearance's `max_embedding_age_seconds` still applies. A stale vector is cleared
and cannot reactivate a pooled track; LEFT remains delayed until the ReID timeout.
Both defaults are five seconds. If increasing the window, adjust embedding age only
if the longer memory retention is intended. Repeated successful observations refresh
both clocks; there is no evidence refresh during a miss.

## Re-identification cost

Let `s` be cosine similarity to the stored EMA. Let `d` be the smaller normalized
center distance to the Kalman prediction or last observed box. The last-observation
anchor accommodates people pausing or reversing while hidden, without allowing an
unbounded appearance-only search. Both anchors are image-local; distance is not a
physical-world speed constraint. The prediction is bounded at the image edges.

```text
valid = recent age <= window AND usable embedding
        AND s >= min_similarity AND d <= max_center_distance

cost = [appearance_weight*(1-s) + motion_weight*(d/max_center_distance)]
       / (appearance_weight + motion_weight)
```

Default weights 0.7/0.3 sum to one. Hungarian or greedy assignment consumes these
pre-gated costs with deterministic input-order ties and one-to-one matching.
The generic Jake Hungarian solver is unchanged. Active tracks have priority and
cannot lose an already-assigned detection to the pool. There is no appearance-only
override for distant candidates. The last box and prediction can drift apart;
this two-anchor heuristic trades some false-match risk for re-entry tolerance.

Pool cost construction is O(R×U×E) for R recent tracks, U unmatched detections,
and embedding dimension E. Hungarian assignment is O((R+U)^3) time and O((R+U)^2)
space; greedy sorting is O(RU log(RU)). Retained vectors use O(R×E) memory within
the time window. High detection churn can still increase transient memory use.

## Configuration and live validation

Add this table to `config/local.toml` without duplicating a table:

```toml
[tracking.reid]
enabled = true
window_seconds = 5.0
min_similarity = 0.65
max_center_distance = 0.30
appearance_weight = 0.7
motion_weight = 0.3
```

Five seconds covers the tested 2.3-second out/in gap while bounding memory.
Similarity 0.65 is stricter than active matching's 0.45 because a longer gap and
wider 0.30 spatial gate create more opportunities for false matches. Appearance
gets greater weight in this stage. These are tested starting values, not calibrated
probabilities or guarantees on real footage. Values must be finite and positive;
similarity, gate, and weights must be at most one. Enabled must be boolean and
unknown settings fail loading. The example opts in; omitted configuration stays
disabled for backwards compatibility.

With the Phase 2A local model pair and dependencies already installed:

```powershell
uv run --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --reid --events --debug-tracks
```

`--reid` explicitly enables this mode and implies appearance if not otherwise
specified. It requires `--track --tracker kalman`; explicitly combining it with
`--no-appearance` is rejected. A configuration-enabled ReID applies only when the
appearance Kalman mode is enabled. `--no-reid` reproduces Phase 2A with the same
encoder. Direct Python callers use `RecentlyLostPersonTracker`; the original
`AppearanceKalmanPersonTracker` remains unchanged in behavior.

Debug labels include `RECENTLY_LOST ID 7 age=2.1s` and
`REACTIVATED ID 7 | app 0.91`. Omit `--debug-tracks` for the normal preview.
q/Q and Ctrl+C retain cleanup. No new model download, network access, telemetry,
image saving, embedding persistence, face recognition, or later-phase features
are introduced. See [appearance setup/privacy](appearance-tracking.md).

## Synthetic benchmark

Run `uv run --extra tracking python -m jake.reid_benchmark`.
Six independent sessions use 256-dimensional synthetic vectors. People establish
confirmation, disappear beyond active retention, then return within five seconds.
The final empty frames drain all track state. Scoring uses exact measurement
associations against fixture truth; truth never enters the tracker.

| Scenario | IDs baseline/ReID | Switches | Fragmentation | False reactivations | ENTERED | LEFT |
| --- | --- | --- | --- | --- | --- | --- |
| Walk out/in | 2/1 | 1/0 | 1/0 | 0/0 | 2/1 | 2/1 |
| Full occlusion | 2/1 | 1/0 | 1/0 | 0/0 | 2/1 | 2/1 |
| Doorway return | 2/1 | 1/0 | 1/0 | 0/0 | 2/1 | 2/1 |
| Edge loss | 2/1 | 1/0 | 1/0 | 0/0 | 2/1 | 2/1 |
| Two-person return | 4/2 | 2/0 | 2/0 | 0/0 | 4/2 | 4/2 |
| Different person, same clothing vector | 2/1 | 0/0 | 0/0 | 0/1 | 2/1 | 2/1 |
| Total | 14/7 | 6/0 | 6/0 | 0/1 | 14/7 | 14/7 |

IDs include tentative births. Switches compare each real fixture person's observed
ID with their previous ID. Fragmentation counts distinct IDs per fixture person
beyond the first. False reactivations count reactivation transitions to an ID first
owned by a different fixture person (not every subsequent wrong observation).
One Windows run took approximately 0.27 ms total for the ReID stage over 54 updates;
the baseline has no ReID stage (0 ms). This excludes active assignment, model
inference, filter correction, display, and camera acquisition. Timing varies.

The similar-clothing failure is intentional: fewer events are not automatically
better events. Another person with indistinguishable appearance at a plausible
location can inherit a recently-lost ID and its event duration. This fixture tests
matching policy, not real model accuracy. Live validation is required. No claim of
resident identity is made; clothing, lighting, occlusion, and crop changes remain
limitations. Existing geometry and appearance benchmarks are preserved.
