# Phase 1G: stabilization and event quality

Live Phase 1F showed track fragmentation, producing many short household events.
`StabilizedKalmanPersonTracker` reuses Jake's filter and correction implementation
with a new lifecycle and spatial cost policy. No model tracking, appearance,
recognition, persistence, network access, or image history is introduced.

## Lifecycle and event timing

New tracks start TENTATIVE with one successful observation. Three successful
observations by default confirm a track, including observations separated by
short misses. Only CONFIRMED visible snapshots qualify for ENTERED/PRESENT.
A confirmed miss becomes LOST and remains confirmed in the public event contract;
its positive missed count suppresses PRESENT. Reacquisition restores CONFIRMED
and retains its ID and event history. Tentative misses remain TENTATIVE.

At each update, before association, elapsed UTC capture time since the last
successful observation is compared to `max_missed_seconds`. Exceeding the timeout
transitions either tentative or confirmed/lost tracks to EXPIRED and removes
them from the active set. Equality with the timeout retains them. A detection
arriving after timeout receives a new ID, even if spatially identical. Tentative
expiration emits nothing; confirmed/lost expiration produces one LEFT through
the unchanged, separate `PersonEventGenerator`. EXPIRED is transient and is not
returned as an active track or retained indefinitely for display.

ENTERED time and duration begin at **confirmation time**, not first tentative
observation. LEFT uses the frame on which expiration is observed, including the
grace period. Confirmation has a hit-count delay (about two frame intervals for
three uninterrupted hits); retention is time-based, independent of FPS except
for up to one processed-frame interval of expiration quantization. Tests compare
10 and 30 FPS. Equal timestamps are allowed; decreasing capture timestamps reject
the update without mutating state. UTC subtraction handles timezone offset changes.
The Kalman prediction dt clamp remains separate from the uncapped retention clock.

Tentative tracks can accumulate hits across brief misses because detector dropouts
need not reset useful evidence. A timeout resets that evidence by removing the
candidate. Recurring nearby false detections can still confirm; confirmation does
not prove a physical identity. Existing normal preview boxes include candidates
and retained tracks; active count is not confirmed household occupancy.

## Motion cost and gating

For predicted and measured normalized centers, `d = hypot(zx-px, zy-py)`.
Let `g = max_center_distance` and `I = IoU(prediction, detection)`:

```text
valid = d <= g AND (I >= min_iou OR d <= g/2)
cost = (iou_weight * (1-I) + distance_weight * d/g)
       / (iou_weight + distance_weight)
```

Thus a close low-IoU box can reconnect within the inner half of the gate, while
large overlapping boxes still cannot bypass the outer center-distance gate.
Distance is normalized by the gate for cost scaling. Both weights must be finite
and in (0,1]; they are normalized by their sum, which need not be one.
The gate must be in (0,1]. These are spatial heuristics, not covariance-based
Mahalanobis gating. Fast movement beyond the gate still fragments tracks.

`motion_matching.py` owns this policy. Greedy picks lowest cost with deterministic
index ties. Hungarian uses the existing generic Jake solver and explicit unmatched
nodes: maximize valid match count, then minimize cost. Forbidden pairs never match.
One track consumes at most one detection and vice versa. Costs remain in [0,1].
Candidate work is O(TD); greedy sorting is O(TD log(TD)); Hungarian uses O((T+D)^3)
time and O((T+D)^2) space. The generic solver and baseline IoU matcher are unchanged.

## Defaults and migration

The CLI `--tracker kalman` now selects stabilization. `--tracker kalman-baseline`
reproduces Phase 1D–1F frame-count retention, immediate confirmation, and IoU gating.
`--tracker iou` and both assignment options remain available. Direct Python callers
use `StabilizedKalmanPersonTracker`; `KalmanPersonTracker` keeps baseline defaults
so existing integrations and benchmarks do not silently change.

Add these keys under your existing `[tracking]` table:

```toml
confirmation_hits = 3
max_missed_seconds = 1.5
max_center_distance = 0.15
iou_weight = 0.6
distance_weight = 0.4
```

Three hits reject isolated false positives while keeping confirmation delay short.
The 1.5-second allowance covers the tested half-second dropouts without depending
on the 10/30 FPS rate. A 0.15 outer gate limits implausible spatial jumps, with a
0.075 low-overlap recovery gate; 0.6/0.4 favors overlap while allowing motion proximity
to contribute. These defaults pass deterministic regressions, not a calibrated
home-surveillance dataset, and should be validated on the user's camera.
`max_missed_frames` is ignored by stabilized Kalman and retained for baselines.
Old TOML files load with the new defaults. Missed seconds must be finite and positive;
confirmation hits must be an integer >=1. Invalid keys/types fail startup.

Run physical validation with local model weights already installed:

```sh
uv run --extra detection jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --events --debug-tracks
```

Debug labels show `TENTATIVE 2/3`, `CONFIRMED`, or `LOST 0.7s` with the track ID.
Omit `--debug-tracks` for the normal display. q/Q and Ctrl+C still release resources.
For a prepared offline environment replace `--extra detection` with `--offline --no-sync`.

## Deterministic comparison

Run `uv run --extra tracking python -m jake.stabilization_benchmark`.
The 10 FPS fixture combines two crossing people, jitter, a six-frame dropout,
partial box-width occlusion, fast lateral motion, and seven isolated clutter detections.
Exact measurement associations map tracks to fixture ground truth only for scoring.
No ground truth is provided to the tracker. A false short event pair is a completed
entry/departure under two seconds for a track associated exclusively with clutter.
Switches include changes in a real person's observed ID, including across occlusion.

| Metric | Kalman baseline | Stabilized |
| --- | ---: | ---: |
| Total IDs created | 13 | 9 |
| Confirmed IDs | 13 | 2 |
| ID switches | 5 | 4 |
| ENTERED | 13 | 2 |
| LEFT | 13 | 2 |
| False short event pairs | 7 | 0 |

This demonstrates suppressed noisy events and reduced fragmentation in this fixture.
It does **not** eliminate crossing ID switches: four remain. A switch can transfer
presence history to another person without creating a new household event.
Geometry alone cannot establish identity. The previous `jake.benchmarks` results
remain the independent Phase 1C–1E baselines. Physical validation is still needed.
