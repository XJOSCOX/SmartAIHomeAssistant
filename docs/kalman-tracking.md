# Phase 1D: motion-aware box tracking

Jake implements its own linear Kalman filter using NumPy. Neither OpenCV's
KalmanFilter nor Ultralytics tracking is used. `KalmanPersonTracker` implements
the unchanged `PersonTracker` protocol. Geometry and greedy association remain
in `matching.py`; this phase changes the boxes supplied to that matcher.

## Model and dimensions

The state is the column vector:

```text
x = [cx, cy, w, h, vx, vy, vw, vh]^T
z = [cx, cy, w, h]^T

F(dt) = [ I4   dt*I4 ]       H = [ I4  04 ]
        [ 04    I4   ]
```

Centers and sizes use normalized image coordinates. All four velocities use
normalized units per second. The state assumes constant velocity between
measurements, including constant rates of width/height change.

| Quantity | Shape | Meaning |
| --- | --- | --- |
| x | 8×1 | Position, size, and velocity estimate |
| z, y | 4×1 | Measurement and innovation |
| F | 8×8 | Advances positions/sizes by velocity × dt |
| H | 4×8 | Observes the first four state components |
| P | 8×8 | State uncertainty and cross-covariances |
| Q | 8×8 | Process noise added each prediction |
| R, S | 4×4 | Measurement noise and innovation covariance |
| K | 8×4 | Kalman gain |

NumPy stores x and z as one-dimensional arrays with shapes `(8,)` and `(4,)`,
equivalent to the mathematical column vectors in these matrix products.
The reusable `KalmanFilter` also supports other state/measurement dimensions;
`motion.py` supplies Jake's eight-dimensional box model.

## Prediction and correction

```text
x_pred = F x
P_pred = F P F^T + Q
y      = z - H x_pred
S      = H P_pred H^T + R
K      = P_pred H^T S^-1
x_new  = x_pred + K y
```

The implementation obtains K using `numpy.linalg.solve(S, (P_pred H^T)^T)^T`,
not an explicit inverse. For covariance it uses the Joseph form:

```text
A     = I - K H
P_new = A P_pred A^T + K R K^T
```

This equals `(I - K H) P_pred` in exact arithmetic for the Kalman gain above,
but better resists cancellation and loss of positive semidefiniteness. All
arithmetic uses float64. P is symmetrized after correction; dimensions,
finiteness, symmetry, and covariance eigenvalues are validated. The PSD check
permits only a small roundoff tolerance of `1e-12`; R must be positive definite.
Bad matrices or failed solves raise `KalmanError`, wrapped as `TrackerError` at
the tracker boundary. No silent covariance reset or pseudoinverse is used.

Predict/correct return new filter instances. Input/output arrays are copied,
and an unsuccessful tracker update leaves the previous session state intact.

## Noise configuration

Defaults in `config/jake.example.toml`:

```toml
[tracking.kalman]
process_noise_position = 0.0001
process_noise_velocity = 0.001
measurement_noise = 0.001
initial_position_variance = 0.01
initial_velocity_variance = 1.0
```

These are variances/noise intensities, **not standard deviations**. All five must
be finite positive numbers; zero, negative, NaN, infinity, booleans, strings, and
unknown keys are rejected. Existing configurations receive these defaults.

```text
Q(dt) = diag([process_noise_position*dt] × 4,
             [process_noise_velocity*dt] × 4)
R     = measurement_noise * I4
P0    = diag([initial_position_variance] × 4,
             [initial_velocity_variance] × 4)
x0    = [measured cx, cy, w, h, 0, 0, 0, 0]^T
```

Q models independent position/size and velocity random walks with variance
growth proportional to elapsed time. It is a simple diagonal process-noise
model, not an integrated white-acceleration model. Position intensity has units
of normalized-coordinate variance per second; velocity intensity has units of
velocity variance per second. Increasing measurement noise trusts the motion
estimate more; increasing process noise allows more uncertainty and adaptation.
The defaults are development starting points, not a calibration for every camera.

## Time and lifecycle

dt comes from consecutive `FrameContext.captured_at` timestamps. Positive deltas
are clamped to **0.001–1.0 seconds**. Equal or backward timestamps fall back to
**1/30 second**, accommodating wall-clock correction without negative motion.
The first frame uses the same fallback policy but only initializes new tracks;
there is no previous state to advance. Timestamps are required by FrameContext;
there is no missing-timestamp sentinel. Long stalls therefore advance motion by
at most one second per update, rather than extrapolating arbitrarily far.
These bounds/fallback are explicit constants in `motion.py`.

Every active track is predicted before matching. Greedy IoU compares detections
to these **predicted** boxes. Matched tracks are corrected and preserve their IDs;
unmatched tracks retain the prediction and accumulate misses. The Phase 1C
expiration rule is unchanged: keep through `max_missed_frames` unmatched updates,
remove on the next. Misses and age count updates, not elapsed time or sequence
gaps. A new session restarts IDs; invalid camera IDs or non-increasing sequences
are rejected without state changes.

Each internal track owns its filter (x and P), current and predicted boxes,
last measurement, confidence, created/last-seen timestamps, age, visible count,
and missed count. `PersonTrack` remains minimal. Public boxes are corrected
estimates when matched and moving predictions when missed. Confidence stays at
the last detector confidence; it is not a probability that a missed person is
still present.

Before matching or public exposure, width/height are clamped to `[1e-6, 1]` and
centers are bounded so the entire box fits within the image. The internal linear
state remains unmodified by this geometry projection. Predictions fully outside
the image may sit at an edge publicly until they expire; no image history is kept.

## Running and diagnostics

Use the existing webcam and local weight configuration:

```sh
uv run --extra detection jake-camera --config config/local.toml --track --tracker kalman
```

Normal rendering retains `ID 1 | PERSON 94.2%`. Add `--debug-tracks` for orange
pre-correction predictions, blue measurements (only when matched), and per-track
ID/missed-count text. Green boxes are the current public estimate. Diagnostics
are immutable metadata snapshots supplied by the tracker; the preview owns no
motion or lifecycle state. Debug output is drawn locally and not persisted.
Use `--tracker iou` (the default) for the physically validated Phase 1C baseline.

## Deterministic comparison

```sh
uv run --extra tracking python -m jake.benchmarks
```

This needs only NumPy, no webcam, model weights, GPU, or network. Two known
synthetic people move in separate image bands. One disappears for three frames
in the occlusion case. IDs are associated back to visible synthetic detections
by IoU; the tool reports unique IDs created, changes from a person's last visible
ID, observed detections, and unmatched detections. Regression tests check these
results:

| Trajectory | IoU births / ID changes | Kalman births / ID changes |
| --- | --- | --- |
| Continuous motion | 2 / 0 | 2 / 0 |
| Three-frame occlusion | 3 / 1 | 2 / 0 |

Both methods associate every visible detection on these trajectories. This is a
repeatable behavior comparison, not a general MOT accuracy or hardware-speed
benchmark. The trajectories are deliberately simple and do not validate crowded
crossings.

## Why it helps and what remains

Last-observed boxes stay behind a moving person during occlusion. Estimated
velocity moves Jake's predicted box toward the likely reappearance location,
allowing IoU matching to reconnect the existing ID. Several initial observations
are needed to learn useful velocity. Sudden direction/size changes, long
occlusions, camera movement, and ambiguous crossings can still fragment or switch
IDs. Geometry alone cannot recover identity. A retained track can still attach
to another person; an edge-clamped prediction may match an unrelated edge detection.

Greedy association is unchanged: O(T×D) candidate generation and worst-case
O((T×D) log(T×D)) sorting, with O(T×D) candidate memory. Fixed 8×8 filter operations
add O(T) work and state. Phase 1E now provides selectable
[Hungarian/global assignment](global-assignment.md); use `--assignment greedy`
to reproduce the Phase 1D association baseline, or `--assignment hungarian`
for the new strategy. Its different complexity and gating policy are documented there.
Appearance embeddings and identity recognition are outside this phase. Track IDs
remain session-local, with no network calls or image/database persistence.
