# Phase 1E: Jake-owned global assignment

`hungarian.py` implements dense Hungarian linear assignment using row/column
potentials, reduced costs, and shortest augmenting paths. It uses Python's
standard library, with no SciPy solver, YOLO tracking, or external assignment
implementation. Kalman mathematics and public domain contracts are unchanged.

## Greedy versus global

Greedy sorts valid pairs by descending IoU, accepting each pair when both members
are unmatched. It can consume a detection needed by another track. Global
assignment considers the whole matrix and can sacrifice the strongest individual
pair to find a better combination. Both strategies are deterministic and one-to-one.

For Kalman tracking, costs use predicted boxes. For the IoU tracker, they use
last-observed boxes. Geometry and gating live in `matching.py`:

```text
cost[i,j] = 1 - IoU(track_box[i], detection_box[j])
valid[i,j] = IoU(track_box[i], detection_box[j]) >= min_iou
```

## Gating and unmatched choices

Below-threshold pairs are forbidden even when the matrix is square. Merely
solving then rejecting bad pairs could discard a valid alternative, so unmatched
choices are represented inside the optimization. For T tracks and D detections,
the wrapper constructs an N=T+D square matrix:

```text
                  D real detections     T dummy columns
T real tracks     valid cost / M        B
D dummy rows      B                     0

B = min(T, D) + 1
M = 2 * B * (T + D + 1)
```

Dummy columns allow unmatched tracks; dummy rows allow unmatched detections.
For k valid matches, total cost is `B*(T+D-2*k) + sum(valid match costs)`.
Since valid costs lie in [0,1], this prioritizes **maximum valid match count**,
then **minimum total 1-IoU cost**. M exceeds the all-unmatched solution, so an
optimal assignment never needs a forbidden pair. Output extraction defensively
checks the gate as well. This explicit policy treats every gate-passing pair as
eligible; weak geometric connections still need an appropriate IoU threshold.

Unmatched detections create tracks; unmatched tracks retain predictions and use
the existing missed-frame expiration. Zero tracks or detections produce no pairs.
Both rectangular orientations are supported without reusing rows or columns.

The generic `linear_assignment()` accepts a finite rectangular cost matrix,
pads it to max(rows, columns) with zero-cost dummy cells, and returns exactly
min(rows, columns) pairs. It has no tracking gate: the wrapper supplies the
additional unmatched nodes and penalties described above.

## Determinism, validation, and complexity

Rows and columns are scanned in ascending order; predecessor choices change only
on strictly lower cost. Output is sorted by row. Identical ordered inputs give
identical results, without random tie-breaking or epsilon perturbations. This
does not promise lexicographically minimal solutions or physical identity.

Ragged, non-real, boolean, NaN, and infinite costs raise `AssignmentError`.
Negative finite costs are allowed; inputs are never modified. Uniform positive
scaling bounds potential arithmetic for large finite costs without changing the
mathematical optimum. Computation uses floating-point, not arbitrary precision.
Tests compare small rectangular problems against exhaustive enumeration.

Hungarian time is approximately **O(n³)** and storage **O(n²)**. The generic
solver uses n=max(T,D); the gated tracking wrapper uses **n=T+D**. IoU cost
construction is O(T×D). Greedy remains O(T×D) for candidate construction and
worst-case O((T×D) log(T×D)) for sorting. Kalman filter work is separate.

## Configuration and commands

```toml
[tracking]
min_iou = 0.3
max_missed_frames = 10
assignment = "hungarian"
```

Allowed strategies are `greedy` and `hungarian`. Omitted settings retain the
Phase 1C/1D greedy default; the updated example explicitly selects Hungarian.
CLI `--assignment` overrides configuration for that run without writing the file.
It requires `--track` and works with either tracker implementation.

```sh
uv run --extra detection jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian
```

Use `--assignment greedy` for the physically validated Kalman baseline. Add
`--debug-tracks` to display the strategy, predictions, measurements, and missed
counts. Normal preview is unchanged.

## Benchmark and limitations

```sh
uv run --extra tracking python -m jake.benchmarks
```

Existing Phase 1C/1D comparisons remain. New crossing scenarios compare Kalman +
GREEDY and Kalman + HUNGARIAN, reporting total IDs created, ID switches, and total
assignment time over 26 frames. `perf_counter()` timing includes cost construction,
gating, and solving, excluding Kalman prediction/correction, inference, camera
acquisition, evaluation, and display. Timings vary; tests do not assert speed.

Two people cross in the same image band. The jittered variant adds known position
errors to both detections on frame 11. Ground truth is used only for evaluation:
the benchmark reads exact measurement associations from tracker diagnostics,
rather than judging outputs with another greedy spatial matcher. No frame has
identical measurement boxes; all 52 observations are accounted for.

| Scenario | Greedy IDs / switches | Hungarian IDs / switches |
| --- | --- | --- |
| Smooth crossing | 2 / 0 | 2 / 0 |
| Crossing with detector jitter | 2 / 4 | 2 / 4 |

The equal jittered result is reported honestly: global geometry is not identity.
A separate gated-bottleneck integration regression shows greedy creating a third
track while Hungarian preserves two valid track/detection connections.

Incorrect predictions, detector jitter, heavy overlap, long occlusion, and stale
edge predictions can still cause switches. The solver optimizes the specified
geometry objective, not real-world identity. No appearance embeddings, face
recognition, resident identity, or persistence are added. Track IDs are session-local;
the code makes no network calls and saves no images. Physical Phase 1E validation
on the user's camera remains outstanding.
