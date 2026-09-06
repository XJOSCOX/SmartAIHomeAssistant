# Phase 2A: short-term appearance-assisted tracking

This phase associates detected body appearances within a camera session. It does
not recognize faces, assign resident names, remember visitors, or establish a
long-term identity. Clothing and pose are temporary evidence for track continuity.
No training, database, cloud inference, image persistence, or later-phase AI is added.

Phase 2A.1 optionally extends unexpired confirmed tracks through a recently-lost
pool. The active-retention expiration described here remains the `--no-reid`
baseline. See [recently-lost design](recently-lost.md) for continuous events and timing.

## Architecture and model

`AppearanceEncoder.encode(Frame, BoundingBox)` returns an immutable, L2-normalized
`AppearanceEmbedding` containing a fixed-length float tuple. Dimension is fixed per
encoder/session, not hard-coded into the tracker. The domain validates unit length
and finite values; vectors are excluded from repr. `PersonDetection` gains an optional
appearance field, preserving two-argument construction. `PersonTrack` and semantic
events expose no vectors. The tracker validates dimensions before changing state.

The pipeline and preview inject the encoder between detection and tracking, using
`encode_detections`. Only the encoder sees RGB pixels. The tracker still implements
`PersonTracker.update(FrameContext, detections)`, receives metadata, and imports no
OpenVINO or other re-identification framework. `AppearanceKalmanPersonTracker` reuses
the stabilized lifecycle and Jake-owned Kalman filter. Cost policy is isolated in
`appearance_matching.py`; the generic Hungarian solver is unchanged. Both assignment
strategies, the IoU baseline, baseline Kalman, and stabilized geometry remain available.

The first adapter uses Open Model Zoo's **person-reidentification-retail-0287**, a
small OmniScaleNet/LCT person model (0.595 million parameters, 0.564 GFLOPs) with a
256-float output. The adapter crops the detected region, safely floors/ceils and
clips pixel bounds, converts RGB to BGR, resizes to 128×256, and supplies float32
NCHW `[1,3,256,128]` values in 0–255 range. No extra mean/scale is applied outside
the IR model. Output `[1,256]` is L2-normalized. See the
[official model specification](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0287/README.md).

Only matching IR XML/BIN models with this preprocessing contract are supported by
this adapter, not arbitrary model architectures merely sharing the same shapes.
A different encoder adapter can produce another dimension. CPU is deliberate for
the first implementation; model/framework replacement does not change tracking.
There is one synchronous encoder call per detection. CPU cost increases with people
count; no guaranteed FPS is claimed. Preview reports encoder time separately from
detector inference and includes both in overall preview FPS. Assignment benchmarks
exclude crop encoding, inference, camera acquisition, rendering, and filter work.

## Setup and live command (Windows PowerShell)

Stop any running preview before updating its environment. With the existing YOLO
setup already complete, install the extra and explicitly download the model pair
once, from the repository root:

```powershell
uv sync --locked --extra detection --extra appearance
New-Item -ItemType Directory -Force models
$reidBase = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0287/FP32"
Invoke-WebRequest -Uri "$reidBase/person-reidentification-retail-0287.xml" -OutFile models/person-reidentification-retail-0287.xml
Invoke-WebRequest -Uri "$reidBase/person-reidentification-retail-0287.bin" -OutFile models/person-reidentification-retail-0287.bin
```

These URLs are listed in the upstream
[model manifest](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0287/model.yml).
Weights are not bundled and Jake never automatically downloads missing files.
Alternatively copy an existing matching pair to the machine and configure the XML
path; its sibling BIN must share the same basename. Paths resolve from the working
directory. The model uses the upstream Apache-2.0 license linked in that manifest.

Add this optional table to `config/local.toml` without duplicating an existing table:

```toml
[tracking.appearance]
enabled = true
model = "models/person-reidentification-retail-0287.xml"
weight = 0.35
min_similarity = 0.45
ema_alpha = 0.8
max_embedding_age_seconds = 5.0
```

Run:

```powershell
uv run --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --events --debug-tracks
```

For a prepared offline environment:

```powershell
uv run --offline --no-sync jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --events --debug-tracks
```

`--appearance` enables the mode even when the file says false; `--no-appearance`
forces geometry-only Kalman. The file defaults to false for backwards compatibility.
The setting only applies to `--track --tracker kalman`; other tracker modes ignore
it. An explicit `--appearance` with an incompatible tracker is rejected. Debug labels
can include `ID 7 | CONFIRMED | app 0.91`, the similarity to the pre-update track
representation. New tracks, stale representations, and missed tracks show no score.
Omit `--debug-tracks` for the normal preview. q/Q and Ctrl+C keep existing cleanup.

## Representation, age, and matching

Unit vectors use cosine similarity `s = a·b` (the general formula is
`a·b / (||a|| ||b||)`). Compatible matched observations update the representation:

```text
e = normalize(alpha * e_old + (1-alpha) * e_new)
```

New/stale representations initialize directly from the current observation.
Defaults weight history by 0.8 to damp small changes while adapting to new views.
`appearance_at` advances only on matched encoded observations; lost tracks cannot
refresh it. Once age exceeds `max_embedding_age_seconds`, the vector is dropped
before matching, even if the track remains active. Missing/stale appearance falls
back to the original stabilized geometry policy and initializes from the next match.

**Appearance never resurrects expired IDs.** The existing `max_missed_seconds`
(default 1.5 seconds) still determines track expiry; the five-second embedding limit
does not extend that timeout. It matters when retention is configured longer or
the embedding-age limit is configured shorter. Entry confirmation and event timing
are unchanged: tentative tracks are silent, LOST does not imply LEFT, and duration
begins at confirmation. Embeddings are neither copied into events nor into a gallery.

For predicted-box IoU `I`, normalized center distance `d`, outer spatial gate `g`,
and usable appearance cosine `s`:

```text
valid = d <= g AND s >= min_similarity
cost = [w_iou*(1-I) + w_motion*(d/g) + w_appearance*(1-s)]
       / (w_iou + w_motion + w_appearance)
```

With default 0.6/0.4/0.35 weights, effective normalized weights are approximately
0.444/0.296/0.259. Appearance rejects incompatible pairs even with high IoU and
allows low-IoU matches throughout the outer gate when similarity passes. It cannot
override the hard outer distance gate. Without usable appearance, the original
inner-half low-IoU gate and geometry-only normalization apply. Hungarian still
maximizes valid match count before minimizing cost; greedy sorts ascending cost
with deterministic index ties. One-to-one association is preserved.

The 0.45 similarity threshold is a permissive starting gate; it is not calibrated
as an identity probability. The 0.35 weight contributes without dominating geometry.
The five-second age cap keeps evidence bounded by recent observations. All settings
are validated: weight (0,1], similarity [0,1], alpha [0,1), finite positive age,
boolean enabled, and local XML path. Thresholds need camera-specific live validation.

Candidate cost construction is O(T×D×E) for embedding dimension E, greedy sorting
O(TD log(TD)), and Hungarian O((T+D)³). Appearance storage is O(T×E), with no historical
crop buffer or expired-track embedding archive. Continuously matched tracks refresh
their rolling representation; age limits time since observation, not session length.

## Privacy and limitations

OpenVINO is pinned to the inspected 2026.3.1 runtime because its top-level import
can invoke conversion-tool telemetry. **Before importing it**, the adapter blocks
the optional `openvino_telemetry` module for the Jake process, causing OpenVINO to
use its bundled no-op telemetry stub. It does not modify the user's global consent
file. If OpenVINO was already imported with telemetry available, startup rejects
the session and asks for a fresh process. This policy is process-wide; use a dedicated
Jake process and re-audit imports when upgrading the runtime. The optional dependency
installs the telemetry package transitively, but Jake blocks its use. CPU compilation
disables disk model caching. No crop saving, uploads, telemetry calls, network
inference, or embedding writes occur in Jake's encoding/tracking paths.

Frames/crops and framework inference buffers exist transiently in memory. Jake's
encoder stores only the compiled model; the runtime may retain its latest input
buffer until reuse or process exit. This is not guaranteed secure memory erasure.
Events remain metadata-only. Appearance vectors can still reveal personal attributes;
session-local use does not make them anonymous. Do not add persistence without a
separate retention and access-control design.

Similar outfits, clothing changes, lighting, pose, truncated crops, detector errors,
and heavy occlusion can defeat the encoder. Wrong matches can contaminate EMA history.
Motion beyond the spatial gate or re-entry after timeout always produces new IDs.
This is short-term body-appearance continuity, not face recognition or resident identity.

## Reproducible synthetic comparison

```sh
uv run --extra tracking python -m jake.appearance_benchmark
```

Six independent 12-update fixtures use 256-dimensional synthetic descriptors.
Ground truth scores exact measured-box associations only; it is never given to
tracking. The descriptors deliberately distinguish some people and make others
nearly identical. These tests validate association behavior, **not pretrained-model
accuracy on real footage**. Metrics:

- IDs created: unique track IDs observed, including tentative tracks.
- ID switches: changes in a fixture person's assigned ID between observations.
- Fragmentation: distinct IDs for each person beyond their first, summed.
- False re-associations: observations assigned to an ID initially owned by another
  fixture person. Continued wrong assignments count repeatedly, not just once.
- Assignment latency: total cost/gating/solver time over the 12 updates.

| Scenario | IDs geometry/app | Switches geometry/app | Fragmentation geometry/app | False associations geometry/app |
| --- | --- | --- | --- | --- |
| Temporary occlusion | 2 / 1 | 1 / 0 | 1 / 0 | 0 / 0 |
| Edge/truncated box | 2 / 1 | 1 / 0 | 1 / 0 | 0 / 0 |
| Low-IoU re-entry | 2 / 1 | 1 / 0 | 1 / 0 | 0 / 0 |
| Crossing | 2 / 2 | 2 / 0 | 2 / 0 | 18 / 0 |
| Appearance noise at crossing | 2 / 2 | 2 / 0 | 2 / 0 | 18 / 0 |
| Similar outfits | 2 / 2 | 2 / 2 | 2 / 2 | 18 / 18 |
| Total (six independent sessions) | 12 / 9 | 9 / 2 | 9 / 2 | 54 / 18 |

One Windows run measured approximately 1.49 ms geometry versus 2.76 ms appearance
assignment across all 72 updates. Timing varies; encoder inference is excluded.
The similar-outfit case intentionally remains unresolved. Previous Phase 1C–1G
benchmarks are preserved. Webcam and pretrained-model live validation remain pending.
