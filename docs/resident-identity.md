# Phase 2B: local resident face identity

## Boundaries

`Frame → person detection → body tracking → FaceIdentityService → IdentityMatch / IdentityEvent`

`FaceDetector`, `FaceEncoder`, `IdentityStore`, and `IdentityMatcher` are independent
protocols in `identity_ports.py`. `identity_domain.py` defines immutable face boxes,
normalized embeddings, resident profiles, and identity results. OpenCV model details
stay in `adapters/opencv_faces.py`; persistence stays in `LocalIdentityStore`.
The service only examines visible confirmed tracks and crops their person regions.
It skips regions with zero/multiple faces and overlapping observations of the same
face in two tracks. Enrollment uses the full frame and requires exactly one face.

Body appearance vectors remain short-term memory. Face templates are a distinct
type, model space, and persistent store. Neither resident names nor either kind of
embedding is attached to `PersonTrack`. Its new integer `continuity_epoch` communicates
a body reactivation boundary without exposing a specific tracker implementation.
Existing trackers, benchmarks, person event timing, and `PerceptionPipeline.process()`
return values are preserved. The optional pipeline service exposes separate
`identity_matches` and `identity_events` metadata after each call. Create fresh
stateful services per camera session; do not share across threads.

## Models and explicit setup

The initial adapters use OpenCV's CPU DNN backend, available in the `vision` extra:

| Component | Local model | Input / output | License |
| --- | --- | --- | --- |
| YuNet detector | `face_detection_yunet_2023mar.onnx` | BGR person crop, input size set per crop; boxes, five landmarks, scores | MIT |
| SFace encoder | `face_recognition_sface_2021dec.onnx` | Five-landmark alignment to 112×112, OpenCV RGB blob preprocessing; 128 values normalized to unit length | Apache-2.0 |

Model information and license files are published in the official OpenCV Zoo
[YuNet directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
and [SFace directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
The [OpenCV alignment/preprocessing implementation](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_recognize.cpp)
is used by the adapter. Models and weights are not bundled with Jake. Preserve the
upstream licenses when distributing them.

For later live setup, these are explicit network operations performed by the user,
not by Jake. From the repository root, Windows PowerShell:

```powershell
uv sync --locked --extra vision
New-Item -ItemType Directory -Force models
Invoke-WebRequest -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models/face_detection_yunet_2023mar.onnx"
Invoke-WebRequest -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" -OutFile "models/face_recognition_sface_2021dec.onnx"
```

Configure `[identity]` using `config/jake.example.toml`; existing configurations
receive the documented defaults. Paths are relative to the working directory,
not the TOML file. Both ONNX files must already exist; missing/invalid models fail
clearly without downloading. A SHA-256 fingerprint of the encoder weights is stored
with its model identifier. A different encoder file or vector dimension fails closed
and requires deliberate deletion/re-enrollment, not silent cross-model comparison.

SFace is a small CPU-oriented face encoder. YuNet operates per person crop, then
alignment and SFace run only for a quality-approved face. Recognition attempts are
limited to about 3.3 per second per visible track by default. Cost grows with people,
crop resolution, and CPU speed, and adds to person detection/appearance cost. No GPU
is required or selected by these adapters. No real-model latency or recognition
accuracy is claimed from mock tests; physical Phase 2B validation remains outstanding.

## Quality and enrollment

Quality is checked before encoding. Rejections have explicit reasons:

- Detector confidence below 0.90 or face width/height below 64 pixels.
- Grayscale Laplacian variance below 80 (a lighting/resolution-dependent blur proxy).
- A clipped face at the person crop boundary or within 0.5% of an image edge.
- Missing five landmarks, invalid eye/mouth geometry, excessive roll, or extreme pose.
  Normalized landmark proxies limit absolute yaw to 0.35, nose vertical position
  between eye/mouth levels to 0.20–0.85, and roll to 0.30. These are not calibrated
  pose angles. Small yaw variations form center/left/right sample bins.

Enrollment requires resident consent and a separate explicit command:

```sh
uv run --extra vision jake-enroll-resident --config config/local.toml --name "Joseph" --consent
```

Explain to each participating resident that Jake will save a face template locally,
how it is used, and how to delete it. `--consent` records deliberate operator intent
for the workflow; it is not an automated determination of another person's consent.
There is no visitor auto-enrollment. One consenting person should occupy the preview.
Progress shows accepted samples out of ten; slight head turns supply pose diversity.
Both `q`/`Q` and Ctrl+C cancel and release the camera. Partial enrollment is discarded.

Accepted samples must be at least 0.5 seconds apart, span at least two pose bins,
and have cosine similarity below 0.995 to every previously accepted sample (duplicate
rejection). Each must also be at least 0.60 similar to the existing samples to reject
inconsistent enrollment. These conservative heuristics can require more attempts.
The final sample cannot complete enrollment without pose diversity.

For accepted unit vectors, the single stored template is
`centroid = normalize(mean(e1, ..., en))`. It represents all accepted observations,
not an arbitrary frame. Original samples are released after enrollment. One centroid
minimizes stored biometric data; limited multiple templates are supported by the
domain/store but are not generated in this phase. Duplicate names (case-insensitive)
are rejected; deletion is explicit before re-enrollment.

## Matching and temporal policy

For unit vectors, cosine similarity is `s = dot(observation, template)`. A profile's
score is its maximum template score. Ties are deterministic by resident UUID.
Similarity is **not a probability** and thresholds require local validation.

| Setting | Default | Meaning |
| --- | --- | --- |
| `candidate_similarity` | 0.45 | Below this, UNKNOWN |
| `resident_similarity` | 0.60 | Minimum score for a supporting resident observation |
| `ambiguity_margin` | 0.08 | Required gap between best and second-best resident score |
| `required_confirmations` | 3 | Separate supporting observations for RESIDENT |
| `confirmation_window_seconds` | 3.0 | Rolling supporting-observation window |
| `observation_interval_seconds` | 0.3 | Minimum interval between face attempts/counts |
| `carry_seconds` | 2.0 | Maximum carry since the last supporting face observation |

An empty store, low score, or ambiguous best match yields UNKNOWN. A single strong
face observation yields only CANDIDATE. Three sufficiently spaced observations of
the same resident within the window yield RESIDENT. Contradictory, unknown, or weak
valid face evidence clears supporting evidence and demotes the match. A poor-quality
face does not count or change the match score. Independently, frame-time carry expiry
eventually clears stale identity. No wall-clock time is used for temporal decisions.
Frame sequences must increase, timestamps cannot decrease, and camera sessions cannot
be mixed. Timestamp leaps expire evidence without fabricating observations.

RESIDENT carries for at most two seconds when the same active/lost track's face is
hidden. It is not copied to a new track. Entering RECENTLY_LOST clears the public
identity; a successful body reactivation increments `continuity_epoch`, which also
clears evidence even if no intermediate lost snapshot was observed. Reactivation
therefore requires three fresh face confirmations before restoring the same resident
UUID/name. This deliberately suspends identity during body-only recovery: an incorrect
clothing match cannot inherit resident status. Person ENTERED/PRESENT/LEFT history and
duration remain continuous and unchanged through legitimate body ReID.

Separate identity events contain a local UUID, frame context, track ID, state, resident
UUID/name when available, and similarity metadata. A transition to RESIDENT emits
`IDENTITY_RESOLVED`; other changes emit `IDENTITY_STATE_CHANGED`. Neither event type
contains vectors or pixels. The preview shows `ID 7 | Joseph | RESIDENT`, with ordinary
tracking diagnostics retained. Console identity transitions are metadata only but
still disclose presence; terminal capture is controlled by the operator.

The full live command, after review and model/enrollment setup, is:

```sh
uv run --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --appearance --reid --events --identity
```

This uses the existing explicitly installed YOLO and OpenVINO appearance models too;
their setup remains in the README. To use geometry-only tracking with face identity,
omit `--extra appearance`, `--appearance`, and `--reid`. `--identity` requires `--track`;
without it no face model is loaded or identity store read.

## Storage, deletion, and limitations

**The version-1 store is plaintext, not encrypted.** A dedicated `.jake-identities/`
directory is ignored by Git. Custom paths must also be excluded or located outside
the repository. Use a newly created dedicated local directory, not a shared folder.
Only UUID, display name, enrollment timestamp/sample count, model fingerprint, and
normalized templates are saved. No image, audio, observation history, or raw crop is
written. Vectors are omitted from domain repr and events.

Writes use an exclusive local writer lock, a same-directory temporary file with
restricted permissions, flush/fsync, and atomic replacement. Readers see a complete
old or new document. Invalid versions, corrupt/non-normalized vectors, duplicate
records, and oversized stores are rejected without replacing records. The store
limits profiles to 100 and reads to 2 MB. A crash may leave a lock or a private
temporary template file; inspect only after all writers stop, then remove stale files.
The store is for one local filesystem, not network storage or distributed writers.

POSIX permissions are 0700/0600; Windows removes inherited ACL grants and grants the
current SID full control before writing templates. Preexisting explicit ACL grants
are not sanitized, another reason to use a fresh dedicated directory. Permission
setup failure aborts saving. Local administrators, same-account processes, backups,
and compromised hosts remain outside this protection. Python offers no secure memory
erasure guarantee. Robust encryption with OS-keychain-managed keys, key rotation, and
recovery is the immediate follow-up; no homemade encryption or committed key is used.

Deletion/re-enrollment procedure: stop running identity sessions first (they cache
profiles for their session), list local UUIDs, then explicitly delete the chosen UUID:

```sh
uv run --extra vision jake-enroll-resident --config config/local.toml --list
uv run --extra vision jake-enroll-resident --config config/local.toml --delete <resident-uuid>
```

Restarting creates a session without that profile. Re-enrollment uses the deliberate
consent command again and creates a new UUID. Remove any separately made backups or
stale temporary files as appropriate; deletion is not a promise of physical secure
erasure on the underlying drive.

Recognition is a development aid, not authorization for locks or sensitive actions.
There is no liveness/anti-spoofing: photos/screens can fool face recognition. Lighting,
occlusion, pose, demographic variation, lookalikes, model calibration, and body track
ID switches can all cause errors. Active-track carry is bounded but cannot eliminate
an undetected body ID switch. No automatic learning, visitor classification, persistent
visitor database, voice, LLM, or behavioral functionality is introduced.

All CI tests use synthetic vectors/images and mocked model/camera boundaries, with
no network, model download, GPU, or webcam. They verify policy and resource behavior,
not biometric accuracy. No live enrollment is part of automated validation.
