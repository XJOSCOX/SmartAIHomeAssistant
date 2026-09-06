# Phase 2G: native camera detail

Camera resolution and model inference resolution are independent. Jake retains the
original immutable packed RGB8 `Frame` through the pipeline. This phase does not
change tracking, resident matching, visitor thresholds, encrypted storage or day counts.

## Configuration and negotiation

Device-only configurations remain valid. `width` and `height` are independently
optional positive integers; `fps` is an optional finite positive number. Omission
means leave the driver's default alone. Booleans are not numbers here. `fourcc` is
optional and must be exactly four printable ASCII characters. No specific list of
resolutions is imposed. Settings omits unset optional values when saving TOML.

The OpenCV adapter opens the explicit local index/backend, requests FOURCC first,
then width, height and FPS. It reads back all four properties and the backend name.
A rejected optional setting produces a warning, not an acquisition failure. Missing
or invalid readback is shown as unknown, never inferred from requested values.
The first and subsequent captured arrays supply authoritative Frame width/height and
update camera metadata if the driver reported stale dimensions.

`CameraInfo` is immutable session metadata outside the perception Frame contract.
It reports requested/actual mode, codec, backend, optional-property warnings and
measured delivered capture FPS. `FrameSource` remains unchanged; alternative adapters
need not implement this optional metadata to work with the desktop controller.

A known resolution mismatch or FPS difference greater than 0.5 FPS is marked
`REQUEST NOT HONORED`. The FPS tolerance allows common 29.97 versus 30 rounding.
Jake continues at the actual resolution. There is no implicit strict-mode failure,
fallback backend retry or attempt to invent a camera capability list.

| Backend | Behavior |
| --- | --- |
| `auto` (default) | OpenCV chooses the installed platform backend. |
| `dshow` | Explicit Windows DirectShow; useful for UVC mode negotiation. |
| `msmf` | Explicit Windows Media Foundation; supported modes can differ from DirectShow. |
| `v4l2` | Explicit Linux Video4Linux2 for USB cameras on PCs, Jetson and Raspberry Pi. |
| `gstreamer` | Selects an installed OpenCV GStreamer backend for a numeric device. It does not add pipeline-string or CSI support. |

An unavailable explicit backend produces a camera-open error and releases the
capture. Windows backends are never selected automatically on Linux. Qt discovery
still provides only device descriptions; its ordering can differ from OpenCV.
MJPG may make larger UVC modes feasible within USB bandwidth, but support and mode
ordering are driver-specific. Actual hardware mode negotiation awaits live review.

## Example camera sections

Replace only `[camera]` in an existing local configuration; retain its local model
paths, identity/visitor opt-ins, household timezone and other settings.

1080p30 request:

```toml
[camera]
device = 0
width = 1920
height = 1080
fps = 30
backend = "auto"
fourcc = "MJPG"
```

4K30 request:

```toml
[camera]
device = 0
width = 3840
height = 2160
fps = 30
backend = "auto"
fourcc = "MJPG"
```

For explicit Windows negotiation, change `backend` to `dshow` or `msmf`. These are
requests, not a promise that a webcam can produce those modes. Omit FOURCC if the
device/backend does not support codec selection. There is no phone-specific camera
integration or assumption about a phone's USB webcam output resolution.

## Image path and face quality

1. OpenCV translates its original BGR array into packed RGB8 bytes without resizing.
2. YOLO receives a contiguous BGR copy at the original dimensions. Ultralytics performs
   its own resizing/letterboxing according to `detector.image_size` (still 640 by
   default). Its returned pixel boxes are normalized by the original source width
   and height. Jake adds no second global resize or coordinate transformation.
3. Body appearance encoding crops the original person region before its existing
   retail-0287 preprocessing to 128x256. Its encoder and tracking algorithm are unchanged.
4. YuNet receives the original person's pixel ROI selected by a normalized track box.
   Returned face coordinates/landmarks map back to the original frame. Face quality
   and SFace alignment/encoding use that same original frame. Visitor evidence comes
   from this face stage, not from a reduced YOLO image or a body embedding.

Increasing genuine capture resolution can put more pixels across a distant face.
Upscaling a low-resolution image cannot recover detail. Megapixels alone cannot
overcome poor optics, lighting, compression, blur, pose or occlusion. The existing
64-pixel minimum face dimension and all other quality/identity safeguards remain.

`FaceQuality` now contains optional `face_width_px`, `face_height_px` and the minimum
pixel threshold. Its machine-readable rejection reason remains stable. Human-facing
diagnostics expand it to, for example:

```text
face too small: 47x53 px; minimum 64 px
accepted; Face 96x112 px
```

Both enrollment interfaces use this message alongside accepted sample counts.
Desktop track diagnostics and CLI `--debug-tracks` show accepted/rejected face sizes;
visitor rejection diagnostics include measured small-face dimensions too. This
metadata is session-only, cleared as observations advance, and never written into
resident or visitor profiles. No vectors are logged.

## Cadence, timing and memory

The existing live identity observation interval is unchanged (0.3 seconds by default,
per visible confirmed track). Face quality and embedding are not run every frame.
Enrollment now also skips face work between attempts, using the greater of that
interval and its existing 0.5-second sample spacing. The preview continues updating
during skipped attempts and retains the last quality message. All cadence uses
Frame timestamps; processing measurements use a monotonic high-resolution timer.

The desktop Settings form adds width, height, FPS, backend and optional FOURCC without
changing navigation. Both live and enrollment workers use the same CameraConfig.
Live status shows requested/actual mode and driver-reported FPS separately from
measured delivered capture FPS and processing FPS. Enrollment shows mode information
with its quality/progress message. CLI previews print current-mode diagnostics and
keep their existing commands and configuration path behavior.

Delivered capture FPS is `(successful frames - 1) / elapsed time between completed
captures`. It starts as unknown. Acquisition is synchronous: slow processing lowers
this rate, and driver buffering can affect it. It is not the sensor's true exposure
rate or the configured FPS. Processing FPS measures session work excluding camera
acquisition; CLI also retains its total preview-throughput FPS. YOLO latency,
appearance latency and face-stage latency are measured separately. A skipped face
observation has negligible face-stage time; this does not imply expensive inference
ran at that speed. Visitor bookkeeping is outside the face-stage timer.

An RGB8 frame is about 6.2 MB at 1080p and 24.9 MB at 4K. Jake keeps the necessary
BGR-to-RGB conversion and immutable byte copy. NumPy reconstructs read-only views of
the bytes; only adapter inputs/crops and the visualization boundary make their needed
owned copies. There is no new full-frame resize, queue, history or frame retention.
The desktop mailbox stays bounded. USB bandwidth, conversion/copy cost, CPU face ROI
size and model inference all affect achievable throughput; no physical FPS or
recognition improvement is claimed by synthetic tests.

## Validation and review

Hardware-free tests cover mode settings/readback, mismatches, backend mapping,
unsupported properties, cleanup, actual image dimensions, source-relative YOLO
boxes from VGA through 4K and arbitrary modes, original identity/visitor evidence,
YuNet/SFace native crops, face-size rejection diagnostics, observation cadence,
enrollment configuration and desktop metadata. Existing algorithm/storage tests remain.

After repository review, the existing live command is:

```powershell
uv run --extra identity --extra detection --extra appearance jake-camera --config config/local.toml --track --tracker kalman --assignment hungarian --identity --events --debug-tracks
```

Visitor memory follows the existing opt-in in the local config; no new opt-in is
implied by this command. The desktop development entry remains
`uv run --extra desktop --extra identity --extra detection --extra appearance jake-desktop`.
No camera or real biometric store is opened as part of Phase 2G automated validation.

Validation result: **669 tests passed, 93% total coverage**. Ruff check, Ruff format
check, strict mypy and Python wheel/source-distribution builds passed. No physical
capture, native camera mode negotiation or distant-face accuracy test was run.

Changed files (22):

- Configuration and camera: `config/jake.example.toml`, `src/jake/config.py`,
  `src/jake/camera_info.py`, `src/jake/adapters/opencv_camera.py`.
- Face metadata: `src/jake/adapters/opencv_faces.py`, `src/jake/identity_domain.py`,
  `src/jake/identity.py`, `src/jake/face_identity.py`.
- Timing and integration: `src/jake/pipeline.py`, `src/jake/preview.py`,
  `src/jake/enroll_cli.py`, `src/jake/application/sessions.py`,
  `src/jake/application/settings.py`, `src/jake/desktop/controller.py`,
  `src/jake/desktop/widgets.py`, `src/jake/desktop/window.py`.
- Tests: `tests/test_camera_modes.py`, `tests/test_desktop.py`, `tests/test_identity.py`.
- Documentation: `README.md`, `docs/desktop-control-center.md`, this guide.
