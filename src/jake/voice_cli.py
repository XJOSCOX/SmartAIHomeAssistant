"""Explicit voice development entry point; metadata diagnostics only."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from threading import Event

from jake.config import load_app_config
from jake.voice_domain import VoiceError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Jake offline voice (no audio/transcript recording)"
    )
    parser.add_argument("--config", type=Path, default=Path("config/local.toml"))
    parser.add_argument(
        "--list-devices", action="store_true", help="List PortAudio devices without opening streams"
    )
    parser.add_argument(
        "--with-camera",
        action="store_true",
        help="Run existing perception on a separate worker for conservative visual association",
    )
    parser.add_argument("--preview", action="store_true", help="Show annotated camera preview")
    args = parser.parse_args(argv)
    if args.preview and not args.with_camera:
        parser.error("--preview requires --with-camera")
    voice = camera = None
    display = None
    try:
        if args.list_devices:
            from jake.adapters.sounddevice_audio import devices

            for device in devices():
                print(
                    f"{device.index}: {device.description} [{device.host_api}] "
                    f"inputs={device.input_channels} outputs={device.output_channels} "
                    f"default_rate={device.default_sample_rate:g}"
                )
            return 0
        from jake.application.voice_composition import compose_voice

        config = load_app_config(args.config)
        voice = compose_voice(config)
        voice.start()
        if args.with_camera:
            from jake.application.voice_camera import VoiceCamera

            camera = VoiceCamera(config, voice, preview=args.preview)
            if args.preview:
                from jake.voice_preview import VoicePreview

                display = VoicePreview(config.interaction.visual_context_max_age_seconds)
            camera.start()
        print("Jake voice active locally; half-duplex. Ctrl+C stops. No transcript logging.")
        delay = Event()
        previous = None
        while not delay.wait(0.01 if display else 0.25):
            if voice.metrics.input_status.startswith("failed") or camera and camera.error:
                raise VoiceError(
                    "Local voice/camera worker failed; check runtime and device configuration"
                )
            if (
                display is not None
                and camera is not None
                and not display.update(camera.preview_snapshot())
            ):
                break
            if camera is not None and camera.finished.is_set():
                break
            metrics = voice.metrics
            if metrics != previous:
                print(
                    f"audio={metrics.input_status} VAD={metrics.vad_state} "
                    f"utterance={metrics.utterance_seconds:.2f}s STT={metrics.stt_ms:.1f}ms "
                    f"TTS={metrics.tts_ms:.1f}ms RTF={metrics.stt_real_time_factor}"
                )
                previous = metrics
            for event in voice.drain_events():
                print(f"{event.kind} track={event.track_id or 'unresolved'}")
    except KeyboardInterrupt:
        return 0
    except VoiceError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Voice setup failed: {type(exc).__name__}; check configuration and voice extra",
            file=sys.stderr,
        )
        return 1
    finally:
        if camera is not None:
            camera.request_stop()
        try:
            if voice is not None:
                voice.close()
        finally:
            if camera is not None:
                camera.close()
            if display is not None:
                display.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
