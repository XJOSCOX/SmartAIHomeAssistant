"""Explicit voice development entry point; metadata diagnostics only."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from threading import Event

from jake.application.perception_session import PerceptionOverrides, configure_perception
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
    for feature in ("identity", "visitors", "appearance", "reid"):
        parser.add_argument(f"--{feature}", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tracker", choices=("kalman", "kalman-baseline", "iou"))
    parser.add_argument("--assignment", choices=("greedy", "hungarian"))
    args = parser.parse_args(argv)
    if args.preview and not args.with_camera:
        parser.error("--preview requires --with-camera")
    if not args.with_camera and any(
        getattr(args, name) is not None
        for name in ("identity", "visitors", "appearance", "reid", "tracker", "assignment")
    ):
        parser.error("perception overrides require --with-camera")
    overrides = PerceptionOverrides(
        args.tracker or "kalman",
        args.assignment,
        args.appearance,
        args.reid,
        args.identity,
        args.visitors,
    )
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
        if args.with_camera:
            # Validate before opening audio or loading speech models.
            config = configure_perception(config, overrides).config
        else:
            config = replace(
                config,
                identity=replace(config.identity, enabled=False),
                visitors=replace(config.visitors, enabled=False),
            )
        voice = compose_voice(config)
        voice.start()
        if args.with_camera:
            from jake.application.voice_camera import VoiceCamera

            camera = VoiceCamera(config, voice, preview=args.preview, overrides=overrides)
            if args.preview:
                from jake.voice_preview import VoicePreview

                display = VoicePreview(config.interaction.visual_context_max_age_seconds)
            camera.start()
        print(f"conversation AI: {'enabled' if config.conversation_ai.enabled else 'disabled'}")
        print("Jake voice active locally; half-duplex. Ctrl+C stops. No transcript logging.")
        delay = Event()
        previous = None
        previous_ai = None
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
            if config.conversation_ai.enabled and voice.ai is not None:
                ai = voice.ai.metrics
                diagnostic = (ai, voice.ai_fallback_used)
                if diagnostic != previous_ai:
                    print(
                        f"conversation AI: {ai.state} load={ai.load_ms}ms "
                        f"LLM latency={ai.generation_ms}ms prompt_tokens={ai.prompt_tokens} "
                        f"output_tokens={ai.output_tokens} tokens/sec={ai.tokens_per_second} "
                        f"fallback used: {voice.ai_fallback_used or ai.fallback_used}"
                    )
                    previous_ai = diagnostic
            for event in voice.drain_events():
                print(f"{event.kind} track={event.track_id or 'unresolved'}")
    except KeyboardInterrupt:
        return 0
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
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
