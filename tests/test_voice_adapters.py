"""Fake native/model boundaries. No microphone, speaker, downloads or weights."""

import os
import socket
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jake.adapters.local_speech import FasterWhisperRecognizer, PiperSynthesizer, WebRtcVAD
from jake.adapters.sounddevice_audio import SoundDeviceInput, SoundDeviceOutput, devices
from jake.voice_config import AudioConfig, SpeechConfig, STTConfig, TTSConfig
from jake.voice_domain import AudioChunk, SpeechRequest, SpeechResult, SpeechSegment, VoiceError

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_device_descriptions_and_stream_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = Mock()
    sd = Mock(RawInputStream=Mock(return_value=stream))
    sd.query_hostapis.return_value = [{"name": "Windows WASAPI"}]
    sd.query_devices.return_value = [
        {
            "name": "USB Mic",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000,
        }
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    found = devices()
    assert found[0].description == "USB Mic" and found[0].host_api == "Windows WASAPI"
    stream.start.assert_not_called()
    source = SoundDeviceInput(AudioConfig(enabled=True, input_device="USB Mic", queue_chunks=2))
    source.start()
    assert sd.RawInputStream.call_args.kwargs["samplerate"] == 16000
    for _ in range(10):
        source._callback(bytes(960), 480, None, False)
    assert source.queue.qsize() == 2 and source.dropped_chunks == 8
    audio = source.read(0.01)
    assert audio is not None and audio.sequence == 9
    assert "pcm" not in repr(audio)
    source.close()
    source.close()
    source._callback(bytes(960), 480, None, False)
    assert source.queue.empty()
    stream.abort.assert_called_once()
    stream.close.assert_called_once()


def test_audio_open_failure_and_disabled_do_not_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = Mock(start=Mock(side_effect=RuntimeError("device failed")))
    sd = Mock(RawInputStream=Mock(return_value=stream))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    with pytest.raises(VoiceError, match="disabled"):
        SoundDeviceInput(AudioConfig()).start()
    sd.RawInputStream.assert_not_called()
    source = SoundDeviceInput(AudioConfig(enabled=True))
    with pytest.raises(VoiceError, match="microphone"):
        source.start()
    stream.abort.assert_called_once()
    stream.close.assert_called_once()


def test_output_uses_model_sample_rate_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = Mock()
    sd = Mock(RawOutputStream=Mock(return_value=stream))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    output = SoundDeviceOutput(AudioConfig(output_device=3))
    output.play(SpeechResult(bytes(4410), 22050, 10), Event())
    assert sd.RawOutputStream.call_args.kwargs["samplerate"] == 22050
    assert sd.RawOutputStream.call_args.kwargs["device"] == 3
    stream.stop.assert_called_once()
    stream.close.assert_called_once()
    assert b"".join(c.args[0] for c in stream.write.call_args_list) == bytes(4410)


def test_vad_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock(is_speech=Mock(return_value=True))
    module = Mock(Vad=Mock(return_value=model))
    monkeypatch.setitem(sys.modules, "webrtcvad", module)
    vad = WebRtcVAD(SpeechConfig(vad_mode=3))
    assert vad.is_speech(AudioChunk(0, NOW, bytes(960)))
    module.Vad.assert_called_once_with(3)
    model.is_speech.assert_called_once_with(bytes(960), 16000)


def test_stt_local_only_quality_filters_and_real_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).touch()
    decoded = [
        SimpleNamespace(text="hello", avg_logprob=-0.2, no_speech_prob=0.1, compression_ratio=1),
        SimpleNamespace(text="bad noise", avg_logprob=-4, no_speech_prob=0.9, compression_ratio=4),
    ]
    engine = Mock(transcribe=Mock(return_value=(iter(decoded), SimpleNamespace(language="en"))))
    module = Mock(WhisperModel=Mock(return_value=engine))
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    recognizer = FasterWhisperRecognizer(STTConfig(model=str(tmp_path)), SpeechConfig())
    assert module.WhisperModel.call_args.kwargs["local_files_only"] is True
    assert module.WhisperModel.call_args.kwargs["cpu_threads"] == 2
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    segment = SpeechSegment(NOW, NOW + timedelta(seconds=0.03), bytes([0, 64]) * 480)
    result = recognizer.transcribe(segment)
    assert result.text == "hello" and result.confidence is None and result.processing_ms >= 0
    assert np.all(engine.transcribe.call_args.args[0] == 0.5)
    assert engine.transcribe.call_args.kwargs["condition_on_previous_text"] is False
    assert engine.transcribe.call_args.kwargs["vad_filter"] is False
    assert "hello" not in repr(result)
    engine.transcribe.side_effect = RuntimeError("private transcript")
    with pytest.raises(VoiceError, match="Local transcription failed"):
        recognizer.transcribe(segment)


def test_missing_models_fail_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = Mock()
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    with pytest.raises(VoiceError, match="existing local"):
        FasterWhisperRecognizer(STTConfig(model=str(tmp_path)), SpeechConfig())
    module.WhisperModel.assert_not_called()
    with pytest.raises(VoiceError, match="existing local"):
        PiperSynthesizer(TTSConfig(voice=str(tmp_path / "voice.onnx")))


def test_piper_in_memory_telemetry_disabled_and_output_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "voice.onnx"
    path.touch()
    Path(str(path) + ".json").write_text('{"phoneme_type":"espeak"}', encoding="utf-8")
    ort = Mock()
    engine = Mock(
        synthesize=Mock(
            return_value=[
                SimpleNamespace(
                    sample_rate=22050,
                    sample_width=2,
                    sample_channels=1,
                    audio_int16_bytes=bytes(100),
                )
            ]
        )
    )
    module = Mock(PiperVoice=Mock(load=Mock(return_value=engine)))
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "piper", module)
    synthesizer = PiperSynthesizer(TTSConfig(voice=str(path)))
    ort.disable_telemetry_events.assert_called_once()
    assert engine.use_tashkeel is False
    assert module.PiperVoice.load.call_args.kwargs["use_cuda"] is False
    result = synthesizer.synthesize(SpeechRequest("Hello"))
    assert result.sample_rate == 22050 and result.pcm == bytes(100)
    assert set(p.name for p in tmp_path.iterdir()) == {"voice.onnx", "voice.onnx.json"}
    engine.synthesize.return_value = [
        SimpleNamespace(
            sample_rate=1, sample_width=2, sample_channels=1, audio_int16_bytes=bytes(122)
        )
    ]
    with pytest.raises(VoiceError):
        synthesizer.synthesize(SpeechRequest("Hello"))


def test_cli_device_listing_without_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jake.adapters.sounddevice_audio import AudioDevice
    from jake.voice_cli import main

    monkeypatch.setattr(
        "jake.adapters.sounddevice_audio.devices",
        Mock(return_value=(AudioDevice(2, "Mic", "Host", 1, 0, 16000),)),
    )
    assert main(["--list-devices"]) == 0
    assert "2: Mic [Host]" in capsys.readouterr().out


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("voice adapters must not use the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def test_download_capable_phonemizers_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "voice.onnx"
    path.touch()
    Path(str(path) + ".json").write_text('{"phoneme_type":"pinyin"}', encoding="utf-8")
    module = Mock()
    monkeypatch.setitem(sys.modules, "piper", module)
    with pytest.raises(VoiceError):
        PiperSynthesizer(TTSConfig(voice=str(path)))
    module.PiperVoice.load.assert_not_called()


def test_unexpected_microphone_stop_is_reported() -> None:
    source = SoundDeviceInput(AudioConfig(enabled=True))
    source._finished.set()
    with pytest.raises(VoiceError, match="stopped unexpectedly"):
        source.read(0.001)
    source.close()
