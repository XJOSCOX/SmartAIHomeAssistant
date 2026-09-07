"""Explicit local voice composition, separate from vision and desktop imports."""

from jake.application.voice import VoiceService
from jake.config import AppConfig
from jake.voice_domain import VoiceError


def compose_voice(config: AppConfig) -> VoiceService:
    if not config.audio.enabled:
        raise VoiceError("Audio disabled; set audio.enabled=true to opt in")
    from jake.adapters.local_speech import FasterWhisperRecognizer, PiperSynthesizer, WebRtcVAD
    from jake.adapters.sounddevice_audio import SoundDeviceInput, SoundDeviceOutput

    return VoiceService(
        config.audio,
        config.speech,
        config.interaction,
        SoundDeviceInput(config.audio),
        WebRtcVAD(config.speech),
        FasterWhisperRecognizer(config.stt, config.speech),
        PiperSynthesizer(config.tts),
        SoundDeviceOutput(config.audio),
    )
