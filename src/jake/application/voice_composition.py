"""Explicit local voice composition, separate from vision and desktop imports."""

from jake.application.voice import VoiceService
from jake.config import AppConfig
from jake.voice_domain import VoiceError


def compose_voice(config: AppConfig) -> VoiceService:
    if not config.audio.enabled:
        raise VoiceError("Audio disabled; set audio.enabled=true to opt in")
    from jake.adapters.local_speech import FasterWhisperRecognizer, PiperSynthesizer, WebRtcVAD
    from jake.adapters.sounddevice_audio import SoundDeviceInput, SoundDeviceOutput

    model = None
    if config.conversation_ai.enabled:
        from jake.adapters.llama_conversation import LlamaCppConversationModel

        model = LlamaCppConversationModel(config.conversation_ai)
    capabilities: tuple[str, ...] = ("local conversation",)
    if config.identity.enabled or config.visitors.enabled:
        capabilities += ("resident recognition",)
    if config.visitors.enabled:
        capabilities += ("visitor recognition",)
    return VoiceService(
        config.audio,
        config.speech,
        config.interaction,
        SoundDeviceInput(config.audio),
        WebRtcVAD(config.speech),
        FasterWhisperRecognizer(config.stt, config.speech),
        PiperSynthesizer(config.tts),
        SoundDeviceOutput(config.audio),
        conversation_model=model,
        conversation_ai=config.conversation_ai,
        timezone=config.home.timezone,
        capabilities=capabilities,
    )
