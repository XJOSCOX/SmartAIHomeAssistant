"""Strict opt-in local conversational model settings."""

from dataclasses import dataclass

from jake.voice_config import integer, number


@dataclass(frozen=True, slots=True)
class ConversationAIConfig:
    enabled: bool = False
    backend: str = "llama_cpp"
    model: str = "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    context_tokens: int = 4096
    max_output_tokens: int = 96
    temperature: float = 0.2
    top_p: float = 0.9
    threads: int = 4
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or self.backend != "llama_cpp":
            raise ValueError("invalid conversation AI opt-in/backend")
        if (
            not isinstance(self.model, str)
            or not self.model.endswith(".gguf")
            or "://" in self.model
            or not self.model.isprintable()
        ):
            raise ValueError("conversation_ai.model must be a local GGUF path")
        integer(self.context_tokens, "context_tokens", 2048, 16384)
        integer(self.max_output_tokens, "max_output_tokens", 16, 256)
        integer(self.threads, "conversation threads", 1, 32)
        number(self.temperature, "temperature", 0, 1)
        number(self.top_p, "top_p", 0.01, 1)
        number(self.timeout_seconds, "conversation timeout", 0.1, 120)
