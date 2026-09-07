"""Local GGUF adapter boundary tests; no real weights or inference."""

import json
import socket
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jake.adapters.llama_conversation import LlamaCppConversationModel, LocalConversationError
from jake.conversation_ai_config import ConversationAIConfig
from jake.conversation_context import BASE_REPLIES, build_context
from jake.conversation_domain import ConversationRequest


def request() -> ConversationRequest:
    return ConversationRequest(
        build_context(
            "session", datetime.now(UTC), None, None, (), timezone="UTC", max_age=2, supported=()
        ),
        "hello",
    )


@pytest.fixture
def backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[LlamaCppConversationModel, Mock, Mock]:
    monkeypatch.setattr(
        socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden"))
    )
    monkeypatch.setattr(
        socket, "getaddrinfo", Mock(side_effect=AssertionError("network forbidden"))
    )
    path = tmp_path / "fake.gguf"
    path.write_bytes(b"fake asset for boundary testing")
    native = Mock()
    native.tokenize.return_value = list(range(10))
    native.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({"reply": BASE_REPLIES[0]})}}],
        "usage": {"prompt_tokens": 33, "completion_tokens": 9},
    }
    factory = Mock(return_value=native)
    monkeypatch.setitem(
        sys.modules, "llama_cpp", SimpleNamespace(Llama=factory, LogitsProcessorList=list)
    )
    model = LlamaCppConversationModel(ConversationAIConfig(enabled=True, model=str(path)))
    return model, native, factory


def test_local_model_flags_usage_and_no_persistence(
    backend: tuple[LlamaCppConversationModel, Mock, Mock], tmp_path: Path
) -> None:
    model, native, factory = backend
    assert model.status.state == "not_loaded" and not factory.called
    before = tuple(tmp_path.iterdir())
    result = model.generate(request())
    assert result.text == BASE_REPLIES[0]
    assert result.prompt_tokens == 33 and result.output_tokens == 9
    assert result.tokens_per_second is None and result.generation_ms >= 0
    assert model.status.load_ms is not None
    flags = factory.call_args.kwargs
    assert flags["n_gpu_layers"] == 0 and flags["verbose"] is False
    assert flags["n_ctx"] == 4096
    options = native.create_chat_completion.call_args.kwargs
    assert not options["stream"] and "tools" not in options
    assert options["response_format"]["schema"]["properties"]["reply"]["enum"]
    native.reset.assert_called_once()
    assert tuple(tmp_path.iterdir()) == before
    model.close()
    native.close.assert_called_once()


def test_missing_file_never_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock()
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=factory))
    model = LlamaCppConversationModel(ConversationAIConfig(model=str(tmp_path / "missing.gguf")))
    with pytest.raises(LocalConversationError, match="local conversation generation failed"):
        model.generate(request())
    factory.assert_not_called()


def test_context_pressure_drops_oldest_history(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend
    # First prompt too large; next prompt fits after oldest exchange is dropped.
    native.tokenize.side_effect = [list(range(n)) for n in (3000, 3000, 10, 10)]
    req = request()
    req = replace(
        req, context=replace(req.context, history=(("user", "oldest"), ("jake", "old reply")))
    )
    model.generate(req)
    payload = json.loads(native.create_chat_completion.call_args.kwargs["messages"][1]["content"])
    assert payload["conversation_history"] == []
    assert len(req.context.history) == 2


def test_context_overflow_fails_closed(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend
    native.tokenize.return_value = list(range(5000))
    with pytest.raises(LocalConversationError):
        model.generate(request())
    native.create_chat_completion.assert_not_called()


@pytest.mark.parametrize("content", ["not json", '{"reply":42}', '{"reply":"hi","secret":"x"}'])
def test_malformed_output_rejected(
    backend: tuple[LlamaCppConversationModel, Mock, Mock], content: str
) -> None:
    model, native, _ = backend
    native.create_chat_completion.return_value = {"choices": [{"message": {"content": content}}]}
    with pytest.raises(LocalConversationError):
        model.generate(request())
    assert model.status.state == "failed"
    native.reset.assert_called_once()


def test_cancellation_uses_supported_logits_hook(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend

    def generate(**options: object) -> None:
        model.cancel()
        callbacks = options["logits_processor"]
        assert isinstance(callbacks, list)
        callbacks[0]([], [])

    native.create_chat_completion.side_effect = generate
    with pytest.raises(LocalConversationError):
        model.generate(request())
    native.reset.assert_called_once()
