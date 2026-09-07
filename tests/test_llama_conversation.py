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
    native.metadata = {"tokenizer.chat_template": "{{ messages }}"}
    native.token_eos.return_value = 1
    native.token_bos.return_value = 2
    native.detokenize.return_value = b"token"
    formatter = Mock(return_value=SimpleNamespace(prompt="formatted prompt", added_special=True))
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp.llama_chat_format",
        SimpleNamespace(Jinja2ChatFormatter=Mock(return_value=formatter)),
    )
    native.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({"reply": BASE_REPLIES[0]})}}],
        "usage": {"prompt_tokens": 33, "completion_tokens": 9},
    }
    factory = Mock(return_value=native)
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=factory, LogitsProcessorList=list, LlamaGrammar=Mock()),
    )
    model = LlamaCppConversationModel(ConversationAIConfig(enabled=True, model=str(path)))
    return model, native, factory


def test_local_model_flags_usage_and_no_persistence(
    backend: tuple[LlamaCppConversationModel, Mock, Mock], tmp_path: Path
) -> None:
    model, native, factory = backend
    assert model.status.state == "not_loaded" and not factory.called
    before = tuple(tmp_path.iterdir())
    model.load()
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
    assert options["grammar"] is not None
    assert "response_format" not in options
    native.reset.assert_called_once()
    assert tuple(tmp_path.iterdir()) == before
    model.close()
    native.close.assert_called_once()


def test_missing_file_never_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock()
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=factory))
    model = LlamaCppConversationModel(ConversationAIConfig(model=str(tmp_path / "missing.gguf")))
    with pytest.raises(LocalConversationError, match="MODEL_NOT_FOUND"):
        model.load()
    factory.assert_not_called()


def test_context_pressure_drops_oldest_history(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend
    # First prompt too large; next prompt fits after oldest exchange is dropped.
    native.tokenize.side_effect = [list(range(n)) for n in (5000, 10)]
    req = request()
    req = replace(
        req, context=replace(req.context, history=(("user", "oldest"), ("jake", "old reply")))
    )
    model.load()
    model.generate(req)
    payload = json.loads(native.create_chat_completion.call_args.kwargs["messages"][1]["content"])
    assert payload["conversation_history"] == []
    assert len(req.context.history) == 2


def test_context_overflow_fails_closed(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend
    native.tokenize.return_value = list(range(5000))
    model.load()
    with pytest.raises(LocalConversationError):
        model.generate(request())
    native.create_chat_completion.assert_not_called()


@pytest.mark.parametrize("content", ["not json", '{"reply":42}', '{"reply":"hi","secret":"x"}'])
def test_malformed_output_rejected(
    backend: tuple[LlamaCppConversationModel, Mock, Mock], content: str
) -> None:
    model, native, _ = backend
    native.create_chat_completion.return_value = {"choices": [{"message": {"content": content}}]}
    model.load()
    with pytest.raises(LocalConversationError):
        model.generate(request())
    assert model.status.state == "ready"
    native.reset.assert_called_once()


def test_cancellation_uses_supported_logits_hook(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend

    def generate(**options: object) -> None:
        model.cancel_current()
        callbacks = options["logits_processor"]
        assert isinstance(callbacks, list)
        callbacks[0]([], [])

    native.create_chat_completion.side_effect = generate
    model.load()
    with pytest.raises(LocalConversationError):
        model.generate(request())
    native.reset.assert_called_once()


def test_invalid_json_and_policy_rejection_recover(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, factory = backend
    model.load()
    valid = native.create_chat_completion.return_value
    for content, code in (
        ("bad json", "INVALID_JSON"),
        ('{"reply":"Joseph is home."}', "POLICY_REJECTED"),
    ):
        native.create_chat_completion.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        with pytest.raises(LocalConversationError, match=code):
            model.generate(request())
        assert model.status.state == "ready"
        native.create_chat_completion.return_value = valid
        assert model.generate(request()).text == BASE_REPLIES[0]
    factory.assert_called_once()


def test_cancel_current_then_next_generation_and_terminal_shutdown(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    from threading import Event

    model, native, _ = backend
    model.load()
    cancelled = Event()
    cancelled.set()
    with pytest.raises(LocalConversationError, match="CANCELLED"):
        model.generate(request(), cancel=cancelled)
    assert model.generate(request()).text == BASE_REPLIES[0]
    model.cancel()
    with pytest.raises(LocalConversationError, match="CANCELLED"):
        model.generate(request())
    model.close()
    assert model.status.state == "closed"


def test_native_load_failure_has_safe_code(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, _, factory = backend
    factory.side_effect = ValueError("secret prompt contents")
    with pytest.raises(LocalConversationError, match="MODEL_LOAD_FAILED") as error:
        model.load()
    assert "secret" not in str(error.value)
    assert error.value.permanent and model.status.state == "failed"


def test_template_and_schema_failure_codes(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend
    native.metadata = {}
    with pytest.raises(LocalConversationError, match="CHAT_TEMPLATE_FAILED"):
        model.load()


def test_schema_compile_failure_never_falls_back_to_generic_json(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend
    model.load()
    sys.modules["llama_cpp"].LlamaGrammar.from_json_schema.side_effect = ValueError("private")
    with pytest.raises(LocalConversationError, match="SCHEMA_FAILED"):
        model.generate(request())
    native.create_chat_completion.assert_not_called()


def test_reset_failure_prevents_reuse(
    backend: tuple[LlamaCppConversationModel, Mock, Mock],
) -> None:
    model, native, _ = backend
    model.load()
    native.reset.side_effect = RuntimeError("private")
    with pytest.raises(LocalConversationError, match="RESET_FAILED"):
        model.generate(request())
    assert model.status.state == "failed"
