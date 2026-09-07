"""Load/generation deadlines and retries without native models or devices."""

from datetime import UTC, datetime
from pathlib import Path
from threading import Event, get_ident
from time import monotonic, sleep
from unittest.mock import Mock

import pytest

from jake.application.conversation_worker import ConversationWorker
from jake.conversation_ai_config import ConversationAIConfig
from jake.conversation_context import BASE_REPLIES, build_context
from jake.conversation_domain import (
    ConversationErrorCode as Code,
)
from jake.conversation_domain import (
    ConversationFailure,
    ConversationRequest,
    ConversationResponse,
    ModelStatus,
)


def request() -> ConversationRequest:
    return ConversationRequest(
        build_context(
            "test", datetime.now(UTC), None, None, (), timezone="UTC", max_age=2, supported=()
        ),
        "Hello.",
    )


class Runtime:
    def __init__(
        self,
        errors: tuple[Code, ...] = (),
        *,
        slow_load: bool = False,
        load_error: bool = False,
        block_first: bool = False,
    ) -> None:
        self.errors, self.slow_load, self.load_error, self.block_first = (
            errors,
            slow_load,
            load_error,
            block_first,
        )
        self.loading, self.release, self.entered = Event(), Event(), Event()
        self.cancelled, self.closed = False, False
        self.calls = 0
        self.owner = self.close_owner = 0
        self.current: Event | None = None
        self.status = ModelStatus()
        self.requests: list[ConversationRequest] = []

    def load(self) -> None:
        self.owner = get_ident()
        self.loading.set()
        if self.slow_load:
            assert self.release.wait(3)
        if self.load_error:
            raise ValueError("private transcript that must not leak")
        self.status = ModelStatus("ready", 50, chat_template_ok=True)

    def generate(
        self, req: ConversationRequest, *, cancel: Event | None = None
    ) -> ConversationResponse:
        self.calls += 1
        self.requests.append(req)
        self.current = cancel
        self.entered.set()
        if self.block_first and self.calls == 1:
            assert cancel is not None and cancel.wait(3)
            raise ConversationFailure(Code.CANCELLED)
        if self.calls <= len(self.errors):
            raise ConversationFailure(self.errors[self.calls - 1])
        return ConversationResponse(BASE_REPLIES[0], 2, 20, 8)

    def cancel_current(self) -> None:
        if self.current is not None:
            self.current.set()

    def cancel(self) -> None:
        self.cancelled = True
        self.release.set()
        self.cancel_current()

    def close(self) -> None:
        self.closed = True
        self.close_owner = get_ident()


def ready(worker: ConversationWorker) -> None:
    deadline = monotonic() + 2
    while worker.metrics.state != "ready" and monotonic() < deadline:
        worker.poll()
        sleep(0.005)
    assert worker.metrics.state == "ready"


def outcome(worker: ConversationWorker) -> object:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            return result
        sleep(0.005)
    raise AssertionError("no outcome")


def test_slow_load_separate_from_generation_budget() -> None:
    model = Runtime(slow_load=True)
    worker = ConversationWorker(model, 0.02, load_timeout_seconds=1)
    worker.start()
    try:
        assert model.loading.wait(1)
        sleep(0.04)
        worker.poll()
        assert worker.metrics.state == "loading" and not worker.submit(request())
        assert worker.metrics.last_error_code is None
        model.release.set()
        ready(worker)
        assert worker.submit(request())
        result = outcome(worker)
        assert result.response is not None  # type: ignore[attr-defined]
        assert worker.metrics.last_error_code is None
    finally:
        worker.close()
    assert model.closed and model.owner == model.close_owner != get_ident()


@pytest.mark.parametrize(
    "code",
    [
        Code.INVALID_JSON,
        Code.INVALID_REPLY,
        Code.POLICY_REJECTED,
        Code.GENERATION_FAILED,
        Code.CONTEXT_OVERFLOW,
    ],
)
def test_recoverable_failures_allow_next_request(code: Code) -> None:
    worker = ConversationWorker(Runtime((code,)), 1)
    worker.start()
    try:
        ready(worker)
        assert worker.submit(request())
        result = outcome(worker)
        assert result.error_code == code  # type: ignore[attr-defined]
        assert worker.metrics.state == "ready"
        assert worker.submit(request())
        result = outcome(worker)
        assert result.response is not None  # type: ignore[attr-defined]
        assert worker.metrics.last_error_code is None
    finally:
        worker.close()


def test_timeout_cleanup_then_retry() -> None:
    model = Runtime(block_first=True)
    worker = ConversationWorker(model, 0.02)
    worker.start()
    try:
        ready(worker)
        assert worker.submit(request())
        assert model.entered.wait(1)
        result = outcome(worker)
        assert result.error_code == Code.GENERATION_TIMEOUT  # type: ignore[attr-defined]
        ready(worker)
        assert not model.cancelled
        assert worker.submit(request())
        result = outcome(worker)
        assert result.response is not None  # type: ignore[attr-defined]
    finally:
        worker.close()
    assert model.cancelled and model.closed


def test_load_failure_is_permanent_and_sanitized() -> None:
    model = Runtime(load_error=True)
    worker = ConversationWorker(model, 1)
    worker.start()
    try:
        deadline = monotonic() + 1
        while worker.metrics.state != "failed" and monotonic() < deadline:
            sleep(0.005)
        assert worker.metrics.last_error_code == Code.MODEL_LOAD_FAILED
        assert "private" not in repr(worker.metrics)
        assert not worker.submit(request())
    finally:
        worker.close()


def test_load_deadline_and_late_completion_cannot_become_ready() -> None:
    model = Runtime(slow_load=True)
    worker = ConversationWorker(model, 1, load_timeout_seconds=0.01)
    worker.start()
    try:
        assert model.loading.wait(1)
        sleep(0.02)
        worker.poll()
        assert worker.metrics.state == "failed"
        assert worker.metrics.last_error_code == Code.MODEL_LOAD_TIMEOUT
        assert not worker.submit(request())
        model.release.set()
    finally:
        worker.close()
    assert model.closed and model.owner == model.close_owner


def test_legacy_timeout_alias_and_new_defaults(tmp_path: Path) -> None:
    from jake.config import load_app_config

    assert ConversationAIConfig().generation_timeout == 30
    assert ConversationAIConfig().load_timeout_seconds == 120
    path = tmp_path / "local.toml"
    path.write_text(
        '[pipeline]\ncamera_id="test"\n[conversation_ai]\ntimeout_seconds=20', encoding="utf-8"
    )
    config = load_app_config(path).conversation_ai
    assert config.generation_timeout == 20 and config.load_timeout_seconds == 120
    with pytest.raises(ValueError):
        ConversationAIConfig(load_timeout_seconds=0)
    with pytest.raises(ValueError):
        ConversationAIConfig(generation_timeout_seconds=float("nan"))


def test_health_command_uses_only_fixed_prompt_no_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jake.voice_cli import main

    config = tmp_path / "config.toml"
    config.write_text('[pipeline]\ncamera_id="test"', encoding="utf-8")
    model = Runtime()
    monkeypatch.setattr(
        "jake.adapters.llama_conversation.LlamaCppConversationModel", Mock(return_value=model)
    )
    devices = Mock(side_effect=AssertionError("must not compose audio or camera"))
    monkeypatch.setattr("jake.application.voice_composition.compose_voice", devices)
    monkeypatch.setattr("jake.application.voice_camera.VoiceCamera", devices)
    assert main(["--config", str(config), "--test-conversation-model"]) == 0
    assert len(model.requests) == 1 and model.requests[0].text == "Hello."
    assert not model.requests[0].context.visible_people
    assert model.closed and model.close_owner == model.owner
    output = capsys.readouterr().out
    assert "Result: READY" in output and "Schema generation: OK" in output
    assert "Hello." not in output
    devices.assert_not_called()


def test_health_accepts_novel_reply_without_devices(capsys: pytest.CaptureFixture[str]) -> None:
    from jake.application.conversation_health import test_model

    class NaturalRuntime(Runtime):
        def generate(
            self, request: ConversationRequest, *, cancel: Event | None = None
        ) -> ConversationResponse:
            self.requests.append(request)
            return ConversationResponse("Hey. What would you like to discuss today?", 2, 20, 12)

    model = NaturalRuntime()
    assert test_model(ConversationAIConfig(), model) == 0
    output = capsys.readouterr().out
    assert "Result: READY" in output and "discuss today" not in output
    assert model.closed
    assert model.requests[0].text == "Hello."
