"""Owner-thread GGUF lifecycle with recoverable requests and safe staged errors."""

import json
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from typing import Any

from jake.conversation_ai_config import ConversationAIConfig
from jake.conversation_context import allowed_replies, prompt_messages
from jake.conversation_domain import (
    ConversationErrorCode as Code,
)
from jake.conversation_domain import (
    ConversationFailure,
    ConversationRequest,
    ConversationResponse,
    ModelStatus,
)

LocalConversationError = ConversationFailure


class LlamaCppConversationModel:
    def __init__(
        self,
        config: ConversationAIConfig,
        *,
        health_diagnostics: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        # Only the fixed model-health composition enables this metadata sink.
        self._health_diagnostics = health_diagnostics
        self._model: Any = None
        self._formatter: Any = None
        self._library: Any = None
        self._shutdown = Event()
        self._lock = Lock()
        self._current: Event | None = None
        self._status = ModelStatus()

    @property
    def status(self) -> ModelStatus:
        return self._status

    def cancel_current(self) -> None:
        with self._lock:
            if self._current is not None:
                self._current.set()

    def cancel(self) -> None:
        """Terminal shutdown signal; native resources close only on their owner."""
        self._shutdown.set()
        self.cancel_current()

    def close(self) -> None:
        self.cancel()
        try:
            if self._model is not None:
                self._model.close()
        finally:
            self._model = self._formatter = None
            self._status = replace(self._status, state="closed")

    def load(self) -> None:
        if self._shutdown.is_set():
            raise ConversationFailure(Code.CANCELLED)
        if self._status.state == "ready":
            return
        if self._status.state == "failed":
            raise ConversationFailure(self._status.last_error_code or Code.MODEL_LOAD_FAILED)
        started = perf_counter()
        self._status = ModelStatus("loading")
        try:
            try:
                exists = Path(self.config.model).is_file()
            except OSError:
                exists = False
            if not exists:
                raise ConversationFailure(Code.MODEL_NOT_FOUND)
            try:
                self._library = import_module("llama_cpp")
                formats = import_module("llama_cpp.llama_chat_format")
            except Exception:
                raise ConversationFailure(Code.BACKEND_IMPORT_FAILED) from None
            try:
                self._model = self._library.Llama(
                    model_path=self.config.model,
                    n_ctx=self.config.context_tokens,
                    n_threads=self.config.threads,
                    n_threads_batch=self.config.threads,
                    n_gpu_layers=0,
                    verbose=False,
                    logits_all=False,
                    embedding=False,
                )
            except Exception as exc:
                code = (
                    Code.CHAT_TEMPLATE_FAILED
                    if type(exc).__module__.startswith("jinja2")
                    else Code.MODEL_LOAD_FAILED
                )
                raise ConversationFailure(code) from None
            if self._shutdown.is_set():
                raise ConversationFailure(Code.CANCELLED)
            try:
                template = self._model.metadata.get("tokenizer.chat_template")
                if not isinstance(template, str) or not template.strip():
                    raise ValueError
                eos, bos = self._model.token_eos(), self._model.token_bos()
                self._formatter = formats.Jinja2ChatFormatter(
                    template=template,
                    eos_token=self._model.detokenize([eos], special=True).decode(
                        "utf-8", errors="replace"
                    )
                    if eos >= 0
                    else "",
                    bos_token=self._model.detokenize([bos], special=True).decode(
                        "utf-8", errors="replace"
                    )
                    if bos >= 0
                    else "",
                    stop_token_ids=[eos] if eos >= 0 else [],
                )
                # Fixed internal template smoke test, no user data or inference.
                self._formatter(
                    messages=[
                        {"role": "system", "content": "You are Jake."},
                        {"role": "user", "content": "Hello."},
                    ]
                )
                # Jinja2ChatFormatter uses eos_token as a textual stop. An empty
                # stop matches every completion at offset zero, even with grammar.
                if not self._formatter.eos_token:
                    raise ValueError
                self._model.chat_handler = self._formatter.to_chat_handler()
            except Exception:
                raise ConversationFailure(Code.CHAT_TEMPLATE_FAILED) from None
            self._status = ModelStatus(
                "ready", (perf_counter() - started) * 1000, chat_template_ok=True
            )
        except ConversationFailure as exc:
            self._status = ModelStatus("failed", (perf_counter() - started) * 1000, exc.code)
            raise

    def generate(
        self, request: ConversationRequest, *, cancel: Event | None = None
    ) -> ConversationResponse:
        if self._shutdown.is_set():
            raise ConversationFailure(Code.CANCELLED)
        if self._status.state != "ready":
            raise ConversationFailure(self._status.last_error_code or Code.MODEL_NOT_READY)
        token = cancel if cancel is not None else Event()
        with self._lock:
            self._current = token
        started = perf_counter()
        self._status = replace(self._status, state="generating", last_error_code=None)

        def check_cancel() -> None:
            if token.is_set() or self._shutdown.is_set():
                raise ConversationFailure(Code.CANCELLED)

        try:
            check_cancel()
            limited = request
            while True:
                try:
                    messages = prompt_messages(limited)
                    formatted = self._formatter(messages=messages)
                except Exception:
                    raise ConversationFailure(Code.CHAT_TEMPLATE_FAILED) from None
                try:
                    count = len(
                        self._model.tokenize(
                            formatted.prompt.encode("utf-8"),
                            add_bos=not formatted.added_special,
                            special=True,
                        )
                    )
                except Exception:
                    raise ConversationFailure(Code.TOKEN_BUDGET_FAILED) from None
                if count + self.config.max_output_tokens <= self.config.context_tokens:
                    break
                if not limited.context.history:
                    raise ConversationFailure(Code.CONTEXT_OVERFLOW)
                limited = replace(
                    limited, context=replace(limited.context, history=limited.context.history[2:])
                )
                check_cancel()
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "reply": {"type": "string", "enum": list(allowed_replies(request.context))}
                    },
                    "required": ["reply"],
                    "additionalProperties": False,
                }
                # Compile explicitly: unlike the response_format helper this cannot
                # silently fall back to unrestricted JSON after a schema compiler error.
                grammar = self._library.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=False
                )
            except Exception:
                raise ConversationFailure(Code.SCHEMA_FAILED) from None

            def cancellation_gate(input_ids: Any, scores: Any) -> Any:
                check_cancel()
                return scores

            check_cancel()
            try:
                completion = self._model.create_chat_completion(
                    messages=messages,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_output_tokens,
                    stream=False,
                    grammar=grammar,
                    logits_processor=self._library.LogitsProcessorList([cancellation_gate]),
                )
            except ConversationFailure:
                raise
            except Exception as exc:
                code = (
                    Code.CHAT_TEMPLATE_FAILED
                    if type(exc).__module__.startswith("jinja2")
                    else Code.GENERATION_FAILED
                )
                raise ConversationFailure(code) from None
            check_cancel()
            if self._health_diagnostics is not None:
                self._report_health_completion(completion)
            try:
                content = completion["choices"][0]["message"]["content"]
                if not isinstance(content, str) or len(content) > 2000:
                    raise ValueError
            except Exception:
                raise ConversationFailure(Code.INVALID_RESPONSE_SHAPE) from None
            try:
                document = json.loads(content.strip())
            except (ValueError, TypeError) as exc:
                if self._health_diagnostics is not None:
                    self._health_diagnostics(f"json_parse_error_type: {type(exc).__name__}")
                raise ConversationFailure(Code.INVALID_JSON) from None
            if (
                not isinstance(document, dict)
                or set(document) != {"reply"}
                or not isinstance(document["reply"], str)
            ):
                raise ConversationFailure(Code.INVALID_RESPONSE_SHAPE)
            if document["reply"] not in allowed_replies(request.context):
                raise ConversationFailure(Code.POLICY_REJECTED)
            usage = completion.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            prompt_tokens, output_tokens = (
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
            prompt_tokens = (
                prompt_tokens if type(prompt_tokens) is int and prompt_tokens >= 0 else None
            )
            output_tokens = (
                output_tokens if type(output_tokens) is int and output_tokens >= 0 else None
            )
            self._status = replace(self._status, state="ready", last_error_code=None)
            return ConversationResponse(
                document["reply"], (perf_counter() - started) * 1000, prompt_tokens, output_tokens
            )
        except ConversationFailure as exc:
            self._status = replace(
                self._status, state="failed" if exc.permanent else "ready", last_error_code=exc.code
            )
            raise
        finally:
            with self._lock:
                self._current = None
            try:
                self._model.reset()
            except Exception:
                self._status = replace(
                    self._status, state="failed", last_error_code=Code.RESET_FAILED
                )
                raise ConversationFailure(Code.RESET_FAILED) from None

    def _report_health_completion(self, completion: Any) -> None:
        """Fixed health request metadata only; never print response or prompt text."""
        emit = self._health_diagnostics
        if emit is None:
            return
        content: Any = None
        finish: Any = None
        try:
            choice = completion["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        # Restrict even metadata values to known types/enums; no native payloads.
        kind = (
            type(content).__name__
            if type(content) in (str, dict, list, int, float, bool)
            else "other"
        )
        emit("grammar_enabled: True")
        emit("chat_template_type: GGUF Jinja2ChatFormatter")
        emit(f"raw_response_type: {kind}")
        emit(f"raw_response_length: {len(content) if isinstance(content, str) else 0}")
        emit(f"starts_with_brace: {isinstance(content, str) and content.strip().startswith('{')}")
        emit(f"ends_with_brace: {isinstance(content, str) and content.strip().endswith('}')}")
        emit(
            f"finish_reason: {finish if finish in ('stop', 'length', 'tool_calls') else 'unknown'}"
        )
