"""Embedded GGUF reasoning adapter. Never downloads, serves HTTP, or saves state."""

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

from jake.conversation_ai_config import ConversationAIConfig
from jake.conversation_context import allowed_replies, prompt_messages
from jake.conversation_domain import ConversationRequest, ConversationResponse, ModelStatus


class LocalConversationError(RuntimeError):
    pass


class LlamaCppConversationModel:
    def __init__(self, config: ConversationAIConfig) -> None:
        self.config = config
        self._model: Any = None
        self._cancel = Event()
        self._status = ModelStatus()

    @property
    def status(self) -> ModelStatus:
        return self._status

    def cancel(self) -> None:
        # Cancellation is terminal for this instance: no race resetting an in-flight signal.
        self._cancel.set()

    def close(self) -> None:
        self.cancel()
        if self._model is not None:
            self._model.close()
            self._model = None
        self._status = ModelStatus("closed", self._status.load_ms)

    def generate(self, request: ConversationRequest) -> ConversationResponse:
        started = perf_counter()
        try:
            if self._cancel.is_set():
                raise LocalConversationError("cancelled")
            llama_cpp = import_module("llama_cpp")
            if self._model is None:
                path = Path(self.config.model)
                if not path.is_file():
                    raise LocalConversationError("local GGUF model unavailable")

                self._status = ModelStatus("loading")
                load_start = perf_counter()
                self._model = llama_cpp.Llama(
                    model_path=str(path),
                    n_ctx=self.config.context_tokens,
                    n_threads=self.config.threads,
                    n_threads_batch=self.config.threads,
                    n_gpu_layers=0,
                    verbose=False,
                    logits_all=False,
                    embedding=False,
                )
                self._status = ModelStatus("ready", (perf_counter() - load_start) * 1000)
            if self._cancel.is_set():
                raise LocalConversationError("cancelled")
            started = perf_counter()
            # Drop oldest session-local turns, using the actual tokenizer. Chat-template
            # overhead has a conservative reserve; native context overflow still fails closed.
            limited = request
            while True:
                messages = prompt_messages(limited)
                count = sum(
                    len(self._model.tokenize(m["content"].encode("utf-8"))) for m in messages
                )
                if count + self.config.max_output_tokens + 256 <= self.config.context_tokens:
                    break
                if not limited.context.history:
                    raise LocalConversationError("conversation exceeds model context")
                limited = replace(
                    limited, context=replace(limited.context, history=limited.context.history[2:])
                )
            if self._cancel.is_set():
                raise LocalConversationError("cancelled")

            def cancellation_gate(input_ids: Any, scores: Any) -> Any:
                if self._cancel.is_set():
                    raise LocalConversationError("cancelled")
                return scores

            completion = self._model.create_chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_output_tokens,
                stream=False,
                logits_processor=llama_cpp.LogitsProcessorList([cancellation_gate]),
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reply": {
                                "type": "string",
                                "enum": list(allowed_replies(request.context)),
                            }
                        },
                        "required": ["reply"],
                        "additionalProperties": False,
                    },
                },
            )
            if self._cancel.is_set():
                raise LocalConversationError("cancelled")
            content = completion["choices"][0]["message"]["content"]
            if not isinstance(content, str) or len(content) > 2000:
                raise LocalConversationError("invalid model response")
            document = json.loads(content)
            if (
                not isinstance(document, dict)
                or set(document) != {"reply"}
                or not isinstance(document["reply"], str)
            ):
                raise LocalConversationError("invalid model response")
            usage = completion.get("usage", {})
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
            return ConversationResponse(
                document["reply"], (perf_counter() - started) * 1000, prompt_tokens, output_tokens
            )
        except Exception:
            self._status = ModelStatus("failed", self._status.load_ms)
            raise LocalConversationError("local conversation generation failed") from None
        finally:
            if self._model is not None:
                # Clear reusable attention state between requests; history is supplied explicitly.
                self._model.reset()
