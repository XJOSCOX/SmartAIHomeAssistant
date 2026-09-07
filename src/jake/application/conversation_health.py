"""Fixed, metadata-only local model health probe; no device or store dependencies."""

from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from jake.application.conversation_worker import ConversationWorker
from jake.conversation_ai_config import ConversationAIConfig
from jake.conversation_context import FALLBACK, build_context, validate_response
from jake.conversation_domain import ConversationModel, ConversationRequest


def test_model(config: ConversationAIConfig, model: ConversationModel | None = None) -> int:
    if model is None:
        from jake.adapters.llama_conversation import LlamaCppConversationModel

        model = LlamaCppConversationModel(config)
    worker = ConversationWorker(
        model, config.generation_timeout, load_timeout_seconds=config.load_timeout_seconds
    )
    print(f"Model: {Path(config.model).name}")
    delay = Event()
    try:
        worker.start()
        print("conversation AI: loading")
        while worker.metrics.state == "loading":
            worker.poll()
            delay.wait(0.01)
        if worker.metrics.state != "ready":
            print(f"Result: {worker.metrics.last_error_code or 'MODEL_LOAD_FAILED'}")
            return 1
        print(f"Load: {worker.metrics.load_ms} ms")
        print(f"Chat template: {'OK' if model.status.chat_template_ok else 'unverified'}")
        request = ConversationRequest(
            build_context(
                "internal-health-check",
                datetime.now(UTC),
                None,
                None,
                (),
                timezone="UTC",
                max_age=2,
                supported=("local conversation",),
            ),
            "Hello.",
        )
        if not worker.submit(request):
            print("Result: MODEL_NOT_READY")
            return 1
        outcome = None
        while outcome is None:
            outcome = worker.poll()
            delay.wait(0.01)
        if outcome.response is None:
            print(f"Result: {outcome.error_code or 'GENERATION_FAILED'}")
            return 1
        if validate_response(outcome.response.text, request) == FALLBACK:
            print("Result: POLICY_REJECTED")
            return 1
        print("Schema generation: OK")
        print(f"Prompt tokens: {outcome.response.prompt_tokens}")
        print(f"Output tokens: {outcome.response.output_tokens}")
        print(f"Generation: {outcome.response.generation_ms} ms")
        print("Result: READY")
        return 0
    except KeyboardInterrupt:
        print("Result: CANCELLED")
        return 130
    finally:
        worker.close()
