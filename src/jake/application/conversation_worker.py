"""Async startup and one recoverable request slot, owned by a single worker."""

from dataclasses import dataclass, field, replace
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic

from jake.conversation_domain import (
    ConversationErrorCode as Code,
)
from jake.conversation_domain import (
    ConversationFailure,
    ConversationModel,
    ConversationRequest,
    ConversationResponse,
)


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    state: str = "disabled"
    load_ms: float | None = None
    generation_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    fallback_used: bool = False
    last_error_code: Code | None = None


@dataclass
class _Job:
    request: ConversationRequest = field(repr=False)
    started: float | None = None
    ended: float | None = None
    done: Event = field(default_factory=Event)
    cancel: Event = field(default_factory=Event)
    response: ConversationResponse | None = field(default=None, repr=False)
    error: Code | None = None
    delivered: bool = False


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    request: ConversationRequest = field(repr=False)
    response: ConversationResponse | None = field(repr=False)
    error_code: Code | None = None


class ConversationWorker:
    def __init__(
        self, model: ConversationModel, timeout_seconds: float, *, load_timeout_seconds: float = 120
    ) -> None:
        self.model = model
        self.timeout = timeout_seconds
        self.load_timeout = load_timeout_seconds
        self._queue: Queue[_Job] = Queue(maxsize=1)
        self._lock = Lock()
        self._closed = Event()
        self._unavailable = False
        self._load_started: float | None = None
        self._job: _Job | None = None
        self._thread: Thread | None = None
        self.metrics = GenerationMetrics()

    def start(self) -> None:
        if self._thread is not None or self._closed.is_set():
            raise RuntimeError("conversation worker is single-use")
        self._load_started = monotonic()
        self.metrics = GenerationMetrics(state="loading")
        self._thread = Thread(target=self._run, name="jake-conversation")
        self._thread.start()

    def submit(self, request: ConversationRequest) -> bool:
        with self._lock:
            if (
                self._job is not None
                or self._closed.is_set()
                or self._unavailable
                or self.metrics.state != "ready"
            ):
                return False
            job = _Job(request)
            self._job = job
            self.metrics = replace(self.metrics, state="generating", fallback_used=False)
            self._queue.put_nowait(job)
            return True

    def _complete_slot(self, job: _Job) -> None:
        if job.error is not None and ConversationFailure(job.error).permanent:
            self._unavailable = True
            self.metrics = replace(self.metrics, state="failed", last_error_code=job.error)
        else:
            self.metrics = replace(self.metrics, state="ready")
        self._job = None

    def poll(self) -> GenerationOutcome | None:
        with self._lock:
            if self.metrics.state == "loading" and self._load_started is not None:
                if monotonic() - self._load_started >= self.load_timeout:
                    self._unavailable = True
                    self.metrics = GenerationMetrics(
                        "failed",
                        (monotonic() - self._load_started) * 1000,
                        fallback_used=True,
                        last_error_code=Code.MODEL_LOAD_TIMEOUT,
                    )
                    self.model.cancel()
                return None
            job = self._job
            if job is None or job.started is None:
                return None
            elapsed = (job.ended if job.ended is not None else monotonic()) - job.started
            if job.delivered:
                if job.done.is_set():
                    self._complete_slot(job)
                return None
            if not job.done.is_set() and elapsed < self.timeout:
                return None
            code = job.error
            if elapsed >= self.timeout:
                code = Code.GENERATION_TIMEOUT
                job.cancel.set()
                self.model.cancel_current()
            result = None if code is not None else job.response
            self.metrics = GenerationMetrics(
                "generating",
                self.model.status.load_ms,
                result.generation_ms if result else elapsed * 1000,
                result.prompt_tokens if result else None,
                result.output_tokens if result else None,
                result.tokens_per_second if result else None,
                code is not None,
                code,
            )
            job.delivered = True
            if job.done.is_set():
                self._complete_slot(job)
            return GenerationOutcome(job.request, result, code)

    def note_policy_rejection(self) -> None:
        with self._lock:
            self.metrics = replace(
                self.metrics, last_error_code=Code.POLICY_REJECTED, fallback_used=True
            )

    def cancel_active(self) -> None:
        with self._lock:
            if self._job is not None:
                self._job.cancel.set()
                self._job.error = Code.CANCELLED
                self.model.cancel_current()

    def close(self) -> None:
        self._closed.set()
        self.cancel_active()
        self.model.cancel()
        if self._thread is not None:
            self._thread.join()  # Native cancellation is cooperative; never reuse before return.
        else:
            self.model.close()
        with self._lock:
            self._job = None
            self.metrics = replace(self.metrics, state="closed")
        while not self._queue.empty():
            self._queue.get_nowait()

    def _run(self) -> None:
        try:
            try:
                self.model.load()
            except Exception as exc:
                code = exc.code if isinstance(exc, ConversationFailure) else Code.MODEL_LOAD_FAILED
                with self._lock:
                    self._unavailable = True
                    if self.metrics.last_error_code != Code.MODEL_LOAD_TIMEOUT:
                        self.metrics = GenerationMetrics(
                            "failed",
                            self.model.status.load_ms,
                            fallback_used=True,
                            last_error_code=code,
                        )
                return
            with self._lock:
                # Also enforce the load deadline when no caller polled during startup.
                if (
                    self._load_started is not None
                    and monotonic() - self._load_started >= self.load_timeout
                ):
                    self._unavailable = True
                    self.metrics = GenerationMetrics(
                        "failed",
                        self.model.status.load_ms,
                        fallback_used=True,
                        last_error_code=Code.MODEL_LOAD_TIMEOUT,
                    )
                if self._unavailable or self._closed.is_set():
                    return
                self.metrics = GenerationMetrics("ready", self.model.status.load_ms)
            while not self._closed.is_set():
                try:
                    job = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    with self._lock:
                        job.started = monotonic()
                    if self._closed.is_set() or job.cancel.is_set():
                        raise ConversationFailure(Code.CANCELLED)
                    job.response = self.model.generate(job.request, cancel=job.cancel)
                    if job.cancel.is_set():
                        raise ConversationFailure(Code.CANCELLED)
                except Exception as exc:
                    job.error = (
                        exc.code if isinstance(exc, ConversationFailure) else Code.GENERATION_FAILED
                    )
                finally:
                    job.ended = monotonic()
                    job.done.set()
                    del job
        finally:
            try:
                self.model.close()  # All native teardown runs on the owner thread.
            except Exception:
                with self._lock:
                    self.metrics = replace(
                        self.metrics, state="failed", last_error_code=Code.RESET_FAILED
                    )
