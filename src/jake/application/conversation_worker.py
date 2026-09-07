"""One bounded conversational generation on its own worker; polling never waits."""

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic

from jake.conversation_domain import ConversationModel, ConversationRequest, ConversationResponse


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    state: str = "idle"
    load_ms: float | None = None
    generation_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    fallback_used: bool = False


@dataclass
class _Job:
    request: ConversationRequest = field(repr=False)
    started: float
    done: Event = field(default_factory=Event)
    response: ConversationResponse | None = field(default=None, repr=False)
    failed: bool = False
    delivered: bool = False


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    request: ConversationRequest = field(repr=False)
    response: ConversationResponse | None = field(repr=False)


class ConversationWorker:
    def __init__(self, model: ConversationModel, timeout_seconds: float) -> None:
        self.model = model
        self.timeout = timeout_seconds
        self._queue: Queue[_Job] = Queue(maxsize=1)
        self._lock = Lock()
        self._closed = Event()
        self._unavailable = False
        self._job: _Job | None = None
        self._thread: Thread | None = None
        self.metrics = GenerationMetrics()

    def start(self) -> None:
        if self._thread is not None or self._closed.is_set():
            raise RuntimeError("conversation worker is single-use")
        self._thread = Thread(target=self._run, name="jake-conversation")
        self._thread.start()

    def submit(self, request: ConversationRequest) -> bool:
        with self._lock:
            if (
                self._job is not None
                or self._closed.is_set()
                or self._unavailable
                or self._thread is None
            ):
                return False
            job = _Job(request, monotonic())
            self._job = job
            self.metrics = GenerationMetrics(state="generating", load_ms=self.model.status.load_ms)
            self._queue.put_nowait(job)
            return True

    def poll(self) -> GenerationOutcome | None:
        with self._lock:
            job = self._job
            if job is None:
                return None
            elapsed = monotonic() - job.started
            if not job.done.is_set() and elapsed < self.timeout:
                return None
            if job.delivered:
                if job.done.is_set():
                    self._job = None
                return None
            timed_out = elapsed >= self.timeout
            failed = timed_out or job.failed
            if failed:
                self._unavailable = True
                self.model.cancel()
            result = None if failed else job.response
            self.metrics = GenerationMetrics(
                "timeout" if timed_out else "failed" if failed else "ready",
                self.model.status.load_ms,
                elapsed * 1000 if failed else result.generation_ms if result else None,
                result.prompt_tokens if result else None,
                result.output_tokens if result else None,
                result.tokens_per_second if result else None,
                failed,
            )
            job.delivered = True
            if job.done.is_set():
                self._job = None
            return GenerationOutcome(job.request, result)

    def cancel_active(self) -> None:
        with self._lock:
            if self._job is not None:
                self._job.failed = True
                self._unavailable = True
                self.model.cancel()

    def close(self) -> None:
        self._closed.set()
        self.model.cancel()
        if self._thread is not None:
            self._thread.join()  # Native cancellation is cooperative at token/eval boundaries.
        self.model.close()
        with self._lock:
            self._job = None
        while not self._queue.empty():
            self._queue.get_nowait()

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                job = self._queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                if not self._closed.is_set() and not self._unavailable:
                    job.response = self.model.generate(job.request)
                else:
                    job.failed = True
            except Exception:
                job.failed = True  # Never keep exceptions containing prompt/user data.
            finally:
                job.done.set()
                del job
