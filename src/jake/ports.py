"""Structural interfaces implemented by future local adapters.

No model framework or camera SDK belongs in these contracts.
"""

from collections.abc import Iterator
from typing import Protocol

from jake.domain import (
    AppearanceEmbedding,
    BoundingBox,
    Frame,
    FrameContext,
    PersonDetection,
    PersonEvent,
    PersonTrack,
)


class FrameSource(Protocol):
    """Yields ordered frames for one camera session.

    The caller owns source startup/shutdown, including cancellation cleanup.
    A concrete camera adapter should expose a context manager for that lifecycle.
    """

    def __iter__(self) -> Iterator[Frame]: ...


class PersonDetector(Protocol):
    """Converts RGB frames to person detections using normalized coordinates."""

    def detect(self, frame: Frame) -> tuple[PersonDetection, ...]: ...


class PersonTracker(Protocol):
    """Maintains IDs within one camera session, including empty detection updates.

    Output is the complete unexpired lifecycle set, including recently-lost pool
    snapshots so event history stays open, and missed tracks with their
    missed_frames count. confirmed marks eligibility for semantic entry; current
    baseline trackers confirm on the first detection; stabilized trackers require hits.
    IDs must never be reused.
    ID assignment, occlusion handling,
    and expiry policies belong to the implementation. Instances are not shared
    across sessions. A future image-based tracker will need a separate contract.
    """

    def update(
        self, context: FrameContext, detections: tuple[PersonDetection, ...]
    ) -> tuple[PersonTrack, ...]: ...


class EventGenerator(Protocol):
    """Derives events from current active tracks, including empty track updates.

    Transition state and deduplication belong to a per-session implementation.
    Only disappearance from the complete active set indicates expiration.
    Call every processed frame with increasing sequences and nondecreasing times.
    """

    def generate(
        self, context: FrameContext, tracks: tuple[PersonTrack, ...]
    ) -> tuple[PersonEvent, ...]: ...


class AppearanceEncoder(Protocol):
    """Encode only a person's crop into a unit vector of fixed dimension per encoder.

    Inference is local. No crops or embeddings may be persisted by implementations.
    Do not share vectors across model versions or camera sessions.
    """

    def encode(self, frame: Frame, box: BoundingBox) -> AppearanceEmbedding: ...
