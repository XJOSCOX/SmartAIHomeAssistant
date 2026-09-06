"""Framework-independent, local face identity boundaries."""

from typing import Protocol

from jake.domain import BoundingBox, Frame
from jake.identity_domain import FaceDetection, FaceEmbedding, IdentityMatch, ResidentProfile


class FaceDetector(Protocol):
    def detect(self, frame: Frame, region: BoundingBox) -> tuple[FaceDetection, ...]: ...


class FaceEncoder(Protocol):
    def encode(self, frame: Frame, face: FaceDetection) -> FaceEmbedding: ...


class IdentityStore(Protocol):
    def profiles(self) -> tuple[ResidentProfile, ...]: ...
    def add(self, profile: ResidentProfile) -> None: ...
    def delete(self, resident_id: str) -> None: ...


class KeyProvider(Protocol):
    """Create a new OS-protected key or retrieve an existing key without replacement."""

    def create(self) -> tuple[str, bytes]: ...
    def get(self, key_id: str) -> bytes: ...


class IdentityMatcher(Protocol):
    def match(
        self, embedding: FaceEmbedding, profiles: tuple[ResidentProfile, ...]
    ) -> IdentityMatch: ...
