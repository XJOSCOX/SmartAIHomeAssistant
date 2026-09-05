from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from jake.domain import BoundingBox, Frame, PersonDetection


@pytest.mark.parametrize(
    "coordinates",
    [(0, 0, 0, 1), (0.5, 0, 0.1, 1), (-1, 0, 1, 1), (0, 0, 2, 1), (0, 0, float("nan"), 1)],
)
def test_invalid_boxes(coordinates: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        BoundingBox(*coordinates)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError):
        PersonDetection(BoundingBox(0, 0, 1, 1), confidence)


def test_frame_pixels_are_private_in_repr_and_frame_is_immutable() -> None:
    frame = Frame("camera", 0, datetime.now(UTC), 1, 1, b"abc")
    assert "abc" not in repr(frame)
    assert "pixels" not in repr(frame)
    with pytest.raises(FrozenInstanceError):
        frame.camera_id = "other"  # type: ignore[misc]  # Exercise runtime immutability.


def test_frame_requires_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Frame("camera", 0, datetime(2026, 1, 1), 1, 1, b"abc")


def test_frame_requires_matching_rgb_buffer() -> None:
    with pytest.raises(ValueError, match="buffer size"):
        Frame("camera", 0, datetime.now(UTC), 2, 1, b"abc")
