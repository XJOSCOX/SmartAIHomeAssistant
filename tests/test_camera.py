from datetime import UTC, datetime
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from jake.adapters.opencv_camera import CameraError, OpenCVCamera
from jake.config import CameraConfig
from jake.ports import FrameSource


@pytest.fixture
def capture() -> Mock:
    device = Mock()
    device.isOpened.return_value = True
    device.read.return_value = (True, np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8))
    return device


def test_rgb_translation_metadata_and_sequence(capture: Mock) -> None:
    factory = Mock(return_value=capture)
    before = datetime.now(UTC)
    with OpenCVCamera("door", CameraConfig(2), capture_factory=factory) as camera:
        source: FrameSource = camera  # Static verification of structural compatibility.
        iterator = iter(source)
        frames = [next(iterator) for _ in range(3)]
        capture.release.assert_not_called()
    after = datetime.now(UTC)
    factory.assert_called_once_with(2)
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    for frame in frames:
        assert frame.pixels == bytes([30, 20, 10, 60, 50, 40])
        assert (frame.width, frame.height, frame.camera_id) == (2, 1, "door")
        assert frame.captured_at.tzinfo is UTC
        assert before <= frame.captured_at <= after
    capture.release.assert_called_once_with()


def test_non_contiguous_image_is_packed_row_major(capture: Mock) -> None:
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[:, ::2, :]
    assert not image.flags.c_contiguous
    capture.read.return_value = True, image
    with OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture) as camera:
        frame = next(camera)
    assert (frame.width, frame.height) == (2, 3)
    assert frame.pixels == image[:, :, ::-1].tobytes(order="C")


def test_camera_open_failure_releases(capture: Mock) -> None:
    capture.isOpened.return_value = False
    with (
        pytest.raises(CameraError, match="Cannot open local camera index 4"),
        OpenCVCamera("door", CameraConfig(4), capture_factory=lambda _: capture),
    ):
        pytest.fail("failed camera must never enter the context")
    capture.release.assert_called_once_with()


@pytest.mark.parametrize("error", [RuntimeError("consumer failed"), KeyboardInterrupt()])
def test_consumer_failure_and_interrupt_release(capture: Mock, error: BaseException) -> None:
    with (
        pytest.raises(type(error)),
        OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture) as camera,
    ):
        next(camera)
        raise error
    capture.release.assert_called_once_with()


def test_early_break_and_explicit_close_release_once(capture: Mock) -> None:
    with OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture) as camera:
        for _ in camera:
            break
        camera.close()
    camera.close()
    capture.release.assert_called_once_with()


@pytest.mark.parametrize(
    "ok,image",
    [
        (False, None),
        (True, None),
        (True, np.empty((0, 0, 3), dtype=np.uint8)),
        (True, np.zeros((2, 2), dtype=np.uint8)),
        (True, np.zeros((2, 2, 4), dtype=np.uint8)),
        (True, np.zeros((2, 2, 3), dtype=np.float64)),
    ],
)
def test_invalid_reads_release(capture: Mock, ok: bool, image: NDArray[np.generic] | None) -> None:
    capture.read.return_value = ok, image
    with OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture) as camera:
        with pytest.raises(CameraError):
            next(camera)
        capture.release.assert_called_once_with()
    capture.release.assert_called_once_with()


@pytest.mark.parametrize("error", [cv2.error("read failed"), KeyboardInterrupt()])
def test_read_exception_releases(capture: Mock, error: BaseException) -> None:
    capture.read.side_effect = error
    expected = CameraError if isinstance(error, cv2.error) else KeyboardInterrupt
    with OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture) as camera:
        with pytest.raises(expected):
            next(camera)
        capture.release.assert_called_once_with()
    capture.release.assert_called_once_with()


def test_conversion_exception_releases(capture: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "cvtColor", Mock(side_effect=cv2.error("conversion failed")))
    with (
        OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture) as camera,
        pytest.raises(CameraError, match="converting"),
    ):
        next(camera)
    capture.release.assert_called_once_with()


def test_context_required_and_session_cannot_reopen(capture: Mock) -> None:
    camera = OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture)
    with pytest.raises(CameraError, match="not open"):
        next(camera)
    with camera:
        assert next(camera).sequence == 0
        with pytest.raises(CameraError, match="single-use"):
            camera.__enter__()
    with pytest.raises(CameraError, match="single-use"):
        camera.__enter__()
    with pytest.raises(CameraError, match="not open"):
        next(camera)
    capture.release.assert_called_once_with()


def test_native_open_exception_releases(capture: Mock) -> None:
    capture.isOpened.side_effect = cv2.error("backend failure")
    with (
        pytest.raises(CameraError, match="Cannot open"),
        OpenCVCamera("door", CameraConfig(), capture_factory=lambda _: capture),
    ):
        pytest.fail("unexpected successful open")
    capture.release.assert_called_once_with()
