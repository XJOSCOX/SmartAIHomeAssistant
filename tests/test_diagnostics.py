from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from jake.diagnostics import measure_detection
from jake.domain import Frame


def test_detector_timing_excludes_acquisition_and_display(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = Mock()
    detector.detect.return_value = ()
    clock = Mock(side_effect=[100.0, 100.025])
    monkeypatch.setattr("jake.diagnostics.perf_counter", clock)
    frame = Frame("test", 0, datetime.now(UTC), 1, 1, bytes(3))
    measurement = measure_detection(detector, frame)
    detector.detect.assert_called_once_with(frame)
    assert measurement.inference_ms == pytest.approx(25.0)
    assert measurement.detection_fps == pytest.approx(40.0)
    assert measurement.detections == ()


def test_zero_elapsed_time_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jake.diagnostics.perf_counter", lambda: 100.0)
    detector = Mock()
    detector.detect.return_value = ()
    frame = Frame("test", 0, datetime.now(UTC), 1, 1, bytes(3))
    assert measure_detection(detector, frame).detection_fps == 0.0
