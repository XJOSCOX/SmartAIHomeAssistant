from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from jake.config import KalmanConfig
from jake.domain import BoundingBox, FrameContext
from jake.kalman import KalmanError, KalmanFilter
from jake.motion import (
    FALLBACK_DT,
    MAX_DT,
    MIN_DT,
    bounded_box,
    frame_dt,
    initialize,
    measurement,
    observation_matrix,
    process_noise,
    transition_matrix,
)

BOX = BoundingBox(0.2, 0.2, 0.4, 0.6)


def test_stationary_object_and_covariance_remain_stable() -> None:
    config = KalmanConfig()
    motion = initialize(BOX, config)
    for _ in range(100):
        motion = motion.predict(transition_matrix(0.1), process_noise(config, 0.1))
        motion = motion.correct(
            measurement(BOX), observation_matrix(), np.eye(4) * config.measurement_noise
        )
        assert np.linalg.eigvalsh(motion.covariance).min() >= -1e-12
    np.testing.assert_allclose(motion.state[:4], measurement(BOX), atol=1e-12)
    np.testing.assert_allclose(motion.state[4:], 0, atol=1e-12)
    np.testing.assert_allclose(motion.covariance, motion.covariance.T, atol=1e-12)


def test_constant_velocity_and_dt_scale_prediction() -> None:
    config = KalmanConfig()
    state = initialize(BOX, config).state
    state[4:] = [0.1, -0.05, 0.02, 0.01]
    motion = KalmanFilter(state, np.eye(8))
    for dt in (0.1, 0.5, 1.0):
        predicted = motion.predict(transition_matrix(dt), process_noise(config, dt))
        np.testing.assert_allclose(predicted.state[:4], state[:4] + state[4:] * dt)
        np.testing.assert_allclose(predicted.state[4:], state[4:])
    np.testing.assert_array_equal(motion.state, state)


def test_hand_calculated_prediction_gain_and_covariance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np.linalg, "inv", Mock(side_effect=AssertionError("no explicit inverse")))
    motion = KalmanFilter(np.array([0.2, 0.1]), np.diag([0.04, 0.09]))
    predicted = motion.predict(np.array([[1.0, 2.0], [0.0, 1.0]]), np.diag([0.01, 0.02]))
    np.testing.assert_allclose(predicted.state, [0.4, 0.1])
    np.testing.assert_allclose(predicted.covariance, [[0.41, 0.18], [0.18, 0.11]])
    corrected = predicted.correct(np.array([0.6]), np.array([[1.0, 0.0]]), np.array([[0.05]]))
    gain = np.array([0.41, 0.18]) / 0.46
    np.testing.assert_allclose(corrected.state, np.array([0.4, 0.1]) + gain * 0.2)
    expected = predicted.covariance - np.outer(gain, [0.41, 0.18])
    np.testing.assert_allclose(corrected.covariance, expected)
    assert 0.4 < corrected.state[0] < 0.6


def test_shapes_matrices_and_defensive_array_ownership() -> None:
    config = KalmanConfig()
    motion = initialize(BOX, config)
    assert motion.state.shape == (8,)
    assert motion.covariance.shape == (8, 8)
    assert measurement(BOX).shape == (4,)
    assert observation_matrix().shape == (4, 8)
    np.testing.assert_array_equal(observation_matrix(), np.hstack((np.eye(4), np.zeros((4, 4)))))
    np.testing.assert_array_equal(transition_matrix(0.5)[:4, 4:], np.eye(4) * 0.5)
    np.testing.assert_allclose(np.diag(process_noise(config, 0.5)), [0.00005] * 4 + [0.0005] * 4)
    np.testing.assert_array_equal(np.diag(motion.covariance), [0.01] * 4 + [1.0] * 4)
    state, covariance = motion.state, motion.covariance
    state[:] = 0
    covariance[:] = 0
    assert motion.state[0] > 0
    assert motion.covariance[0, 0] > 0


@pytest.mark.parametrize(
    "delta,expected",
    [(0.1, 0.1), (0.0, FALLBACK_DT), (-1.0, FALLBACK_DT), (5.0, MAX_DT), (0.0001, MIN_DT)],
)
def test_timestamp_policy(delta: float, expected: float) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    previous = FrameContext("test", 0, start)
    current = FrameContext("test", 1, start + timedelta(seconds=delta))
    assert frame_dt(previous, current) == expected
    assert frame_dt(None, current) == FALLBACK_DT


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_time_step_rejected(dt: float) -> None:
    with pytest.raises(ValueError):
        transition_matrix(dt)
    with pytest.raises(ValueError):
        process_noise(KalmanConfig(), dt)


def test_public_projection_is_bounded_without_modifying_linear_state() -> None:
    state = np.array([10.0, -10.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    original = state.copy()
    box = bounded_box(state)
    assert 0 <= box.left < box.right <= 1
    assert 0 <= box.top < box.bottom <= 1
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize("state", [np.zeros((8, 1)), np.array([]), np.array([float("nan")])])
def test_invalid_state_rejected(state: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> None:
    with pytest.raises(KalmanError):
        KalmanFilter(state, np.eye(state.size))


def test_invalid_matrices_rejected() -> None:
    with pytest.raises(KalmanError):
        KalmanFilter(np.zeros(2), np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(KalmanError):
        KalmanFilter(np.zeros(2), np.diag([1.0, -1.0]))
    motion = initialize(BOX, KalmanConfig())
    with pytest.raises(KalmanError):
        motion.predict(np.eye(7), np.eye(8))
    with pytest.raises(KalmanError):
        motion.correct(np.zeros(4), observation_matrix(), np.zeros((4, 4)))
    with pytest.raises(KalmanError):
        motion.correct(np.zeros((4, 1)), observation_matrix(), np.eye(4))
    with pytest.raises(ValueError):
        bounded_box(np.full(8, float("nan")))


def test_solve_failure_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    motion = initialize(BOX, KalmanConfig())
    monkeypatch.setattr(np.linalg, "solve", Mock(side_effect=np.linalg.LinAlgError("singular")))
    with pytest.raises(KalmanError, match="cannot be solved"):
        motion.correct(measurement(BOX), observation_matrix(), np.eye(4))
