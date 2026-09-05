"""Reusable linear Kalman mathematics implemented in NumPy, with no vision SDK."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class KalmanError(ValueError):
    """Invalid dimensions, covariance, or non-finite filter arithmetic."""


def _array(value: Array, shape: tuple[int, ...], name: str) -> Array:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise KalmanError(f"{name} must be finite with shape {shape}")
    return result


def _covariance(value: Array, size: int, name: str, *, positive: bool = False) -> Array:
    matrix = _array(value, (size, size), name)
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise KalmanError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < -1e-12 or (positive and eigenvalues.min() <= 0):
        raise KalmanError(f"{name} must be positive {'definite' if positive else 'semidefinite'}")
    return matrix


class KalmanFilter:
    """Functional updates return new filters; callers cannot mutate stored arrays.

    x is a length-n vector (the mathematical n×1 column), P is n×n. predict()
    accepts F/Q; correct() accepts z/H/R, allowing reuse with other linear models.
    """

    def __init__(self, state: Array, covariance: Array) -> None:
        if state.ndim != 1 or state.size == 0:
            raise KalmanError("state must be a non-empty one-dimensional vector")
        self._state = _array(state, (state.size,), "state")
        self._covariance = _covariance(covariance, state.size, "P")

    @property
    def state(self) -> Array:
        return self._state.copy()

    @property
    def covariance(self) -> Array:
        return self._covariance.copy()

    def predict(self, transition: Array, process_noise: Array) -> "KalmanFilter":
        size = self._state.size
        transition = _array(transition, (size, size), "F")
        noise = _covariance(process_noise, size, "Q")
        return KalmanFilter(
            transition @ self._state,
            transition @ self._covariance @ transition.T + noise,
        )

    def correct(
        self, measurement: Array, observation: Array, measurement_noise: Array
    ) -> "KalmanFilter":
        if measurement.ndim != 1 or measurement.size == 0:
            raise KalmanError("measurement must be a non-empty vector")
        size, observed = self._state.size, measurement.size
        measurement = _array(measurement, (observed,), "z")
        observation = _array(observation, (observed, size), "H")
        noise = _covariance(measurement_noise, observed, "R", positive=True)
        innovation = measurement - observation @ self._state
        cross_covariance = self._covariance @ observation.T
        innovation_covariance = observation @ cross_covariance + noise
        try:
            gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
        except np.linalg.LinAlgError as exc:
            raise KalmanError("innovation covariance cannot be solved") from exc
        residual = np.eye(size) - gain @ observation
        # Joseph form equals (I-KH)P in exact arithmetic; it is more robust to
        # floating-point cancellation and preserves positive semidefiniteness.
        covariance = residual @ self._covariance @ residual.T + gain @ noise @ gain.T
        return KalmanFilter(self._state + gain @ innovation, (covariance + covariance.T) * 0.5)
