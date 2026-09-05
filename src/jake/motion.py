"""Normalized constant-velocity box model and explicit timestamp policy."""

from math import isfinite

import numpy as np

from jake.config import KalmanConfig
from jake.domain import BoundingBox, FrameContext
from jake.kalman import Array, KalmanFilter

MIN_DT = 0.001
MAX_DT = 1.0
FALLBACK_DT = 1.0 / 30.0
MIN_SIZE = 1e-6


def frame_dt(previous: FrameContext | None, current: FrameContext) -> float:
    """Seconds: clamp positive deltas to [1 ms, 1 s]; otherwise use 1/30 s."""
    if previous is None:
        return FALLBACK_DT
    elapsed = (current.captured_at - previous.captured_at).total_seconds()
    return min(MAX_DT, max(MIN_DT, elapsed)) if elapsed > 0 else FALLBACK_DT


def measurement(box: BoundingBox) -> Array:
    return np.array(
        [
            (box.left + box.right) / 2,
            (box.top + box.bottom) / 2,
            box.right - box.left,
            box.bottom - box.top,
        ],
        dtype=np.float64,
    )


def observation_matrix() -> Array:
    return np.eye(4, 8, dtype=np.float64)


def transition_matrix(dt: float) -> Array:
    if not isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    transition = np.eye(8, dtype=np.float64)
    transition[:4, 4:] = np.eye(4) * dt
    return transition


def process_noise(config: KalmanConfig, dt: float) -> Array:
    if not isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    return np.diag(
        np.array([config.process_noise_position] * 4 + [config.process_noise_velocity] * 4) * dt
    )


def initialize(box: BoundingBox, config: KalmanConfig) -> KalmanFilter:
    state = np.concatenate((measurement(box), np.zeros(4)))
    covariance = np.diag(
        [config.initial_position_variance] * 4 + [config.initial_velocity_variance] * 4
    )
    return KalmanFilter(state, covariance)


def bounded_box(state: Array) -> BoundingBox:
    """Project only the exposed geometry; leave the internal linear state intact.

    Width/height are clamped to [1e-6, 1]; centers are clamped so the whole box
    fits inside the image. Out-of-view predictions can therefore sit at an edge.
    """
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError("box state must be finite with shape (8,)")
    width, height = (min(1.0, max(MIN_SIZE, float(value))) for value in state[2:4])
    cx = min(1 - width / 2, max(width / 2, float(state[0])))
    cy = min(1 - height / 2, max(height / 2, float(state[1])))
    return BoundingBox(
        max(0.0, cx - width / 2),
        max(0.0, cy - height / 2),
        min(1.0, cx + width / 2),
        min(1.0, cy + height / 2),
    )
