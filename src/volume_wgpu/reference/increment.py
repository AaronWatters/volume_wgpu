from __future__ import annotations

import numpy as np


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input volume must be a NumPy array.")
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array with shape (z, y, x).")
    if volume.size == 0:
        raise ValueError("Input volume must be nonempty.")
    if volume.shape[0] <= 0 or volume.shape[1] <= 0 or volume.shape[2] <= 0:
        raise ValueError("Input volume dimensions must be positive integers.")
    if volume.dtype != np.float32:
        raise TypeError("Input volume must have dtype float32.")
    return np.ascontiguousarray(volume)


def increment_reference(volume: np.ndarray, increment: float | np.floating) -> np.ndarray:
    """Reference implementation that increments every element by a scalar."""
    validated = _validate_volume(volume)
    scalar = np.float32(increment)
    if not np.isscalar(increment):
        raise TypeError("Increment must be a scalar value.")
    return validated + scalar
