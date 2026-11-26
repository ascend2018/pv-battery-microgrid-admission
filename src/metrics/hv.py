"""Validated hypervolume and frozen affine normalization for 3-D minimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.instrumentation import record_hv_call


def _points_3d(points: np.ndarray, *, allow_empty: bool = True) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError("points must have shape (N, 3)")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("reference set must contain at least one point")
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    return array


def _reference_3d(ref: np.ndarray) -> np.ndarray:
    array = np.asarray(ref, dtype=float)
    if array.shape != (3,):
        raise ValueError("reference point must have shape (3,)")
    if not np.isfinite(array).all():
        raise ValueError("reference point must be finite")
    return array


@dataclass(frozen=True)
class HVNormalization:
    """Candidate-independent affine normalization frozen for one guard context."""

    minimum: np.ndarray
    scale: np.ndarray
    range_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=float).copy()
        scale = np.asarray(self.scale, dtype=float).copy()
        if minimum.shape != (3,) or scale.shape != (3,):
            raise ValueError("normalization vectors must have shape (3,)")
        if not np.isfinite(minimum).all() or not np.isfinite(scale).all():
            raise ValueError("normalization vectors must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("normalization scale must be positive")
        if not np.isfinite(self.range_epsilon) or self.range_epsilon <= 0.0:
            raise ValueError("range_epsilon must be positive and finite")
        minimum.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(cls, reference_set: np.ndarray, *, range_epsilon: float = 1e-12) -> "HVNormalization":
        points = _points_3d(reference_set, allow_empty=False)
        minimum = points.min(axis=0)
        ranges = points.max(axis=0) - minimum
        scale = np.where(ranges <= range_epsilon, 1.0, ranges)
        return cls(minimum=minimum, scale=scale, range_epsilon=range_epsilon)

    def transform(self, points: np.ndarray) -> np.ndarray:
        array = _points_3d(points)
        return (array - self.minimum) / self.scale


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """Return the exact dominated union volume for a 3-D minimization set."""

    record_hv_call()
    candidates = _points_3d(points)
    reference = _reference_3d(ref)
    if candidates.shape[0] == 0:
        return 0.0
    if np.any(candidates >= reference):
        raise ValueError("reference point must strictly dominate every point")

    volume = 0.0
    x_values = np.unique(candidates[:, 0])
    for index, x_left in enumerate(x_values):
        x_right = x_values[index + 1] if index + 1 < len(x_values) else reference[0]
        active = candidates[candidates[:, 0] <= x_left]
        volume += (x_right - x_left) * _hv2d(active[:, 1:], reference[1:])
    return float(volume)


def _hv2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Return the exact dominated union area for validated 2-D rectangles."""

    area = 0.0
    x_values = np.unique(points[:, 0])
    for index, x_left in enumerate(x_values):
        x_right = x_values[index + 1] if index + 1 < len(x_values) else ref[0]
        active_y = points[points[:, 0] <= x_left, 1]
        height = ref[1] - float(np.min(active_y))
        area += (x_right - x_left) * height
    return float(area)
