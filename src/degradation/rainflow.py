"""PyYeti-aligned finite-record Rainflow counting and causal online state.

The counting convention matches ``pyyeti==1.4.6.5`` with
``cyclecount.findap(signal, tol=0.0)`` followed by its three-point Rainflow
state machine. PyYeti serves as an independent comparison reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class RainflowCycle:
    """One emitted cycle range, mean SoC, and half/full count."""

    dod: float
    soc_mean: float
    count: float


@dataclass(frozen=True)
class RainflowResidualState:
    """Causal residual state retained between physical transitions."""

    peaks: tuple[float, ...]
    endpoint: float | None
    direction: int


def _validate_soc(soc: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(soc, dtype=float)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError("SoC path must be 1-D with length >=1")
    if np.any(~np.isfinite(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("SoC values must be finite and in [0,1]")
    return arr


def _require_zero_tolerance(eps: float) -> None:
    if float(eps) != 0.0:
        raise ValueError("The frozen Rainflow convention requires eps=0.0")


def _turning_points(soc: Sequence[float] | np.ndarray, eps: float = 0.0) -> List[float]:
    """Return first, strict reversals, and final point with exact deduplication."""

    _require_zero_tolerance(eps)
    arr = _validate_soc(soc)
    compressed = [float(arr[0])]
    for value in arr[1:]:
        value = float(value)
        if value != compressed[-1]:
            compressed.append(value)
    if len(compressed) <= 2:
        return compressed

    peaks = [compressed[0]]
    for left, middle, right in zip(compressed, compressed[1:], compressed[2:]):
        if (middle - left) * (right - middle) < 0.0:
            peaks.append(middle)
    peaks.append(compressed[-1])
    return peaks


def _accept_peak(stack: list[float], peak: float) -> list[RainflowCycle]:
    cycles: list[RainflowCycle] = []
    stack.append(float(peak))
    while len(stack) > 2:
        previous_range = abs(stack[-3] - stack[-2])
        newest_range = abs(stack[-2] - stack[-1])
        if newest_range < previous_range:
            break
        mean = 0.5 * (stack[-3] + stack[-2])
        if len(stack) == 3:
            cycles.append(RainflowCycle(dod=previous_range, soc_mean=mean, count=0.5))
            stack[:] = stack[1:]
        else:
            cycles.append(RainflowCycle(dod=previous_range, soc_mean=mean, count=1.0))
            del stack[-3:-1]
    return cycles


def count(soc: Sequence[float] | np.ndarray, eps: float = 0.0) -> List[RainflowCycle]:
    """Count a complete finite SoC record under the frozen convention."""

    stack: list[float] = []
    cycles: list[RainflowCycle] = []
    for peak in _turning_points(_validate_soc(soc), eps=eps):
        cycles.extend(_accept_peak(stack, peak))
    for left, right in zip(stack, stack[1:]):
        cycles.append(
            RainflowCycle(
                dod=abs(right - left),
                soc_mean=0.5 * (left + right),
                count=0.5,
            )
        )
    return cycles


class OnlineRainflowCounter:
    """Causal Rainflow counter with immutable, explicit residual-state snapshots."""

    def __init__(self, *, eps: float = 0.0) -> None:
        _require_zero_tolerance(eps)
        self._peaks: list[float] = []
        self._endpoint: float | None = None
        self._direction = 0
        self._finished = False

    @staticmethod
    def _validate_sample(sample: float) -> float:
        value = float(sample)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("SoC values must be finite and in [0,1]")
        return value

    def push(self, sample: float) -> list[RainflowCycle]:
        """Consume one SoC sample and emit cycles resolved by this prefix."""

        if self._finished:
            raise RuntimeError("cannot push after finish")
        value = self._validate_sample(sample)
        if self._endpoint is None:
            self._peaks.append(value)
            self._endpoint = value
            return []

        difference = value - self._endpoint
        if difference == 0.0:
            return []
        direction = 1 if difference > 0.0 else -1
        if self._direction == 0:
            self._direction = direction
            self._endpoint = value
            return []
        if direction == self._direction:
            self._endpoint = value
            return []

        cycles = _accept_peak(self._peaks, self._endpoint)
        self._endpoint = value
        self._direction = direction
        return cycles

    def finish(self) -> list[RainflowCycle]:
        """Confirm the final endpoint and emit terminal residual half cycles once."""

        if self._finished:
            raise RuntimeError("finish may only be called once")
        if self._endpoint is None:
            raise ValueError("cannot finish an empty record")

        cycles: list[RainflowCycle] = []
        if self._peaks[-1] != self._endpoint:
            cycles.extend(_accept_peak(self._peaks, self._endpoint))
        for left, right in zip(self._peaks, self._peaks[1:]):
            cycles.append(
                RainflowCycle(
                    dod=abs(right - left),
                    soc_mean=0.5 * (left + right),
                    count=0.5,
                )
            )
        self._finished = True
        return cycles

    @property
    def residual_state(self) -> RainflowResidualState:
        return RainflowResidualState(
            peaks=tuple(self._peaks),
            endpoint=self._endpoint,
            direction=self._direction,
        )

    def encode(self, *, max_points: int) -> np.ndarray:
        """Encode ``(peaks, endpoint, direction)`` without discarding stack entries."""

        if max_points < 1:
            raise ValueError("max_points must be positive")
        if len(self._peaks) > max_points:
            raise RuntimeError("Rainflow residual stack exceeds frozen observation capacity")
        encoded = np.zeros(2 * max_points + 2, dtype=float)
        if self._peaks:
            encoded[: len(self._peaks)] = self._peaks
            encoded[max_points : max_points + len(self._peaks)] = 1.0
        encoded[2 * max_points] = 0.0 if self._endpoint is None else self._endpoint
        encoded[2 * max_points + 1] = float(self._direction)
        return encoded
