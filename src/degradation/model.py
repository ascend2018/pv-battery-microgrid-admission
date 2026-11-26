"""Damage and cost model based on cycle counting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .rainflow import RainflowCycle


@dataclass(frozen=True)
class DegradationParams:
    """Parameters for damage and monetization."""

    dmin: float
    a: float
    b: float
    replacement_cost_y: float  # C_rep

    def __post_init__(self) -> None:
        values = (self.dmin, self.a, self.b, self.replacement_cost_y)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("degradation parameters must be finite")
        if self.dmin < 0.0:
            raise ValueError("dmin must be non-negative")
        if self.a <= 0.0:
            raise ValueError("a must be positive")
        if self.b <= 0.0:
            raise ValueError("b must be positive")
        if self.replacement_cost_y < 0.0:
            raise ValueError("replacement_cost_y must be non-negative")


def episode_damage(cycles: Sequence[RainflowCycle], params: DegradationParams) -> float:
    """Compute Miner’s damage fraction."""

    dmg = 0.0
    for c in cycles:
        d = c.dod
        if d < params.dmin:
            continue
        n_allow = params.a * (d ** (-params.b))
        if n_allow <= 0:
            continue
        dmg += c.count / n_allow
    return float(max(dmg, 0.0))


def episode_cost_y(damage: float, params: DegradationParams) -> float:
    """Convert damage to monetary cost."""

    return float(params.replacement_cost_y * max(damage, 0.0))
