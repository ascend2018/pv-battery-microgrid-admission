"""Standalone implementation of Eq. (20) and refined-first slot allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np

from src.metrics.hv import HVNormalization, hypervolume


DOMINANCE_TOLERANCE = 1e-12
HV_GAIN_TOLERANCE = 1e-9
ADMISSION_REFERENCE = (1.1, 1.1, 1.1)
Policy = TypeVar("Policy")


@dataclass(frozen=True)
class Admission:
    """Indices of retained policies in parent–child order and the decision reason."""

    kept: tuple[int, ...]
    reason: str


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(
        np.all(a <= b + DOMINANCE_TOLERANCE)
        and np.any(a < b - DOMINANCE_TOLERANCE)
    )


def admit_pair(
    pair_objectives: np.ndarray | None,
    pair_feasible: Sequence[bool] | None,
    *,
    archive: Sequence[Sequence[float]],
    normalization: HVNormalization | None,
) -> Admission:
    """Decide a pair evaluated on a common schedule using a frozen context.

    Row 0 is the parent and row 1 is the candidate. Pass None for a failed
    assessment. The caller fits the normalization before SAC refinement from
    the feasible population and archive objectives, and reuses it for each pair.
    """
    if pair_objectives is None or pair_feasible is None:
        return Admission((0,), "invalid_pair")
    try:
        objectives = np.asarray(pair_objectives, dtype=float)
        feasible = np.asarray(pair_feasible)
    except (TypeError, ValueError, OverflowError):
        return Admission((0,), "invalid_pair")
    if (
        objectives.shape != (2, 3)
        or not np.isfinite(objectives).all()
        or feasible.shape != (2,)
        or feasible.dtype.kind != "b"
    ):
        return Admission((0,), "invalid_pair")

    parent_feasible, child_feasible = feasible
    if child_feasible and not parent_feasible:
        return Admission((1,), "feasibility_replace")
    if not child_feasible:
        return Admission((0,), "infeasible_reject")

    parent, child = objectives
    if _dominates(child, parent):
        return Admission((1,), "dominance_replace")
    if _dominates(parent, child):
        return Admission((0,), "dominance_reject")

    if normalization is None:
        return Admission((0,), "invalid_HV")
    try:
        parent_union = normalization.transform(np.vstack([*archive, parent]))
        child_union = normalization.transform(np.vstack([*archive, child]))
        reference = np.asarray(ADMISSION_REFERENCE)
        if (
            not np.isfinite(parent_union).all()
            or not np.isfinite(child_union).all()
            or np.any(parent_union >= reference)
            or np.any(child_union >= reference)
        ):
            return Admission((0,), "invalid_HV")
        # The union volume is invariant to removal of dominated points.
        parent_hv = hypervolume(parent_union, reference)
        child_hv = hypervolume(child_union, reference)
    except (TypeError, ValueError, OverflowError):
        return Admission((0,), "invalid_HV")

    if child_hv > parent_hv + HV_GAIN_TOLERANCE:
        return Admission((1,), "HV_replace")
    return Admission((0, 1), "coexist")


def place_next_population(
    retained: Sequence[Policy], offspring: Sequence[Policy], population_size: int
) -> list[Policy]:
    """Append an offspring prefix after the ordered pair outputs in Eq. (21)."""
    remaining = population_size - len(retained)
    if remaining < 0 or len(offspring) < remaining:
        raise ValueError("retained policies and offspring cannot fill the population")
    return [*retained, *offspring[:remaining]]
