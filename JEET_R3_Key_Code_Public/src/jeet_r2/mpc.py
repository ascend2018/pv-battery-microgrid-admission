"""Exact finite-grid MPC with shared action prefixes.

Each prefix carries its full physical state and Rainflow residual history.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
from numbers import Number
from typing import Any, Callable, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.baselines.finite_grid_mpc import FiniteGridMPCConfig, FiniteGridMPCDecision
    from src.degradation.rainflow import RainflowResidualState
    from src.env.microgrid_env import MicrogridConfig, Scenario


@dataclass(frozen=True)
class PrefixRequirements:
    transitions: int
    leaves: int


@dataclass(frozen=True)
class PrefixLeaf:
    action_sequence: tuple[float, ...]
    scenario_costs: tuple[tuple[float, float, float], ...]
    scalar_score: float
    feasible: bool
    terminal_soc_by_scenario: tuple[float, ...]
    terminal_residual_by_scenario: tuple[Any, ...]

    @property
    def action(self) -> float:
        return self.action_sequence[0]


@dataclass(frozen=True)
class PrefixTreeResult:
    best: PrefixLeaf
    sequences_evaluated: int
    transitions_evaluated: int


def _validated_soc(value: Any) -> float:
    soc = np.asarray(value, dtype=float)
    if soc.shape != () or not np.isfinite(soc) or not 0.0 <= float(soc) <= 1.0:
        raise ValueError("MPC SoC must be a finite scalar in [0, 1]")
    return float(soc)


def _validate_residual(value: Any) -> None:
    """Check numeric state without importing or reconstructing source residuals."""
    if value is None or isinstance(value, (str, bytes, bool)):
        return
    if isinstance(value, Number):
        if not np.isfinite(value):
            raise ValueError("MPC residual contains nonfinite numeric state")
    elif isinstance(value, np.ndarray):
        for item in value.flat:
            _validate_residual(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _validate_residual(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _validate_residual(key)
            _validate_residual(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_residual(item)
    else:
        raise ValueError("unsupported MPC residual representation")


def prefix_requirements(*, action_count: int, horizon: int, scenario_count: int) -> PrefixRequirements:
    """Worst complete unit; each forecast context requires its own transitions."""
    if not 1 <= horizon <= 4 or action_count < 1 or scenario_count < 1:
        raise ValueError("positive action/scenario counts and horizon in [1, 4] required")
    return PrefixRequirements(
        transitions=scenario_count * sum(action_count ** depth for depth in range(1, horizon + 1)),
        leaves=action_count ** horizon,
    )


def solve_prefix_tree(
    env_cfg: Any,
    forecast_scenarios: Sequence[Any],
    *,
    horizon: int,
    action_grid: Sequence[float],
    initial_soc: float,
    initial_residual: Any,
    weights: Sequence[float],
    global_t: int,
    global_episode_len: int,
    transition: Callable[..., tuple[float, Any, np.ndarray, bool]],
    reserve: Callable[[PrefixRequirements], None],
    on_leaf: Callable[[PrefixLeaf], None] | None = None,
) -> PrefixTreeResult:
    """Visit leaves in ``itertools.product`` order with exact sequential sums.

    ``transition`` must be pure: it may not mutate its input residual or forecast
    context. Residual state may use numeric containers or dataclass records.
    ``reserve`` must atomically reserve the full requirement or raise, before
    any callback transition is run. Actual callback counts are returned; errors
    propagate and never produce a successful decision.
    """
    scenarios = tuple(forecast_scenarios)
    actions = tuple(float(a) for a in action_grid)
    required = prefix_requirements(action_count=len(actions), horizon=horizon, scenario_count=len(scenarios))
    if any(not np.isfinite(a) or not -1.0 <= a <= 1.0 for a in actions):
        raise ValueError("actions must be finite and in [-1, 1]")
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or np.any(~np.isfinite(w)) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("mpc.weights must contain three finite nonnegative values with positive sum")
    w = w / float(w.sum())
    root_soc = _validated_soc(initial_soc)
    _validate_residual(initial_residual)
    if not callable(reserve):
        raise ValueError("whole-unit reservation callback is required")
    reserve(required)

    best: PrefixLeaf | None = None
    transitions = 0
    leaves = 0

    def visit(sequence, states):
        nonlocal best, transitions, leaves
        offset = len(sequence)
        if offset == horizon:
            leaves += 1
            costs = tuple(state[2] for state in states)
            leaf = PrefixLeaf(
                action_sequence=sequence,
                scenario_costs=tuple(tuple(float(x) for x in F) for F in costs),
                scalar_score=max(float(np.dot(w, F)) for F in costs),
                feasible=bool(all(state[3] for state in states)),
                terminal_soc_by_scenario=tuple(state[0] for state in states),
                terminal_residual_by_scenario=tuple(state[1] for state in states),
            )
            if on_leaf is not None:
                on_leaf(leaf)
            if best is None or (leaf.feasible and not best.feasible) or (
                leaf.feasible == best.feasible and leaf.scalar_score < best.scalar_score
            ):
                best = leaf
            return
        for action in actions:
            children = []
            for scenario, (soc, residual, total, feasible) in zip(scenarios, states):
                transitions += 1
                soc_next, residual_next, costs, step_feasible = transition(
                    env_cfg, scenario, t=offset, soc=soc, residual=residual,
                    action=action, terminal_override=global_t + offset == global_episode_len - 1,
                )
                soc_next = _validated_soc(soc_next)
                _validate_residual(residual_next)
                step_costs = np.asarray(costs, dtype=float)
                if step_costs.shape != (3,) or not np.isfinite(step_costs).all():
                    raise ValueError("MPC transition costs must contain three finite values")
                if not isinstance(step_feasible, (bool, np.bool_)):
                    raise ValueError("MPC transition feasibility must be boolean")
                next_total = total.copy()
                with np.errstate(over="ignore", invalid="ignore"):
                    next_total += step_costs
                if not np.isfinite(next_total).all():
                    raise ValueError("MPC accumulated costs must remain finite")
                children.append((soc_next, residual_next, next_total, feasible and step_feasible))
            visit(sequence + (action,), children)

    visit((), [(root_soc, initial_residual, np.zeros(3, dtype=float), True) for _ in scenarios])
    assert best is not None
    return PrefixTreeResult(best, leaves, transitions)


def solve_finite_grid_mpc_decision(
    env_cfg: MicrogridConfig,
    forecast_scenarios: Sequence[Scenario],
    *,
    initial_soc: float,
    initial_residual: RainflowResidualState,
    weights: Sequence[float],
    global_t: int,
    global_episode_len: int,
    config: FiniteGridMPCConfig,
    reserve: Callable[[PrefixRequirements], None] | None = None,
) -> FiniteGridMPCDecision:
    """Select the first action using feasibility and scalarized cost.

    The reservation callback receives the complete prefix-tree budget before
    transitions are evaluated.
    """
    if reserve is None:
        raise ValueError("source MPC requires a whole-unit reservation callback")
    if not forecast_scenarios:
        raise ValueError("at least one MPC forecast scenario is required")
    horizon = len(forecast_scenarios[0].pv_kw)
    if horizon < 1 or horizon > config.horizon:
        raise ValueError("invalid forecast horizon")
    if any(len(s.pv_kw) != horizon for s in forecast_scenarios):
        raise ValueError("all robust forecast scenarios must share one horizon")

    from src.baselines.finite_grid_mpc import FiniteGridMPCDecision, _planning_config
    from src.baselines.train_planning_baselines import _simulate_grid_transition

    result = solve_prefix_tree(
        _planning_config(env_cfg, horizon), forecast_scenarios,
        horizon=horizon, action_grid=config.action_grid, initial_soc=initial_soc,
        initial_residual=initial_residual, weights=weights, global_t=global_t,
        global_episode_len=global_episode_len, transition=_simulate_grid_transition,
        reserve=reserve,
    )
    leaf = result.best
    return FiniteGridMPCDecision(
        action=leaf.action, action_sequence=leaf.action_sequence,
        scenario_costs=leaf.scenario_costs, scalar_score=leaf.scalar_score,
        feasible=leaf.feasible, terminal_soc_by_scenario=leaf.terminal_soc_by_scenario,
        terminal_residual_by_scenario=leaf.terminal_residual_by_scenario,
        sequences_evaluated=result.sequences_evaluated,
        transitions_evaluated=result.transitions_evaluated,
    )
