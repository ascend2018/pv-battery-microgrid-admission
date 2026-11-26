"""Nominal forecast and configuration for the finite-grid MPC comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.degradation.rainflow import RainflowResidualState
from src.env.microgrid_env import MicrogridConfig, Scenario


@dataclass(frozen=True)
class FiniteGridMPCConfig:
    horizon: int = 4
    action_grid: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    robust_scenario_count: int = 3
    pv_relative_scale: float = 0.15
    load_relative_scale: float = 0.10
    price_relative_scale: float = 0.10
    carbon_relative_scale: float = 0.10
    wall_clock_seconds_hard_cap: float = 600.0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.horizon > 4:
            raise ValueError("finite-grid MPC horizon must be in [1, 4]")
        if not self.action_grid or any(not np.isfinite(a) or a < -1.0 or a > 1.0 for a in self.action_grid):
            raise ValueError("finite-grid MPC action grid must be finite and within [-1, 1]")
        if self.robust_scenario_count != 3:
            raise ValueError("v1.4 robust MPC freezes exactly three forecast scenarios")
        if self.wall_clock_seconds_hard_cap <= 0.0:
            raise ValueError("MPC wall-clock cap must be positive")


@dataclass(frozen=True)
class FiniteGridMPCDecision:
    action: float
    action_sequence: tuple[float, ...]
    scenario_costs: tuple[tuple[float, float, float], ...]
    scalar_score: float
    feasible: bool
    terminal_soc_by_scenario: tuple[float, ...]
    terminal_residual_by_scenario: tuple[RainflowResidualState, ...]
    sequences_evaluated: int
    transitions_evaluated: int


def _bounded_horizon(global_t: int, episode_len: int, requested: int) -> int:
    return min(int(requested), int(episode_len) - int(global_t))


def causal_nominal_forecast(realized: Scenario, *, t: int, horizon: int) -> Scenario:
    """Build current-realized plus three-step forecast data without future leakage."""

    if realized.forecast_windows is None:
        raise ValueError("finite-grid MPC requires causal forecast_windows")
    h = _bounded_horizon(t, len(realized.pv_kw), horizon)
    current = np.array(
        [realized.pv_kw[t], realized.load_kw[t], realized.price_buy_y_per_kwh[t], realized.carbon_kg_per_kwh[t]],
        dtype=float,
    )
    future = np.asarray(realized.forecast_windows[t], dtype=float)
    channels = np.vstack([current, future[: max(0, h - 1)]])
    if channels.shape != (h, 4):
        raise RuntimeError("causal forecast horizon construction failed")
    sell = None
    if realized.price_sell_y_per_kwh is not None:
        ratio = float(realized.price_sell_y_per_kwh[t] / max(realized.price_buy_y_per_kwh[t], 1e-12))
        sell = channels[:, 2] * ratio
    return Scenario(
        pv_kw=channels[:, 0],
        load_kw=channels[:, 1],
        price_buy_y_per_kwh=channels[:, 2],
        price_sell_y_per_kwh=sell,
        carbon_kg_per_kwh=channels[:, 3],
    )


def _planning_config(env_cfg: MicrogridConfig, horizon: int) -> MicrogridConfig:
    values = dict(env_cfg.__dict__)
    values["episode_len"] = int(horizon)
    values["observation_horizon_max"] = max(int(values.get("observation_horizon_max") or horizon), horizon)
    return MicrogridConfig(**values)
