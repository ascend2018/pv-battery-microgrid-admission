"""Microgrid environment implementation for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.degradation import model, rainflow
from src.instrumentation import (
    assert_environment_step_budget,
    record_environment_step,
    record_episode_reset,
)


Array = np.ndarray
RewardVec = Array
PreferenceVec = Array


@dataclass(frozen=True)
class MicrogridConfig:
    """Configuration container."""

    dt_hours: float
    episode_len: int
    e_bat_kwh: float
    soc_min: float
    soc_max: float
    eta_ch: float
    eta_dis: float
    p_ch_max_kw: float
    p_dis_max_kw: float
    p_grid_max_kw: float
    terminal_soc_target: Optional[float]
    terminal_soc_lambda: float
    shed_penalty_y_per_kwh: float
    p_grid_export_max_kw: Optional[float] = None
    sell_price_factor: float = 1.0
    deg_dmin: float = 0.05
    deg_a: float = 1e3
    deg_b: float = 1.5
    deg_replacement_cost_y: float = 1000.0
    deg_rainflow_eps: float = 0.0
    observation_horizon_max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.observation_horizon_max is not None and self.observation_horizon_max < self.episode_len:
            raise ValueError("observation_horizon_max must cover episode_len")


@dataclass(frozen=True)
class Scenario:
    """Deterministic exogenous profiles."""

    pv_kw: Array
    load_kw: Array
    price_buy_y_per_kwh: Array
    carbon_kg_per_kwh: Array
    price_sell_y_per_kwh: Optional[Array] = None
    forecast_windows: Optional[Array] = None


@dataclass(frozen=True)
class StepInfo:
    """Per-step diagnostics and costs."""

    p_bat_kw: float
    p_ch_kw: float
    p_dis_kw: float
    p_grid_kw: float
    p_grid_import_kw: float
    p_grid_export_kw: float
    p_curt_kw: float
    p_shed_kw: float
    soc: float
    soc_next: float
    c_grid_y: float
    c_carbon_kg: float
    c_deg_y: float
    c_shed_y: float
    c_term_y: float
    p_bat_cmd_kw: float
    projection_applied: bool
    rainflow_cycles: Tuple[Tuple[float, float], ...]
    rainflow_terminal_finalized: bool
    rainflow_stack_len: int


class MicrogridEnv:
    """Grid import/export microgrid environment."""

    def __init__(self, cfg: MicrogridConfig):
        self.cfg = cfg
        self._scenario: Optional[Scenario] = None
        self._t: int = 0
        self._soc: float = cfg.soc_min
        self._price_sell: Optional[Array] = None
        self._p_bat_hist: list[float] = []
        self._soc_hist: list[float] = []
        self._deg_params = model.DegradationParams(
            dmin=cfg.deg_dmin,
            a=cfg.deg_a,
            b=cfg.deg_b,
            replacement_cost_y=cfg.deg_replacement_cost_y,
        )
        self._deg_rainflow_eps = cfg.deg_rainflow_eps
        self._rainflow_counter = rainflow.OnlineRainflowCounter(eps=self._deg_rainflow_eps)

    def reset(
        self, *, scenario: Scenario, soc0: Optional[float] = None, seed: Optional[int] = None
    ) -> Tuple[Array, Dict[str, Any]]:
        """Reset environment state."""

        del seed  # deterministic profiles; seed unused
        self._scenario = scenario
        if scenario.price_sell_y_per_kwh is None:
            self._price_sell = scenario.price_buy_y_per_kwh * self.cfg.sell_price_factor
        else:
            self._price_sell = scenario.price_sell_y_per_kwh
        self._t = 0
        if soc0 is None:
            self._soc = float(self.cfg.soc_min)
        else:
            self._soc = float(np.clip(soc0, self.cfg.soc_min, self.cfg.soc_max))
        self._p_bat_hist = []
        self._soc_hist = [self._soc]
        self._rainflow_counter = rainflow.OnlineRainflowCounter(eps=self._deg_rainflow_eps)
        self._rainflow_counter.push(self._soc)
        record_episode_reset(
            soc0=self._soc,
            dt_hours=self.cfg.dt_hours,
            e_bat_kwh=self.cfg.e_bat_kwh,
            replacement_cost_y=self.cfg.deg_replacement_cost_y,
        )
        obs = self._obs()
        return obs, {"soc": self._soc, "t": self._t}

    def step(self, action: float) -> Tuple[Array, RewardVec, bool, bool, Dict[str, Any]]:
        """Apply one control step."""

        assert self._scenario is not None, "call reset first"
        assert_environment_step_budget()
        s = self._scenario
        t = min(self._t, self.cfg.episode_len - 1)
        cfg = self.cfg
        done = False
        truncated = False

        # Action mapping
        u = float(np.clip(action, -1.0, 1.0))
        p_bat_cmd = u * (cfg.p_dis_max_kw if u >= 0 else cfg.p_ch_max_kw)

        # Feasibility projection
        p_ch_cmd = max(-p_bat_cmd, 0.0)
        p_dis_cmd = max(p_bat_cmd, 0.0)
        p_ch_soc = (cfg.soc_max - self._soc) * cfg.e_bat_kwh / (cfg.eta_ch * cfg.dt_hours)
        p_dis_soc = (self._soc - cfg.soc_min) * cfg.e_bat_kwh * cfg.eta_dis / cfg.dt_hours
        p_ch = min(p_ch_cmd, cfg.p_ch_max_kw, p_ch_soc)
        p_dis = min(p_dis_cmd, cfg.p_dis_max_kw, p_dis_soc)
        p_bat = p_dis - p_ch
        projection_applied = (p_ch != p_ch_cmd) or (p_dis != p_dis_cmd)

        # State transition
        soc_next = self._soc + (cfg.eta_ch * p_ch - p_dis / cfg.eta_dis) / cfg.e_bat_kwh * cfg.dt_hours
        soc_next = float(np.clip(soc_next, cfg.soc_min, cfg.soc_max))

        # Power balance
        p_net = float(s.load_kw[t] - s.pv_kw[t] - p_bat)
        export_cap = cfg.p_grid_export_max_kw if cfg.p_grid_export_max_kw is not None else cfg.p_grid_max_kw
        if p_net >= 0.0:
            p_grid_import = float(min(p_net, cfg.p_grid_max_kw))
            p_grid_export = 0.0
            p_shed = max(p_net - cfg.p_grid_max_kw, 0.0)
            p_curt = 0.0
        else:
            p_grid_import = 0.0
            p_grid_export = float(min(-p_net, export_cap))
            p_curt = max(-p_net - export_cap, 0.0)
            p_shed = 0.0
        p_grid = p_grid_import - p_grid_export

        # Costs
        e_imp = p_grid_import * cfg.dt_hours
        e_exp = p_grid_export * cfg.dt_hours
        price_sell = self._price_sell
        if price_sell is None:
            price_sell = s.price_buy_y_per_kwh * cfg.sell_price_factor
        c_grid = s.price_buy_y_per_kwh[t] * e_imp - price_sell[t] * e_exp
        c_shed = cfg.shed_penalty_y_per_kwh * p_shed * cfg.dt_hours
        c_term = 0.0
        if cfg.terminal_soc_target is not None and t == cfg.episode_len - 1:
            c_term = cfg.terminal_soc_lambda * abs(soc_next - cfg.terminal_soc_target)
        c_grid += c_shed + c_term
        c_carbon = s.carbon_kg_per_kwh[t] * e_imp

        # Track realized histories and advance causal degradation accounting.
        self._p_bat_hist.append(p_bat)
        self._soc_hist.append(soc_next)
        terminal = t == cfg.episode_len - 1
        c_deg, new_cycles = self._compute_degradation_cost(
            soc_next=soc_next,
            terminal=terminal,
        )

        reward_vec = np.array([-c_grid, -c_carbon, -c_deg], dtype=float)

        info = StepInfo(
            p_bat_kw=p_bat,
            p_ch_kw=p_ch,
            p_dis_kw=p_dis,
            p_grid_kw=p_grid,
            p_grid_import_kw=p_grid_import,
            p_grid_export_kw=p_grid_export,
            p_curt_kw=p_curt,
            p_shed_kw=p_shed,
            soc=self._soc,
            soc_next=soc_next,
            c_grid_y=c_grid,
            c_carbon_kg=c_carbon,
            c_deg_y=c_deg,
            c_shed_y=c_shed,
            c_term_y=c_term,
            p_bat_cmd_kw=p_bat_cmd,
            projection_applied=projection_applied,
            rainflow_cycles=tuple((cycle.dod, cycle.count) for cycle in new_cycles),
            rainflow_terminal_finalized=terminal,
            rainflow_stack_len=len(self._rainflow_counter.residual_state.peaks),
        ).__dict__

        # Advance
        self._soc = soc_next
        self._t += 1
        if self._t >= cfg.episode_len:
            done = True

        record_environment_step(info)
        obs = self._obs()
        return obs, reward_vec, done, truncated, info

    def get_episode_trajectory(self) -> Mapping[str, Array]:
        """Return realized trajectories."""

        return {
            "soc": np.asarray(self._soc_hist, dtype=float),
            "p_bat": np.asarray(self._p_bat_hist, dtype=float),
        }

    @property
    def observation_dim(self) -> int:
        """Return physical/exogenous features plus lossless Rainflow-state encoding."""

        horizon = self.cfg.observation_horizon_max or self.cfg.episode_len
        max_points = horizon + 1
        forecast_dim = 0 if self._scenario is None or self._scenario.forecast_windows is None else 12
        return 9 + forecast_dim + 2 * max_points + 2

    def _obs(self) -> Array:
        """Construct physical/exogenous observation augmented with Rainflow state."""

        assert self._scenario is not None, "call reset first"
        t = min(self._t, self.cfg.episode_len - 1)
        s = self._scenario
        price_sell = self._price_sell
        if price_sell is None:
            price_sell = s.price_buy_y_per_kwh * self.cfg.sell_price_factor
        t_frac = 0.0 if self.cfg.episode_len <= 1 else float(t) / float(self.cfg.episode_len - 1)
        t_angle = 2.0 * np.pi * t_frac
        physical = np.array(
            [
                self._soc,
                float(s.pv_kw[t]),
                float(s.load_kw[t]),
                float(s.price_buy_y_per_kwh[t]),
                float(price_sell[t]),
                float(s.carbon_kg_per_kwh[t]),
                t_frac,
                float(np.sin(t_angle)),
                float(np.cos(t_angle)),
            ],
            dtype=float,
        )
        horizon = self.cfg.observation_horizon_max or self.cfg.episode_len
        residual = self._rainflow_counter.encode(max_points=horizon + 1)
        forecast = np.array([], dtype=float) if s.forecast_windows is None else s.forecast_windows[t].reshape(-1)
        observation = np.concatenate([physical, forecast, residual])
        if observation.shape != (self.observation_dim,):
            raise RuntimeError("unexpected augmented observation shape")
        return observation

    def _compute_degradation_cost(
        self,
        *,
        soc_next: float,
        terminal: bool,
    ) -> Tuple[float, list[rainflow.RainflowCycle]]:
        """Advance Rainflow state once and return this immutable transition cost."""

        cycles = self._rainflow_counter.push(soc_next)
        if terminal:
            cycles.extend(self._rainflow_counter.finish())

        damage = model.episode_damage(cycles, self._deg_params)
        return model.episode_cost_y(damage, self._deg_params), cycles
