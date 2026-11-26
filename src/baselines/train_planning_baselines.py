"""Physical transition and Rainflow state update used by finite-grid MPC."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.degradation import model, rainflow
from src.env.microgrid_env import MicrogridConfig, Scenario


def _counter_from_residual(
    residual: rainflow.RainflowResidualState,
    *,
    eps: float,
) -> rainflow.OnlineRainflowCounter:
    counter = rainflow.OnlineRainflowCounter(eps=eps)
    counter._peaks = list(residual.peaks)
    counter._endpoint = residual.endpoint
    counter._direction = int(residual.direction)
    counter._finished = False
    return counter


def _simulate_grid_transition(
    cfg: MicrogridConfig,
    scenario: Scenario,
    *,
    t: int,
    soc: float,
    residual: rainflow.RainflowResidualState,
    action: float,
    terminal_override: Optional[bool] = None,
) -> tuple[float, rainflow.RainflowResidualState, np.ndarray, bool]:
    u = float(np.clip(action, -1.0, 1.0))
    p_bat_cmd = u * (cfg.p_dis_max_kw if u >= 0.0 else cfg.p_ch_max_kw)
    p_ch_cmd = max(-p_bat_cmd, 0.0)
    p_dis_cmd = max(p_bat_cmd, 0.0)
    p_ch_soc = (cfg.soc_max - soc) * cfg.e_bat_kwh / (cfg.eta_ch * cfg.dt_hours)
    p_dis_soc = (soc - cfg.soc_min) * cfg.e_bat_kwh * cfg.eta_dis / cfg.dt_hours
    p_ch = min(p_ch_cmd, cfg.p_ch_max_kw, p_ch_soc)
    p_dis = min(p_dis_cmd, cfg.p_dis_max_kw, p_dis_soc)
    p_bat = p_dis - p_ch
    soc_next = soc + (cfg.eta_ch * p_ch - p_dis / cfg.eta_dis) / cfg.e_bat_kwh * cfg.dt_hours
    soc_next = float(np.clip(soc_next, cfg.soc_min, cfg.soc_max))

    p_net = float(scenario.load_kw[t] - scenario.pv_kw[t] - p_bat)
    export_cap = cfg.p_grid_export_max_kw if cfg.p_grid_export_max_kw is not None else cfg.p_grid_max_kw
    if p_net >= 0.0:
        p_grid_import = float(min(p_net, cfg.p_grid_max_kw))
        p_grid_export = 0.0
        p_shed = max(p_net - cfg.p_grid_max_kw, 0.0)
    else:
        p_grid_import = 0.0
        p_grid_export = float(min(-p_net, export_cap))
        p_shed = 0.0

    e_imp = p_grid_import * cfg.dt_hours
    e_exp = p_grid_export * cfg.dt_hours
    price_sell = scenario.price_sell_y_per_kwh
    if price_sell is None:
        price_sell = scenario.price_buy_y_per_kwh * cfg.sell_price_factor
    c_grid = scenario.price_buy_y_per_kwh[t] * e_imp - price_sell[t] * e_exp
    c_shed = cfg.shed_penalty_y_per_kwh * p_shed * cfg.dt_hours
    c_term = 0.0
    terminal = t == cfg.episode_len - 1 if terminal_override is None else bool(terminal_override)
    if cfg.terminal_soc_target is not None and terminal:
        c_term = cfg.terminal_soc_lambda * abs(soc_next - cfg.terminal_soc_target)
    c_grid = float(c_grid + c_shed + c_term)
    c_carbon = float(scenario.carbon_kg_per_kwh[t] * e_imp)

    counter = _counter_from_residual(residual, eps=cfg.deg_rainflow_eps)
    cycles = counter.push(soc_next)
    if terminal:
        cycles.extend(counter.finish())
    deg_params = model.DegradationParams(
        dmin=cfg.deg_dmin,
        a=cfg.deg_a,
        b=cfg.deg_b,
        replacement_cost_y=cfg.deg_replacement_cost_y,
    )
    c_deg = model.episode_cost_y(model.episode_damage(cycles, deg_params), deg_params)
    costs = np.array([c_grid, c_carbon, float(c_deg)], dtype=float)
    feasible = bool(p_shed <= 1e-6 and np.isfinite(costs).all())
    return soc_next, counter.residual_state, costs, feasible
