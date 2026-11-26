"""Exercise the engineering reference components with synthetic inputs."""

import numpy as np

from src.baselines.finite_grid_mpc import FiniteGridMPCConfig, causal_nominal_forecast
from src.baselines.rule_based import RuleBasedConfig, RuleBasedPolicy
from src.degradation.rainflow import OnlineRainflowCounter
from src.env.microgrid_env import MicrogridConfig, Scenario
from src.jeet_r2.mpc import solve_finite_grid_mpc_decision


def check_engineering_references() -> None:
    rule = RuleBasedPolicy(RuleBasedConfig())
    actions = [rule.act(np.array([soc])) for soc in (0.2, 0.3, 0.5, 0.8, 0.9)]
    if actions != [-1, 0, 0, 0, 1]:
        raise AssertionError("Unexpected Rule threshold decisions.")

    cfg = MicrogridConfig(
        dt_hours=1, episode_len=4, e_bat_kwh=50, soc_min=0.2, soc_max=0.9,
        eta_ch=0.95, eta_dis=0.95, p_ch_max_kw=20, p_dis_max_kw=20,
        p_grid_max_kw=40, terminal_soc_target=None, terminal_soc_lambda=0,
        shed_penalty_y_per_kwh=1000,
    )
    future = np.tile([0.0, 18.0, 0.05, 0.5], (4, 3, 1))
    scenario = Scenario(
        pv_kw=np.zeros(4), load_kw=np.full(4, 18.0),
        price_buy_y_per_kwh=np.array([1.0, 0.05, 0.05, 0.05]),
        carbon_kg_per_kwh=np.full(4, 0.5), forecast_windows=future,
    )
    forecast = causal_nominal_forecast(scenario, t=0, horizon=4)
    counter = OnlineRainflowCounter()
    counter.push(0.5)
    charges = []
    decision = solve_finite_grid_mpc_decision(
        cfg, [forecast], initial_soc=0.5, initial_residual=counter.residual_state,
        weights=(1, 0, 0), global_t=0, global_episode_len=4,
        config=FiniteGridMPCConfig(), reserve=charges.append,
    )
    if not decision.feasible or decision.action != 1 or decision.sequences_evaluated != 625:
        raise AssertionError("Unexpected nominal FG-MPC decision in the peak-price fixture.")
    if len(charges) != 1 or charges[0].transitions != decision.transitions_evaluated:
        raise AssertionError("The prefix-tree budget did not match the synthetic decision.")


if __name__ == "__main__":
    check_engineering_references()
    print("Verified nominal FG-MPC and Rule on synthetic inputs.")
