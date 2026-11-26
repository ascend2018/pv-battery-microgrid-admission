"""Run Algorithm 1 on an ordered sequence of supplied training units."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.ea.evaluate import evaluate_population
from src.ea.nsga import nd_sort, select_elites
from src.ea.operators import crossover, mutate
from src.ea.population import Individual
from src.env.microgrid_env import MicrogridConfig, MicrogridEnv, Scenario
from src.hybrid.admission import admit_pair, place_next_population
from src.hybrid.sac_refinement import SACConfig, ScalarizedSAC
from src.metrics.hv import HVNormalization
from src.models.actor_mlp import actor_param_dim


POPULATION_SIZE = 16
ELITE_COUNT = 4

CASE_ENVIRONMENT = {
    "dt_hours": 1.0,
    "observation_horizon_max": 48,
    "e_bat_kwh": 50.0,
    "soc_min": 0.2,
    "soc_max": 0.9,
    "eta_ch": 0.95,
    "eta_dis": 0.95,
    "p_ch_max_kw": 20.0,
    "p_dis_max_kw": 20.0,
    "p_grid_max_kw": 40.0,
    "p_grid_export_max_kw": 40.0,
    "sell_price_factor": 0.7,
    "terminal_soc_target": 0.5,
    "terminal_soc_lambda": 0.0,
    "shed_penalty_y_per_kwh": 1000.0,
}


def scenario_from_mapping(values: dict) -> Scenario:
    """Read one 24- or 48-hour unit with three forecast steps per observation."""
    if not isinstance(values, dict):
        raise ValueError("each training unit must be a JSON object")
    names = ("pv_kw", "load_kw", "price_buy_y_per_kwh", "carbon_kg_per_kwh")
    profiles = {name: np.asarray(values[name], dtype=float) for name in names}
    horizon = profiles["pv_kw"].size
    if horizon not in (24, 48):
        raise ValueError("a training unit must contain 24 or 48 hours")
    if "price_sell_y_per_kwh" in values:
        profiles["price_sell_y_per_kwh"] = np.asarray(values["price_sell_y_per_kwh"], dtype=float)
    for name, array in profiles.items():
        if array.shape != (horizon,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {horizon} finite hourly values")
    forecasts = np.asarray(values["forecast_windows"], dtype=float)
    if forecasts.shape != (horizon, 3, 4) or not np.isfinite(forecasts).all():
        raise ValueError("forecast_windows must have shape (hours, 3, 4) and finite values")
    return Scenario(**profiles, forecast_windows=forecasts)


def _update_archive(archive: list[Individual], population: list[Individual], feasible: np.ndarray):
    # Keep the objective snapshot with each archived policy.
    candidates = archive + [
        Individual(policy.theta.copy(), policy.F.copy())
        for policy, is_feasible in zip(population, feasible)
        if is_feasible
    ]
    if not candidates:
        return []
    objectives = np.stack([policy.F for policy in candidates])
    return [candidates[index] for index in nd_sort(objectives)[0]]


def _refine(parent, agent, env, scenario, preference):
    agent.reset_refinement_actor(parent.theta)
    for _ in range(2):
        obs, _ = env.reset(scenario=scenario)
        for _ in range(env.cfg.episode_len):
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.buffer.add((obs, action, reward, next_obs, done, preference))
            agent.update_with_actor_control(
                update_actor=len(agent.buffer.storage) >= agent.cfg.actor_warmup_transitions
            )
            obs = next_obs
    return Individual(agent.get_theta())


def _offspring(elites, count, generation, rng):
    sigma = 0.1 if generation <= 10 else 0.03
    result = []
    while len(result) < count:
        first, second = rng.integers(len(elites), size=2)
        children = crossover(elites[first].theta, elites[second].theta, rate=0.5, rng=rng)
        result.extend(Individual(mutate(theta, sigma=sigma, p=0.5, rng=rng)) for theta in children)
    return result[:count]


def run_case(schedule: Sequence[Scenario], *, seed: int, device: str) -> dict:
    """Use one generation per scheduled unit and return its final policy sets."""
    if not schedule or type(seed) is not int or seed < 0:
        raise ValueError("a nonempty schedule and nonnegative integer seed are required")
    rng = np.random.default_rng(seed)
    archive: list[Individual] = []
    population: list[Individual] = []

    for generation, scenario in enumerate(schedule):
        horizon = len(scenario.pv_kw)
        env = MicrogridEnv(MicrogridConfig(episode_len=horizon, **CASE_ENVIRONMENT))
        observation, _ = env.reset(scenario=scenario)
        if not population:
            width = actor_param_dim(len(observation))
            population = [Individual(rng.normal(0.0, 0.03, width)) for _ in range(POPULATION_SIZE)]

        objectives, feasible = evaluate_population(population, env, scenario, n_eval=1, seed=seed)
        if not np.isfinite(objectives).all():
            raise ValueError("population evaluation produced nonfinite objectives")
        archive = _update_archive(archive, population, feasible)
        archive_objectives = [policy.F for policy in archive]
        fitting_points = [*objectives[feasible], *archive_objectives]
        normalization = HVNormalization.fit(np.stack(fitting_points)) if fitting_points else None

        eligible = np.flatnonzero(feasible) if feasible.any() else np.arange(POPULATION_SIZE)
        pool = [population[index] for index in eligible]
        elites = [pool[index] for index in select_elites(pool, objectives[eligible], ELITE_COUNT)]
        preferences = rng.dirichlet(np.ones(3), size=POPULATION_SIZE)
        # A generation shares replay and twin critics across its actor refinements.
        sac = SACConfig(batch_size=8, steps_per_episode=2 * horizon, device=device)
        agent = ScalarizedSAC(len(observation), sac, preferences[0], int(rng.integers(1_000_000_000)))
        retained = []
        for parent, preference in zip(elites, preferences):
            child = _refine(parent, agent, env, scenario, preference)
            pair = (Individual(parent.theta), child)
            try:
                pair_objectives, pair_feasible = evaluate_population(pair, env, scenario, n_eval=1, seed=seed)
            except (RuntimeError, ValueError, ArithmeticError):
                pair_objectives, pair_feasible = None, None
            decision = admit_pair(pair_objectives, pair_feasible, archive=archive_objectives, normalization=normalization)
            retained.extend(pair[index] for index in decision.kept)

        offspring = _offspring(elites, POPULATION_SIZE - len(retained), generation, rng)
        population = place_next_population(retained, offspring, POPULATION_SIZE)

    return {
        "seed": seed,
        "generations": len(schedule),
        "observation_dim": len(observation),
        "last_evaluated_archive": [
            {"actor_parameters": policy.theta.tolist(), "objectives": policy.F.tolist()}
            for policy in archive
        ],
        "last_constructed_population": [policy.theta.tolist() for policy in population],
    }
