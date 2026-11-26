"""Population evaluation for EA."""

from __future__ import annotations

import numpy as np
import torch

from src.models.actor_mlp import MLP, actor_param_dim, load_actor_params


def evaluate_population(pop, env, scenario, n_eval: int, seed: int):
    """Evaluate population; return (F, feasible) with F shape (K,3) minimizing costs."""

    if n_eval < 1:
        raise ValueError("n_eval must be positive")
    rng = np.random.default_rng(seed)
    evaluation_seeds = rng.integers(0, 1_000_000_000, size=n_eval)
    K = len(pop)
    F = np.zeros((K, 3), dtype=float)
    feasible = np.ones(K, dtype=bool)
    obs_dim = env._obs().shape[0]
    theta_dim = actor_param_dim(obs_dim)
    actors = []
    for ind in pop:
        if ind.theta.size != theta_dim:
            raise ValueError(f"Individual theta has size {ind.theta.size}, expected {theta_dim}")
        actor = MLP(obs_dim, 1)
        load_actor_params(actor, ind.theta)
        actor.eval()
        actors.append(actor)

    for i, (ind, actor) in enumerate(zip(pop, actors)):
        costs_acc = np.zeros(3, dtype=float)
        feas_i = True
        for evaluation_seed in evaluation_seeds:
            obs, _ = env.reset(scenario=scenario, seed=int(evaluation_seed))
            done = False
            while not done:
                with torch.no_grad():
                    a_raw = actor(torch.as_tensor(obs, dtype=torch.float32)).squeeze().item()
                act = float(np.clip(np.tanh(a_raw), -1.0, 1.0))
                obs, r_vec, done, _, info = env.step(act)
                costs_acc += -r_vec  # reward -> cost
                # Feasible if no shedding occurs and costs finite
                if info.get("p_shed_kw", 0.0) > 1e-6 or not np.isfinite(costs_acc).all():
                    feas_i = False
        F[i] = costs_acc / n_eval
        ind.F = F[i]
        feasible[i] = feas_i
    return F, feasible
