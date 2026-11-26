"""Simple rule-based SoC threshold policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RuleBasedConfig:
    """Configuration for rule baseline."""

    soc_low: float = 0.3
    soc_high: float = 0.8


class RuleBasedPolicy:
    """State-feedback threshold policy."""

    def __init__(self, cfg: RuleBasedConfig):
        self.cfg = cfg

    def act(self, obs: np.ndarray) -> float:
        """Map observation to normalized action in [-1,1]."""

        soc = float(obs[0])
        if soc < self.cfg.soc_low:
            return -1.0  # charge
        if soc > self.cfg.soc_high:
            return 1.0  # discharge
        return 0.0


def run_episode(env, policy: RuleBasedPolicy) -> Tuple[float, float, float, float]:
    """Run one episode and return cumulative objectives."""

    obs, _ = env.reset(scenario=env._scenario or env._scenario)  # scenario should be pre-set
    done = False
    R_scalar = 0.0
    F1 = F2 = F3 = 0.0
    while not done:
        action = policy.act(obs)
        obs, r_vec, done, _, info = env.step(action)
        R_scalar += float(np.dot(np.ones(3), r_vec))
        # r_vec = -c => accumulate costs
        F1 += -r_vec[0]
        F2 += -r_vec[1]
        F3 += -r_vec[2]
    return F1, F2, F3, R_scalar
