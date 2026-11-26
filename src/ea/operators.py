"""Variation operators for EA."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def mutate(theta: np.ndarray, sigma: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian mutation on parameter vector."""

    mask = rng.random(theta.shape) < p
    noise = rng.normal(scale=sigma, size=theta.shape)
    return theta + mask * noise


def crossover(parent1: np.ndarray, parent2: np.ndarray, rate: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover between two parents."""

    mask = rng.random(parent1.shape) < rate
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2
