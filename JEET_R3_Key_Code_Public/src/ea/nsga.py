"""NSGA-II style sorting and selection."""

from __future__ import annotations

from typing import List

import numpy as np

from .population import Individual


def dominates(f_i: np.ndarray, f_j: np.ndarray) -> bool:
    """Return True if f_i Pareto-dominates f_j (minimization)."""

    return bool(np.all(f_i <= f_j) and np.any(f_i < f_j))


def nd_sort(F: np.ndarray) -> List[List[int]]:
    """Non-dominated sorting (minimization)."""

    N = F.shape[0]
    dominates_set: List[List[int]] = [[] for _ in range(N)]
    n_dom = np.zeros(N, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                dominates_set[p].append(q)
            elif dominates(F[q], F[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    """Compute crowding distance for a single front (minimization)."""

    m = F.shape[1]
    front_len = len(front)
    d = np.zeros(front_len, dtype=float)
    if front_len == 0:
        return d
    # boundary points set to inf
    for j in range(m):
        idx_sorted = sorted(range(front_len), key=lambda i: F[front[i], j])
        d[idx_sorted[0]] = d[idx_sorted[-1]] = np.inf
        f_min = F[front[idx_sorted[0]], j]
        f_max = F[front[idx_sorted[-1]], j]
        if f_max == f_min:
            continue
        for k in range(1, front_len - 1):
            prev_f = F[front[idx_sorted[k - 1]], j]
            next_f = F[front[idx_sorted[k + 1]], j]
            d[idx_sorted[k]] += (next_f - prev_f) / (f_max - f_min)
    return d


def select_elites(pop: List[Individual], F: np.ndarray, k: int) -> List[int]:
    """Select elite indices via fronts then crowding (stable tie by index)."""

    fronts = nd_sort(F)
    selected: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            dist = crowding_distance(F, front)
            order = sorted(zip(front, dist), key=lambda x: (-x[1], x[0]))
            for idx, _ in order:
                if len(selected) < k:
                    selected.append(idx)
                else:
                    break
        if len(selected) >= k:
            break
    return selected
