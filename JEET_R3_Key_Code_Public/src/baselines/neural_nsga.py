"""Parent and offspring survival for neural NSGA-II and neural NSGA-III."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Candidate:
    """One source-precision genotype and training-domain assessment.

    source_id MUST bind topology, original dtype/bytes, and execution config; it
    must not be an execution-float32 identity. The backend provides that binding.
    Nonfinite observations are technical failures, never feasible solutions.
    """
    source_id: str
    theta: np.ndarray
    objectives: np.ndarray
    feasible: bool
    shedding: float
    evaluation_key: str | None = None

    def __post_init__(self):
        self.theta = np.asarray(self.theta).copy()
        self.objectives = np.asarray(self.objectives, dtype=float).copy()
        if not self.source_id or self.theta.ndim != 1:
            raise ValueError('source identity and flat source genotype required')
        if self.objectives.shape != (3,) or not np.isfinite(self.objectives).all() or not np.isfinite(self.theta).all():
            raise ValueError('finite source parameters and three objectives required')
        if not np.isfinite(self.shedding) or self.shedding < 0:
            raise ValueError('finite nonnegative shedding required')
        if not isinstance(self.feasible, (bool, np.bool_)):
            raise ValueError('explicit boolean feasibility required')
        if self.evaluation_key is not None and (type(self.evaluation_key) is not str or not self.evaluation_key):
            raise ValueError('evaluation_key must be a nonempty observation token or None')


def _validated(items):
    result = copy.deepcopy(list(items))
    for item in result:
        if not isinstance(item, Candidate):
            raise TypeError('provider must return Candidate records')
        item.__post_init__()
    return result


def _dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def _fronts(items):
    remaining = sorted(range(len(items)), key=lambda i: (items[i].source_id, i))
    result = []
    while remaining:
        front = [i for i in remaining if not any(_dominates(items[j].objectives, items[i].objectives) for j in remaining if j != i)]
        result.append(front)
        remove = set(front)
        remaining = [i for i in remaining if i not in remove]
    return result


def _crowding(items, front):
    distance = {i: 0. for i in front}
    for objective in range(3):
        ordered = sorted(front, key=lambda i: (items[i].objectives[objective], items[i].source_id, i))
        lo, hi = items[ordered[0]].objectives[objective], items[ordered[-1]].objectives[objective]
        if hi == lo:
            continue  # Constant objectives confer no arbitrary boundary advantage.
        distance[ordered[0]] = distance[ordered[-1]] = float('inf')
        for position in range(1,len(ordered)-1):
            distance[ordered[position]] += (items[ordered[position+1]].objectives[objective]-items[ordered[position-1]].objectives[objective])/(hi-lo)
    return distance


def _pool(parents, offspring, k):
    items = _validated([*parents, *offspring])
    if not isinstance(k, int) or k < 0 or k > len(items):
        raise ValueError('survival count must fit parent plus offspring pool')
    feasible = [item for item in items if item.feasible]
    infeasible = sorted([item for item in items if not item.feasible], key=lambda x:(x.shedding, tuple(x.objectives), x.source_id))
    return feasible, infeasible


def nsga2_survival(parents: Sequence[Candidate], offspring: Sequence[Candidate], k: int):
    """True parent+offspring rank/crowding survival; archive has no argument.

    Genotype duplicates retain population multiplicity. Stable source-id and
    original occurrence order resolve ties; identical objective vectors alone do
    not collapse distinct original source identities.
    """
    feasible, infeasible = _pool(parents, offspring, k)
    chosen=[]
    for front in _fronts(feasible):
        if len(chosen)+len(front) <= k:
            chosen.extend(feasible[i] for i in front)
        else:
            crowd = _crowding(feasible,front)
            order=sorted(front,key=lambda i:(-crowd[i],feasible[i].source_id,i))
            chosen.extend(feasible[i] for i in order[:k-len(chosen)])
            break
    return chosen+infeasible[:k-len(chosen)]


def reference_directions():
    """Frozen enumeration: simplex quarter-grid, followed by exact center."""
    return np.array([(i/4,j/4,(4-i-j)/4) for i in range(5) for j in range(5-i)]+[(1/3,1/3,1/3)])


def _normalized_nsga3(points):
    shifted = points-points.min(axis=0)
    # Deb--Jain extreme-point ASF and hyperplane intercepts. Degenerate or
    # nonpositive intercepts fall back to the component maxima of this pool.
    weights = np.full((3,3),1e-6)
    np.fill_diagonal(weights,1.)
    extremes=np.array([shifted[np.argmin(np.max(shifted/weight,axis=1))] for weight in weights])
    fallback=shifted.max(axis=0)
    try:
        plane=np.linalg.solve(extremes,np.ones(3))
        intercept=np.divide(1.,plane,out=np.full(3,np.inf),where=plane!=0)
        if not np.isfinite(intercept).all() or np.any(intercept <= 1e-12):
            intercept=fallback
    except np.linalg.LinAlgError:
        intercept=fallback
    return shifted/np.where(intercept>1e-12,intercept,1.)


def nsga3_survival(parents: Sequence[Candidate], offspring: Sequence[Candidate], k: int):
    """Parent+offspring NSGA-III; fixed 16 directions and deterministic niching.

    Lowest direction index resolves equal occupancy; empty niches select shortest
    perpendicular distance then hash, occupied niches select hash. This freezes
    the otherwise randomized final-front choice without drawing a hidden RNG.
    """
    feasible,infeasible=_pool(parents,offspring,k)
    fronts=_fronts(feasible)
    chosen=[]; last=[]
    for front in fronts:
        if len(chosen)+len(front)<=k:
            chosen.extend(front)
        else:
            last=front
            break
    if last and len(chosen)<k:
        participating=chosen+last
        normalized=_normalized_nsga3(np.array([feasible[i].objectives for i in participating]))
        directions=reference_directions()
        unit=directions/np.linalg.norm(directions,axis=1)[:,None]
        projection=normalized @ unit.T
        distances=np.sqrt(np.maximum(0,np.sum(normalized**2,axis=1)[:,None]-projection**2))
        niche=np.argmin(distances,axis=1)
        position={i:p for p,i in enumerate(participating)}
        occupancy=np.bincount(niche[:len(chosen)],minlength=len(directions))
        available=set(last)
        while len(chosen)<k and available:
            active={int(niche[position[i]]) for i in available}
            direction=min(active,key=lambda d:(occupancy[d],d))
            candidates=[i for i in available if niche[position[i]]==direction]
            if occupancy[direction]==0:
                winner=min(candidates,key=lambda i:(distances[position[i],direction],feasible[i].source_id,i))
            else:
                winner=min(candidates,key=lambda i:(feasible[i].source_id,i))
            chosen.append(winner);available.remove(winner);occupancy[direction]+=1
    return [feasible[i] for i in chosen]+infeasible[:k-len(chosen)]
