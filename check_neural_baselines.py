"""Check neural NSGA-II and NSGA-III survival with synthetic scores."""

import numpy as np

from src.baselines.neural_nsga import Candidate, nsga2_survival, nsga3_survival


def _candidate(name, objectives, shedding=0.0):
    return Candidate(name, np.zeros(2), np.asarray(objectives), shedding == 0, shedding)


def main():
    parents = [
        _candidate("cost", (0, 4, 4)),
        _candidate("wear", (4, 4, 0)),
        _candidate("dominated", (6, 6, 6)),
    ]
    offspring = [
        _candidate("carbon", (4, 0, 4)),
        _candidate("balanced", (2, 2, 2)),
        _candidate("infeasible", (0, 0, 0), shedding=1.0),
    ]
    cases = (
        (nsga2_survival, ["carbon", "cost"]),
        (nsga3_survival, ["cost", "carbon"]),
    )
    for select, expected in cases:
        if [row.source_id for row in select(parents, offspring, 2)] != expected:
            raise AssertionError(f"Unexpected split-front selection from {select.__name__}")
        survivors = select(parents, offspring, 6)
        if survivors[-1].source_id != "infeasible" or not all(row.feasible for row in survivors[:-1]):
            raise AssertionError(f"Unexpected feasibility ordering from {select.__name__}")
    print("Verified neural NSGA-II crowding and neural NSGA-III reference-direction survival.")


if __name__ == "__main__":
    main()
