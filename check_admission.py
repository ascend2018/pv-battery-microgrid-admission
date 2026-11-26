"""Exercise the paper's eight admission outcomes with synthetic objective vectors."""

import numpy as np

from src.hybrid.admission import Admission, admit_pair, place_next_population
from src.metrics.hv import HVNormalization


def main() -> None:
    normalization = HVNormalization.fit(np.array([[0, 0, 0], [1, 1, 1]]))
    parent = (0.2, 0.8, 0.8)
    child = (0.3, 0.3, 0.3)
    equal_volume_child = (0.8, 0.2, 0.8)
    cases = (
        (None, None, (), Admission((0,), "invalid_pair")),
        ((parent, child), (False, True), (), Admission((1,), "feasibility_replace")),
        ((parent, child), (True, False), (), Admission((0,), "infeasible_reject")),
        ((parent, (0.1, 0.7, 0.7)), (True, True), (), Admission((1,), "dominance_replace")),
        ((parent, (0.4, 0.9, 0.9)), (True, True), (), Admission((0,), "dominance_reject")),
        ((parent, child), (True, True), ((1.1, 0.9, 0.9),), Admission((0,), "invalid_HV")),
        ((parent, child), (True, True), (), Admission((1,), "HV_replace")),
        ((parent, equal_volume_child), (True, True), (), Admission((0, 1), "coexist")),
    )
    for pair, feasible, archive, expected in cases:
        result = admit_pair(
            pair, feasible, archive=archive, normalization=normalization
        )
        if result != expected:
            raise AssertionError(f"Expected {expected}, got {result}")

    offspring = ["offspring 1", "offspring 2", "offspring 3"]
    population = place_next_population(["parent", "child"], offspring, population_size=4)
    if population != ["parent", "child", "offspring 1", "offspring 2"]:
        raise AssertionError(f"Unexpected population order: {population}")
    print(f"Verified {len(cases)} admission branches and refined-first slot allocation.")


if __name__ == "__main__":
    main()
