"""Check selected adapted CE-CMEPS operations with synthetic inputs."""

import numpy as np

from src.jeet_r2.ce import (
    CEConfig,
    CascadeGenome,
    ScoredGenome,
    cascade_offspring,
    newest_node_mutation_mask,
    parameter_count,
    update_memory,
)


def main() -> None:
    rng = np.random.default_rng(0)
    initial = CascadeGenome.from_vector(2, 0, np.zeros(parameter_count(2, 0)))
    grown = cascade_offspring(initial, rng, mutation_probability=0, hidden_node_cap=64)
    if grown.hidden_count != 1:
        raise AssertionError("The zero-hidden-node genome should grow on its first update.")

    at_cap = CascadeGenome.from_vector(2, 64, np.zeros(parameter_count(2, 64)))
    child = cascade_offspring(
        at_cap, rng, add_probability=1, mutation_probability=1, hidden_node_cap=64
    )
    changed = child.parameter_vector() != at_cap.parameter_vector()
    if child.hidden_count != 64 or not np.array_equal(changed, newest_node_mutation_mask(at_cap)):
        raise AssertionError("At the node cap, only latest-node mutation should apply.")

    first = ScoredGenome(initial, "synthetic", (1.0, 1.0, 1.0), 0.0)
    dominated = ScoredGenome(grown, "synthetic", (2.0, 2.0, 2.0), 0.0)
    if update_memory((first,), (dominated, first)) != (first,):
        raise AssertionError("Memory should retain one copy of the nondominated genome.")
    if sum(CEConfig().candidate_schedule) != 96:
        raise AssertionError("Unexpected initial CE candidate count.")
    print("Verified cascade growth, latest-node mutation at the cap, memory, and CE schedule size.")


if __name__ == "__main__":
    main()
