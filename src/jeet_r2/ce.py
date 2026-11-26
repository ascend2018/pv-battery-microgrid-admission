"""Adapted CE-CMEPS with cross-entropy initialization and cascade evolution.

The caller supplies policy evaluation and evaluation-budget callbacks.
"""
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from src.compact_storage.bundles import ArrayPayload, ModelBundle


def parameter_count(input_count: int, hidden_count: int) -> int:
    if type(input_count) is not int or input_count <= 0:
        raise ValueError("positive integer input count required")
    if type(hidden_count) is not int or not 0 <= hidden_count <= 64:
        raise ValueError("hidden_count must be an integer from 0 to 64")
    return sum(input_count + i + 1 for i in range(hidden_count)) + input_count + hidden_count + 1


@dataclass(frozen=True)
class CascadeGenome:
    """Fully connected feedforward cascade, source bytes never mutated.

    Node order: inputs, oldest-to-newest ReLU hidden nodes, single tanh output.
    Vector order: each hidden's incoming weights then bias, output weights then
    bias. The output receives every input/hidden node, including direct skips.
    Original precision is f4/f8; execution dtype is independently frozen.
    """
    input_count: int
    hidden_count: int
    dtype: str
    parameters: bytes
    execution_dtype: str = "f4"

    def __post_init__(self):
        count = parameter_count(self.input_count, self.hidden_count)
        if self.dtype not in ("<f4", "<f8") or self.execution_dtype not in ("f4", "f8"):
            raise ValueError("explicit original/execution floating dtype required")
        if type(self.parameters) is not bytes or len(self.parameters) != count * np.dtype(self.dtype).itemsize:
            raise ValueError("immutable exact-size original parameter bytes required")
        if not np.isfinite(self.parameter_vector()).all():
            raise ValueError("nonfinite original parameter")

    @classmethod
    def from_vector(cls, input_count, hidden_count, values, *, execution_dtype="f4"):
        values = np.asarray(values)
        if values.ndim != 1 or values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
            raise ValueError("one-dimensional original f4/f8 parameter vector required")
        dtype = values.dtype.newbyteorder("<")
        return cls(input_count, hidden_count, dtype.str, values.astype(dtype, copy=False).tobytes(), execution_dtype)

    def parameter_vector(self):
        # bytes-backed view cannot be made writable by setting its flags.
        return np.frombuffer(self.parameters, dtype=self.dtype)

    @property
    def edges(self):
        return tuple((source, target) for target in range(self.input_count, self.input_count + self.hidden_count + 1)
                     for source in range(target))

    @property
    def source_hash(self):
        return self.to_bundle().source_hash

    def to_bundle(self):
        """Actual compact-storage graph grammar, raw original-dtype arrays."""
        n, h = self.input_count, self.hidden_count
        vector, offset = self.parameter_vector(), 0
        weights = []
        biases = np.zeros(n + h + 1, dtype=self.dtype)
        for target in range(n, n + h + 1):
            weights.extend(vector[offset:offset + target])
            biases[target] = vector[offset + target]
            offset += target + 1
        weights = np.asarray(weights, dtype=self.dtype)
        arrays = {
            "edge_weight": ArrayPayload(np.dtype(self.dtype).str[1:], weights.shape, weights.tobytes()),
            "node_bias": ArrayPayload(np.dtype(self.dtype).str[1:], biases.shape, biases.tobytes()),
        }
        return ModelBundle("graph", arrays, {
            "architecture": {"type": "graph", "input_nodes": list(range(n)), "output_nodes": [n + h],
                             "weight": "edge_weight", "bias": "node_bias"},
            "topology": {"nodes": list(range(n + h + 1)), "edges": [list(edge) for edge in self.edges]},
            "activations": ["identity"] * n + ["relu"] * h + ["tanh"],
            "node_flags": [True] * (n + h + 1), "edge_flags": [True] * len(self.edges),
            "inference_state": {"type": "stateless"}, "enabled": True,
            "normalizer": {"type": "identity"}, "decoder": {"type": "identity"},
            "execution_dtype": self.execution_dtype,
            "array_specs": {name: {"dtype": value.dtype, "shape": list(value.shape),
                                   "role": "graph_weight" if name == "edge_weight" else "bias"}
                            for name, value in arrays.items()},
            "adapter_schema": "jeet-r2-cascade-v1",
        })

    @classmethod
    def from_bundle(cls, bundle):
        if not isinstance(bundle, ModelBundle) or bundle.kind != "graph":
            raise ValueError("compact-storage graph ModelBundle required")
        m = bundle.metadata
        if m.get("adapter_schema") != "jeet-r2-cascade-v1":
            raise ValueError("unsupported cascade graph schema")
        n = len(m["architecture"]["input_nodes"])
        h = len(m["topology"]["nodes"]) - n - 1
        w, b = bundle.arrays["edge_weight"], bundle.arrays["node_bias"]
        if w.dtype != b.dtype:
            raise ValueError("mixed source precision unsupported")
        weights, biases = np.frombuffer(w.data, dtype="<" + w.dtype), np.frombuffer(b.data, dtype="<" + b.dtype)
        values, offset = [], 0
        for target in range(n, n + h + 1):
            values.extend(weights[offset:offset + target])
            values.append(biases[target])
            offset += target
        result = cls.from_vector(n, h, np.asarray(values, dtype="<" + w.dtype), execution_dtype=m["execution_dtype"])
        if result.to_bundle().encode() != bundle.encode():
            raise ValueError("bundle is not the canonical complete cascade layout")
        return result


def add_hidden_node(genome: CascadeGenome, new_values) -> CascadeGenome:
    """Append one ReLU node and preserve the existing parameters bit-for-bit.

    ``new_values`` contains weights from the inputs and earlier nodes,
    the new node's bias, and its weight to the output.
    """
    n, h = genome.input_count, genome.hidden_count
    parameter_count(n, h + 1)  # Check the node bound before allocating parameters.
    values = np.asarray(new_values, dtype=genome.dtype)
    if values.shape != (n + h + 2,) or not np.isfinite(values).all():
        raise ValueError("new-node weights/bias must have exact finite shape")
    old = genome.parameter_vector()
    output_start = len(old) - (n + h + 1)
    result = np.concatenate((old[:output_start], values[:-1], old[output_start:-1], values[-1:], old[-1:]))
    return CascadeGenome.from_vector(n, h + 1, result, execution_dtype=genome.execution_dtype)


def newest_node_mutation_mask(genome: CascadeGenome):
    if genome.hidden_count == 0:
        raise ValueError("no newest hidden node; explicit bootstrap required")
    target = genome.input_count + genome.hidden_count - 1
    start = sum(genome.input_count + i + 1 for i in range(genome.hidden_count - 1))
    mask = np.zeros(len(genome.parameter_vector()), dtype=bool)
    mask[start:start + target + 1] = True  # incoming weights and this bias
    mask[-2] = True  # newest hidden -> output, output bias stays frozen
    return mask


def _probability(value):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must lie in [0,1]")


def mutate_latest_node(genome, rng, *, probability=.1, scale=.05):
    _probability(probability)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("finite nonnegative mutation scale required")
    mask = newest_node_mutation_mask(genome)
    mutable = np.flatnonzero(mask)
    selected = mutable[rng.random(len(mutable)) < probability]
    values = genome.parameter_vector().copy()
    values[selected] += rng.normal(0., scale, len(selected))
    return CascadeGenome.from_vector(genome.input_count, genome.hidden_count, values, execution_dtype=genome.execution_dtype)


def cascade_offspring(genome, rng, *, add_probability=.1, mutation_probability=.1, mutation_scale=.05,
                      hidden_node_cap=None):
    _probability(add_probability)
    if hidden_node_cap is not None and (type(hidden_node_cap) is not int or hidden_node_cap != 64):
        raise ValueError('hidden_node_cap must be 64 when specified')
    if hidden_node_cap is not None and genome.hidden_count > hidden_node_cap:
        raise ValueError('hidden_count exceeds hidden_node_cap')
    if (hidden_node_cap is None or genome.hidden_count < hidden_node_cap) and (
            genome.hidden_count == 0 or rng.random() < add_probability):
        genome = add_hidden_node(genome, rng.normal(0., .5, genome.input_count + genome.hidden_count + 2))
    return mutate_latest_node(genome, rng, probability=mutation_probability, scale=mutation_scale)


def graph_forward_cuda(genome, observation, *, device, execution_dtype, return_details=False):
    """Evaluate the cascade on CUDA with ReLU hidden nodes and a tanh output.

    Supply the observation vector in the input representation used by the model.
    """
    if type(device) is not str or not (device == "cuda" or device.startswith("cuda:")):
        raise ValueError("specify a CUDA device")
    dtype_code = {"float32": "f4", "float64": "f8", "f4": "f4", "f8": "f8"}.get(execution_dtype)
    if dtype_code != genome.execution_dtype:
        raise ValueError("execution dtype differs from frozen bundle decoder")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cascade inference requires an available CUDA device")
    selected = torch.device(device)
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise RuntimeError("selected CUDA device unavailable")
    dtype = torch.float32 if dtype_code == "f4" else torch.float64
    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=dtype, device=selected)
        if obs.ndim < 1 or obs.shape[-1] != genome.input_count or not torch.isfinite(obs).all():
            raise ValueError("finite observation with exact graph input width required")
        weights = torch.tensor(genome.parameter_vector().copy(), dtype=dtype, device=selected)
        nodes, offset = list(obs.unbind(-1)), 0
        for target in range(genome.input_count, genome.input_count + genome.hidden_count + 1):
            value = (torch.stack(nodes, dim=-1) * weights[offset:offset + target]).sum(-1) + weights[offset + target]
            nodes.append(torch.tanh(value) if target == genome.input_count + genome.hidden_count else torch.relu(value))
            offset += target + 1
        raw = value.unsqueeze(-1)
        output = nodes[-1].unsqueeze(-1)
        if not torch.isfinite(raw).all() or not torch.isfinite(output).all():
            raise ValueError("nonfinite CUDA graph output")
        return {'raw_output':raw,'bounded_command':output} if return_details else output


def population_moments(vectors, *, variance_floor=1e-6):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("finite nonempty vector cohort required")
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("positive finite variance floor required")
    return values.mean(axis=0), np.sqrt(np.maximum(values.var(axis=0, ddof=0), variance_floor))


def smooth_diagonal(previous_mean, previous_std, elite_vectors, *, alpha=.7, variance_floor=1e-6):
    _probability(alpha)
    mean, std = population_moments(elite_vectors, variance_floor=variance_floor)
    pm, ps = np.asarray(previous_mean), np.asarray(previous_std)
    if pm.shape != mean.shape or ps.shape != std.shape or not np.isfinite(pm).all() or not np.isfinite(ps).all() or (ps < 0).any():
        raise ValueError("finite matching previous mean/std required")
    return alpha * mean + (1. - alpha) * pm, np.maximum(alpha * std + (1. - alpha) * ps, np.sqrt(variance_floor))


@dataclass(frozen=True)
class ScoredGenome:
    genome: CascadeGenome
    context_id: str
    objectives: tuple
    shedding: float

    def __post_init__(self):
        if not isinstance(self.genome, CascadeGenome) or type(self.context_id) is not str or not self.context_id:
            raise ValueError("graph and immutable nonempty full-context identity required")
        if type(self.objectives) is not tuple or len(self.objectives) != 3 or not np.isfinite(self.objectives).all():
            raise ValueError("three finite minimization objectives required; invalid evaluation stops")
        if not np.isfinite(self.shedding) or self.shedding < 0:
            raise ValueError("finite nonnegative cumulative shedding required")


def _same_context_rows(rows):
    rows = tuple(rows)
    if len({row.context_id for row in rows}) > 1:
        raise ValueError("cannot rank or merge mixed context evaluations")
    unique = {}
    for row in rows:
        key = row.genome.source_hash
        if key in unique and unique[key] != row:
            raise ValueError("inconsistent exact-genome evaluation in same context")
        unique[key] = row
    # Genome identity permits score sharing, not removal of population slots.
    # Stable sort retains occurrence order among identical genome hashes.
    return tuple(sorted(rows, key=lambda row: row.genome.source_hash))


def _same_context_unique(rows):
    unique = {row.genome.source_hash: row for row in _same_context_rows(rows)}
    return tuple(unique.values())


def _dominates(a, b):
    return all(x <= y for x, y in zip(a.objectives, b.objectives)) and any(x < y for x, y in zip(a.objectives, b.objectives))


def _fronts(rows):
    remaining = list(rows)
    while remaining:
        front = tuple(row for row in remaining if not any(_dominates(other, row) for other in remaining))
        yield front
        hashes = {row.genome.source_hash for row in front}
        remaining = [row for row in remaining if row.genome.source_hash not in hashes]


def _crowding(front):
    # Distances belong to pool occurrences, including exact duplicate genomes.
    distances = [0.] * len(front)
    if len(front) <= 2:
        return [float("inf")] * len(front)
    for axis in range(3):
        ordered = sorted(range(len(front)), key=lambda i: (front[i].objectives[axis], front[i].genome.source_hash, i))
        span = front[ordered[-1]].objectives[axis] - front[ordered[0]].objectives[axis]
        if span == 0:
            continue
        distances[ordered[0]] = distances[ordered[-1]] = float("inf")
        for i in range(1, len(ordered) - 1):
            distances[ordered[i]] += (front[ordered[i + 1]].objectives[axis] - front[ordered[i - 1]].objectives[axis]) / span
    return distances


def select_survivors(rows: Iterable[ScoredGenome], capacity=16):
    if type(capacity) is not int or capacity < 0:
        raise ValueError("nonnegative integer survivor capacity required")
    rows = _same_context_rows(rows)
    feasible = [row for row in rows if row.shedding == 0]
    selected = []
    for front in _fronts(feasible):
        crowding = _crowding(front)
        order = sorted(range(len(front)), key=lambda i: (-crowding[i], front[i].genome.source_hash, i))
        selected.extend(front[i] for i in order[:max(0, capacity - len(selected))])
        if len(selected) >= capacity:
            return tuple(selected)
    infeasible = sorted((row for row in rows if row.shedding > 0),
                        key=lambda row: (row.shedding, row.objectives, row.genome.source_hash))
    selected.extend(infeasible[:max(0, capacity - len(selected))])
    return tuple(selected)


def update_memory(previous, survivors, *, capacity=16):
    rows = _same_context_unique(tuple(previous) + tuple(survivors))
    feasible = tuple(row for row in rows if row.shedding == 0)
    if not feasible:
        return ()
    return select_survivors(next(_fronts(feasible)), capacity)


class ReservationDenied(RuntimeError):
    pass


class ContextEvaluator:
    """Cache policy evaluations by genome and evaluation context.

    ``reserve(unit, worst_case_candidates)`` checks the budget before evaluation.
    Returning False raises ReservationDenied and stops the unit.
    """
    def __init__(self, evaluate: Callable, reserve: Callable):
        if not callable(evaluate) or not callable(reserve):
            raise ValueError("explicit evaluator and whole-unit reservation callbacks required")
        self._evaluate, self._reserve = evaluate, reserve
        self._cache = {}
        self.evaluation_count = 0

    def reserve(self, unit, count):
        if type(unit) is not str or not unit or type(count) is not int or count <= 0:
            raise ValueError("explicit unit and positive whole-candidate reservation required")
        token = self._reserve(unit, count)
        if token is False:
            raise ReservationDenied(unit)
        return token

    def evaluate(self, genomes, context_id, *, unit, reserve_candidates):
        genomes = tuple(genomes)
        if type(context_id) is not str or not context_id:
            raise ValueError("immutable full-context identity required")
        if type(reserve_candidates) is not int or reserve_candidates < len({g.source_hash for g in genomes}):
            raise ValueError("reservation must cover every unique candidate in complete unit")
        self.reserve(unit, reserve_candidates)
        return self._evaluate_reserved(genomes, context_id)

    def _evaluate_reserved(self, genomes, context_id):
        rows = []
        for genome in genomes:
            key = context_id, genome.source_hash
            if key not in self._cache:
                objectives, shedding = self._evaluate(genome, context_id)
                row = ScoredGenome(genome, context_id, tuple(objectives), shedding)
                self._cache[key] = row
                self.evaluation_count += 1
            rows.append(self._cache[key])
        return tuple(rows)


@dataclass(frozen=True)
class CEConfig:
    population: int = 16
    sample_population: int = 32
    elite_count: int = 16
    stages: int = 4
    smoothing: float = .7
    initial_std: float = .5
    variance_floor: float = 1e-6

    def __post_init__(self):
        # These values are this frozen adaptation, never tuned by outcomes.
        if (self.population, self.sample_population, self.elite_count, self.stages,
            self.smoothing, self.initial_std, self.variance_floor) != (16, 32, 16, 4, .7, .5, 1e-6):
            raise ValueError("Use the CE configuration specified for this baseline")

    @property
    def candidate_schedule(self):
        # Source Algorithm 1 starts at t=1 and iterates while t < T.
        return (self.population, self.sample_population) + (self.sample_population - self.elite_count,) * (self.stages - 1)


@dataclass(frozen=True)
class CEResult:
    elites: tuple
    memory: tuple
    final_sampled_unreturned: tuple
    evaluated_candidates: int


def run_ce(input_count, rng, evaluator: ContextEvaluator, context_id, *, dtype="float64", execution_dtype="f4", config=CEConfig()):
    """Initialize a cascade population with cross-entropy search.

    The initial unit reserves 96 candidate evaluations. The returned elites
    come from the third selection, before evaluation of the final 16 samples.
    """
    evaluator.reserve("ce-initial-complete-unit", sum(config.candidate_schedule))
    start_count = evaluator.evaluation_count
    width = parameter_count(input_count, 0)

    def sample(count, mean, std):
        vectors = rng.normal(mean, std, (count, width)).astype(dtype)
        genomes = tuple(CascadeGenome.from_vector(input_count, 0, v, execution_dtype=execution_dtype) for v in vectors)
        return evaluator._evaluate_reserved(genomes, context_id)

    cohort = sample(config.population, 0., config.initial_std)
    mean, std = population_moments([r.genome.parameter_vector() for r in cohort], variance_floor=config.variance_floor)
    population = sample(config.sample_population, mean, std)
    for _ in range(1, config.stages):
        elites = select_survivors(population, config.elite_count)
        if len(elites) != config.elite_count:
            raise ValueError("deduplication collapsed CE cohort; no silent population refill")
        mean, std = smooth_diagonal(mean, std, [r.genome.parameter_vector() for r in elites],
                                    alpha=config.smoothing, variance_floor=config.variance_floor)
        sampled = sample(config.sample_population - config.elite_count, mean, std)
        population = elites + sampled
    return CEResult(elites, update_memory((), elites), sampled, evaluator.evaluation_count - start_count)


@dataclass(frozen=True)
class CascadeGeneration:
    population: tuple
    memory: tuple
    evaluated_candidates: int
    reserved_candidates: int


def run_cascade_generation(parents, memory, rng, evaluator: ContextEvaluator, context_id, *,
                           hidden_node_cap=None, offspring_observer=None):
    """Whole common-context parent16/memory<=16/offspring16 unit, reserve48.

    Caller assigns the next cyclic training profile and seed offset; profiles
    are never chosen here using objective outcomes. Memory is the report output.
    """
    parents, memory = tuple(parents), tuple(memory)
    if len(parents) != 16 or len(memory) > 16:
        raise ValueError("frozen parent16/memory<=16 layout required")
    evaluator.reserve("cascade-complete-generation", 48)
    start_count = evaluator.evaluation_count
    def offspring_for(row):
        child = cascade_offspring(row.genome, rng, hidden_node_cap=hidden_node_cap)
        if offspring_observer is not None:
            offspring_observer(row.genome, child)
        return child
    offspring = tuple(offspring_for(row) for row in parents)
    all_genomes = tuple(r.genome for r in parents + memory) + offspring
    rows = evaluator._evaluate_reserved(all_genomes, context_id)
    parents_now, memory_now, offspring_now = rows[:16], rows[16:16 + len(memory)], rows[16 + len(memory):]
    population = select_survivors(parents_now + offspring_now, 16)
    if len(population) != 16:
        raise ValueError("deduplication collapsed survivor population; no silent refill")
    return CascadeGeneration(population, update_memory(memory_now, population), evaluator.evaluation_count - start_count, 48)
