"""Experiments that incrementally change the number of samples N."""

import multiprocessing as mp
import os
import random
from functools import lru_cache, partial
from itertools import product
from typing import Iterable, List, Generator

import numpy as np
import stim
import tqdm

import graph_state.graph_state as gs

INCREMENTAL_STEP_DEFAULT = 5
INITIAL_SAMPLE_SIZE_DEFAULT = 1_000


def generate_bsqn_samples(
    g: gs.GraphState,
    error_model: str,
    fidelity: float,
    seed: int = None,
) -> Generator[np.ndarray, int, None]:
    """Generate a number of BSQN samples incrementally."""
    n_samples = yield  # initial amount of samples

    if not error_model == "depolarizing":
        circuit = (
            g.get_graph_state_circuit(0)
            + g.get_graph_state_circuit(g.n)
            + g.get_noise_circuit(fidelity, error_model, 0)
            + g.get_noise_circuit(fidelity, error_model, g.n)
            + g.get_bell_sampling_circuit()
        )

        sampler = circuit.compile_sampler(seed=seed)
        while True:
            n_samples = yield sampler.sample(n_samples, bit_packed=True)

    # Polarizing circuit case
    base_circ = g.get_graph_state_circuit(0) + g.get_graph_state_circuit(g.n)
    bell_circ = g.get_bell_sampling_circuit()

    circuits = [
        base_circ + noise_model_circ + bell_circ
        for noise_model_circ in [
            stim.Circuit(),
            g.get_noise_circuit(fidelity, "fully-dephased", 0),
            g.get_noise_circuit(fidelity, "fully-dephased", g.n),
            g.get_noise_circuit(fidelity, "fully-dephased", 0)
                + g.get_noise_circuit(fidelity, "fully-dephased", g.n),
        ]
    ]

    samplers = [c.compile_sampler(seed=seed) for c in circuits]

    N = 2 ** g.n
    p = fidelity - (1 - fidelity) / (N - 1)
    random.seed(seed)

    while True:
        circuit_indices = random.choices(
            [0, 1, 2, 3],
            weights=[p**2, p * (1 - p), p * (1 - p), (1 - p)**2],
            k=n_samples,
        )
        n_samples = yield np.array([
            samplers[i].sample(1, bit_packed=True)[0]
            for i in circuit_indices
        ])


@lru_cache(maxsize=32)
def _generate_or_get_stabilizers(g: gs.GraphState):
    """
    Get or generate all stabilizers for a given graph.
    
    We must generate 2^n stabilizers for each experiment, so it is less
    costly (in time) to generate all of them at once and save it in a cache.
    A graph state is hashable by its underlying graph.
    """
    return list(g.generate_all_int_stabilizers())


def _run_single_instance_scalability_experiment(
    combination: tuple[gs.GraphState, str, float, float, int],
    norm_order: int,
    output_dir: str,
    overwrite: bool,
    incremental_step: int,
    initial_sample_size: int,
) -> np.ndarray:
    g, error_type, fidelity, epsilon, repeat = combination

    # Construct filename and check for overwrite
    filename = (
        f"bell_incremental_{g.n}q_F{fidelity:.3f}_err_{error_type}_"
        f"epsilon{epsilon:.3f}_repeat{repeat}.npy"
    )
    filepath = os.path.join(output_dir, filename)
    if not overwrite and os.path.exists(filepath):
        return np.load(filepath)

    # Get stabilizers (hopefully cached)
    stabilizers = _generate_or_get_stabilizers(g)

    # To avoid making the Walsh-Hadamard transform in every iteration, we optimize
    # with respect to ||dw|| <= sqrt(2^n) epsilon
    a_true = gs.get_true_diagonals(g.n, fidelity, error_type)
    w_true = gs.fwht(a_true)
    epsilon_w = (2 ** (g.n / 2)) * epsilon

    def estimate_w(samples: np.ndarray) -> float:
        """Helper function to be used on iterative process."""
        expectations = np.array(
            [1.0] +    # Expectation always 1 for identity
            [
                gs.expectation_value_of_observables_from_bell_bitpacked(
                    np.packbits(s, bitorder="little"), samples
                )
                for s in stabilizers
            ]
        )
        return np.sqrt(np.maximum(0, expectations))

    # We start with a considerable big amount of samples...
    sample_generator = generate_bsqn_samples(
        g, error_type, fidelity, seed=1_000 * g.n + repeat
    )
    sample_generator.send(None)
    samples = sample_generator.send(initial_sample_size)
    w_estimate = estimate_w(samples)
    delta_w_norm = np.linalg.norm(w_estimate - w_true, ord=norm_order)

    # If the norm is smaller than epsilon, we decrease it...
    while delta_w_norm < epsilon_w:
        samples = samples[:-incremental_step, :]
        w_estimate = estimate_w(samples)
        delta_w_norm = np.linalg.norm(w_estimate - w_true, ord=norm_order)

    # If the norm is bigger than epsilon, we increase it...
    while delta_w_norm > epsilon_w:
        samples = np.concatenate((samples, sample_generator.send(incremental_step)), axis=0)
        w_estimate = estimate_w(samples)
        delta_w_norm = np.linalg.norm(w_estimate - w_true, ord=norm_order)

    samples_total = len(samples) * 2
    a_estimate = gs.get_diagonals_from_all_stabilizer_observables(g, w_estimate[1:])
    delta_a_norm = np.linalg.norm(a_estimate - a_true, ord=norm_order)
    result = np.array([
        g.n, error_type, fidelity, epsilon, norm_order, delta_a_norm, repeat, samples_total
    ])

    np.save(filepath, result)
    return result


def bell_diagonal_incremental_scalability_experiment(
    graphs: gs.GraphState | List[gs.GraphState],
    err_model: str | List[str],
    fidelity: float | Iterable[float],
    epsilon: float | Iterable[float],
    num_repeats: int,
    norm_order: int,
    output_dir: str,
    overwrite: bool,
    incremental_step: int = INCREMENTAL_STEP_DEFAULT,
    initial_sample_size: int = INITIAL_SAMPLE_SIZE_DEFAULT,
):
    # Normalize all inputs to be lists
    graphs_list = [graphs] if isinstance(graphs, gs.GraphState) else list(graphs)
    err_models = [err_model] if isinstance(err_model, str) else list(err_model)
    fidelities = [fidelity] if isinstance(fidelity, (float, int)) else list(fidelity)
    epsilons = [epsilon] if isinstance(epsilon, (float, int)) else list(epsilon)
    repeats = list(range(1, num_repeats + 1))

    # Create all combinations of the parameters
    combinations = list(product(graphs_list, err_models, fidelities, epsilons, repeats))

    # Create the directory for saving results
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment data to: '{output_dir}/'")

    print(
        f"Starting Bell incremental sampling experiment for "
        f"{len(graphs_list)} graph(s), {len(err_models)} error model(s), "
        f"{len(fidelities)} fidelity value(s), {len(epsilons)} epsilons, "
        f"{num_repeats} trials, for the {norm_order}-norm of ||da||.\n"
        f"Total combinations (jobs) to run: {len(combinations)}."
    )

    # Set up the partial function for the worker
    worker_partial = partial(
        _run_single_instance_scalability_experiment,
        norm_order=norm_order,
        output_dir=output_dir,
        overwrite=overwrite,
        incremental_step=incremental_step,
        initial_sample_size=initial_sample_size,
    )

    # Set up and run the multiprocessing pool
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores free
    print(f"Running on {num_processes} processes...")

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm.tqdm(
            pool.imap_unordered(worker_partial, combinations),
            total=len(combinations),
            desc="Running Experiments",
        ):
            results += [result]

    return np.array(results)
