import os
from functools import partial
from itertools import product
from typing import Iterable, List, Union

import multiprocess as mp
import tqdm
import numpy as np

import graph_state.graph_state as gs


def _print_statistics_for_parallelized_experiments(results):
    print("\n--- Experiment Run Summary ---")
    saved_count = sum(1 for r in results if r.startswith("Saved"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    failed_count = sum(1 for r in results if r.startswith("Failed"))
    print(f"Successfully saved: {saved_count}")
    print(f"Skipped (existed):  {skipped_count}")
    print(f"Failed:             {failed_count}")
    if failed_count > 0:
        print("\nFailures:")
        for r in results:
            if r.startswith("Failed"):
                print(f"  - {r}")


def _run_bell_sampling_worker(combination: tuple, num_repeats: int, output_dir: str, overwrite: bool) -> str:
    """
    A single-process worker function for bell_sampling_fidelity_experiment.
    'combination' is a tuple: (g: gs.GraphState, err: error model (str), fidelity: float, shots: int, stab_factor: string indicating how many stabilizer elements we want to pick)
    The process will run and save the data to the output_dir.

    Returns:
        A status string for logging.
    """
    g, err, fid, shots, stab_factor = combination

    if stab_factor == '2n':
        numstab = g.n * 2
    elif stab_factor == 'n^2':
        numstab = g.n ** 2
    elif stab_factor.isdigit():
        numstab = int(stab_factor)
    else:
        return f"Failed (param error): Stabilizer factor '{stab_factor}' not recognized"

    # Construct filename and check for overwrite
    filename = f"bell_fidelity_{g.n}q_F{fid:.3f}_err_{err}_shots_{shots}_numstab_{numstab}.npy"
    filepath = os.path.join(output_dir, filename)
    
    if not overwrite and os.path.exists(filepath):
        return f"Skipped (exists): {filename}"

    # Run the actual experiment
    try:
        # all_fidelities = [fidelity_estimation_via_random_sampling_bitpacked(g, numstab, samples)]
        samples = gs.bell_sampling(g, err, fid, shots * num_repeats, seed=shots * num_repeats)
        all_fidelities = [
            gs.fidelity_estimation_via_random_sampling_bitpacked(g, numstab, sample_split)
            for sample_split in np.split(samples, num_repeats)
        ]

        # for seed_i in range(num_repeats):
        #    samples = gs.bell_sampling(g, err, fid, shots, seed=seed_i)
        #    est_fidelity = gs.fidelity_estimation_via_random_sampling_bitpacked(g, numstab, samples)
        #    all_fidelities.append(est_fidelity)
            
        # Save the result
        np.save(filepath, np.array(all_fidelities))
        return f"Saved ({len(all_fidelities)} repeats): {filename}"
    except Exception as e:
        return f"Failed (runtime error): {filename} with error: {e}"

def bell_sampling_fidelity_experiment(
    graphs: Union[gs.GraphState, List[gs.GraphState]],
    err_model: Union[str, List[str]],
    fidelity: Union[float, Iterable[float]],
    num_shots: Union[int, Iterable[int]],
    num_repeats: int,
    stabilizer_factors: Union[str, List[str]],
    output_dir: str,
    overwrite: bool = False
):
    """
    Runs a Bell sampling fidelity experiment in parallel for all combinations of parameters. 
    Each combination is run in a separate process.
    Arguments can be passed in as either a single instance or as a list and will be expanded.
    """
    # Normalize all inputs to be lists
    graphs_list = [graphs] if isinstance(graphs, gs.GraphState) else list(graphs)
    err_models = [err_model] if isinstance(err_model, str) else list(err_model)
    fidelities = [fidelity] if isinstance(fidelity, (float, int)) else list(fidelity)
    shots_list = [num_shots] if isinstance(num_shots, int) else list(num_shots)
    stabilizer_factors_list = [stabilizer_factors] if isinstance(stabilizer_factors, str) else list(stabilizer_factors)

    # Create the directory for saving results
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment data to: '{output_dir}/'")

    # Create all combinations of the parameters
    combinations = list(product(graphs_list, err_models, fidelities, shots_list, stabilizer_factors_list))
    
    print(f"Starting Bell random sampling experiment for {len(graphs_list)} graph(s), "
          f"{len(err_models)} error model(s), {len(fidelities)} fidelity value(s), "
          f"{len(shots_list)} shot setting(s), and {len(stabilizer_factors_list)} stabilizer factor(s).")
    print(f"Total combinations (jobs) to run: {len(combinations)}")
    
    # Set up the partial function for the worker
    worker_partial = partial(
        _run_bell_sampling_worker,
        num_repeats=num_repeats,
        output_dir=output_dir,
        overwrite=overwrite
    )
    
    # Set up and run the multiprocessing pool
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores free
    print(f"Running on {num_processes} processes...")

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(worker_partial, combinations), total=len(combinations), desc="Running Experiments"):
            results.append(result)

    _print_statistics_for_parallelized_experiments(results)

def _run_tomo_worker(combination: tuple[gs.GraphState, str, float, int], overlap_observables: bool, num_repeats: int, output_dir: str, overwrite: bool) -> str:
    """
    A single-process worker function for partial_tomo_experiment.
    'combination' is a tuple: (err, fid, total_shots)
    
    Returns:
        A status string for logging.
    """
    try:
        g, err, fid, total_shots = combination

        filename = f"tomo_{g.n}q_F{fid:.3f}_err_{err}_shots_{total_shots}.npy"
        filepath = os.path.join(output_dir, filename)

        if not overwrite and os.path.exists(filepath):
            return f"Skipped (exists): {filename}"

        all_diags_for_combo = []
        for seed_i in range(num_repeats):
            exps = gs.dge_combined(g, err, fid, total_shots, overlap_observables=overlap_observables, seed=seed_i)
            exps = np.maximum(0, exps)
            diags = gs.get_diagonals_from_all_stabilizer_observables(g, exps)
            all_diags_for_combo.append(diags)
            
        # Stack the results into a single 2D NumPy array
        stacked_diags = np.array(all_diags_for_combo) # shape (num_repeats, number_of_diagonals)

        # 5. Save the result
        np.save(filepath, stacked_diags)
        return f"Saved ({stacked_diags.shape}): {filename}"
    
    except Exception as e:
        # Provide more context in the error message
        return f"Failed: {filename} with error: {e}"

def partial_tomo_experiment_parallelized(
    graphs: Union[gs.GraphState, List[gs.GraphState]],
    err_model: Union[str, List[str]],
    fidelity: Union[float, Iterable[float]],
    total_shots: Union[int, Iterable[int]],
    overlap_observables: bool,
    num_repeats: int,
    output_dir: str,
    overwrite: bool,
):
    """
    Runs a partial tomography experiment in parallel for all combinations
    of parameters. Each combination is run in a separate process.

    Args:
        g (gs.GraphState): The graph state object.
        err_model (Union[str, List[str]]): A single error model string or a list of them.
        fidelity (Union[float, Iterable[float]]): A single fidelity value or an iterable.
        total_shots (Union[int, Iterable[int]]): A single total_shot (count) value, 
                                                 or an iterable of values.
        num_repeats (int): The number of times to repeat the experiment
                           for each parameter combination.
        output_dir (str): The directory where the output .npy files will be saved.
        overwrite (bool): If False, skips the calculation if the output file
                          already exists. If True, it will always run and overwrite
                          any existing file.
    """
    # Normalize all inputs to be lists
    graphs = [graphs] if isinstance(graphs, gs.GraphState) else list(graphs)
    err_models = [err_model] if isinstance(err_model, str) else list(err_model)
    fidelities = [fidelity] if isinstance(fidelity, (float, int)) else list(fidelity)
    shots_list = [total_shots] if isinstance(total_shots, int) else list(total_shots)

    # Create the directory for saving results
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment data to: '{output_dir}/'")

    # Create all combinations of the parameters
    combinations = list(product(graphs, err_models, fidelities, shots_list))
    
    print(f"Starting tomography experiment for {len(graphs)} graphs ({len(err_models)} error model(s), "
          f"{len(fidelities)} fidelity value(s), and {len(shots_list)} shot setting(s).")
    print(f"Total combinations (jobs) to run: {len(combinations)}")
    
    # Set up the partial function for the worker
    worker_partial = partial(
        _run_tomo_worker,
        overlap_observables=overlap_observables,
        num_repeats=num_repeats,
        output_dir=output_dir,
        overwrite=overwrite
    )
    
    # Set up and run the multiprocessing pool
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores free
    print(f"Running on {num_processes} processes...")

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(worker_partial, combinations), total=len(combinations), desc="Running Experiments"):
            results.append(result)

    _print_statistics_for_parallelized_experiments(results)

def _run_bell_sampling_diagonal_worker(combination: tuple, g: gs.GraphState, num_repeats: int, output_dir: str, overwrite: bool) -> str:
    # Unpack the combination tuple
    err, fid, shots = combination

    # Construct filename and check for overwrite
    filename = f"bell_diag_{g.n}q_F{fid:.3f}_err_{err}_shots_{shots}.npy"
    filepath = os.path.join(output_dir, filename)

    if not overwrite and os.path.exists(filepath):
        return f"Skipped (exists): {filename}"

    actual_shots = (shots + 1) // 2
    bell_samples = gs.bell_sampling(g, err, fid, actual_shots * num_repeats, seed=num_repeats)
    samples_split = np.split(bell_samples, num_repeats)

    all_diags_for_combo = []

    for samples in samples_split:
        exps = np.array(
            [
                gs.expectation_value_of_observables_from_bell_bitpacked(
                    np.packbits(stab, bitorder="little"), samples
                )
                for stab in g.generate_all_int_stabilizers()
            ]
        )
        sqrt_exps_safe = np.sqrt(np.maximum(0, exps))
        diags = gs.get_diagonals_from_all_stabilizer_observables(g, sqrt_exps_safe)
        all_diags_for_combo.append(diags)

    stacked_diags = np.array(all_diags_for_combo) # Shape (num_repeats, number_of_diagonals)
    np.save(filepath, stacked_diags)
    return f"Saved ({stacked_diags.shape}): {filename}"

def bell_sampling_diagonal_experiment(
    g: gs.GraphState,
    err_model: Union[str, List[str]],
    fidelity: Union[float, Iterable[float]],
    num_shots: Union[int, Iterable[int]],
    num_repeats: int,
    output_dir: str,
    overwrite: bool,
):
    err_models = [err_model] if isinstance(err_model, str) else list(err_model)
    fidelities = [fidelity] if isinstance(fidelity, (float, int)) else list(fidelity)
    shots_list = [num_shots] if isinstance(num_shots, int) else list(num_shots)

    # Create the directory for saving results if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment data to: '{output_dir}/'")

    # Create all combinations of the parameters
    combinations = list(product(err_models, fidelities, shots_list))
    
    print(f"Starting Bell diagonal experiment for {g.n} qubits, {len(err_models)} error model(s), {len(fidelities)} fidelity value(s), and {len(shots_list)} shot setting(s).")
    print(f"Starting experiment for {len(combinations)} parameter combinations...")
    
    # Set up the partial function for the worker
    worker_partial = partial(
        _run_bell_sampling_diagonal_worker,
        num_repeats=num_repeats,
        g=g,
        output_dir=output_dir,
        overwrite=overwrite
    )
    
    # Set up and run the multiprocessing pool
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores free
    print(f"Running on {num_processes} processes...")

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(worker_partial, combinations), total=len(combinations), desc="Running Experiments"):
            results.append(result)

    _print_statistics_for_parallelized_experiments(results)


def _run_bell_sampling_diagonal_single_run_worker(combination: tuple[gs.GraphState, str, float, int, int], output_dir: str, overwrite: bool) -> str:
    g, err, fid, shots, repeat_idx = combination

    # Construct filename and check for overwrite
    filename = f"bell_scalability_{g.n}q_F{fid:.3f}_err_{err}_shots_{shots}_repeat_{repeat_idx}.npy"
    filepath = os.path.join(output_dir, filename)

    if not overwrite and os.path.exists(filepath):
        return f"Skipped (exists): {filename}"

    actual_shots = (shots + 1) // 2
    bell_samples = gs.bell_sampling(g, err, fid, actual_shots, seed=1000 * g.n + repeat_idx)

    exps = np.array(
        [
            gs.expectation_value_of_observables_from_bell_bitpacked(
                np.packbits(stab, bitorder="little"), bell_samples
            )
            for stab in g.generate_all_int_stabilizers()
        ]
    )
    sqrt_exps_safe = np.sqrt(np.maximum(0, exps))
    diags = gs.get_diagonals_from_all_stabilizer_observables(g, sqrt_exps_safe)

    np.save(filepath, diags)
    return f"Saved ({diags.shape}): {filename}"

def bell_diagonal_scalability_experiment(
    graphs: Union[gs.GraphState, List[gs.GraphState]],
    err_model: Union[str, List[str]],
    fidelity: Union[float, Iterable[float]],
    num_shots: Union[int, Iterable[int]],
    num_repeats: int,
    output_dir: str,
    overwrite: bool,
):
    # Normalize all inputs to be lists
    graphs_list = [graphs] if isinstance(graphs, gs.GraphState) else list(graphs)
    err_models = [err_model] if isinstance(err_model, str) else list(err_model)
    fidelities = [fidelity] if isinstance(fidelity, (float, int)) else list(fidelity)
    shots_list = [num_shots] if isinstance(num_shots, int) else list(num_shots)
    repeat_indices = [i for i in range(num_repeats)]

    # Create the directory for saving results
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment data to: '{output_dir}/'")

    # Create all combinations of the parameters
    combinations = list(product(graphs_list, err_models, fidelities, shots_list, repeat_indices))
    
    print(f"Starting Bell random sampling experiment for {len(graphs_list)} graph(s), "
          f"{len(err_models)} error model(s), {len(fidelities)} fidelity value(s), "
          f"{len(shots_list)} shot setting(s), and {num_repeats} trials.")
    print(f"Total combinations (jobs) to run: {len(combinations)}")
    
    # Set up the partial function for the worker
    worker_partial = partial(
        _run_bell_sampling_diagonal_single_run_worker,
        output_dir=output_dir,
        overwrite=overwrite
    )
    
    # Set up and run the multiprocessing pool
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores free
    print(f"Running on {num_processes} processes...")

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(worker_partial, combinations), total=len(combinations), desc="Running Experiments"):
            results.append(result)

    _print_statistics_for_parallelized_experiments(results)