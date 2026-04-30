"""Module with functions not used anywhere in the experiments."""

from functools import partial

import multiprocess as mp
import numpy as np
import stim

from graph_state.graph_state import (
    GraphState, expectation_value_of_observables_from_bell_bitpacked
)

def _worker_calculate_exp_value_packed(packed_s_item, meas_samples_fixed):
    """
    Worker function for multiprocessing. Calls the original expectation value function.
    'packed_s_item' is one row from the Packed_S_matrix.
    'meas_samples_fixed' is the constant meas_samples array.
    """
    return expectation_value_of_observables_from_bell_bitpacked(packed_s_item, meas_samples_fixed)


def expectation_value_of_observables_from_bell_bitpacked_parallelized(g: GraphState, bell_samples):
    stabilizers_unpacked_list = list(g.generate_all_int_stabilizers())
    S_matrix = np.array(stabilizers_unpacked_list)
    Packed_S_matrix = np.packbits(S_matrix, axis=1, bitorder='little')
    task_function = partial(_worker_calculate_exp_value_packed, meas_samples_fixed=bell_samples)
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores for other tasks
    packed_s_tasks = [row for row in Packed_S_matrix]
    with mp.Pool(processes=num_processes) as pool:
        exps_list_results = pool.map(task_function, packed_s_tasks)
        exps = np.array(exps_list_results)
    return exps


def fidelity_estimation_via_random_sampling_parallelized(g: GraphState, num_obs: int, bell_samples):
    ps = np.packbits(g.sample_int_stabilizers(num_obs), axis=1, bitorder='little')
    task_function = partial(_worker_calculate_exp_value_packed, meas_samples_fixed=bell_samples)
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores for other tasks
    packed_s_tasks = [row for row in ps]
    with mp.Pool(processes=num_processes) as pool:
        exps_list_results = pool.map(task_function, packed_s_tasks)
        exps = np.array(exps_list_results)
    return np.mean(np.sqrt(np.maximum(0, exps)))


def expectation_value_of_observables_bitpacked(int_paulis: np.ndarray, measurement_results: np.ndarray):
    # tables for eigenvalues
    #      | Phi+ | Phi- | Psi+ | Psi- |
    #  XX  |   1  |  -1  |   1  |  -1  |  check Z
    #  YY  |  -1  |   1  |   1  |  -1  |  xor 00
    #  ZZ  |   1  |   1  |  -1  |  -1  |  check X
    gn = lambda x: np.bitwise_count(np.bitwise_xor.reduce(x & int_paulis, axis=1)) % 2
    n = len(measurement_results)
    return max((n - 2.0 * np.sum(gn(measurement_results))) / n, 0)