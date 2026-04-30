"""Module that contains the graph state object and BSQN functions."""

from functools import cache, partial

import igraph as ig
import multiprocess as mp
import numpy as np
import stim

from graph_state.array_helper import (
    probabilistic_select_rows,
    constructive_disjoint_complete_graph_stabilizer_grouping,
)


def _validate_fidelity(fidelity: float):
    if fidelity < 0 or fidelity > 1:
        raise ValueError(f"Fidelity must be between 0 and 1 (given {fidelity})")
    return


def _validate_error_model(error_model: str):
    if error_model not in [
        "no-error",
        "depolarizing",
        "single-qubit-dephasing",
        "bimodal",
    ]:
        raise ValueError(f"unknown error_model (given {error_model})")
    return


class GraphState:

    def __init__(self, n: int, graph_type="complete", edges: list[tuple[int, int]] = None):
        """unless graph type is specified as 'manual' the edges parameters are ignored
        all the graphs are guaranteed to be single connected component with the exception of manual graph_type
        """
        if graph_type not in [
            "manual",
            "complete",
            "random-tree",
            "star",
            "path",
            "ring",
        ]:
            raise ValueError(f"unknown graph_type (given {graph_type})")

        if graph_type == "manual" and edges is None:
            raise ValueError("edge list must be given for manual graph type")

        # attribute template
        self.n = n
        self.type = graph_type
        self.graph = ig.Graph()
        self.adj_mat = np.array([[]])
        self.stab_generators: list[str] = []
        self.int_stab_generators = np.array([])

        # attribute assignment
        if graph_type == "complete":
            self.graph = ig.Graph.Full(n)
        elif graph_type == "random-tree":
            self.graph = ig.Graph.Tree_Game(n)
        elif graph_type == "star":
            self.graph = ig.Graph.Star(n)
        elif graph_type == "path":
            self.graph = ig.Graph.Ring(n, circular=False)
        elif graph_type == "ring":
            self.graph = ig.Graph.Ring(n, circular=True)
        else:  # manual
            # TODO: add some checks
            self.graph = ig.Graph(n=n, edges=edges)

        self.adj_mat = self.graph.get_adjacency()
        self._populate_generators()
        self._powers_of_2_for_selection = 2 ** np.arange(self.n)

    def _populate_generators(self):
        """enumerate all stabilizer generators as 
            1. strings and store them to .stab_generators.
            2. int and store them to .int_stab_generators (1 for Z, 2 for X, and 0 for I).
        """
        for u, vs in enumerate(self.adj_mat):
            paulis = ["Z" if v == 1 else "_" for v in vs]
            paulis[u] = "X"
            self.stab_generators.append("".join(paulis))

        # populate the integer version for easy sampling
        self.int_stab_generators = np.array(self.graph.get_adjacency())
        for u in range(self.n):
            self.int_stab_generators[u][u] = 2

    def generate_all_int_stabilizers(self):
        """
        Generates all non-identity stabilizer elements from the base generators.

        The return format for each stabilizer is a Boolean numpy array of
        shape (3 * n_qubits), where the first n_qubits elements denote
        Pauli X locations, the next n_qubits for Z locations, and the
        final n_qubits for Y locations.

        Assumes Pauli encoding: I=0, Z=1, X=2, Y=3.
        """
        for i in range(1, 1 << self.n):
            # The j-th generator is selected if the j-th bit of 'i' is set.
            selection_mask = (i & self._powers_of_2_for_selection) != 0
            selected_generators = self.int_stab_generators[selection_mask]

            # Compute the Pauli product for this combination.
            # This is a column-wise bitwise XOR sum of the selected generators.
            # The result, 'pauli_product_vector', is a 1D array of length self.n_qubits,
            # where each element is the integer representation of the resulting
            # Pauli operator on that qubit (e.g., X, Y, Z, or I).
            pauli_product_vector = np.bitwise_xor.reduce(selected_generators, axis=0)

            # Based on the integer encoding (e.g., X=2, Z=1, Y=3):
            # Create boolean arrays indicating the presence of X, Z, or Y Paulis
            # at each qubit position. Each is of shape (self.n_qubits,).
            is_X = pauli_product_vector == 2
            is_Z = pauli_product_vector == 1
            is_Y = pauli_product_vector == 3

            # Concatenate these boolean arrays to form the specified extended format:
            # [X_q0, X_q1,..., X_qn-1, Z_q0,..., Z_qn-1, Y_q0,..., Y_qn-1]
            yield np.concatenate((is_X, is_Z, is_Y))

    def sample_int_stabilizers(self, shots: int, seed: int = None):
        """
        xx = int_paulis == 2
        zz = int_paulis == 1
        yy = int_paulis == 3
        """
        rng = np.random.default_rng(seed=seed)
        stabilizers = rng.random((shots, self.n)) > 0.5
        return np.array(
            [
                [
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 2,
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 1,
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 3,
                ]
                for si in stabilizers
            ]
        ).reshape(shots, self.n * 3)

    def get_graph_state_circuit(self, offset: int) -> stim.Circuit:
        """return stim.Circuit representing the graph state initialization with qubit index (offset, offset + graph.n)"""
        circuit = stim.Circuit(f"""H {' '.join(map(str, range(offset, offset + self.n)))}""")
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if self.adj_mat[u][v]:
                    circuit.append("CZ", [offset + u, offset + v])

        return circuit

    def get_noise_circuit(self, fidelity: float, error_model: str, offset: int) -> stim.Circuit:
        if error_model not in [
            "no-error",
            "single-qubit-dephasing",
            "bimodal",
            "fully-dephased",
        ]:
            raise ValueError(f"unknown error_model (given {error_model})")
        if fidelity < 0 or fidelity > 1:
            raise ValueError(f"Fidelity must be between 0 and 1 (given {fidelity})")
        circuit = stim.Circuit()

        if fidelity == 1 or error_model == "no-error":
            return circuit

        if error_model == "single-qubit-dephasing":
            p = 1 - (fidelity ** (1.0 / self.n))
            return stim.Circuit(f'Z_ERROR({p}) {" ".join(map(str, range(offset, offset + self.n)))}')

        if error_model == 'fully-dephased':
            return stim.Circuit(f'Z_ERROR({0.5}) {" ".join(map(str, range(offset, offset + self.n)))}')

        if error_model == "bimodal":
            p = 1 - fidelity
            return stim.Circuit(f"Z_ERROR({p}) {offset}")

        raise RuntimeError("reaching end of noise circuit generation without getting valid noise circuit.")

    def get_bell_sampling_circuit(self) -> stim.Circuit:
        sampling_circuit = stim.Circuit(
            f"CX {' '.join([f'{i} {self.n + i}' for i in range(self.n)])}\n"
            f"H {' '.join(map(str, range(self.n)))}\n"
            # these CX's and X are for measuring YY
            f"CX {' '.join([f'{i} {2 * self.n + i}' for i in range(self.n)])}\n"
            f"CX {' '.join([f'{self.n + i} {2 * self.n + i}' for i in range(self.n)])}\n"
            f"X {' '.join(map(str,range(2 * self.n, 3 * self.n)))}\n"
            f"MZ {' '.join(map(str, range(3 * self.n)))}"
        )
        return sampling_circuit

    def get_partial_tomo_measurement_circuit(self) -> stim.Circuit:
        stim_pauli_prods = []
        for st in self.stab_generators:
            words = [f'{p}{i}' for i, p in enumerate(st) if p != '_']
            words = '*'.join(words)
            stim_pauli_prods.append(words)

        return stim.Circuit(f'MPP {' '.join(stim_pauli_prods)}')


@cache
def get_true_diagonals(num_qubits: int, fidelity: float, error_model: str):
    """Returns the true diagonal vector in the graph-state basis.
    
    Recall that the vector is independent to the underlying graph and only depends on the noise model.
    """
    n = num_qubits
    N = 2 ** n
    us = np.array([0.0] * N, dtype=np.float64)
    if error_model == "no-error":
        us[0] = 1
        return us
    
    if error_model == "depolarizing":
        us[0] = fidelity
        us[1:] = (1 - fidelity) / (N - 1)
        return us
    
    if error_model == "single-qubit-dephasing":
        p = 1 - (fidelity ** (1.0 / n))
        us = np.array(
            [
                p ** i.bit_count() * (1 - p) ** (n - i.bit_count())
                for i in range(N)
            ]
        )
        return us
    
    if error_model == "bimodal":
        us[0] = fidelity
        us[1] = 1 - fidelity
        return us

    raise ValueError(f"unknown error_model (given {error_model})")


def bell_sampling(g: GraphState, error_model: str, fidelity: float, shots: int, seed: int = None):
    """steps to perform Bell samping.
        1. create the circuit of graph state.
        2. add noise according to the given noise model and fidelity
        3. generate samples and return.
    """
    # TODO: write the output format of the samples
    _validate_error_model(error_model)
    _validate_fidelity(fidelity)
    
    if error_model != "depolarizing":
        circuit = g.get_graph_state_circuit(0) + g.get_graph_state_circuit(g.n)
        circuit += g.get_noise_circuit(fidelity, error_model, 0) + g.get_noise_circuit(fidelity, error_model, g.n)
        circuit += g.get_bell_sampling_circuit()
        # return circuit
        return circuit.compile_sampler(seed=seed).sample(shots, bit_packed=True)
    
    """error model must be depolarizing, we need to build 4 circuits:
    1. no error/no error
    2. no error/fully dephased
    3. fully dephased/no error
    4. fully dephased/fully dephased
    and sample this based on the given fidelity
    """
    base_circ = g.get_graph_state_circuit(0) + g.get_graph_state_circuit(g.n)
    bell_circ = g.get_bell_sampling_circuit()

    circ_1 = base_circ + bell_circ
    circ_2 = base_circ + g.get_noise_circuit(fidelity, 'fully-dephased', 0) + bell_circ
    circ_3 = base_circ + g.get_noise_circuit(fidelity, 'fully-dephased', g.n) + bell_circ
    circ_4 = base_circ + g.get_noise_circuit(fidelity, 'fully-dephased', 0) + g.get_noise_circuit(fidelity, 'fully-dephased', g.n) + bell_circ

    samples_1 = circ_1.compile_sampler(seed=seed).sample(shots, bit_packed=True)
    samples_2 = circ_2.compile_sampler(seed=seed).sample(shots, bit_packed=True)
    samples_3 = circ_3.compile_sampler(seed=seed).sample(shots, bit_packed=True)
    samples_4 = circ_4.compile_sampler(seed=seed).sample(shots, bit_packed=True)

    N = 2 ** g.n
    p = fidelity - (1 - fidelity) / (N - 1)

    samples = probabilistic_select_rows([samples_1, samples_2, samples_3, samples_4], [p**2, p * (1 - p), p * (1 - p), (1 - p)**2], seed=seed)
    return samples


def expectation_value_of_observables_from_bell_bitpacked(
    int_paulis: np.ndarray,
    measurement_results: np.ndarray,
) -> float:
    """
    Performs expectation values calculation of a given observable defined by Pauli string
    into custom binary format (but bitpacked into integers),
    where the format is constructed by expanding length n Pauli string into length 3n.
    The 3n bits are separated into 3 blocks of length n each.
    The 1s in the first block indicates whether or not the Pauli string has X on that position.
    Similarly for 2nd and 3rd block for Z and Y.

    For example, _X_YZ would translate to
    01000 00001 00010 -> bitpack into 01000000 0100010_ = 128 66
    (I could get the endian wrong)

    The expected measurement results are to be bitpacked and each entry has length 3n,
    where each block of n represents XX, ZZ, and YY parities (recall BSM).
    The YY is added by xoring the results of first two blocks.

    The calculation goes as follows.
    1. For each measurement result, perform bitwise and (&) with the Pauli string.
       (this tells the sign contribution from each bit)
    2. We see if the total is an odd or even parity (total sign of + or -)
    3. We then sum all of them and divided by the total number of measurements

    A longer explanation of this magical piece of code:
        - `measurement_results` is a list of outcomes for each shot in the simulation;
        - `x & int_paulis` computes what outcomes will contribute to a -1 in the total
            outcome of the Pauli product;
        - `np.bitwise_xor.reduce()` is the first step in the reduction process. It xors
            each element in a measurement result array. Then, `np.bitwise_count() % 2`
            finishes things off by xor'ing the rest of the values and concluding if the
            final result is positive or negative;
        - All negative results will reduce the value of the expectation value. With
            this in mind, the return line makes sense: start with the biggest possible
            value and decrease it based on all contributions from negative outcomes.
    """
    # tables for eigenvalues
    #      | Phi+ | Phi- | Psi+ | Psi- |
    #  XX  |   1  |  -1  |   1  |  -1  |  check Z
    #  YY  |  -1  |   1  |   1  |  -1  |  xor 00
    #  ZZ  |   1  |   1  |  -1  |  -1  |  check X
    gn = lambda x: np.bitwise_count(np.bitwise_xor.reduce(x & int_paulis, axis=1)) % 2
    n = len(measurement_results)
    return (n - 2.0 * np.sum(gn(measurement_results))) / n


def fidelity_estimation_via_random_sampling_bitpacked(g: GraphState, num_stabilizers: int, bell_samples):
    # bitpacked in little endian, the same way as stim does
    fn = lambda x: np.sqrt(np.maximum(expectation_value_of_observables_from_bell_bitpacked(x, bell_samples), 0))
    stabilizers = np.packbits(g.sample_int_stabilizers(num_stabilizers), axis=1, bitorder='little')
    sum_sqrt_cp = np.sum([fn(p) for p in stabilizers])
    return sum_sqrt_cp / len(stabilizers)


def _post_process_dge_for_complete_graph_overlap(g: GraphState, samples, seed: int = None):
    """postprocess dge for complete graph state with non-overlapping stabilizer observables"""
    # since samples are bitpacked (little-endian) we count the chunks instead of the number of qubits
    shots, num_measurement_chunks = samples.shape # (M, C)
    N = 2**g.n
    num_groups = (
        N // 2 + 1
    )  # note here that we have N/2 (odd product of generators) + 1 (all even; i.e., all Ys)
    rng = np.random.default_rng(seed=seed)

    if g.n > 64:
        raise ValueError(f"n={g.n} is too large for this optimized function (max 64).")
    if num_groups > shots:
        raise ValueError(
            f"Not enough shots to split between all stabilizers (shots given: {shots}; total circuits to run {num_groups})"
        )

    all_indices = np.arange(N, dtype=np.uint64)  # array of length N
    all_bit_counts = np.bitwise_count(
        all_indices
    )  # precompute the parity of bitstrings
    all_numbers_bitpacked = np.array(
        [
            list(i.to_bytes(num_measurement_chunks, byteorder="little", signed=False))
            for i in range(N)
        ],
        dtype=np.uint8,
    )

    # Filter odd/even
    odd_mask = (all_bit_counts % 2) == 1
    odd_indices = all_indices[odd_mask]
    bitpacked_odd = all_numbers_bitpacked[odd_mask]

    even_mask = ((all_bit_counts % 2) == 0) & (all_indices != 0)  # we don't want 0
    even_indices = all_indices[even_mask]
    bitpacked_even = all_numbers_bitpacked[even_mask]

    # assign how many shots to each group
    sample_sizes = np.full(num_groups, shots // num_groups, dtype=int)
    left_over_shots = shots % num_groups
    lucky_stabilizers = rng.choice(
        num_groups, left_over_shots, replace=False
    )  # we don't add to the same group twice
    sample_sizes[lucky_stabilizers] += 1

    # 1. Split the samples into groups based on sample_sizes
    split_indices = np.cumsum(sample_sizes)[:-1]
    sample_groups = np.split(samples, split_indices, axis=0)
    odd_product_parities_list = []

    # 2. Process the first (num_groups - 1) groups against the odd stabilizers
    for i in range(num_groups - 1):
        group_samples = sample_groups[i]
        stabilizer = bitpacked_odd[i]

        # we perform bitwise_and to see post-process the generator into stabilizer measurement
        # i.g., if stabilizer we want is g1.g4.g5 = 10011 we bitwise_and with the measurement result
        # and compute the measurement result as 0/1 by looking at the even/odd parity
        bitwise_and_result = group_samples & stabilizer
        total_bit_counts_per_sample = np.bitwise_count(bitwise_and_result).sum(axis=1)
        parities = (total_bit_counts_per_sample % 2).astype(np.int8)

        odd_product_parities_list.append(parities)

    # 3. Process the last group against all *even* stabilizers
    last_group_samples = sample_groups[-1]

    # We want to compute (s & g) for all samples s in the last group
    # and all stabilizers g in the even list.
    s_broadcast = last_group_samples[:, np.newaxis, :]
    g_broadcast = bitpacked_even[np.newaxis, :, :]

    bitwise_and_result = s_broadcast & g_broadcast

    # Sum bit counts over the C dimension (chunks)
    total_bit_counts = np.bitwise_count(bitwise_and_result).sum(axis=2)  # Shape (M, K); M is number of shots; K is number of stabilizer elements

    # Compute the parities
    even_products_parities_group = (total_bit_counts % 2).astype(np.int8)
    # all_parities_list.append(even_products_parities_group)

    # 4. (a) Process odd stabilizer groups (first N//2)
    exp_vals_odd = np.array(
        [np.mean(parities * -2 + 1) for parities in odd_product_parities_list], dtype=float
    )
    #    (b) Process even stabilizer group (last one)
    eigenvalues_last_group = even_products_parities_group * -2 + 1  # Shape (M, K)
    exp_vals_even = np.mean(eigenvalues_last_group, axis=0)  # Shape (K,)

    # 6. Create final ordered 1D array of size (N-1)
    # Use advanced indexing (scatter) to build the final array.
    final_expectation_values = np.full(N - 1, np.nan, dtype=float)

    # we need -1 because array is 0-indexed (for stabilizers 1 to N-1)
    final_expectation_values[odd_indices - 1] = exp_vals_odd
    final_expectation_values[even_indices - 1] = exp_vals_even

    return final_expectation_values


def _post_process_dge_for_complete_graph_non_overlap(g: GraphState, samples, seed: int = None):
    """
    New version of post-processing where even stabilizers are further
    grouped into bitwise-disjoint sets.
    """
    # since samples are bitpacked (little-endian) we count the chunks instead of the number of qubits
    shots, num_measurement_chunks = samples.shape # shape (M, C)
    N = 2**g.n

    if g.n > 64:
        raise ValueError(f"n={g.n} is too large for this optimized function (max 64).")

    all_indices = np.arange(N, dtype=np.uint64)  # array of length N
    all_bit_counts = np.bitwise_count(
        all_indices
    )  # precompute the parity of bitstrings
    all_numbers_bitpacked = np.array(
        [
            list(i.to_bytes(num_measurement_chunks, byteorder="little", signed=False))
            for i in range(N)
        ],
        dtype=np.uint8,
    )

    # Odd stabilizers
    odd_mask = (all_bit_counts % 2) == 1
    odd_indices = all_indices[odd_mask]
    bitpacked_odds = all_numbers_bitpacked[odd_mask]  # Shape (N//2, C)

    # Even stabilizers
    even_mask = ((all_bit_counts % 2) == 0) & (all_indices != 0)
    even_indices = all_indices[even_mask]
    bitpacked_evens = all_numbers_bitpacked[even_mask]  # Shape (K, C)

    # We group the stabilizer elements without any overlapped supports
    even_groups_indices = constructive_disjoint_complete_graph_stabilizer_grouping(even_indices, g.n)
    num_even_groups = len(even_groups_indices)
    # Create a lookup map for index -> bitpacked array
    bitpacked_even_map = {idx: arr for idx, arr in zip(even_indices, bitpacked_evens)}

    # Total number of groups is odd groups + new even groups
    num_odd_groups = N // 2
    num_groups = num_odd_groups + num_even_groups

    # validation
    if num_groups > shots:
        raise ValueError(
            f"Not enough shots to split between all stabilizers (shots given: {shots}; total circuits to run {num_groups})"
        )

    rng = np.random.default_rng(seed=seed)
    sample_sizes = np.full(num_groups, shots // num_groups, dtype=int)
    left_over_shots = shots % num_groups

    lucky_stabilizers = rng.choice(num_groups, left_over_shots, replace=False)
    sample_sizes[lucky_stabilizers] += 1

    # 1. Split the samples into groups
    split_indices = np.cumsum(sample_sizes)[:-1]
    sample_groups = np.split(samples, split_indices, axis=0)

    all_parities_list = []

    # 2. Process the odd stabilizer groups
    for i in range(num_odd_groups):
        group_samples = sample_groups[i]

        if group_samples.shape[0] == 0:
            all_parities_list.append(np.array([], dtype=np.int8))
            continue

        stabilizer = bitpacked_odds[i]
        bitwise_and_result = group_samples & stabilizer
        total_bit_counts_per_sample = np.bitwise_count(bitwise_and_result).sum(axis=1)
        parities = (total_bit_counts_per_sample % 2).astype(np.int8)
        all_parities_list.append(parities)

    # 3. Process the even stabilizer groups (non-overlapped)
    for k in range(num_even_groups):
        current_samples = sample_groups[
            num_odd_groups + k
        ]  # Get the k-th even sample group
        current_group_indices = even_groups_indices[k]

        # Build the (K_k, C) stabilizer array for this group
        current_stabilizers_bitpacked = np.array(
            [bitpacked_even_map[idx] for idx in current_group_indices]
        )

        # Broadcast samples (M_k, C) against stabilizers (K_k, C)
        s_broadcast = current_samples[:, np.newaxis, :]  # Shape (M_k, 1, C)
        g_broadcast = current_stabilizers_bitpacked[
            np.newaxis, :, :
        ]  # Shape (1, K_k, C)

        bitwise_and_result = s_broadcast & g_broadcast  # Shape (M_k, K_k, C)

        total_bit_counts = np.bitwise_count(bitwise_and_result).sum(
            axis=2
        )  # Shape (M_k, K_k)
        parities_group = (total_bit_counts % 2).astype(np.int8)

        all_parities_list.append(parities_group)

    final_expectation_values = np.full(N - 1, np.nan, dtype=float)

    # 4. (a) Process odd stabilizer groups
    exp_vals_odd = np.array(
        [np.mean(parities * -2 + 1) for parities in all_parities_list[:num_odd_groups]],
        dtype=float,
    )

    final_expectation_values[odd_indices - 1] = exp_vals_odd

    #    (b) Process even stabilizer groups (non-overlap)
    for k in range(num_even_groups):
        parities_k = all_parities_list[
            num_odd_groups + k
        ]  # Get k-th even parity result
        indices_k = even_groups_indices[k]

        eigenvalues_k = parities_k * -2 + 1  # Shape (M_k, K_k)
        exp_vals_k = np.mean(eigenvalues_k, axis=0)  # Shape (K_k,)
        # Scatter these values into the final array
        final_expectation_values[np.array(indices_k) - 1] = exp_vals_k

    return final_expectation_values


def dge_combined(
    g: GraphState,
    error_model: str,
    fidelity: float,
    shots: int,
    overlap_observables: bool,
    seed: int = None,
):
    """steps to perform DGE specific to ONLY complete-graph graph states
    (to get expectation values over all obsevables defined from stabilizer elements).
    1. create the circuit of graph state.
    2. add noise according to the given noise model and fidelity
    3. add stabilizer generator measurement parts (we get n measurements each run)
    4. transform these n measurements to expectation values over all observables defined from stabilizer elements.
    5. return results.
    """
    if g.type != "complete":
        raise ValueError("currently this version of DGE only supports complete graphs.")
    _validate_error_model(error_model)
    _validate_fidelity(fidelity)

    if error_model != "depolarizing":
        circuit = g.get_graph_state_circuit(0)
        circuit += g.get_noise_circuit(fidelity, error_model, 0)
        circuit += g.get_partial_tomo_measurement_circuit()
        samples = circuit.compile_sampler(seed=seed).sample(shots, bit_packed=True)
    else:
        base_circ = g.get_graph_state_circuit(0)
        meas_circ = g.get_partial_tomo_measurement_circuit()

        circ_1 = base_circ + meas_circ
        circ_2 = (
            base_circ + g.get_noise_circuit(fidelity, "fully-dephased", 0) + meas_circ
        )

        samples_1 = circ_1.compile_sampler(seed=seed).sample(shots, bit_packed=True)
        samples_2 = circ_2.compile_sampler(seed=seed).sample(shots, bit_packed=True)

        N = 2**g.n
        p = fidelity - (1 - fidelity) / (N - 1)

        rng = np.random.default_rng(seed=seed)
        mask = rng.random(shots) < p
        mask_reshaped = mask[:, np.newaxis]
        samples = np.where(mask_reshaped, samples_1, samples_2)

    if overlap_observables:
        return _post_process_dge_for_complete_graph_overlap(g, samples)
    else:
        return _post_process_dge_for_complete_graph_non_overlap(g, samples)


def fwht(x: np.array, inverse: bool = False):
    """
    Fast Walsh–Hadamard transform.
    
    Taken from SPyRiT's source code, which in turn is adapted from Amit
    Portnoy's hadamard-transform library.
    """
    original_shape = x.shape

    # create batch if x is 1D
    if len(original_shape) == 1:
        x = x.reshape(1, -1)  # shape (1, n)

    *batch, d = x.shape  # batch is tuple and d is int
    h = 2

    while h <= d:
        x = x.reshape(*batch, d // h, h)
        half1, half2 = np.split(x, 2, axis=-1)
        x = np.concatenate((half1 + half2, half1 - half2), axis=-1)
        h *= 2

    x = x.reshape(original_shape)

    if inverse:
        x = x / x.size

    return x


def get_diagonals_from_all_stabilizer_observables(g: GraphState, expvals):
    N_fwhm = 1 << g.n  # 2**n_qubits, size of the FWHT vector

    # Create the input vector for FWHT of size N_fwhm
    fwht_input = np.zeros(N_fwhm, dtype=float)

    # Get the integer indices corresponding to the stabilizers in 'exps'
    # These indices are assumed to be 1, 2, ..., N_fwhm-1 if exps covers all non-identity Paulis
    stabilizer_integer_indices = np.arange(1, 2**g.n)

    # Populate fwht_input:
    fwht_input[stabilizer_integer_indices] = expvals
    fwht_input[0] = 1.0 # expectation values of I^n is always 1

    # Calculate the Fast Walsh-Hadamard Transform
    # The result 'transformed_coeffs[i]' = sum_s (fwht_input[s] * (-1)**<i,s>)
    # where <i,s> is the bitwise dot product (popcount(i&s) % 2)
    transformed_coeffs = fwht(fwht_input)

    # Calculate the final diagonal values
    diagonals = transformed_coeffs / N_fwhm
    return diagonals
