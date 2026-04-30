import os
import re

import numpy as np
import pandas as pd

import graph_state.graph_state as gs


def _calculate_metrics_from_diags(diags_array: np.ndarray, true_diags: np.ndarray):
    """
    Calculates various metrics from diagonal measurements for each repeat.
    
    Args:
        diags_array: A 2D array of shape (num_repeats, num_diagonals).
        true_diags: A 1D array of the true diagonal values.
        input_fidelity: The fidelity value used as input for this experiment.

    Returns:
        A tuple containing three 1D arrays (one value per repeat):
        - delta_a_norm_2: The L2 norm of the difference vector.
        - estimated_fidelities: The calculated fidelity for each repeat.
        - delta_fidelities: The difference between input and estimated fidelity.
    """

    original_true_diags = np.array(true_diags)
    true_diags = true_diags[np.newaxis, :]

    # L2 norm of the difference vector (delta_a) for each repeat
    delta_a_norm_2 = np.linalg.norm(diags_array - true_diags, axis=1)
    delta_a_norm_1 = np.linalg.norm(diags_array - true_diags, ord=1, axis=1)
    
    estimated_fidelities = diags_array[:, 0]
    delta_fidelities = np.abs(original_true_diags[0] - estimated_fidelities)
    
    return delta_a_norm_2, delta_fidelities, delta_a_norm_1

def load_multiruns_diagonal_data(output_dir: str, file_prefix: str, df_prefix = None):
    """Loads all .npy files, calculates metrics, and returns pandas DataFrames."""
    all_data = []
    
    file_pattern = re.compile(
        rf"{file_prefix}_(\d+)q_F([\d.]+)_err_(.*?)_shots_(\d+).npy"
    )

    for filename in os.listdir(output_dir):
        match = file_pattern.match(filename)
        if match:
            qubits, F_str, err, shots = match.groups()
            qubits = int(qubits)
            input_F = float(F_str)
            
            filepath = os.path.join(output_dir, filename)
            diags_data = np.load(filepath)
            true_diags = gs.get_true_diagonals(qubits, input_F, err)
            
            # Calculate all metrics for the loaded data
            delta_norms_2, delta_fidelities, delta_norms_1 = _calculate_metrics_from_diags(diags_data, true_diags)

            # testing the second element
            estimated_second_element = diags_data[:, 1]
            
            # Delta b=1
            delta_b_1 = np.abs(np.array([true_diags[1] for _ in estimated_second_element]) - estimated_second_element)
            
            # Append one record per repeat
            for i in range(len(delta_norms_2)):
                all_data.append({
                    "prefix": file_prefix if df_prefix is None else df_prefix,
                    "qubits": int(qubits),
                    "input_fidelity": input_F,
                    "error_model": err,
                    "total_shots": int(shots),
                    "repeats": i,
                    "diag_sanity": np.sum(diags_data[i]), # sums the vector; it should sum to 1
                    "vector_a_norm_1": np.linalg.norm(diags_data[i], ord=1),
                    "vector_a_norm_2": np.linalg.norm(diags_data[i]),
                    "delta_a_norm_1": delta_norms_1[i],
                    "delta_a_norm_2": delta_norms_2[i],
                    "est_fidelity": diags_data[i][0],
                    "delta_fidelity": delta_fidelities[i],
                    "delta_b=1": delta_b_1[i],
                })
                
    if not all_data:
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    
    return df

def load_scalability_data(output_dir: str, file_prefix: str, df_prefix: str = None):
    """Loads all .npy files, calculates metrics, and returns pandas DataFrames."""
    all_data = []
    
    file_pattern = re.compile(
        rf"{file_prefix}_(\d+)q_F([\d.]+)_err_(.*?)_shots_(\d+)_repeat_(\d+).npy"
    )

    for filename in os.listdir(output_dir):
        match = file_pattern.match(filename)
        if match:
            qubits, F_str, err, shots, repeat_idx = match.groups()
            qubits = int(qubits)
            input_F = float(F_str)
            
            filepath = os.path.join(output_dir, filename)
            diags_data_single = np.array([np.load(filepath)])
            true_diags = gs.get_true_diagonals(qubits, input_F, err)
            
            # Calculate all metrics for the loaded data
            delta_norms_2, delta_fidelities, delta_norms_1 = _calculate_metrics_from_diags(diags_data_single, true_diags)
            # print(delta_norms_2, delta_fidelities, delta_norms_1)

            # testing the second element
            estimated_second_element = diags_data_single[:, 1]
            
            # Delta b=1
            delta_b_1 = np.abs(np.array([true_diags[1] for _ in estimated_second_element]) - estimated_second_element)
            
            # Append one record per repeat
            all_data.append({
                "prefix": file_prefix if df_prefix is None else df_prefix,
                "qubits": int(qubits),
                "input_fidelity": input_F,
                "error_model": err,
                "total_shots": int(shots),
                "repeats": repeat_idx,
                "diag_sanity": np.sum(diags_data_single), # sums the vector; it should sum to 1
                "vector_a_norm_1": np.linalg.norm(diags_data_single, ord=1),
                "vector_a_norm_2": np.linalg.norm(diags_data_single),
                "delta_a_norm_1": delta_norms_1[0],
                "delta_a_norm_2": delta_norms_2[0],
                "est_fidelity": diags_data_single[0],
                "delta_fidelity": delta_fidelities[0],
                "delta_b=1": delta_b_1[0],
            })
                
    if not all_data:
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    
    return df

def load_bsqn_fidelity_estimation_data(output_dir: str):
    """
    Loads all 'bell_fidelity' data, calculates metrics, and returns a DataFrame.
    """
    all_data = []
    file_pattern = re.compile(
        r"bell_fidelity_(\d+)q_F([\d.]+)_err_(.*?)_shots_(\d+)_numstab_(\d+).npy"
    )
    
    if not os.path.isdir(output_dir):
        print(f"Warning: Fidelity data directory '{output_dir}' not found. Returning empty DataFrame.")
        return pd.DataFrame()

    files = (
        os.path.join(r, f)
        for r, _, filelist in os.walk(output_dir)
        for f in filelist
    )

    for filepath in files:
        match = file_pattern.match(os.path.basename(filepath))
        if match:
            qubits, F_str, err, shots, numstab = match.groups()
            qubits = int(qubits)
            input_F = float(F_str)
            shots = int(shots)
            numstab = int(numstab)

            fidelities_array = np.load(filepath)
            
            for i, est_fidelity in enumerate(fidelities_array):
                all_data.append({
                    "qubits": qubits,
                    "input_fidelity": input_F,
                    "error_model": err,
                    "shots": shots,
                    "numstab": numstab,
                    "repeat": i,
                    "estimated_fidelity": est_fidelity,
                    "fidelity_error": np.abs(est_fidelity - input_F) # (est - true)
                })
    return pd.DataFrame(all_data)