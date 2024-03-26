"""Functions for solving linear assignment problems."""


import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import]
from numba import jit  # type: ignore[import]
import streamlit as st  # type: ignore[import]

import distance
from type_defs import AssignmentSolution, ArrU8


@jit(cache=True, nopython=True, parallel=True, fastmath=True, boundscheck=False, nogil=True)
def greedy_random(D: ArrU8) -> AssignmentSolution:
    """Solve an assignment problem using a greedy algorithm.

    Arguments:
    D: distance matrix. May be modified in-place by this function.
    """
    N = D.shape[0]
    sol_src = np.arange(N, dtype=np.int64)
    sol_tgt = np.zeros(N, dtype=np.int64)
    for i in np.random.permutation(np.arange(N)):
        sol_tgt[i] = np.argmin(D[i])
        D[:, sol_tgt[i]] = np.iinfo(np.uint8).max
    return sol_src, sol_tgt


def compute_assignment(X: ArrU8, Y: ArrU8, algorithm: str) -> AssignmentSolution:
    """Compute assignment using specified algorithm."""
    D = distance.compute_distance_matrix(X, Y)
    assignment_func_map = {
        "greedy_random": greedy_random,
        "jonker_volgenant": linear_sum_assignment,
    }
    assignment_func = assignment_func_map[algorithm]
    sol_src, sol_tgt = assignment_func(D)
    sol_src = sol_src.astype(np.int64)
    sol_tgt = sol_tgt.astype(np.int64)
    return sol_src, sol_tgt


def compute_assignment_batched(
    X: ArrU8, Y: ArrU8, X_batch_size: int, Y_batch_size: int, algorithm: str
) -> AssignmentSolution:
    """Compute assignment using randomized batches.

    Together X_batch_size and Y_batch_size determine the max distance matrix and assignment problem size,
    which is X_batch_size x Y_batch_size.
    """
    if X_batch_size > Y_batch_size:
        msg = f"X_batch_size ({X_batch_size}) > Y_batch_size ({Y_batch_size}), which is not allowed."
        raise ValueError(msg)
    N = X.shape[0]
    num_batches = int(np.ceil(N / X_batch_size))

    X_idxs = np.arange(X.shape[0])
    Y_idxs = np.arange(Y.shape[0])
    unused_Y_idxs = set(Y_idxs)
    X_rand_idxs = np.random.permutation(X_idxs)
    a = np.zeros(N, dtype=np.int64)
    a_start = 0
    progress_bar = st.progress(0.0, "Solving Assignment Batches")
    for i in range(num_batches):
        X_batch_start = i * X_batch_size
        X_batch_stop = (i + 1) * X_batch_size
        X_batch_idxs = X_rand_idxs[X_batch_start:X_batch_stop]

        Y_batch_idxs = np.random.permutation(list(unused_Y_idxs))[0:Y_batch_size]

        X_batch_len = len(X_batch_idxs)
        Y_batch_len = len(Y_batch_idxs)

        progress_bar.progress(
            (i + 1) / num_batches,
            f"Solving assignment batch {i+1} of {num_batches} with size {X_batch_len} x {Y_batch_len}",
        )

        X_batch = X[X_batch_idxs]
        Y_batch = Y[Y_batch_idxs]

        _, sol_batch_tgt = compute_assignment(X_batch, Y_batch, algorithm)
        Y_batch_idxs_a = Y_batch_idxs[sol_batch_tgt]

        # Update
        a_stop = a_start + len(Y_batch_idxs_a)
        a[a_start:a_stop] = Y_batch_idxs_a
        a_start = a_stop
        unused_Y_idxs -= set(Y_batch_idxs_a)

    X_rand_idxs_argsort_idxs = np.argsort(X_rand_idxs)
    sol_src = np.arange(N, dtype=np.int64)
    sol_tgt = a[X_rand_idxs_argsort_idxs]
    progress_bar.empty()
    return sol_src, sol_tgt
