"""Functions for solving linear assignment problems."""

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import jit

import distance


@jit(cache=True, nopython=True, parallel=True, fastmath=True, boundscheck=False, nogil=True)
def greedy_random(D):
    """Solve an assignment problem using a greedy algorithm.

    Arguments:
    D: distance matrix. May be modified in-place by this function.
    """
    N = D.shape[0]
    y = np.zeros(N, dtype=np.uint64)
    for i in np.random.permutation(np.arange(N)):
        y[i] = np.argmin(D[i])
        D[:, y[i]] = np.iinfo(np.uint8).max
    return np.arange(N), y


def compute_assignment(X, Y, algorithm: str) -> Any:
    D = distance.compute_distance_matrix(X, Y)
    assignment_func_map = {
        "greedy_random": greedy_random,
        "jonker_volgenant": linear_sum_assignment,
    }
    assignment_func = assignment_func_map[algorithm]
    return assignment_func(D)


def compute_assignment_batched(X, Y, X_batch_size: int, Y_batch_size: int, algorithm: str):
    """Compute assignment using randomized batches.

    Together X_batch_size and Y_batch_size determine the max distance matrix and assignment problem size,
    which is X_batch_size x Y_batch_size.
    """
    N = X.shape[0]
    num_batches = int(np.ceil(N / X_batch_size))

    X_idxs = np.arange(X.shape[0])
    Y_idxs = np.arange(Y.shape[0])
    unused_Y_idxs = set(Y_idxs)
    X_rand_idxs = np.random.permutation(X_idxs)
    a = []
    for i in range(num_batches):
        Y_batch_idxs = np.random.permutation(list(unused_Y_idxs))[0:Y_batch_size]

        start = i * X_batch_size
        stop = (i + 1) * X_batch_size
        X_batch_idxs = X_rand_idxs[start:stop]

        X_batch = X[X_batch_idxs]
        Y_batch = Y[Y_batch_idxs]

        sol_batch = compute_assignment(X_batch, Y_batch, algorithm)
        Y_batch_idxs_a = Y_batch_idxs[sol_batch[1]]
        a.extend(Y_batch_idxs_a)
        unused_Y_idxs -= set(Y_batch_idxs_a)
    a = np.array(a)

    X_rand_idxs_argsort_idxs = np.argsort(X_rand_idxs)
    sol_1 = np.arange(N)
    sol_2 = a[X_rand_idxs_argsort_idxs]
    return sol_1, sol_2
