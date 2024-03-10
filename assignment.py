"""Functions for solving linear assignment problems."""

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import jit


@jit(cache=True, nopython=True, parallel=True, fastmath=True, boundscheck=False, nogil=True)
def greedy_random(D):
    N = D.shape[0]
    Dc = np.copy(D)
    y = np.zeros(N, dtype=np.int64)
    for i in np.random.permutation(np.arange(N)):
        y[i] = np.argmin(Dc[i])
        Dc[:, y[i]] = np.iinfo(np.uint16).max
    return np.arange(N), y


def compute_assignment(D, algorithm: str) -> Any:
    assignment_func_map = {
        "greedy_random": greedy_random,
        "jonker_volgenant": linear_sum_assignment,
    }
    assignment_func = assignment_func_map[algorithm]
    return assignment_func(D)


def compute_quality(D, sol):
    return np.sum(D[sol[0], sol[1]])
