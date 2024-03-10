"""Functions for computing distance between pixels and images."""

import numpy as np
from numba import jit, prange


@jit(cache=True, nopython=True, parallel=True, fastmath=True, boundscheck=False, nogil=True)
def compute_distance_matrix(X, Y):
    nx = X.shape[0]
    ny = Y.shape[0]
    nz = X.shape[1]
    D = np.zeros((nx, ny), dtype=np.uint16)
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                D[i, j] += abs(X[i, k] - Y[j, k])
    return D
