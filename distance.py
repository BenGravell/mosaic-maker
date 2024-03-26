"""Functions for computing distance between pixels and images."""

import numpy as np
from numba import jit, prange  # type: ignore[import]

from type_defs import ArrU8


@jit(cache=True, nopython=True, parallel=True, fastmath=True, boundscheck=False, nogil=True)
def compute_distance_matrix(X: ArrU8, Y: ArrU8) -> ArrU8:
    nx = X.shape[0]
    ny = Y.shape[0]
    nz = X.shape[1]
    # Pre-allocate using the smallest usable dtype, which is uint8 for our purposes for 8-bit RGB colors.
    D = np.zeros((nx, ny), dtype=np.uint8)
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                # Use integer division by nz to ensure
                # 1. The summand is integer valued
                # 2. The sum does not exceed the size of np.uint8 in any entry of D
                D[i, j] += abs(X[i, k] - Y[j, k]) // nz
    return D
