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
            # Reset accumulator
            acc = 0
            for k in prange(nz):
                # Use custom branching logic for the absolute difference
                # to prevent underflow that would happen with uint8 subtraction.
                x = X[i, k]
                y = Y[j, k]
                d = y - x if y > x else x - y
                acc += d
            # Use integer division by nz to ensure
            # 1. The summand is integer valued
            # 2. The sum does not exceed the size of np.uint8 in any entry of D
            # 3. Computation happens fast
            D[i, j] = np.uint8(acc // nz)
    return D
