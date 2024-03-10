"""Misc utilities."""

import numpy as np


def logspace_2_and_3(start: int, stop: int):
    x = np.logspace(start, stop, base=2, num=(stop - start) + 1)
    y = 0.75 * x
    return np.sort(np.concatenate([x, y])).astype(int)
