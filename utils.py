import numpy as np

from type_defs import ArrI64


def fourspace(start: int, stop: int) -> ArrI64:
    """Create an array of integers all divisible by 4 spaced exponentially with base 2."""
    base = np.array([4, 5, 6, 7])
    min_scale_log2 = int(np.floor(np.log2(np.floor(start / base[0]))))
    max_scale_log2 = int(np.ceil(np.log2(np.ceil(stop / base[-1]))))
    scales = 2 ** np.arange(min_scale_log2, max_scale_log2 + 1, dtype=np.int64)
    x = np.concatenate([base * scale for scale in scales])
    x = x[x >= start]
    x = x[x <= stop]
    return x
