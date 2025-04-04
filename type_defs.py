"""Type definitions."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt


ArrU8 = npt.NDArray[np.uint8]
ArrS64 = npt.NDArray[np.int64]
ArrF32 = npt.NDArray[np.float32]
ArrF64 = npt.NDArray[np.float64]

AssignmentSolution: TypeAlias = tuple[ArrS64, ArrS64]
