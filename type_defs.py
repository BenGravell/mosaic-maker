"""Type definitions."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt


ArrU8 = npt.NDArray[np.uint8]
ArrI64 = npt.NDArray[np.int64]

AssignmentSolution: TypeAlias = tuple[ArrI64, ArrI64]
