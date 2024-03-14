import numpy as np

import utils

# NOTE: max value should not exceed sqrt(60_000) since we only have 60_000 images in CIFAR-100 and CIFAR-10
TARGET_RESOLUTION_OPTIONS = utils.logspace_2_and_3(2, 8)
TARGET_RESOLUTION_OPTIONS = TARGET_RESOLUTION_OPTIONS[TARGET_RESOLUTION_OPTIONS <= int(60_000**0.5)]
TARGET_RESOLUTION_OPTIONS = np.concatenate([TARGET_RESOLUTION_OPTIONS, [244]])
assert np.max(TARGET_RESOLUTION_OPTIONS) ** 2 <= 60_000
