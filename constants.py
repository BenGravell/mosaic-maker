import numpy as np

from utils import fourspace

DATASET_NAME_OPTIONS = ["CIFAR100", "CIFAR10"]

ASSIGNMENT_ALGORITHM_OPTIONS = ["jonker_volgenant", "greedy_random"]


def ASSIGNMENT_ALGORITHM_OPTIONS_FORMAT_FUNC(s: str) -> str:
    return s.replace("_", " ").title().replace(" ", "-")


# NOTE: max value should not exceed sqrt(60_000) since we only have 60_000 images in each of CIFAR-100 and CIFAR-10
TARGET_RESOLUTION_OPTIONS = fourspace(16, 60_000**0.5)
TARGET_RESOLUTION_OPTIONS = TARGET_RESOLUTION_OPTIONS[TARGET_RESOLUTION_OPTIONS <= int(60_000**0.5)]
assert np.max(TARGET_RESOLUTION_OPTIONS) ** 2 <= 60_000

X_BATCH_SIZE_OPTIONS = fourspace(64, 1_024)
Y_BATCH_SIZE_OPTIONS = fourspace(1_024, 40_000)
# This check is to prevent out-of-memory issues when deployed with <1GB RAM resources
assert np.max(X_BATCH_SIZE_OPTIONS) * np.max(Y_BATCH_SIZE_OPTIONS) < 100_000_000
# This check is to prevent errors when passing options to the assignment solver
assert np.max(X_BATCH_SIZE_OPTIONS) <= np.max(Y_BATCH_SIZE_OPTIONS)

TARGET_IMG_BLEND_ALPHA_OPTIONS = np.concatenate(
    [np.arange(0.00, 0.30, 0.01), np.arange(0.30, 0.60, 0.02), np.arange(0.60, 1.05, 0.05)],
).round(2)
