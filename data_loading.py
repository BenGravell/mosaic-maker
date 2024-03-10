"""Functions for loading data."""


from PIL.Image import Image
import torchvision


def load_source_images(dataset_name: str = "CIFAR100") -> list[Image]:
    cls = getattr(torchvision.datasets, dataset_name)
    dataset_trn = cls(root="./data", train=True, download=True)
    dataset_tst = cls(root="./data", train=False, download=True)
    return [x[0] for x in dataset_trn] + [x[0] for x in dataset_tst]
