from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

from banditdl.data.dataset_utils import get_default_root
from banditdl.data.femnist import load_femnist_pool


@dataclass(frozen=True)
class SamplePool:
    train_dataset: Dataset
    eval_dataset: Dataset
    targets: np.ndarray
    owners: np.ndarray | None = None

    def __post_init__(self) -> None:
        size = len(self.train_dataset)
        if len(self.eval_dataset) != size or len(self.targets) != size:
            raise ValueError("dataset views and targets must contain the same samples")
        if self.owners is not None and len(self.owners) != size:
            raise ValueError("owners must contain one value per sample")


class DatasetProvider(Protocol):
    def load(self) -> SamplePool: ...


_MNIST_TRANSFORM = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
_BASIC_TRANSFORM = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
_CIFAR_TRAIN_TRANSFORM = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
_CIFAR_EVAL_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

_TORCHVISION_DATASETS = {
    "mnist": ("MNIST", _MNIST_TRANSFORM, _MNIST_TRANSFORM),
    "fashionmnist": ("FashionMNIST", _BASIC_TRANSFORM, T.ToTensor()),
    "emnist": ("EMNIST", _MNIST_TRANSFORM, _MNIST_TRANSFORM),
    "cifar10": ("CIFAR10", _CIFAR_TRAIN_TRANSFORM, _CIFAR_EVAL_TRANSFORM),
    "cifar100": ("CIFAR100", _CIFAR_TRAIN_TRANSFORM, _CIFAR_EVAL_TRANSFORM),
}


class TorchvisionProvider:
    def __init__(self, name: str, root: str | None = None):
        self.name = name.lower()
        self.root = Path(root).expanduser() if root else get_default_root()

    def load(self) -> SamplePool:
        try:
            class_name, train_transform, eval_transform = _TORCHVISION_DATASETS[self.name]
        except KeyError as exc:
            raise ValueError(f"unsupported torchvision dataset: {self.name!r}") from exc

        dataset_class = getattr(torchvision.datasets, class_name)
        train_dataset = dataset_class(
            root=self.root,
            train=True,
            download=True,
            transform=train_transform,
        )
        eval_dataset = dataset_class(
            root=self.root,
            train=True,
            download=True,
            transform=eval_transform,
        )
        targets = torch.as_tensor(train_dataset.targets, dtype=torch.long).numpy()
        return SamplePool(train_dataset, eval_dataset, targets)


class FemnistProvider:
    def load(self) -> SamplePool:
        dataset, owners = load_femnist_pool()
        return SamplePool(dataset, dataset, dataset.targets.numpy(), owners)
