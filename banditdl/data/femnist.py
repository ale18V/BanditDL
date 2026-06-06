from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms as T
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import Dataset

FEMNIST_HF_NAME = "flwrlabs/femnist"
FEMNIST_NUM_CLASSES = 62
_TRANSFORM = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


@lru_cache(maxsize=1)
def _load_hf_dataset() -> DatasetDict:
    return load_dataset(FEMNIST_HF_NAME)


class FemnistDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self._label_column = _column(split, "character", "label", "labels")
        self.targets = torch.tensor(split[self._label_column], dtype=torch.long)

    def __len__(self):
        return len(self._split)

    def __getitem__(self, index):
        row = self._split[int(index)]
        return _TRANSFORM(row["image"]), int(row[self._label_column])


def load_femnist_pool() -> tuple[FemnistDataset, np.ndarray]:
    source = _load_hf_dataset()
    if "train" not in source:
        raise ValueError(f"FEMNIST has no train split; available={list(source)}")
    split = (
        concatenate_datasets([source["train"], source["test"]])
        if "test" in source
        else source["train"]
    )
    writer_column = _column(split, "writer_id", "user", "client_id")
    return FemnistDataset(split), np.asarray(split[writer_column]).astype(str)


def _column(split, *candidates):
    for candidate in candidates:
        if candidate in split.column_names:
            return candidate
    raise ValueError(f"none of {candidates!r} exists in dataset columns {split.column_names}")
