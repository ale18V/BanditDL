"""FEMNIST data loading.

FEMNIST is the LEAF Federated EMNIST benchmark: 62 classes (digits + lowercase +
uppercase letters), naturally non-IID by writer. This module reads the
`flwrlabs/femnist` HuggingFace dataset and exposes two integration paths:

- `load_femnist_writer_loaders` — one writer per node. A fraction of writers is
  held out as the shared global test pool; the remaining writers are sampled as
  per-node trainers, and each trainer's samples are uniformly split into local
  train / local test loaders.

- `build_femnist_pool_dataset` — concatenate all writers' training samples into
  one `torch.utils.data.Dataset` exposing a `targets` attribute. The global test
  holdout is then performed uniformly inside
  `make_train_validation_test_datasets` (matching the torchvision-dataset path).

The HF dataset is cached transparently under `HF_DATASETS_CACHE` (defaults to
~/.cache/huggingface/datasets). On the EPFL GPU cluster, override that env var to
a shared/scratch path and pre-download once from the login node.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms as T
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset, Subset

FEMNIST_HF_NAME = "flwrlabs/femnist"
FEMNIST_NUM_CLASSES = 62

_DEFAULT_TRANSFORM = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


def _label_column(hf_split) -> str:
    for candidate in ("character", "label", "labels"):
        if candidate in hf_split.column_names:
            return candidate
    raise ValueError(
        f"FEMNIST split has no label column among 'character'/'label'/'labels'; "
        f"found {hf_split.column_names}"
    )


def _writer_column(hf_split) -> str:
    for candidate in ("writer_id", "user", "client_id"):
        if candidate in hf_split.column_names:
            return candidate
    raise ValueError(
        f"FEMNIST split has no writer column among 'writer_id'/'user'/'client_id'; "
        f"found {hf_split.column_names}"
    )


@lru_cache(maxsize=2)
def _load_hf_dataset() -> DatasetDict:
    """Load (and cache) the FEMNIST HuggingFace dataset."""
    return load_dataset(FEMNIST_HF_NAME)


class _FEMNISTWrappedDataset(Dataset):
    """Wrap one HF split as a torch Dataset with a `.targets` attribute."""

    def __init__(self, hf_split, transform=None):
        self._hf = hf_split
        self._label_col = _label_column(hf_split)
        self._transform = transform if transform is not None else _DEFAULT_TRANSFORM
        self.targets = torch.tensor(self._hf[self._label_col], dtype=torch.long)

    def __len__(self):
        return len(self._hf)

    def __getitem__(self, idx):
        row = self._hf[int(idx)]
        return self._transform(row["image"]), int(row[self._label_col])


def _resolve_split(hf, name: str):
    if name in hf:
        return hf[name]
    raise KeyError(f"FEMNIST HF dataset has no split named {name!r}; available={list(hf.keys())}")


def _writer_groups(hf_split) -> dict[str, list[int]]:
    """Return {writer_id: [sample_index, ...]} over the given HF split."""
    writer_col = _writer_column(hf_split)
    writers = hf_split[writer_col]
    groups: dict[str, list[int]] = {}
    for idx, w in enumerate(writers):
        groups.setdefault(str(w), []).append(idx)
    return groups


def _full_pool_split(hf) -> HFDataset:
    """Return the pooled HF split (train + test concatenated if both exist).

    The official FEMNIST distribution we use only has a `train` split, but
    `concatenate_datasets` is a no-op for that case and lets us tolerate the
    rare case where both splits are present.
    """
    train_split = _resolve_split(hf, "train")
    if "test" in hf:
        from datasets import concatenate_datasets

        return concatenate_datasets([train_split, hf["test"]])
    return train_split


def _split_writers_for_global_test(
    pooled_split, global_test_ratio: float, seed: int
) -> tuple[dict[str, list[int]], list[int]]:
    """Hold out `global_test_ratio` of writers as the global test pool.

    Returns ({remaining_writer: [indices], ...}, [global_test_indices]).
    """
    all_writers = _writer_groups(pooled_split)
    writer_ids = sorted(all_writers.keys())
    rng = np.random.default_rng(seed)
    nb_eval = max(1, round(global_test_ratio * len(writer_ids)))
    nb_eval = min(nb_eval, len(writer_ids) - 1)
    eval_choice = set(rng.choice(len(writer_ids), size=nb_eval, replace=False).tolist())
    train_writers: dict[str, list[int]] = {}
    eval_indices: list[int] = []
    for i, writer in enumerate(writer_ids):
        if i in eval_choice:
            eval_indices.extend(all_writers[writer])
        else:
            train_writers[writer] = all_writers[writer]
    return train_writers, eval_indices


def _select_writers(
    writers_to_indices: dict[str, list[int]],
    nb_writers: int,
    seed: int,
    writers_cap: int | None = None,
) -> list[str]:
    """Pick `nb_writers` writer IDs deterministically from seed; capped by writers_cap if given."""
    pool = sorted(writers_to_indices.keys())
    if writers_cap is not None:
        pool = pool[: int(writers_cap)]
    if nb_writers > len(pool):
        raise ValueError(
            f"Requested {nb_writers} FEMNIST writers but only {len(pool)} are available "
            f"(writers_cap={writers_cap})"
        )
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(pool), size=nb_writers, replace=False)
    return [pool[int(i)] for i in chosen_idx]


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _uniform_local_split(
    indices: list[int], local_test_ratio: float, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """Uniform shuffle + split into (train_indices, local_test_indices)."""
    indices = list(indices)
    if not indices:
        return [], []
    rng.shuffle(indices)
    test_size = round(len(indices) * local_test_ratio)
    if len(indices) <= 1:
        return indices, []
    test_size = max(0, min(test_size, len(indices) - 1))
    if test_size == 0:
        return indices, []
    return indices[test_size:], indices[:test_size]


def load_femnist_writer_loaders(
    nb_honest: int,
    train_batch: int,
    test_batch: int,
    *,
    global_test_ratio: float = 0.1,
    local_test_ratio: float = 0.2,
    split_seed: int = 0,
    writers_cap: int | None = None,
    return_stats: bool = False,
):
    """Per-writer training + per-writer local test loaders + shared global test loader.

    1. Hold out `global_test_ratio` of writers as the global test pool.
    2. Select `nb_honest` writers from the remaining as per-node trainers.
    3. Split each trainer's samples uniformly into local train / local test by
       `local_test_ratio`.

    Returns the three loaders, plus partition statistics when `return_stats=True`.
    """
    hf = _load_hf_dataset()
    pooled_split = _full_pool_split(hf)
    pooled_wrapped = _FEMNISTWrappedDataset(pooled_split)

    train_writer_groups, global_test_indices = _split_writers_for_global_test(
        pooled_split, global_test_ratio, seed=split_seed
    )
    selected = _select_writers(
        train_writer_groups, nb_honest, seed=split_seed, writers_cap=writers_cap
    )

    train_loaders: dict[int, DataLoader] = {}
    local_test_loaders: dict[int, DataLoader] = {}
    distribution_stats = {}
    for worker_id, writer in enumerate(selected):
        writer_indices = train_writer_groups[writer]
        labels, counts = np.unique(
            pooled_wrapped.targets[writer_indices].numpy(),
            return_counts=True,
        )
        distribution_stats[worker_id] = {
            "writer": writer,
            "total": len(writer_indices),
            "labels": {int(label): int(count) for label, count in zip(labels, counts, strict=True)},
        }
        rng = np.random.default_rng(split_seed + 1 + worker_id)
        train_idx, local_test_idx = _uniform_local_split(writer_indices, local_test_ratio, rng)
        train_loaders[worker_id] = _make_loader(
            Subset(pooled_wrapped, train_idx), batch_size=train_batch, shuffle=True
        )
        local_test_loaders[worker_id] = _make_loader(
            Subset(pooled_wrapped, local_test_idx), batch_size=test_batch, shuffle=False
        )

    global_test_loader = _make_loader(
        Subset(pooled_wrapped, global_test_indices), batch_size=test_batch, shuffle=False
    )
    loaders = train_loaders, local_test_loaders, global_test_loader
    return (*loaders, distribution_stats) if return_stats else loaders


def build_femnist_pool_dataset() -> Dataset:
    """Return the full pooled FEMNIST training dataset (all writers concatenated).

    Used by the pool-mode path in `make_train_validation_test_datasets`: the
    caller then carves out a uniform global test set and partitions the remainder
    across workers via the standard Dirichlet/pathological/grouped machinery.
    """
    hf = _load_hf_dataset()
    pooled_split = _full_pool_split(hf)
    return _FEMNISTWrappedDataset(pooled_split)
