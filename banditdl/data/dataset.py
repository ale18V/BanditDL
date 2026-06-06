from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader, Subset

from banditdl.data.partitioning import PartitionStrategy
from banditdl.data.providers import DatasetProvider


@dataclass(frozen=True)
class DatasetBundle:
    train: dict[int, DataLoader]
    local_test: dict[int, DataLoader]
    global_test: DataLoader
    tracking_test: DataLoader
    audit: dict


@dataclass(frozen=True)
class DatasetBuildConfig:
    nodes: int
    train_batch: int
    test_batch: int
    global_test_ratio: float
    local_test_ratio: float
    seed: int


def build_dataset_bundle(
    provider: DatasetProvider,
    partitioner: PartitionStrategy,
    config: DatasetBuildConfig,
) -> DatasetBundle:
    pool = provider.load()
    partition = partitioner.partition(
        pool,
        config.nodes,
        config.global_test_ratio,
        np.random.default_rng(config.seed),
    )

    train, local_test = {}, {}
    distribution = {}
    for node, indices in partition.node_indices.items():
        train_indices, test_indices = _local_split(
            indices,
            config.local_test_ratio,
            np.random.default_rng(config.seed + node + 1),
        )
        train[node] = DataLoader(
            Subset(pool.train_dataset, train_indices),
            batch_size=config.train_batch,
            shuffle=True,
        )
        local_test[node] = DataLoader(
            Subset(pool.eval_dataset, test_indices),
            batch_size=config.test_batch,
            shuffle=False,
        )
        distribution[node] = _distribution(pool.targets, indices)

    global_test = DataLoader(
        Subset(pool.eval_dataset, partition.global_test_indices),
        batch_size=config.test_batch,
        shuffle=False,
    )
    tracking_size = max(
        1,
        min(
            round(
                len(pool.targets)
                * (1 - config.global_test_ratio)
                / config.nodes
                * config.local_test_ratio
            ),
            len(partition.global_test_indices),
        ),
    )
    tracking_test = DataLoader(
        Subset(pool.eval_dataset, partition.global_test_indices[:tracking_size]),
        batch_size=config.test_batch,
        shuffle=False,
    )
    return DatasetBundle(
        train,
        local_test,
        global_test,
        tracking_test,
        {
            "partition": {"seed": config.seed, **partition.audit},
            "distribution": distribution,
        },
    )


def _local_split(indices, test_ratio, rng):
    indices = list(indices)
    if len(indices) <= 1:
        return indices, []
    rng.shuffle(indices)
    test_size = min(max(round(len(indices) * test_ratio), 0), len(indices) - 1)
    return (indices, []) if test_size == 0 else (indices[test_size:], indices[:test_size])


def _distribution(targets, indices):
    labels, counts = np.unique(targets[indices], return_counts=True)
    return {
        "total": len(indices),
        "labels": {int(label): int(count) for label, count in zip(labels, counts, strict=True)},
    }
