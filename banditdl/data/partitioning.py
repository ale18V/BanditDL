from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from banditdl.data.dataset_utils import partition_hierarchical
from banditdl.data.providers import SamplePool


@dataclass(frozen=True)
class PartitionResult:
    node_indices: dict[int, list[int]]
    global_test_indices: list[int]
    audit: dict


class PartitionStrategy(Protocol):
    def partition(
        self,
        pool: SamplePool,
        nodes: int,
        global_test_ratio: float,
        rng: np.random.Generator,
    ) -> PartitionResult: ...


class SyntheticPartitionStrategy:
    def __init__(self, method: str, **params):
        self.method = method
        self.config = {"method": method, **params}

    def partition(self, pool, nodes, global_test_ratio, rng) -> PartitionResult:
        client_indices, global_indices = _sample_holdout(
            len(pool.targets), nodes, global_test_ratio, rng
        )
        local = partition_hierarchical(
            pool.targets[client_indices],
            nodes,
            int(np.max(pool.targets)) + 1,
            self.config,
            rng,
        )
        assignments = {
            node: [int(client_indices[index]) for index in indices]
            for node, indices in local.items()
        }
        return PartitionResult(
            assignments,
            global_indices.tolist(),
            {
                "strategy": self.method,
                **self.config,
                "resolved_clusters": self.config.get("clusters") or nodes,
            },
        )


class NaturalOwnerPartitionStrategy:
    def __init__(self, writers_limit: int | None = None):
        self.writers_limit = writers_limit

    def partition(self, pool, nodes, global_test_ratio, rng) -> PartitionResult:
        if pool.owners is None:
            raise ValueError("natural-owner partitioning requires sample owner metadata")

        owners = np.asarray(pool.owners).astype(str)
        owner_ids = np.unique(owners)
        if len(owner_ids) < 2:
            raise ValueError("natural-owner partitioning requires at least two owners")
        held_out_count = min(
            max(1, round(global_test_ratio * len(owner_ids))),
            len(owner_ids) - 1,
        )
        held_out = owner_ids[rng.choice(len(owner_ids), size=held_out_count, replace=False)]
        available = owner_ids[~np.isin(owner_ids, held_out)]
        if self.writers_limit is not None:
            available = available[: self.writers_limit]
        if nodes > len(available):
            raise ValueError(
                f"requested {nodes} nodes but only {len(available)} sample owners are available"
            )

        selected = available[rng.choice(len(available), size=nodes, replace=False)]
        assignments = {
            node: np.flatnonzero(owners == owner).tolist() for node, owner in enumerate(selected)
        }
        global_indices = np.flatnonzero(np.isin(owners, held_out)).tolist()
        return PartitionResult(
            assignments,
            global_indices,
            {
                "strategy": "natural_owner",
                "selected_owners": selected.tolist(),
                "held_out_owners": held_out.tolist(),
                "writers_limit": self.writers_limit,
            },
        )


def _sample_holdout(size, nodes, ratio, rng):
    if size <= 1 or nodes >= size:
        raise ValueError(f"nodes ({nodes}) must be smaller than dataset size ({size})")
    indices = rng.permutation(size)
    holdout_size = min(max(round(size * ratio), 1), size - nodes)
    return indices[holdout_size:], indices[:holdout_size]
