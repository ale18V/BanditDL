from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image

from banditdl.data import femnist
from banditdl.data.dataset import DatasetBuildConfig, build_dataset_bundle
from banditdl.data.partitioning import NaturalOwnerPartitionStrategy
from banditdl.data.providers import FemnistProvider


def _split(samples_per_writer, rng):
    images, labels, writers = [], [], []
    for writer, count in samples_per_writer.items():
        for _ in range(count):
            images.append(Image.fromarray(rng.integers(0, 255, (28, 28), dtype=np.uint8)))
            labels.append(int(rng.integers(0, femnist.FEMNIST_NUM_CLASSES)))
            writers.append(writer)
    return Dataset.from_dict({"image": images, "character": labels, "writer_id": writers})


@pytest.fixture
def synthetic_femnist(monkeypatch):
    rng = np.random.default_rng(0)
    source = DatasetDict(
        {
            "train": _split({f"writer-{i}": 6 for i in range(10)}, rng),
            "test": _split({f"writer-{i}": 3 for i in range(10)}, rng),
        }
    )
    monkeypatch.setattr(femnist, "_load_hf_dataset", lambda: source)
    return source


def test_provider_exposes_targets_and_owners(synthetic_femnist):
    pool = FemnistProvider().load()

    assert len(pool.train_dataset) == 90
    assert len(pool.targets) == len(pool.owners) == 90
    assert len(np.unique(pool.owners)) == 10
    image, label = pool.train_dataset[0]
    assert image.shape == (1, 28, 28)
    assert isinstance(label, int)


def test_natural_partition_assigns_one_writer_per_node(synthetic_femnist):
    pool = FemnistProvider().load()
    result = NaturalOwnerPartitionStrategy().partition(
        pool,
        nodes=4,
        global_test_ratio=0.2,
        rng=np.random.default_rng(4),
    )

    assert set(result.node_indices) == {0, 1, 2, 3}
    for indices in result.node_indices.values():
        assert len(set(pool.owners[indices])) == 1
        assert len(indices) == 9
    assert len(result.global_test_indices) == 18


def test_natural_partition_rejects_too_many_nodes(synthetic_femnist):
    pool = FemnistProvider().load()

    with pytest.raises(ValueError, match=r"only .* owners"):
        NaturalOwnerPartitionStrategy(writers_limit=2).partition(
            pool,
            nodes=3,
            global_test_ratio=0.2,
            rng=np.random.default_rng(0),
        )


def test_femnist_bundle_builds_all_loader_views(synthetic_femnist):
    bundle = build_dataset_bundle(
        FemnistProvider(),
        NaturalOwnerPartitionStrategy(),
        DatasetBuildConfig(3, 2, 4, 0.2, 0.2, 0),
    )

    assert set(bundle.train) == set(bundle.local_test) == {0, 1, 2}
    assert sum(len(loader.dataset) for loader in bundle.train.values()) == 21
    assert sum(len(loader.dataset) for loader in bundle.local_test.values()) == 6
    assert len(bundle.global_test.dataset) == 18
    assert bundle.audit["partition"]["strategy"] == "natural_owner"
    assert all(
        label.dtype == torch.int64
        for _, label in (next(iter(loader)) for loader in bundle.train.values())
    )
