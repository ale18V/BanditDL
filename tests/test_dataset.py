import numpy as np
import torch
from torch.utils.data import Dataset

from banditdl.data.dataset import DatasetBuildConfig, build_dataset_bundle
from banditdl.data.partitioning import SyntheticPartitionStrategy
from banditdl.data.providers import SamplePool


class _View(Dataset):
    def __init__(self, targets, view):
        self.targets = targets
        self.view = view

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.view, int(self.targets[index])


class _Provider:
    def load(self):
        targets = np.repeat(np.arange(2), 20)
        return SamplePool(_View(targets, "train"), _View(targets, "eval"), targets)


def test_bundle_uses_eval_view_outside_training():
    bundle = build_dataset_bundle(
        _Provider(),
        SyntheticPartitionStrategy("dirichlet", alpha=1.0),
        DatasetBuildConfig(2, 4, 4, 0.2, 0.2, 0),
    )

    assert bundle.train[0].dataset.dataset.view == "train"
    assert bundle.local_test[0].dataset.dataset.view == "eval"
    assert bundle.global_test.dataset.dataset.view == "eval"
    assert bundle.tracking_test.dataset.dataset.view == "eval"


def test_synthetic_partition_uses_every_sample_once():
    pool = _Provider().load()
    result = SyntheticPartitionStrategy(
        "pathological",
        classes_per_group=1,
    ).partition(pool, nodes=2, global_test_ratio=0.2, rng=np.random.default_rng(0))

    assigned = [index for node_indices in result.node_indices.values() for index in node_indices]
    assert sorted(assigned + result.global_test_indices) == list(range(len(pool.targets)))
    assert len(set(assigned) & set(result.global_test_indices)) == 0


def test_bundle_audit_reports_full_node_distributions():
    bundle = build_dataset_bundle(
        _Provider(),
        SyntheticPartitionStrategy("dirichlet", alpha=1.0),
        DatasetBuildConfig(2, 4, 4, 0.2, 0.2, 0),
    )

    assert sum(node["total"] for node in bundle.audit["distribution"].values()) == 32
    assert torch.is_tensor(next(iter(bundle.train[0]))[1])
