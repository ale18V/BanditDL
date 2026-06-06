import numpy as np
import torch

from banditdl.data.dataset_utils import partition_hierarchical


def _toy_targets(samples_per_label=20, numb_labels=5):
    return torch.tensor([label for label in range(numb_labels) for _ in range(samples_per_label)])


def test_partition_hierarchical_zero_data_loss():
    # Verify the cumulative sum logic fixes the systematic data loss bug
    # (0.2, 0.3, 0.5) over 10 samples should be exactly [2, 3, 5]
    targets = torch.tensor([0] * 10)
    rng = np.random.default_rng(0)

    # Manually mock the matrix for this test to be precise
    from banditdl.data.dataset_utils import _draw_hierarchical
    matrix = np.array([[0.2], [0.3], [0.5]]) # 3 clusters x 1 label
    workers_per_group = [1, 1, 1]

    samples = _draw_hierarchical(targets, matrix, workers_per_group, numb_labels=1, rng=rng)

    assert len(samples[0]) == 2
    assert len(samples[1]) == 3
    assert len(samples[2]) == 5
    assert sorted(samples[0] + samples[1] + samples[2]) == list(range(10))


def test_pathological_mode_respects_group_logic():
    targets = _toy_targets(samples_per_label=20, numb_labels=10)
    rng = np.random.default_rng(0)

    # 2 clusters, 2 workers each. Each group gets 2 distinct labels.
    config = {
        "method": "pathological",
        "clusters": 2,
        "classes_per_group": 2,
        "group_overlap": 0
    }

    worker_samples = partition_hierarchical(targets, nb_workers=4, numb_labels=10, config=config, rng=rng)

    # Cluster 0 (Workers 0,1) gets labels {0, 1}
    # Cluster 1 (Workers 2,3) gets labels {2, 3}
    group0_labels = {0, 1}
    group1_labels = {2, 3}

    for w_id in [0, 1]:
        labels_seen = set(int(targets[i].item()) for i in worker_samples[w_id])
        assert labels_seen <= group0_labels

    for w_id in [2, 3]:
        labels_seen = set(int(targets[i].item()) for i in worker_samples[w_id])
        assert labels_seen <= group1_labels


def test_dirichlet_mode_covers_all_workers():
    targets = _toy_targets(samples_per_label=100, numb_labels=5)
    rng = np.random.default_rng(42)

    # 10 workers, node-level heterogeneity (clusters=10)
    config = {"alpha": 0.5, "clusters": 10}

    worker_samples = partition_hierarchical(targets, nb_workers=10, numb_labels=5, config=config, rng=rng)

    assert len(worker_samples) == 10
    all_indices = []
    for indices in worker_samples.values():
        assert len(indices) > 0
        all_indices.extend(indices)

    assert len(set(all_indices)) == 500
    assert len(all_indices) == 500


def test_grouped_pathological_with_overlap():
    targets = _toy_targets(samples_per_label=50, numb_labels=10)
    rng = np.random.default_rng(0)

    # 3 clusters, overlap of 1 label
    # G0: {0,1,2}, G1: {2,3,4}, G2: {4,5,6}
    config = {
        "method": "pathological",
        "clusters": 3,
        "classes_per_group": 3,
        "group_overlap": 1
    }

    worker_samples = partition_hierarchical(targets, nb_workers=3, numb_labels=10, config=config, rng=rng)

    # Label 2 should be shared by G0 and G1
    # Label 4 should be shared by G1 and G2
    g0_labels = set(int(targets[i].item()) for i in worker_samples[0])
    g1_labels = set(int(targets[i].item()) for i in worker_samples[1])
    g2_labels = set(int(targets[i].item()) for i in worker_samples[2])

    assert g0_labels == {0, 1, 2}
    assert g1_labels == {2, 3, 4}
    assert g2_labels == {4, 5, 6}
