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

    matrix = np.array([[0.2], [0.3], [0.5]])  # 3 clusters x 1 label
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
    config = {"method": "pathological", "clusters": 2, "classes_per_group": 2, "group_overlap": 0}

    worker_samples = partition_hierarchical(
        targets, nb_workers=4, numb_labels=10, config=config, rng=rng
    )

    group0 = worker_samples[0] + worker_samples[1]
    group1 = worker_samples[2] + worker_samples[3]
    group0_labels = {int(targets[i]) for i in group0}
    group1_labels = {int(targets[i]) for i in group1}

    assert {0, 1} <= group0_labels
    assert not ({0, 1} & group1_labels)
    assert {2, 3} <= group1_labels
    assert not ({2, 3} & group0_labels)
    assert sorted(group0 + group1) == list(range(len(targets)))


def test_dirichlet_mode_covers_all_workers():
    targets = _toy_targets(samples_per_label=100, numb_labels=5)
    rng = np.random.default_rng(42)

    # 10 workers, node-level heterogeneity (clusters=10)
    config = {"alpha": 0.5, "clusters": 10}

    worker_samples = partition_hierarchical(
        targets, nb_workers=10, numb_labels=5, config=config, rng=rng
    )

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
    config = {"method": "pathological", "clusters": 3, "classes_per_group": 3, "group_overlap": 1}

    worker_samples = partition_hierarchical(
        targets, nb_workers=3, numb_labels=10, config=config, rng=rng
    )

    # Label 2 should be shared by G0 and G1
    # Label 4 should be shared by G1 and G2
    g0_labels = set(int(targets[i].item()) for i in worker_samples[0])
    g1_labels = set(int(targets[i].item()) for i in worker_samples[1])
    g2_labels = set(int(targets[i].item()) for i in worker_samples[2])

    assert {0, 1, 2} <= g0_labels
    assert {2, 3, 4} <= g1_labels
    assert {4, 5, 6} <= g2_labels
    all_indices = sum(worker_samples.values(), [])
    assert sorted(all_indices) == list(range(len(targets)))


def test_null_clusters_means_one_cluster_per_worker():
    targets = _toy_targets(samples_per_label=50, numb_labels=5)
    config = {"method": "dirichlet", "alpha": 0.5, "clusters": None}

    samples = partition_hierarchical(
        targets,
        nb_workers=5,
        numb_labels=5,
        config=config,
        rng=np.random.default_rng(7),
    )

    assert len(samples) == 5
    assert sorted(sum(samples.values(), [])) == list(range(len(targets)))


def test_clusters_must_divide_worker_count():
    with np.testing.assert_raises_regex(ValueError, "divisible"):
        partition_hierarchical(
            _toy_targets(),
            nb_workers=5,
            numb_labels=5,
            config={"method": "dirichlet", "alpha": 0.5, "clusters": 2},
            rng=np.random.default_rng(0),
        )
