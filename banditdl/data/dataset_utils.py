"""Unified dataset partitioning logic using hierarchical group assignment."""

import os
import pathlib

import numpy as np


def get_default_root():
    """Lazy-initialize and return the dataset root directory path."""
    override = os.environ.get("BANDITDL_DATASET_ROOT")
    if override:
        default_root = pathlib.Path(override).expanduser()
    else:
        default_root = pathlib.Path(__file__).parent.parent / "datasets" / "cache"
    default_root.mkdir(parents=True, exist_ok=True)
    return default_root


def build_dirichlet_matrix(nb_groups, nb_labels, alpha, rng):
    """Build a (Groups x Labels) probability matrix using Dirichlet distribution."""
    return rng.dirichlet([alpha] * nb_groups, size=nb_labels).T


def build_pathological_matrix(nb_groups, nb_labels, labels_per_group, overlap):
    """Build a (Groups x Labels) probability matrix using a sparse mask."""
    stride = labels_per_group - overlap
    mask = np.zeros((nb_groups, nb_labels))

    for g in range(nb_groups):
        start = (g * stride) % nb_labels
        for i in range(labels_per_group):
            mask[g, (start + i) % nb_labels] = 1.0

    col_sums = mask.sum(axis=0)
    uncovered = col_sums == 0
    mask[:, uncovered] = 1.0
    col_sums = mask.sum(axis=0)
    return mask / col_sums


def partition_hierarchical(targets, nb_workers, numb_labels, config, rng):
    """Unified entry point for hierarchical partitioning."""
    nb_groups = config.get("clusters") or nb_workers
    if nb_groups > nb_workers:
        raise ValueError(f"clusters ({nb_groups}) cannot exceed nb_workers ({nb_workers})")
    if nb_workers % nb_groups != 0:
        raise ValueError(f"nb_workers ({nb_workers}) must be divisible by clusters ({nb_groups})")

    group_size = nb_workers // nb_groups
    workers_per_group = [group_size] * nb_groups

    # 1. Build the base heterogeneity matrix
    method = config.get("method", "dirichlet")
    alpha = config.get("alpha")

    if alpha is not None:
        matrix = build_dirichlet_matrix(nb_groups, numb_labels, alpha, rng)
    elif method == "pathological":
        matrix = build_pathological_matrix(
            nb_groups,
            numb_labels,
            config.get("classes_per_group", 1),
            config.get("group_overlap", 0),
        )
    else:
        matrix = np.full((nb_groups, numb_labels), 1.0 / nb_groups)

    # 2. Apply Interpolation (Gamma Similarity)
    gamma = config.get("gamma_similarity")
    if gamma is not None:
        iid_matrix = np.full((nb_groups, numb_labels), 1.0 / nb_groups)
        matrix = gamma * iid_matrix + (1 - gamma) * matrix

    # 3. Draw samples using the matrix and distribute IID within groups
    return _draw_hierarchical(targets, matrix, workers_per_group, numb_labels, rng)


def _draw_hierarchical(targets, matrix, workers_per_group, numb_labels, rng):
    """Draw indices based on the group-class matrix and split IID within groups."""
    nb_groups = len(workers_per_group)
    worker_samples = {}

    targets = np.asarray(targets)
    indices_per_label = []
    for label in range(numb_labels):
        indices = np.flatnonzero(targets == label).tolist()
        rng.shuffle(indices)
        indices_per_label.append(indices)

    group_pools = {g: [] for g in range(nb_groups)}

    for c in range(numb_labels):
        label_indices = indices_per_label[c]
        n_samples = len(label_indices)
        expected = matrix[:, c] * n_samples
        counts = np.floor(expected).astype(int)
        remainder = n_samples - counts.sum()
        if remainder:
            fractions = expected - counts
            tie_break = rng.permutation(nb_groups)
            order = tie_break[np.argsort(fractions[tie_break])[::-1]]
            counts[order[:remainder]] += 1

        start = 0
        for group_id, count in enumerate(counts):
            group_pools[group_id].extend(label_indices[start : start + count])
            start += count

    cursor = 0
    for g in range(nb_groups):
        pool = group_pools[g]
        rng.shuffle(pool)
        worker_chunks = np.array_split(pool, workers_per_group[g])
        for chunk in worker_chunks:
            worker_samples[cursor] = chunk.tolist()
            cursor += 1

    return worker_samples
