"""Dataset utility functions."""

import pathlib
import numpy as np


def get_default_root():
    """Lazy-initialize and return the default dataset root directory path."""
    default_root = pathlib.Path(__file__).parent.parent / "datasets" / "cache"
    default_root.mkdir(parents=True, exist_ok=True)
    return default_root


def draw_indices(samples_distribution, indices_per_label, nb_workers):
    """Return the indices of the training datapoints selected for each honest worker.

    Used in case of Dirichlet distribution.
    """
    worker_samples = {worker: [] for worker in range(nb_workers)}

    for label, label_distribution in enumerate(samples_distribution):
        last_sample = 0
        number_samples_label = len(indices_per_label[label])
        for worker, worker_proportion in enumerate(label_distribution):
            samples_for_worker = int(worker_proportion * number_samples_label)
            worker_samples[worker].extend(
                indices_per_label[label][last_sample:last_sample + samples_for_worker]
            )
            last_sample += samples_for_worker

    return worker_samples
