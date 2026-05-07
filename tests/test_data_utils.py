import numpy as np

from banditdl.data.dataset_utils import draw_indices


def test_draw_indices_advances_offsets_cumulatively():
    samples_distribution = np.array([[0.2, 0.3, 0.5]])
    indices_per_label = {0: list(range(10))}

    samples = draw_indices(samples_distribution, indices_per_label, nb_workers=3)

    assert samples == {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8, 9]}
