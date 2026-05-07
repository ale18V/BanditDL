import pytest
import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.utils.math_utils import average_nearest_neighbors


def test_average_nearest_neighbors_without_pivot_returns_one_average_per_vector():
    vectors = torch.tensor([[0.0], [1.0], [10.0]])

    mixed = average_nearest_neighbors(vectors, f=1)

    assert mixed.shape == vectors.shape
    assert mixed.squeeze().tolist() == pytest.approx([0.5, 0.5, 5.5])


def test_nearest_neighbor_mixing_aggregator_does_not_crash():
    aggregator = RobustAggregator(
        "nnm",
        "average",
        False,
        nb_workers=3,
        nb_byz=1,
        bucket_size=1,
        model_size=1,
        device="cpu",
    )

    result = aggregator.aggregate(
        [torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([10.0])]
    )

    assert result.tolist() == pytest.approx([2.1666667])
