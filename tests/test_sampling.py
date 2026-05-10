import pytest
import torch

from banditdl.core.sampling import (
    Exp3NeighborSampler,
    MultiArmedBanditSampler,
    ParameterDistanceReward,
    SamplerContext,
    make_neighbor_sampler,
    make_reward_strategy,
)
from banditdl.experiments.engine import _best_fixed_subset


def test_bandit_sampler_prefers_high_reward_arm():
    sampler = MultiArmedBanditSampler(epsilon=0.0)
    sampler.update([1, 2, 3], [0.1, 0.9, 0.2])

    assert sampler.sample([1, 2, 3], 1) == [2]


def test_neighbor_sampler_factory():
    assert make_neighbor_sampler("uniform").sample([1, 2, 3], 2)
    assert isinstance(make_neighbor_sampler("bandit"), MultiArmedBanditSampler)


def test_exp3_sampler_factory_uses_context_horizon():
    context = SamplerContext(worker_id=0, nodes=4, k=2, horizon=100, seed=1)
    sampler = make_neighbor_sampler("exp3", context=context, params={"gamma": "auto"})

    assert isinstance(sampler, Exp3NeighborSampler)
    assert sampler.horizon == 100
    assert len(sampler.sample([1, 2, 3], 2)) == 2


def test_exp3_sampler_prefers_high_reward_arm():
    sampler = Exp3NeighborSampler(gamma=0.5, seed=0, horizon=100)
    population = [1, 2, 3]
    sampler.sample(population, 1)
    for _ in range(20):
        sampler.update(population, [0.0, 1.0, 0.0])

    assert sampler._probabilities[1] > sampler._probabilities[0]
    assert sampler._probabilities[1] > sampler._probabilities[2]


def test_sampler_probability_distributions_sum_to_one():
    context = SamplerContext(worker_id=0, nodes=4, k=2, horizon=100, seed=1)
    population = [1, 2, 3]

    for sampler in [
        make_neighbor_sampler("uniform", context=context),
        make_neighbor_sampler("epsilon_greedy", context=context),
        make_neighbor_sampler("exp3", context=context),
    ]:
        probabilities = sampler.probabilities(population, k=2)
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert set(probabilities) == set(population)


def test_parameter_distance_reward():
    reward = ParameterDistanceReward()

    assert reward.score(torch.tensor([1.0]), [torch.tensor([1.0])]) == [1.0]
    assert reward.score(torch.tensor([1.0]), [torch.tensor([3.0])]) == pytest.approx(
        [1 / 3]
    )


def test_reward_strategy_factory():
    assert isinstance(
        make_reward_strategy("parameter_distance"), ParameterDistanceReward
    )


def test_best_fixed_subset_excludes_self():
    selected, reward = _best_fixed_subset(torch.tensor([0.9, 0.5, 0.8, 0.1]), 0, 2)

    assert selected.tolist() == [2, 1]
    assert reward == pytest.approx(0.65)
