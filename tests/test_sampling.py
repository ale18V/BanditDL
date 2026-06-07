import pytest
import torch

from banditdl.core.sampling import (
    CosineSimilarityReward,
    CTSNeighborSampler,
    CUCBNeighborSampler,
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
    assert isinstance(make_neighbor_sampler("cucb"), CUCBNeighborSampler)
    assert isinstance(make_neighbor_sampler("cts"), CTSNeighborSampler)
    assert make_neighbor_sampler(
        "discounted_cucb", params={"gamma": 0.8}
    ).discount == pytest.approx(0.8)
    assert make_neighbor_sampler("discounted_cts", params={"gamma": 0.7}).discount == pytest.approx(
        0.7
    )


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
        make_neighbor_sampler("cucb", context=context),
        make_neighbor_sampler("cts", context=context),
    ]:
        sampler.sample(population, 2)
        diagnostics = sampler.diagnostics(population, k=2)
        assert sum(diagnostics.weights.values()) == pytest.approx(1.0)
        assert sum(diagnostics.probabilities.values()) == pytest.approx(1.0)
        assert set(diagnostics.weights) == set(population)
        assert set(diagnostics.probabilities) == set(population)


def test_parameter_distance_reward():
    reward = ParameterDistanceReward()

    assert reward.score(torch.tensor([1.0]), [torch.tensor([1.0])]) == [1.0]
    assert reward.score(torch.tensor([1.0]), [torch.tensor([3.0])]) == pytest.approx([1 / 3])


def test_reward_strategy_factory():
    assert isinstance(make_reward_strategy("parameter_distance"), ParameterDistanceReward)
    assert isinstance(make_reward_strategy("cosine_similarity"), CosineSimilarityReward)


def test_cosine_similarity_reward_is_shifted_to_unit_interval():
    reward = CosineSimilarityReward()
    local = torch.tensor([1.0, 0.0])

    values = reward.score(
        local,
        [
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([-1.0, 0.0]),
            torch.zeros(2),
        ],
    )

    assert values == pytest.approx([1.0, 0.5, 0.0, 0.5])


def test_cucb_learns_high_reward_top_k():
    sampler = CUCBNeighborSampler(seed=0)
    population = [1, 2, 3]

    selections = []
    for _ in range(100):
        selected = sampler.sample(population, 2)
        sampler.update(selected, [1.0 if arm in {2, 3} else 0.0 for arm in selected])
        selections.append(set(selected))

    assert sum(selection == {2, 3} for selection in selections[-20:]) >= 18


def test_discounted_cucb_adapts_to_changed_best_arm():
    sampler = CUCBNeighborSampler(discount=0.9, seed=0)
    population = [1, 2]

    for _ in range(80):
        selected = sampler.sample(population, 1)
        sampler.update(selected, [1.0 if selected[0] == 1 else 0.0])
    late_selections = []
    for _ in range(80):
        selected = sampler.sample(population, 1)
        sampler.update(selected, [1.0 if selected[0] == 2 else 0.0])
        late_selections.append(selected[0])

    assert late_selections[-20:].count(2) >= 15


def test_cts_learns_high_reward_arm():
    sampler = CTSNeighborSampler(seed=0)
    population = [1, 2, 3]

    selections = []
    for _ in range(200):
        selected = sampler.sample(population, 1)
        sampler.update(selected, [1.0 if selected[0] == 2 else 0.0])
        selections.append(selected[0])

    assert selections[-50:].count(2) >= 40


def test_discounted_cts_adapts_to_changed_best_arm():
    sampler = CTSNeighborSampler(discount=0.95, seed=0)
    population = [1, 2]

    for _ in range(100):
        selected = sampler.sample(population, 1)
        sampler.update(selected, [1.0 if selected[0] == 1 else 0.0])
    late_selections = []
    for _ in range(150):
        selected = sampler.sample(population, 1)
        sampler.update(selected, [1.0 if selected[0] == 2 else 0.0])
        late_selections.append(selected[0])

    assert late_selections[-50:].count(2) >= 40


def test_cts_diagnostics_do_not_change_sampling_rng():
    first = CTSNeighborSampler(seed=7, diagnostic_samples=32)
    second = CTSNeighborSampler(seed=7, diagnostic_samples=32)
    population = [1, 2, 3]

    assert first.sample(population, 2) == second.sample(population, 2)
    first.diagnostics(population, 2)
    assert first.sample(population, 2) == second.sample(population, 2)


@pytest.mark.parametrize("sampler", [Exp3NeighborSampler, CTSNeighborSampler])
def test_monte_carlo_diagnostics_require_samples(sampler):
    with pytest.raises(ValueError, match="diagnostic_samples"):
        sampler(diagnostic_samples=0)


def test_best_fixed_subset_excludes_self():
    selected, reward = _best_fixed_subset(torch.tensor([0.9, 0.5, 0.8, 0.1]), 0, 2)

    assert selected.tolist() == [2, 1]
    assert reward == pytest.approx(0.65)
