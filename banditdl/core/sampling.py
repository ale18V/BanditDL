from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Any

import torch
from mabwiser.mab import LearningPolicy, MAB


@dataclass(frozen=True)
class SamplerContext:
    worker_id: int
    nodes: int
    k: int
    horizon: int
    seed: int


class RewardStrategy(ABC):
    @abstractmethod
    def score(self, local_weights, neighbor_weights) -> list[float]:
        """Compute one reward per selected neighbor."""


class ParameterDistanceReward(RewardStrategy):
    def score(self, local_weights, neighbor_weights) -> list[float]:
        return [
            1 / (1 + torch.norm(weight - local_weights).item())
            for weight in neighbor_weights
        ]


def make_reward_strategy(name):
    if name == "parameter_distance":
        return ParameterDistanceReward()
    raise ValueError(f"Unknown bandit reward strategy: {name}")


class UniformNeighborSampler:
    """Uniformly sample neighbors without replacement."""

    def sample(self, population, k, rng=None):
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > len(population):
            raise ValueError("k cannot exceed population size")
        if rng is None:
            return random.sample(population, k)
        return rng.sample(population, k)

    def update(self, population, rewards) -> None:
        return None


class EpsilonGreedyNeighborSampler:
    """MABWiser-backed epsilon-greedy neighbor sampler."""

    def __init__(self, epsilon=0.1, initial_value=0.0, seed=123456):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.seed = seed
        self._mab = None
        self._arms = set()

    def _ensure_mab(self, population):
        arms = set(population)
        if self._mab is not None and arms == self._arms:
            return
        self._arms = arms
        self._mab = MAB(
            arms=list(population),
            learning_policy=LearningPolicy.EpsilonGreedy(epsilon=self.epsilon),
            seed=self.seed,
        )
        self._mab.fit(
            decisions=list(population),
            rewards=[self.initial_value] * len(population),
        )

    def sample(self, population, k, rng=None):
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > len(population):
            raise ValueError("k cannot exceed population size")
        if k == 0:
            return []

        rng = rng or random
        population = list(population)
        self._ensure_mab(population)

        if k == 1:
            return [self._mab.predict()]

        if rng.random() < self.epsilon:
            return rng.sample(population, k)

        rng.shuffle(population)
        expectations = self._mab.predict_expectations()
        return sorted(
            population,
            key=lambda arm: expectations.get(arm, self.initial_value),
            reverse=True,
        )[:k]

    def update(self, population, rewards) -> None:
        population = list(population)
        rewards = list(rewards)
        if not population:
            return None
        if self._mab is None or any(arm not in self._arms for arm in population):
            self._ensure_mab(population)
        self._mab.partial_fit(decisions=population, rewards=rewards)
        return None


MultiArmedBanditSampler = EpsilonGreedyNeighborSampler


def make_neighbor_sampler(
    name,
    *,
    context: SamplerContext | None = None,
    params: dict[str, Any] | None = None,
    **legacy_kwargs,
):
    params = dict(params or {})
    params.update(
        {key: value for key, value in legacy_kwargs.items() if value is not None}
    )
    seed = params.pop("seed", context.seed if context is not None else 123456)

    if name == "uniform":
        return UniformNeighborSampler()
    if name in {"bandit", "epsilon_greedy"}:
        epsilon = float(params.pop("epsilon", params.pop("bandit_epsilon", 0.1)))
        initial_value = float(
            params.pop("initial_value", params.pop("bandit_initial_value", 0.0))
        )
        return EpsilonGreedyNeighborSampler(
            epsilon=epsilon,
            initial_value=initial_value,
            seed=seed,
        )
    raise ValueError(f"Unknown neighbor sampler: {name}")
