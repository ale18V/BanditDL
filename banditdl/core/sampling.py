from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import random
from typing import Any

import numpy as np
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

    def probabilities(self, population, k=None) -> dict[Any, float]:
        population = list(population)
        if not population:
            return {}
        probability = 1.0 / len(population)
        return {arm: probability for arm in population}


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

    def probabilities(self, population, k=None) -> dict[Any, float]:
        population = list(population)
        if not population:
            return {}
        self._ensure_mab(population)
        if k is None:
            k = 1
        k = max(1, min(int(k), len(population)))
        exploration = self.epsilon / len(population)
        probabilities = {arm: exploration for arm in population}
        expectations = self._mab.predict_expectations()
        greedy_arms = sorted(
            population,
            key=lambda arm: expectations.get(arm, self.initial_value),
            reverse=True,
        )[:k]
        exploitation = (1.0 - self.epsilon) / k
        for arm in greedy_arms:
            probabilities[arm] += exploitation
        return probabilities


MultiArmedBanditSampler = EpsilonGreedyNeighborSampler


class Exp3NeighborSampler:
    """EXP3 neighbor sampler for rewards in [lower, lower + amplitude]."""

    def __init__(
        self,
        gamma="auto",
        lower=0.0,
        amplitude=1.0,
        seed=123456,
        horizon=None,
    ):
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        self.gamma = gamma
        self.lower = float(lower)
        self.amplitude = float(amplitude)
        self.seed = int(seed)
        self.horizon = None if horizon is None else int(horizon)
        self._rng = np.random.default_rng(self.seed)
        self._arms: list[Any] = []
        self._arm_to_index: dict[Any, int] = {}
        self._weights = np.array([], dtype=float)
        self._probabilities = np.array([], dtype=float)

    def _resolve_gamma(self, nb_arms: int) -> float:
        if self.gamma == "auto":
            if self.horizon is None or self.horizon <= 0:
                raise ValueError("EXP3 gamma='auto' requires a positive horizon")
            gamma = math.sqrt(2 * math.log(nb_arms) / (self.horizon * nb_arms))
            return min(1.0, max(1e-12, gamma))
        gamma = float(self.gamma)
        if gamma <= 0 or gamma > 1:
            raise ValueError("EXP3 gamma must be in (0, 1]")
        return gamma

    def _ensure_arms(self, population):
        population = list(population)
        if population == self._arms:
            return
        if not population:
            self._arms = []
            self._arm_to_index = {}
            self._weights = np.array([], dtype=float)
            self._probabilities = np.array([], dtype=float)
            return
        self._arms = population
        self._arm_to_index = {arm: idx for idx, arm in enumerate(population)}
        self._weights = np.ones(len(population), dtype=float)
        self._refresh_probabilities()

    def _refresh_probabilities(self):
        total_weight = self._weights.sum()
        if not np.isfinite(total_weight) or total_weight <= 0:
            self._weights.fill(1.0)
            total_weight = self._weights.sum()
        nb_arms = len(self._weights)
        gamma = self._resolve_gamma(nb_arms)
        exploitation = (1 - gamma) * (self._weights / total_weight)
        exploration = gamma / nb_arms
        self._probabilities = exploitation + exploration
        self._probabilities /= self._probabilities.sum()

    def sample(self, population, k, rng=None):
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > len(population):
            raise ValueError("k cannot exceed population size")
        if k == 0:
            return []
        self._ensure_arms(population)
        indices = self._rng.choice(
            len(self._arms),
            size=k,
            replace=False,
            p=self._probabilities,
        )
        return [self._arms[int(index)] for index in indices]

    def update(self, population, rewards) -> None:
        population = list(population)
        rewards = list(rewards)
        if not population:
            return None
        has_unknown_arm = any(arm not in self._arm_to_index for arm in population)
        if len(self._arms) == 0 or has_unknown_arm:
            self._ensure_arms(population)
        gamma = self._resolve_gamma(len(self._arms))
        for arm, reward in zip(population, rewards, strict=True):
            arm_index = self._arm_to_index[arm]
            normalized_reward = (float(reward) - self.lower) / self.amplitude
            normalized_reward = min(1.0, max(0.0, normalized_reward))
            estimated_reward = normalized_reward / self._probabilities[arm_index]
            self._weights[arm_index] *= math.exp(
                gamma * estimated_reward / len(self._arms)
            )
        self._refresh_probabilities()
        return None

    def probabilities(self, population, k=None) -> dict[Any, float]:
        self._ensure_arms(population)
        if len(self._arms) == 0:
            return {}
        return {
            arm: float(self._probabilities[index])
            for index, arm in enumerate(self._arms)
        }


# Backwards-compatible alias for older tests/imports.
Exp3Sampler = Exp3NeighborSampler


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
    if name == "exp3":
        horizon = params.pop("horizon", None)
        if horizon is None and context is not None:
            horizon = context.horizon
        return Exp3NeighborSampler(
            gamma=params.pop("gamma", "auto"),
            lower=float(params.pop("lower", 0.0)),
            amplitude=float(params.pop("amplitude", 1.0)),
            seed=seed,
            horizon=horizon,
        )
    raise ValueError(f"Unknown neighbor sampler: {name}")
