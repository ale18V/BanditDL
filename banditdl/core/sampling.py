from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from mabwiser.mab import MAB, LearningPolicy

DEFAULT_DIAGNOSTIC_SAMPLES = 256


@dataclass(frozen=True)
class SamplerContext:
    worker_id: int
    nodes: int
    k: int
    horizon: int
    seed: int


@dataclass(frozen=True)
class SamplerDiagnostics:
    weights: dict[Any, float]
    probabilities: dict[Any, float]


class RewardStrategy(ABC):
    @abstractmethod
    def score(self, local_weights, neighbor_weights) -> list[float]:
        """Compute one reward per selected neighbor."""


class ParameterDistanceReward(RewardStrategy):
    def score(self, local_weights, neighbor_weights) -> list[float]:
        return [1 / (1 + torch.norm(weight - local_weights).item()) for weight in neighbor_weights]


class CosineSimilarityReward(RewardStrategy):
    def score(self, local_weights, neighbor_weights) -> list[float]:
        if not neighbor_weights:
            return []
        neighbors = torch.stack(neighbor_weights)
        local = local_weights.unsqueeze(0).expand_as(neighbors)
        similarities = F.cosine_similarity(neighbors, local, dim=1, eps=1e-12)
        zero_norm = (neighbors.norm(dim=1) == 0) | (local_weights.norm() == 0)
        similarities = torch.where(zero_norm, 0.0, similarities)
        return ((similarities + 1) / 2).clamp(0, 1).tolist()


def make_reward_strategy(name):
    if name == "parameter_distance":
        return ParameterDistanceReward()
    if name == "cosine_similarity":
        return CosineSimilarityReward()
    raise ValueError(f"Unknown bandit reward strategy: {name}")


def _validate_sample(population, k):
    population = list(population)
    if k < 0:
        raise ValueError("k must be non-negative")
    if k > len(population):
        raise ValueError("k cannot exceed population size")
    return population


def _normalize(arms, values) -> dict[Any, float]:
    values = np.asarray(values, dtype=float)
    infinite = np.isposinf(values)
    if infinite.any():
        values = infinite.astype(float)
    else:
        values = np.maximum(values, 0)
    total = values.sum()
    if total <= 0 or not np.isfinite(total):
        values = np.ones(len(arms), dtype=float)
        total = len(arms)
    return {arm: float(values[index] / total) for index, arm in enumerate(arms)}


def _top_k_mass(arms, selected, k) -> dict[Any, float]:
    if not arms:
        return {}
    if k == 0:
        return {arm: 0.0 for arm in arms}
    selected = set(selected)
    return {arm: (1.0 / k if arm in selected else 0.0) for arm in arms}


class UniformNeighborSampler:
    """Uniformly sample neighbors without replacement."""

    def sample(self, population, k, rng=None):
        population = _validate_sample(population, k)
        if rng is None:
            return random.sample(population, k)
        return rng.sample(population, k)

    def update(self, population, rewards) -> None:
        return None

    def diagnostics(self, population, k) -> SamplerDiagnostics:
        population = list(population)
        if not population:
            return SamplerDiagnostics({}, {})
        probability = 1.0 / len(population)
        uniform = {arm: probability for arm in population}
        return SamplerDiagnostics(uniform, uniform)

    def probabilities(self, population, k=None) -> dict[Any, float]:
        return self.diagnostics(population, k or 1).probabilities


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
        population = _validate_sample(population, k)
        if k == 0:
            return []

        rng = rng or random
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

    def diagnostics(self, population, k) -> SamplerDiagnostics:
        population = list(population)
        if not population:
            return SamplerDiagnostics({}, {})
        self._ensure_mab(population)
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
        weights = _normalize(
            population,
            [expectations.get(arm, self.initial_value) for arm in population],
        )
        return SamplerDiagnostics(weights, probabilities)

    def probabilities(self, population, k=None) -> dict[Any, float]:
        return self.diagnostics(population, k or 1).probabilities


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
        diagnostic_samples=DEFAULT_DIAGNOSTIC_SAMPLES,
    ):
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        self.gamma = gamma
        self.lower = float(lower)
        self.amplitude = float(amplitude)
        self.seed = int(seed)
        self.horizon = None if horizon is None else int(horizon)
        self._rng = np.random.default_rng(self.seed)
        self._diagnostic_rng = np.random.default_rng(self.seed + 1_000_003)
        self.diagnostic_samples = int(diagnostic_samples)
        if self.diagnostic_samples <= 0:
            raise ValueError("diagnostic_samples must be positive")
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
        population = _validate_sample(population, k)
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
            self._weights[arm_index] *= math.exp(gamma * estimated_reward / len(self._arms))
        self._refresh_probabilities()
        return None

    def diagnostics(self, population, k) -> SamplerDiagnostics:
        self._ensure_arms(population)
        if len(self._arms) == 0:
            return SamplerDiagnostics({}, {})
        gumbels = self._diagnostic_rng.gumbel(size=(self.diagnostic_samples, len(self._arms)))
        scores = np.log(np.maximum(self._probabilities, 1e-300)) + gumbels
        selected = np.argpartition(scores, -k, axis=1)[:, -k:]
        counts = np.bincount(selected.ravel(), minlength=len(self._arms))
        probabilities = _normalize(self._arms, counts)
        return SamplerDiagnostics(
            _normalize(self._arms, self._weights),
            probabilities,
        )

    def probabilities(self, population, k=None) -> dict[Any, float]:
        return self.diagnostics(population, k or 1).probabilities


class CUCBNeighborSampler:
    def __init__(self, exploration=2.0, discount=1.0, seed=123456):
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        self.exploration = float(exploration)
        self.discount = float(discount)
        self._rng = np.random.default_rng(seed)
        self._arms: list[Any] = []
        self._index: dict[Any, int] = {}
        self._counts = np.array([], dtype=float)
        self._reward_sums = np.array([], dtype=float)
        self._time = 0.0
        self._last_scores = np.array([], dtype=float)
        self._last_selected: list[Any] = []

    def _ensure_arms(self, population):
        population = list(population)
        if population == self._arms:
            return
        self._arms = population
        self._index = {arm: index for index, arm in enumerate(population)}
        self._counts = np.zeros(len(population), dtype=float)
        self._reward_sums = np.zeros(len(population), dtype=float)
        self._last_scores = np.array([], dtype=float)
        self._last_selected = []

    def _scores(self):
        scores = np.full(len(self._arms), np.inf)
        observed = self._counts > 0
        if observed.any():
            means = self._reward_sums[observed] / self._counts[observed]
            bonus = np.sqrt(
                self.exploration * math.log(max(2.0, self._time + 1.0)) / self._counts[observed]
            )
            scores[observed] = means + bonus
        return scores

    def sample(self, population, k, rng=None):
        population = _validate_sample(population, k)
        self._ensure_arms(population)
        if k == 0:
            return []
        self._last_scores = self._scores()
        tie_break = self._rng.random(len(self._arms))
        order = np.lexsort((tie_break, -self._last_scores))
        self._last_selected = [self._arms[index] for index in order[:k]]
        return self._last_selected.copy()

    def update(self, population, rewards) -> None:
        if not population:
            return
        self._counts *= self.discount
        self._reward_sums *= self.discount
        self._time = self.discount * self._time + 1
        for arm, reward in zip(population, rewards, strict=True):
            index = self._index[arm]
            self._counts[index] += 1
            self._reward_sums[index] += float(np.clip(reward, 0, 1))

    def diagnostics(self, population, k) -> SamplerDiagnostics:
        self._ensure_arms(population)
        scores = self._last_scores if len(self._last_scores) else self._scores()
        selected = self._last_selected
        if not selected and k:
            selected = self.sample(population, k)
        return SamplerDiagnostics(
            _normalize(self._arms, scores),
            _top_k_mass(self._arms, selected, k),
        )

    def probabilities(self, population, k=None) -> dict[Any, float]:
        return self.diagnostics(population, k or 1).probabilities


class CTSNeighborSampler:
    def __init__(
        self,
        discount=1.0,
        seed=123456,
        diagnostic_samples=DEFAULT_DIAGNOSTIC_SAMPLES,
    ):
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        self.discount = float(discount)
        self._rng = np.random.default_rng(seed)
        self._diagnostic_rng = np.random.default_rng(seed + 1_000_003)
        self.diagnostic_samples = int(diagnostic_samples)
        if self.diagnostic_samples <= 0:
            raise ValueError("diagnostic_samples must be positive")
        self._arms: list[Any] = []
        self._index: dict[Any, int] = {}
        self._successes = np.array([], dtype=float)
        self._failures = np.array([], dtype=float)
        self._last_scores = np.array([], dtype=float)

    def _ensure_arms(self, population):
        population = list(population)
        if population == self._arms:
            return
        self._arms = population
        self._index = {arm: index for index, arm in enumerate(population)}
        self._successes = np.zeros(len(population), dtype=float)
        self._failures = np.zeros(len(population), dtype=float)
        self._last_scores = np.array([], dtype=float)

    def _posterior(self):
        return 1 + self._successes, 1 + self._failures

    def sample(self, population, k, rng=None):
        population = _validate_sample(population, k)
        self._ensure_arms(population)
        if k == 0:
            return []
        alpha, beta = self._posterior()
        self._last_scores = self._rng.beta(alpha, beta)
        indices = np.argpartition(self._last_scores, -k)[-k:]
        return [self._arms[index] for index in indices]

    def update(self, population, rewards) -> None:
        if not population:
            return
        self._successes *= self.discount
        self._failures *= self.discount
        for arm, reward in zip(population, rewards, strict=True):
            value = float(np.clip(reward, 0, 1))
            index = self._index[arm]
            self._successes[index] += value
            self._failures[index] += 1 - value

    def diagnostics(self, population, k) -> SamplerDiagnostics:
        self._ensure_arms(population)
        alpha, beta = self._posterior()
        scores = self._last_scores if len(self._last_scores) else alpha / (alpha + beta)
        draws = self._diagnostic_rng.beta(
            alpha,
            beta,
            size=(self.diagnostic_samples, len(self._arms)),
        )
        selected = np.argpartition(draws, -k, axis=1)[:, -k:]
        counts = np.bincount(selected.ravel(), minlength=len(self._arms))
        return SamplerDiagnostics(
            _normalize(self._arms, scores),
            _normalize(self._arms, counts),
        )

    def probabilities(self, population, k=None) -> dict[Any, float]:
        return self.diagnostics(population, k or 1).probabilities


# Backwards-compatible alias for older tests/imports.
Exp3Sampler = Exp3NeighborSampler


def make_neighbor_sampler(
    name,
    *,
    context: SamplerContext | None = None,
    params: dict[str, Any] | None = None,
):
    params = dict(params or {})
    seed = params.pop("seed", context.seed if context is not None else 123456)

    if name == "uniform":
        return UniformNeighborSampler()
    if name in {"bandit", "epsilon_greedy"}:
        epsilon = float(params.pop("epsilon", 0.1))
        initial_value = float(params.pop("initial_value", 0.0))
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
            diagnostic_samples=int(params.pop("diagnostic_samples", DEFAULT_DIAGNOSTIC_SAMPLES)),
        )
    if name in {"cucb", "discounted_cucb"}:
        discount = 1.0 if name == "cucb" else float(params.pop("gamma", 0.99))
        return CUCBNeighborSampler(
            exploration=float(params.pop("exploration", 2.0)),
            discount=discount,
            seed=seed,
        )
    if name in {"cts", "discounted_cts"}:
        discount = 1.0 if name == "cts" else float(params.pop("gamma", 0.99))
        return CTSNeighborSampler(
            discount=discount,
            seed=seed,
            diagnostic_samples=int(params.pop("diagnostic_samples", DEFAULT_DIAGNOSTIC_SAMPLES)),
        )
    raise ValueError(f"Unknown neighbor sampler: {name}")
