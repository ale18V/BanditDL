from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np


class MetricKey(StrEnum):
    VALIDATION_ACCURACY = "validation_accuracy"
    VALIDATION_LOSS = "validation_loss"
    GLOBAL_ACCURACY = "global_accuracy"
    TRAIN_LOSS = "train_loss"
    TEST_ACCURACY = "test_accuracy"
    REWARD_ALGORITHM = "reward_algorithm"
    REWARD_ORACLE = "reward_oracle"
    REWARD_SELECTED_MIN = "reward_selected_min"
    REWARD_SELECTED_MAX = "reward_selected_max"
    REGRET = "regret"
    NORMALIZED_REGRET = "normalized_regret"
    NEIGHBOR_DISAGREEMENT = "neighbor_disagreement"
    CONSENSUS_DRIFT = "consensus_drift"
    GRADIENT_NORMS = "gradient_norms"
    SAMPLER_WEIGHTS = "sampler_weights"
    SAMPLER_PROBABILITIES = "sampler_probabilities"
    SAMPLER_KL_TO_UNIFORM = "sampler_kl_to_uniform"
    SAMPLER_ENTROPY = "sampler_entropy"
    SAMPLER_MIN_PROBABILITY = "sampler_min_probability"
    SAMPLER_MAX_PROBABILITY = "sampler_max_probability"
    SAMPLER_WEIGHT_KL_TO_UNIFORM = "sampler_weight_kl_to_uniform"
    SAMPLER_WEIGHT_ENTROPY = "sampler_weight_entropy"
    SAMPLER_MIN_WEIGHT = "sampler_min_weight"
    SAMPLER_MAX_WEIGHT = "sampler_max_weight"


ALIASES = {}


NPY_CANDIDATES = {
    MetricKey.VALIDATION_ACCURACY: ("validation_accuracy.npy",),
    MetricKey.VALIDATION_LOSS: ("validation_loss.npy",),
    MetricKey.GLOBAL_ACCURACY: ("global_accuracy.npy",),
    MetricKey.TRAIN_LOSS: ("train_loss.npy",),
    MetricKey.TEST_ACCURACY: ("test_accuracy.npy",),
}


HIGHER_IS_WORSE = {
    MetricKey.REGRET,
    MetricKey.NORMALIZED_REGRET,
    MetricKey.NEIGHBOR_DISAGREEMENT,
    MetricKey.CONSENSUS_DRIFT,
    MetricKey.GRADIENT_NORMS,
    MetricKey.VALIDATION_LOSS,
    MetricKey.TRAIN_LOSS,
}


@dataclass(frozen=True)
class MetricData:
    x: np.ndarray
    values: np.ndarray


class Transform(Protocol):
    def __call__(self, values: np.ndarray) -> np.ndarray: ...


class TimeAverage:
    def __call__(self, values: np.ndarray) -> np.ndarray:
        time_axis = _time_axis(values)
        local_time = np.arange(1, values.shape[time_axis] + 1, dtype=float)
        shape = [1] * values.ndim
        shape[time_axis] = values.shape[time_axis]
        return np.divide(
            values,
            local_time.reshape(tuple(shape)),
            out=np.full_like(values, np.nan, dtype=float),
            where=np.isfinite(values),
        )


@dataclass(frozen=True)
class Aggregation:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]


def _has_seed_axis(values: np.ndarray) -> bool:
    return values.ndim >= 3


def _time_axis(values: np.ndarray) -> int:
    return 1 if _has_seed_axis(values) else 0


def _node_axis(values: np.ndarray) -> int:
    return 2 if _has_seed_axis(values) else 1


def _seed_outer_node_reduce(values: np.ndarray, reducer: Callable) -> np.ndarray:
    if values.ndim < 2:
        return values
    per_seed_or_curve = reducer(values, axis=_node_axis(values))
    if _has_seed_axis(values):
        return np.nanmean(per_seed_or_curve, axis=0)
    return per_seed_or_curve


mean = Aggregation("average", lambda values: _seed_outer_node_reduce(values, np.nanmean))
median = Aggregation("median", lambda values: _seed_outer_node_reduce(values, np.nanmedian))
max_ = Aggregation("max", lambda values: _seed_outer_node_reduce(values, np.nanmax))
min_ = Aggregation("min", lambda values: _seed_outer_node_reduce(values, np.nanmin))

SAMPLER_PROBABILITY_DERIVED = {
    MetricKey.SAMPLER_KL_TO_UNIFORM,
    MetricKey.SAMPLER_ENTROPY,
    MetricKey.SAMPLER_MIN_PROBABILITY,
    MetricKey.SAMPLER_MAX_PROBABILITY,
}

SAMPLER_WEIGHT_DERIVED = {
    MetricKey.SAMPLER_WEIGHT_KL_TO_UNIFORM,
    MetricKey.SAMPLER_WEIGHT_ENTROPY,
    MetricKey.SAMPLER_MIN_WEIGHT,
    MetricKey.SAMPLER_MAX_WEIGHT,
}


def resolve_metric(metric: MetricKey | str) -> MetricKey:
    if isinstance(metric, MetricKey):
        return metric
    key = str(metric)
    return ALIASES.get(key, MetricKey(key))


def sampler_distribution_summary(metric: MetricKey, distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=float)
    if distribution.ndim < 2:
        raise ValueError("sampler distribution must have worker and arm axes")

    workers, arms = distribution.shape[-2:]
    mask = np.ones((workers, arms), dtype=bool)
    for worker_id in range(min(workers, arms)):
        mask[worker_id, worker_id] = False
    valid = distribution[..., mask].reshape(*distribution.shape[:-2], workers, arms - 1)
    totals = np.nansum(valid, axis=-1, keepdims=True)
    valid = np.divide(
        valid,
        totals,
        out=np.full_like(valid, np.nan, dtype=float),
        where=totals > 0,
    )

    if metric in {MetricKey.SAMPLER_MIN_PROBABILITY, MetricKey.SAMPLER_MIN_WEIGHT}:
        return np.nanmin(valid, axis=-1)
    if metric in {MetricKey.SAMPLER_MAX_PROBABILITY, MetricKey.SAMPLER_MAX_WEIGHT}:
        return np.nanmax(valid, axis=-1)
    if metric in {
        MetricKey.SAMPLER_KL_TO_UNIFORM,
        MetricKey.SAMPLER_WEIGHT_KL_TO_UNIFORM,
    }:
        uniform = 1.0 / (arms - 1)
        safe = np.maximum(valid, 1e-12)
        return np.nansum(valid * np.log(safe / uniform), axis=-1)
    if metric in {MetricKey.SAMPLER_ENTROPY, MetricKey.SAMPLER_WEIGHT_ENTROPY}:
        safe = np.maximum(valid, 1e-12)
        return -np.nansum(valid * np.log(safe), axis=-1)
    raise ValueError(f"{metric.value} is not a sampler distribution summary")


def trim_unwritten_rounds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    time_axis = 1 if values.ndim == 4 else 0
    other_axes = tuple(axis for axis in range(values.ndim) if axis != time_axis)
    completed = ~np.all(np.isnan(values), axis=other_axes)
    completed_indices = np.flatnonzero(completed)
    if completed_indices.size == 0:
        return np.take(values, [], axis=time_axis)
    return np.take(
        values,
        np.arange(completed_indices[-1] + 1),
        axis=time_axis,
    )


def interpolate_to_steps(
    values: np.ndarray,
    source_steps: np.ndarray,
    target_steps: np.ndarray,
) -> np.ndarray:
    time_axis = _time_axis(values)
    if source_steps.shape[0] != values.shape[time_axis]:
        return values
    if source_steps.shape[0] == target_steps.shape[0] and np.allclose(source_steps, target_steps):
        return values
    if source_steps.shape[0] == 1:
        return np.repeat(values, repeats=target_steps.shape[0], axis=time_axis)

    moved = np.moveaxis(values, time_axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    interpolated = np.zeros((target_steps.shape[0], flat.shape[1]), dtype=float)
    for series_id in range(flat.shape[1]):
        interpolated[:, series_id] = np.interp(target_steps, source_steps, flat[:, series_id])
    reshaped = interpolated.reshape((target_steps.shape[0], *moved.shape[1:]))
    return np.moveaxis(reshaped, 0, time_axis)


class MetricLoader:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def load(self, metric: MetricKey | str, *, interpolate_eval: bool = False) -> MetricData:
        key = resolve_metric(metric)
        values = self.load_values(key)
        x = np.arange(values.shape[_time_axis(values)])
        if interpolate_eval and key in {
            MetricKey.VALIDATION_ACCURACY,
            MetricKey.VALIDATION_LOSS,
            MetricKey.GLOBAL_ACCURACY,
        }:
            try:
                source_steps = self._load_evaluation_steps()
            except FileNotFoundError:
                pass
            else:
                values = self._align_to_evaluation_steps(values, source_steps)
                x = source_steps
                try:
                    target_steps = self.full_round_axis()
                except FileNotFoundError:
                    pass
                else:
                    values = interpolate_to_steps(values, source_steps, target_steps)
                    x = target_steps
        return MetricData(x=x, values=values)

    def load_values(self, metric: MetricKey | str) -> np.ndarray:
        key = resolve_metric(metric)
        if key == MetricKey.NORMALIZED_REGRET:
            return TimeAverage()(self.load_values(MetricKey.REGRET))
        if key in SAMPLER_PROBABILITY_DERIVED:
            try:
                return sampler_distribution_summary(
                    key, self.load_values(MetricKey.SAMPLER_PROBABILITIES)
                )
            except FileNotFoundError:
                pass
        if key in SAMPLER_WEIGHT_DERIVED:
            return sampler_distribution_summary(key, self.load_values(MetricKey.SAMPLER_WEIGHTS))

        for filename in NPY_CANDIDATES.get(key, (f"{key.value}.npy",)):
            for path in self._npy_paths(filename):
                if path.exists():
                    values = np.load(path)
                    if values.size == 0:
                        raise ValueError(f"{path} is empty")
                    if key in {
                        MetricKey.SAMPLER_PROBABILITIES,
                        MetricKey.SAMPLER_WEIGHTS,
                    }:
                        values = trim_unwritten_rounds(values)
                    return values
        raise FileNotFoundError(self.run_dir / f"{key.value}.npy")

    def load_seed_values(self, metric: MetricKey | str) -> np.ndarray:
        """Load metric values with seed as axis 0.

        Args:
            metric: MetricKey | str
                Metric key or alias to load.
            return: np.ndarray
                Values with an explicit leading seed axis.
        """
        key = resolve_metric(metric)
        if key == MetricKey.NORMALIZED_REGRET:
            return TimeAverage()(self.load_seed_values(MetricKey.REGRET))
        if key in SAMPLER_PROBABILITY_DERIVED:
            try:
                return sampler_distribution_summary(
                    key, self.load_seed_values(MetricKey.SAMPLER_PROBABILITIES)
                )
            except FileNotFoundError:
                pass
        if key in SAMPLER_WEIGHT_DERIVED:
            return sampler_distribution_summary(
                key, self.load_seed_values(MetricKey.SAMPLER_WEIGHTS)
            )

        for filename in NPY_CANDIDATES.get(key, (f"{key.value}.npy",)):
            by_seed_path, path = self._npy_paths(filename)
            if by_seed_path.exists():
                values = np.load(by_seed_path)
                if values.size == 0:
                    raise ValueError(f"{by_seed_path} is empty")
                if key in {
                    MetricKey.SAMPLER_PROBABILITIES,
                    MetricKey.SAMPLER_WEIGHTS,
                }:
                    values = trim_unwritten_rounds(values)
                return values
            if path.exists():
                values = np.load(path)
                if values.size == 0:
                    raise ValueError(f"{path} is empty")
                if key in {
                    MetricKey.SAMPLER_PROBABILITIES,
                    MetricKey.SAMPLER_WEIGHTS,
                }:
                    values = trim_unwritten_rounds(values)
                return values[np.newaxis, ...]
        raise FileNotFoundError(self.run_dir / f"{key.value}.npy")

    def full_round_axis(self) -> np.ndarray:
        max_step: int | None = None
        for key in (
            MetricKey.NEIGHBOR_DISAGREEMENT,
            MetricKey.CONSENSUS_DRIFT,
            MetricKey.REWARD_ALGORITHM,
            MetricKey.REGRET,
            MetricKey.GRADIENT_NORMS,
            MetricKey.SAMPLER_WEIGHTS,
            MetricKey.SAMPLER_PROBABILITIES,
            MetricKey.SAMPLER_KL_TO_UNIFORM,
        ):
            path = self.run_dir / f"{key.value}.npy"
            if path.exists():
                max_step = max(max_step or 0, self.load_values(key).shape[0] - 1)
            by_seed_path = self.run_dir / f"{key.value}_by_seed.npy"
            if by_seed_path.exists():
                max_step = max(max_step or 0, self.load_seed_values(key).shape[1] - 1)
        try:
            evaluation_steps = self._load_evaluation_steps()
        except FileNotFoundError:
            pass
        else:
            max_step = max(max_step or 0, int(np.max(evaluation_steps)))
        if max_step is None:
            raise FileNotFoundError(self.run_dir / "evaluation_steps.npy")
        return np.arange(max_step + 1)

    def _load_evaluation_steps(self) -> np.ndarray:
        path = self.run_dir / "evaluation_steps.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        steps = np.asarray(np.load(path), dtype=float)
        steps = steps[np.isfinite(steps)]
        if steps.size == 0:
            raise ValueError(f"{path} is empty")
        return steps

    def _align_to_evaluation_steps(
        self,
        values: np.ndarray,
        steps: np.ndarray,
    ) -> np.ndarray:
        time_axis = _time_axis(values)
        if values.shape[time_axis] == steps.shape[0]:
            return values
        if values.shape[time_axis] < steps.shape[0]:
            raise ValueError("evaluation_steps.npy has more steps than metric values")
        return np.take(values, np.arange(steps.shape[0]), axis=time_axis)

    def _npy_paths(self, filename: str) -> tuple[Path, Path]:
        stem = Path(filename).stem
        return self.run_dir / f"{stem}_by_seed.npy", self.run_dir / filename


def scalar_reduce(metric: MetricKey | str, values: np.ndarray, direction: str) -> float:
    return scalar_reduce_seed_outer(metric, values[np.newaxis, ...], direction)


def scalar_reduce_seed_outer(metric: MetricKey | str, values: np.ndarray, direction: str) -> float:
    """Reduce seed-stacked metric values with seed averaging as the last step.

    Args:
        metric: MetricKey | str
            Metric key used to decide whether larger values are worse.
        values: np.ndarray
            Metric values with seed as axis 0 and all inner axes belonging to a
            single run.
        direction: str
            Reduction direction, either `avg`, `worse`, or `best`.
        return: float
            Mean across seeds of the per-seed scalar reduction.
    """
    key = resolve_metric(metric)
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    inner_axes = tuple(range(1, values.ndim))
    if direction == "avg":
        per_seed = values if not inner_axes else np.nanmean(values, axis=inner_axes)
        return float(np.nanmean(per_seed))
    if direction == "worse":
        if not inner_axes:
            per_seed = values
        elif key in HIGHER_IS_WORSE:
            per_seed = np.nanmax(values, axis=inner_axes)
        else:
            per_seed = np.nanmin(values, axis=inner_axes)
        return float(np.nanmean(per_seed))
    if direction == "best":
        if not inner_axes:
            per_seed = values
        elif key in HIGHER_IS_WORSE:
            per_seed = np.nanmin(values, axis=inner_axes)
        else:
            per_seed = np.nanmax(values, axis=inner_axes)
        return float(np.nanmean(per_seed))
    raise ValueError(f"Unsupported direction: {direction}")
