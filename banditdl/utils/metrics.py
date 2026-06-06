from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np


class MetricKey(StrEnum):
    VALIDATION_ACCURACIES = "validation_accuracies"
    VALIDATION_LOSSES = "validation_losses"
    TRAIN_LOSSES = "train_losses"
    REWARD_ALGORITHM = "reward_algorithm"
    REWARD_ORACLE = "reward_oracle"
    REWARD_SELECTED_MIN = "reward_selected_min"
    REWARD_SELECTED_MAX = "reward_selected_max"
    REGRET = "regret"
    NORMALIZED_REGRET = "normalized_regret"
    NEIGHBOR_DISAGREEMENT = "neighbor_disagreement"
    CONSENSUS_DRIFT = "consensus_drift"
    GRADIENT_NORMS = "gradient_norms"
    SAMPLER_PROBABILITIES = "sampler_probabilities"
    SAMPLER_KL_TO_UNIFORM = "sampler_kl_to_uniform"
    SAMPLER_MIN_PROBABILITY = "sampler_min_probability"
    SAMPLER_MAX_PROBABILITY = "sampler_max_probability"
    VALIDATION = "validation"
    VALIDATION_WORST = "validation_worst"
    VALIDATION_LOSS = "validation_loss"
    TRAIN_LOSS = "train_loss"
    TEST = "test"


ALIASES = {
    "accuracies": MetricKey.VALIDATION_ACCURACIES,
    "val_accuracy": MetricKey.VALIDATION_ACCURACIES,
    "eval": MetricKey.VALIDATION,
    "eval_worst": MetricKey.VALIDATION_WORST,
    "loss": MetricKey.VALIDATION_LOSS,
}


NPY_CANDIDATES = {
    MetricKey.VALIDATION_ACCURACIES: ("validation_accuracies.npy", "accuracies.npy"),
}


TEXT_CANDIDATES = {
    MetricKey.VALIDATION: ("validation", "eval"),
    MetricKey.VALIDATION_WORST: ("validation_worst", "eval_worst"),
    MetricKey.VALIDATION_LOSS: ("validation_loss",),
    MetricKey.TRAIN_LOSS: ("train_loss",),
    MetricKey.TEST: ("test",),
}


HIGHER_IS_WORSE = {
    MetricKey.REGRET,
    MetricKey.NORMALIZED_REGRET,
    MetricKey.NEIGHBOR_DISAGREEMENT,
    MetricKey.CONSENSUS_DRIFT,
    MetricKey.GRADIENT_NORMS,
    MetricKey.VALIDATION_LOSSES,
    MetricKey.TRAIN_LOSSES,
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
    MetricKey.SAMPLER_MIN_PROBABILITY,
    MetricKey.SAMPLER_MAX_PROBABILITY,
}


def resolve_metric(metric: MetricKey | str) -> MetricKey:
    if isinstance(metric, MetricKey):
        return metric
    key = str(metric)
    return ALIASES.get(key, MetricKey(key))


def sampler_probability_summary(metric: MetricKey, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim < 2:
        raise ValueError("sampler probabilities must have worker and arm axes")

    workers, arms = probabilities.shape[-2:]
    mask = np.ones((workers, arms), dtype=bool)
    for worker_id in range(min(workers, arms)):
        mask[worker_id, worker_id] = False
    valid = probabilities[..., mask].reshape(*probabilities.shape[:-2], workers, arms - 1)

    if metric == MetricKey.SAMPLER_MIN_PROBABILITY:
        return np.nanmin(valid, axis=-1)
    if metric == MetricKey.SAMPLER_MAX_PROBABILITY:
        return np.nanmax(valid, axis=-1)
    if metric == MetricKey.SAMPLER_KL_TO_UNIFORM:
        uniform = 1.0 / (arms - 1)
        safe = np.maximum(valid, 1e-12)
        return np.nansum(valid * np.log(safe / uniform), axis=-1)
    raise ValueError(f"{metric.value} is not a sampler probability summary")


def trim_unwritten_probability_rounds(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    time_axis = 1 if probabilities.ndim == 4 else 0
    other_axes = tuple(axis for axis in range(probabilities.ndim) if axis != time_axis)
    completed = ~np.all(np.isnan(probabilities), axis=other_axes)
    completed_indices = np.flatnonzero(completed)
    if completed_indices.size == 0:
        return np.take(probabilities, [], axis=time_axis)
    return np.take(
        probabilities,
        np.arange(completed_indices[-1] + 1),
        axis=time_axis,
    )


def read_eval(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    skipped = 0
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 2:
                # Surface but tolerate corrupted lines (typically interleaved writes
                # when multiple jobs collide on the same hydra run dir).
                skipped += 1
                continue
            try:
                step = float(tokens[0])
                value = float(tokens[1])
            except ValueError:
                skipped += 1
                continue
            steps.append(step)
            values.append(value)
    if skipped:
        import warnings

        warnings.warn(
            f"{path}: skipped {skipped} malformed line(s); the file may be from a "
            "collided run-dir (check hydra.run.dir uniqueness).",
            stacklevel=2,
        )
    return np.asarray(steps), np.asarray(values)


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
        if key in TEXT_CANDIDATES:
            return self._load_text(key)

        values = self.load_values(key)
        x = np.arange(values.shape[_time_axis(values)])
        if interpolate_eval and key == MetricKey.VALIDATION_ACCURACIES:
            source_steps = self._load_text(MetricKey.VALIDATION).x
            x = self.full_round_axis()
            values = interpolate_to_steps(values, source_steps, x)
        return MetricData(x=x, values=values)

    def load_values(self, metric: MetricKey | str) -> np.ndarray:
        key = resolve_metric(metric)
        if key == MetricKey.NORMALIZED_REGRET:
            return TimeAverage()(self.load_values(MetricKey.REGRET))
        if key in TEXT_CANDIDATES:
            return self._load_text(key).values
        if key in SAMPLER_PROBABILITY_DERIVED:
            try:
                return sampler_probability_summary(
                    key, self.load_values(MetricKey.SAMPLER_PROBABILITIES)
                )
            except FileNotFoundError:
                pass

        for filename in NPY_CANDIDATES.get(key, (f"{key.value}.npy",)):
            for path in self._npy_paths(filename):
                if path.exists():
                    values = np.load(path)
                    if values.size == 0:
                        raise ValueError(f"{path} is empty")
                    if key == MetricKey.SAMPLER_PROBABILITIES:
                        values = trim_unwritten_probability_rounds(values)
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
        if key in TEXT_CANDIDATES:
            return self._load_seed_text(key)
        if key in SAMPLER_PROBABILITY_DERIVED:
            try:
                return sampler_probability_summary(
                    key, self.load_seed_values(MetricKey.SAMPLER_PROBABILITIES)
                )
            except FileNotFoundError:
                pass

        for filename in NPY_CANDIDATES.get(key, (f"{key.value}.npy",)):
            by_seed_path, path = self._npy_paths(filename)
            if by_seed_path.exists():
                values = np.load(by_seed_path)
                if values.size == 0:
                    raise ValueError(f"{by_seed_path} is empty")
                if key == MetricKey.SAMPLER_PROBABILITIES:
                    values = trim_unwritten_probability_rounds(values)
                return values
            if path.exists():
                values = np.load(path)
                if values.size == 0:
                    raise ValueError(f"{path} is empty")
                if key == MetricKey.SAMPLER_PROBABILITIES:
                    values = trim_unwritten_probability_rounds(values)
                return values[np.newaxis, ...]
        raise FileNotFoundError(self.run_dir / f"{key.value}.npy")

    def full_round_axis(self) -> np.ndarray:
        for key in (
            MetricKey.NEIGHBOR_DISAGREEMENT,
            MetricKey.CONSENSUS_DRIFT,
            MetricKey.REWARD_ALGORITHM,
            MetricKey.REGRET,
            MetricKey.GRADIENT_NORMS,
            MetricKey.SAMPLER_PROBABILITIES,
            MetricKey.SAMPLER_KL_TO_UNIFORM,
        ):
            path = self.run_dir / f"{key.value}.npy"
            if path.exists():
                return np.arange(self.load_values(key).shape[0])
            by_seed_path = self.run_dir / f"{key.value}_by_seed.npy"
            if by_seed_path.exists():
                return np.arange(self.load_seed_values(key).shape[1])
        validation = self._load_text(MetricKey.VALIDATION)
        return np.arange(int(np.max(validation.x)) + 1)

    def _load_text(self, metric: MetricKey) -> MetricData:
        for filename in TEXT_CANDIDATES[metric]:
            path = self.run_dir / filename
            if path.exists():
                x, values = read_eval(path)
                if values.size == 0:
                    raise ValueError(f"{path} is empty")
                return MetricData(x=x, values=values)
        raise FileNotFoundError(self.run_dir / TEXT_CANDIDATES[metric][0])

    def _load_seed_text(self, metric: MetricKey) -> np.ndarray:
        for filename in TEXT_CANDIDATES[metric]:
            by_seed_path = self.run_dir / f"{filename}_by_seed.npy"
            if by_seed_path.exists():
                values = np.load(by_seed_path)
                if values.size == 0:
                    raise ValueError(f"{by_seed_path} is empty")
                return values
            path = self.run_dir / filename
            if path.exists():
                _, values = read_eval(path)
                if values.size == 0:
                    raise ValueError(f"{path} is empty")
                return values[np.newaxis, ...]
        raise FileNotFoundError(self.run_dir / TEXT_CANDIDATES[metric][0])

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
