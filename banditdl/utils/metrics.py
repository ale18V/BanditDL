from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Protocol

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
        local_time = np.arange(1, len(values) + 1, dtype=float)
        shape = (len(values),) + (1,) * (values.ndim - 1)
        return np.divide(
            values,
            local_time.reshape(shape),
            out=np.full_like(values, np.nan, dtype=float),
            where=np.isfinite(values),
        )


@dataclass(frozen=True)
class Aggregation:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]


mean = Aggregation("average", lambda values: np.nanmean(values, axis=1))
median = Aggregation("median", lambda values: np.nanmedian(values, axis=1))
max_ = Aggregation("max", lambda values: np.nanmax(values, axis=1))
min_ = Aggregation("min", lambda values: np.nanmin(values, axis=1))


def resolve_metric(metric: MetricKey | str) -> MetricKey:
    if isinstance(metric, MetricKey):
        return metric
    key = str(metric)
    return ALIASES.get(key, MetricKey(key))


def read_eval(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            step, value, *_ = line.split()
            steps.append(float(step))
            values.append(float(value))
    return np.asarray(steps), np.asarray(values)


def interpolate_to_steps(
    values: np.ndarray,
    source_steps: np.ndarray,
    target_steps: np.ndarray,
) -> np.ndarray:
    if source_steps.shape[0] != values.shape[0]:
        return values
    if source_steps.shape[0] == target_steps.shape[0] and np.allclose(source_steps, target_steps):
        return values
    if source_steps.shape[0] == 1:
        return np.repeat(values, repeats=target_steps.shape[0], axis=0)

    out = np.zeros((target_steps.shape[0], values.shape[1]), dtype=float)
    for worker_id in range(values.shape[1]):
        out[:, worker_id] = np.interp(target_steps, source_steps, values[:, worker_id])
    return out


class MetricLoader:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def load(self, metric: MetricKey | str, *, interpolate_eval: bool = False) -> MetricData:
        key = resolve_metric(metric)
        if key in TEXT_CANDIDATES:
            return self._load_text(key)

        values = self.load_values(key)
        x = np.arange(values.shape[0])
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

        for filename in NPY_CANDIDATES.get(key, (f"{key.value}.npy",)):
            path = self.run_dir / filename
            if path.exists():
                values = np.load(path)
                if values.size == 0:
                    raise ValueError(f"{path} is empty")
                return values
        raise FileNotFoundError(self.run_dir / f"{key.value}.npy")

    def full_round_axis(self) -> np.ndarray:
        for key in (
            MetricKey.NEIGHBOR_DISAGREEMENT,
            MetricKey.CONSENSUS_DRIFT,
            MetricKey.REWARD_ALGORITHM,
            MetricKey.REGRET,
            MetricKey.SAMPLER_KL_TO_UNIFORM,
        ):
            path = self.run_dir / f"{key.value}.npy"
            if path.exists():
                return np.arange(np.load(path).shape[0])
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


def scalar_reduce(metric: MetricKey | str, values: np.ndarray, direction: str) -> float:
    key = resolve_metric(metric)
    if direction == "avg":
        return float(np.nanmean(values))
    if direction == "worse":
        if key in HIGHER_IS_WORSE:
            return float(np.nanmax(values))
        return float(np.nanmin(values))
    raise ValueError(f"Unsupported direction: {direction}")
