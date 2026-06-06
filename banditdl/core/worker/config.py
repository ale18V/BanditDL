from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for a decentralized worker."""

    # Core training params
    model: str
    learning_rate: float
    learning_rate_decay: int
    learning_rate_decay_delta: int
    weight_decay: float
    loss: str
    momentum: float
    device: str
    nb_local_steps: int

    # Topology and Byzantine params
    nb_workers: int
    nb_byz: int
    nb_real_byz: int
    b_hat: int
    attack: str | None = None
    rag: bool = False

    # Data params
    numb_labels: int | None = None
    labelflipping: bool = False

    # Optimization/Stability
    aggregator: str = "average"
    pre_aggregator: str | None = None
    gradient_clip: float | None = None
    server_clip: bool = False
    bucket_size: int = 1

    # Sampling params
    sampling_ratio: float | None = None
    neighbor_sampler: Any | None = None
    reward_strategy: Any | None = None
    mimic_learning_phase: int | None = None
