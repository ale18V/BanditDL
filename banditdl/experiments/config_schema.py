from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    dataset: str = "mnist"
    model: str = "MnistNet"
    numb_labels: int = 10
    provider: dict[str, Any] = field(default_factory=dict)
    partitioner: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    learning_rate: float = 0.5
    learning_rate_decay: int = 5000
    learning_rate_decay_delta: int = 1
    weight_decay: float = 0.0
    loss: str = "NLLLoss"
    momentum_worker: float = 0.99
    nb_local_steps: int = 1
    batch_size: int = 32
    rounds: int = 100
    nb_steps: int | None = None


@dataclass
class AdversaryConfig:
    attack: str | None = None
    byzcount: int = 0
    byzantine_budget: int | None = None
    mimic_learning_phase: int | None = None


@dataclass
class AggregatorConfig:
    aggregator: str = "average"
    pre_aggregator: str | None = None
    rag: bool = False
    server_clip: bool = False
    bucket_size: int = 1


@dataclass
class TopologyConfig:
    nodes: int = 10
    sampling: float = 0.2


@dataclass
class EvaluationConfig:
    evaluation_delta: int = 100
    evaluate_test: bool = False
    global_test_ratio: float = 0.1
    local_test_ratio: float = 0.2
    split_seed: int = 0


@dataclass
class HeterogeneityConfig:
    _target_: str = "banditdl.data.partitioning.SyntheticPartitionStrategy"
    method: str = "dirichlet"
    clusters: int | None = None  # Number of clusters. Defaults to topology.nodes.

    alpha: float | None = None
    classes_per_group: int | None = None
    group_overlap: int = 0
    gamma_similarity: float | None = None


@dataclass
class BanditDLConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    adversary: AdversaryConfig = field(default_factory=AdversaryConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    heterogeneity: HeterogeneityConfig = field(default_factory=HeterogeneityConfig)

    seed: int = 42
    num_seeds: int = 1
    device: str = "auto"
    sampler: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_sampler_name(self) -> str:
        return str(self.sampler.get("name", "uniform"))

    @property
    def resolved_reward_name(self) -> str:
        return str(self.sampler.get("reward", "parameter_distance"))

    @property
    def uses_natural_partition(self) -> bool:
        target = str(self.dataset.partitioner.get("_target_", ""))
        return target.endswith("NaturalOwnerPartitionStrategy")

    @property
    def partitioner_config(self) -> dict[str, Any]:
        return self.dataset.partitioner or vars(self.heterogeneity)

    @property
    def total_nodes(self) -> int:
        return self.topology.nodes

    @property
    def nb_honests(self) -> int:
        return self.topology.nodes - self.adversary.byzcount

    @property
    def resolved_clusters(self) -> int:
        return self.heterogeneity.clusters or self.nb_honests

    @property
    def effective_rounds(self) -> int:
        if self.optimization.nb_steps is not None:
            return self.optimization.nb_steps
        return self.optimization.rounds
