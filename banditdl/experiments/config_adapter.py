from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.config_schema import BanditDLConfig


@dataclass(frozen=True)
class EngineRunConfig:
    config: BanditDLConfig
    run_name: str


def build_engine_config(cfg: DictConfig) -> EngineRunConfig:
    merged = OmegaConf.merge(OmegaConf.structured(BanditDLConfig), cfg)
    OmegaConf.resolve(merged)
    obj: BanditDLConfig = OmegaConf.to_object(merged)
    _validate_config(obj)
    byz_budget = (
        obj.adversary.byzantine_budget
        if obj.adversary.byzantine_budget is not None
        else obj.adversary.byzcount
    )

    return EngineRunConfig(
        config=obj,
        run_name=_run_name(obj, byz_budget),
    )


def _validate_config(c: BanditDLConfig) -> None:
    _validate_topology(c)
    _validate_heterogeneity(c)
    _validate_dataset(c)


def _validate_topology(c: BanditDLConfig) -> None:
    if c.topology.nodes < 2:
        raise ValueError("topology.nodes must be >= 2")
    if not 0 <= c.adversary.byzcount < c.topology.nodes:
        raise ValueError("adversary.byzcount must satisfy 0 <= byzcount < topology.nodes")
    if not 0 < c.topology.sampling <= 1:
        raise ValueError("topology.sampling must be in (0, 1]")
    if c.effective_rounds < 1:
        raise ValueError("optimization.rounds must be >= 1")


def _validate_heterogeneity(c: BanditDLConfig) -> None:
    clusters = c.resolved_clusters
    if clusters <= 0 or clusters > c.nb_honests:
        raise ValueError("heterogeneity.clusters must be between 1 and the honest-node count")
    if c.nb_honests % clusters:
        raise ValueError("the honest-node count must be divisible by heterogeneity.clusters")
    if c.heterogeneity.gamma_similarity is not None and not 0 <= c.heterogeneity.gamma_similarity <= 1:
        raise ValueError("heterogeneity.gamma_similarity must be in [0, 1]")

    method = c.heterogeneity.method
    if method == "dirichlet":
        if c.heterogeneity.alpha is None or c.heterogeneity.alpha <= 0:
            raise ValueError("heterogeneity.alpha must be > 0 for dirichlet partitioning")
    elif method == "pathological":
        classes = c.heterogeneity.classes_per_group
        if classes is None or not 1 <= classes <= c.dataset.numb_labels:
            raise ValueError("classes_per_group must be between 1 and dataset.numb_labels")
        if not 0 <= c.heterogeneity.group_overlap < classes:
            raise ValueError("group_overlap must satisfy 0 <= overlap < classes_per_group")
    else:
        raise ValueError(f"unsupported heterogeneity.method: {method!r}")


def _validate_dataset(c: BanditDLConfig) -> None:
    if c.dataset.mode == "writer_per_node" and c.dataset.dataset != "femnist":
        raise ValueError("dataset.mode='writer_per_node' is only valid for FEMNIST")


def _run_name(c: BanditDLConfig, byz_budget: int) -> str:
    return (
        f"{c.dataset.dataset}_n_{c.topology.nodes}"
        f"_model_{c.dataset.model}"
        f"_attack_{c.adversary.attack}"
        f"_agg_{c.aggregator.aggregator}"
        f"_sampling_{c.topology.sampling}"
        f"_sampler_{c.resolved_sampler_name}"
        f"_f_{c.adversary.byzcount}"
        f"_{_partition_token(c)}"
        f"_byz_budget_{byz_budget}"
        f"_nb_local_{c.optimization.nb_local_steps}"
    )


def _partition_token(c: BanditDLConfig) -> str:
    if c.dataset.mode == "writer_per_node":
        return _femnist_writer_token(c)

    method = c.heterogeneity.method
    # Common prefix
    prefix = f"{method}"
    if c.heterogeneity.clusters is not None:
        prefix = f"clustered_{c.resolved_clusters}_{prefix}"

    details = ""
    if method == "dirichlet":
        if c.heterogeneity.alpha is None:
            raise ValueError("alpha is required for dirichlet")
        details = f"alpha_{c.heterogeneity.alpha}"
    elif method == "pathological":
        if c.heterogeneity.classes_per_group is None:
            raise ValueError("classes_per_group is required for pathological")
        details = f"c_{c.heterogeneity.classes_per_group}"
        if c.heterogeneity.group_overlap:
            details += f"_ov_{c.heterogeneity.group_overlap}"
    if c.heterogeneity.gamma_similarity:
        details += f"_gamma_{c.heterogeneity.gamma_similarity}"

    return f"{prefix}_{details}" if details else prefix


def _femnist_writer_token(c: BanditDLConfig) -> str:
    cap = c.dataset.nb_writers_limit
    return "femnist_writers" if cap is None else f"femnist_writers_cap_{cap}"


def resolve_device(cfg: DictConfig) -> str:
    configured = str(cfg.device)
    if configured and configured != "auto":
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"
