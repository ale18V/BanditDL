from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import DictConfig, OmegaConf

from banditdl.core.worker.config import WorkerConfig
from banditdl.experiments.config_schema import BanditDLConfig


@dataclass(frozen=True)
class EngineRunConfig:
    config: BanditDLConfig
    run_mode: str
    run_name: str
    nb_neighbors: int
    byzantine_budget: int

    def to_worker_config(self, device: str) -> WorkerConfig:
        """Convert engine parameters to a structured WorkerConfig."""
        c = self.config
        is_dynamic = self.run_mode == "dynamic"

        sampler_name = (
            c.sampler.get("name", c.topology.neighbor_sampler)
            if c.sampler
            else c.topology.neighbor_sampler
        )

        return WorkerConfig(
            model=c.dataset.model,
            learning_rate=c.optimization.learning_rate,
            learning_rate_decay=c.optimization.learning_rate_decay,
            learning_rate_decay_delta=c.optimization.learning_rate_decay_delta,
            weight_decay=c.optimization.weight_decay,
            loss=c.optimization.loss,
            momentum=c.optimization.momentum_worker,
            device=device,
            nb_local_steps=c.optimization.nb_local_steps,
            nb_workers=c.topology.nodes,
            nb_byz=c.adversary.byzcount,
            nb_real_byz=c.adversary.byzcount,
            b_hat=self.byzantine_budget,
            attack=c.adversary.attack,
            rag=c.aggregator.rag or is_dynamic,
            numb_labels=c.dataset.numb_labels,
            labelflipping=c.adversary.attack == "LF",
            gradient_clip=None,
            server_clip=c.aggregator.server_clip,
            bucket_size=c.aggregator.bucket_size,
            aggregator=c.aggregator.aggregator,
            pre_aggregator=c.aggregator.pre_aggregator,
            nb_neighbors=self.nb_neighbors,
            sampling_ratio=c.topology.sampling,
            neighbor_sampler=None,
            reward_strategy=None,
            mimic_learning_phase=c.adversary.mimic_learning_phase,
            method=c.topology.method or sampler_name,
            comm_graph=None,
            dissensus=c.adversary.attack == "dissensus",
            epsilon=1.0,
        )


def build_engine_config(cfg: DictConfig) -> EngineRunConfig:
    # Use the user's preferred pattern to get a typed object
    merged = OmegaConf.merge(OmegaConf.structured(BanditDLConfig), cfg)
    OmegaConf.resolve(merged)
    obj: BanditDLConfig = OmegaConf.to_object(merged)

    is_dynamic = obj.topology.sampling is not None
    nb_neighbors = (
        max(1, min(obj.topology.nodes - 1, round((obj.topology.nodes - 1) * obj.topology.sampling)))
        if is_dynamic
        else obj.topology.degree
    )

    byz_budget = (
        obj.adversary.byzantine_budget
        if obj.adversary.byzantine_budget is not None
        else obj.adversary.byzcount
    )

    return EngineRunConfig(
        config=obj,
        run_mode="dynamic" if is_dynamic else "fixed",
        run_name=_run_name(obj, byz_budget, nb_neighbors),
        nb_neighbors=nb_neighbors,
        byzantine_budget=byz_budget,
    )


def _run_name(c: BanditDLConfig, byz_budget: int, nb_neighbors: int) -> str:
    sampler = (
        c.sampler.get("name", c.topology.neighbor_sampler)
        if c.sampler
        else c.topology.neighbor_sampler
    )
    topology_token = (
        f"-sampling_{c.topology.sampling}"
        if c.topology.sampling is not None
        else f"-degree_{nb_neighbors}"
    )

    return (
        f"{c.dataset.dataset}-n_{c.topology.nodes}"
        f"-model_{c.dataset.model}"
        f"-attack_{c.adversary.attack}"
        f"-agg_{c.aggregator.aggregator}"
        f"{topology_token}"
        f"-sampler_{sampler}"
        f"-f_{c.adversary.byzcount}"
        f"-{_partition_token(c)}"
        f"-byz_budget_{byz_budget}"
        f"-nb-local_{c.optimization.nb_local_steps}"
    )


def _partition_token(c: BanditDLConfig) -> str:
    if c.dataset.mode == "writer_per_node":
        return _femnist_writer_token(c)

    method = c.heterogeneity.method
    # Common prefix
    prefix = f"{method}"
    if c.heterogeneity.clusters and c.heterogeneity.clusters < c.topology.nodes:
        prefix = f"clustered_{c.heterogeneity.clusters}_{prefix}"

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
    if c.dataset.dataset != "femnist":
        raise ValueError("dataset.mode='writer_per_node' is only valid for FEMNIST")
    cap = c.dataset.nb_writers_limit
    return "femnist_writers" if cap is None else f"femnist_writers_cap_{cap}"


def resolve_device(cfg: DictConfig) -> str:
    configured = str(cfg.device)
    if configured and configured != "auto":
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"
