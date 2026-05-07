from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class EngineRunConfig:
    params: dict[str, Any]
    run_mode: str
    run_name: str
    nb_neighbors: int
    byzantine_budget: int


def _get(section: DictConfig, *names: str, default=None):
    for name in names:
        if name in section:
            return section.get(name)
    return default


def _nodes(cfg: DictConfig) -> int:
    nodes = OmegaConf.select(cfg, "topology.nodes")
    if nodes is None:
        nodes = OmegaConf.select(cfg, "nodes")
    if nodes is None:
        raise ValueError("Missing topology.nodes")
    return int(nodes)


def is_dynamic_topology(topology_cfg: DictConfig) -> bool:
    has_sampling = "sampling" in topology_cfg
    has_degree = "degree" in topology_cfg
    if has_sampling == has_degree:
        raise ValueError("Topology must define exactly one of 'sampling' or 'degree'")
    return has_sampling


def _neighbor_count(cfg: DictConfig, nodes: int, is_dynamic: bool) -> int:
    if is_dynamic:
        sampling = float(cfg.topology.sampling)
        return max(1, min(nodes - 1, int(round((nodes - 1) * sampling))))
    return int(cfg.topology.degree)


def _sampler_name(cfg: DictConfig) -> str:
    if "sampler" in cfg:
        return str(cfg.sampler.name)
    return str(cfg.topology.neighbor_sampler)


def _sampler_params(cfg: DictConfig) -> dict[str, Any]:
    if "sampler" not in cfg or "params" not in cfg.sampler:
        return {}
    params = OmegaConf.to_container(cfg.sampler.params, resolve=True)
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("sampler.params must be a mapping")
    return params


def _sampler_reward(cfg: DictConfig) -> str:
    if "sampler" in cfg:
        return str(cfg.sampler.get("reward", "parameter_distance"))
    return str(cfg.topology.get("bandit_reward", "parameter_distance"))


def _run_name(cfg: DictConfig, byzantine_budget: int, nb_neighbors: int) -> str:
    nodes = _nodes(cfg)
    is_dynamic = is_dynamic_topology(cfg.topology)
    sampler = _sampler_name(cfg)
    if is_dynamic:
        topology_token = f"-sampling_{cfg.topology.sampling}"
    else:
        topology_token = f"-degree_{nb_neighbors}"
    base = (
        f"{cfg.dataset.dataset}-n_{nodes}"
        f"-model_{cfg.dataset.model}"
        f"-attack_{cfg.adversary.attack}"
        f"-agg_{cfg.aggregator.aggregator}"
        f"{topology_token}"
        f"-sampler_{sampler}"
        f"-f_{cfg.adversary.byzcount}"
        f"-alpha_{cfg.heterogeneity.alpha}"
        f"-byz_budget_{byzantine_budget}"
        f"-nb-local_{cfg.optimization.nb_local_steps}"
    )
    if sampler != "uniform":
        sampler_params = _sampler_params(cfg)
        param_token = "_".join(
            f"{key}_{value}" for key, value in sampler_params.items()
        )
        if param_token:
            base += f"-{param_token}"
    return base


def build_engine_config(cfg: DictConfig) -> EngineRunConfig:
    nodes = _nodes(cfg)
    is_dynamic = is_dynamic_topology(cfg.topology)
    nb_neighbors = _neighbor_count(cfg, nodes, is_dynamic)
    sampler = _sampler_name(cfg)
    rounds = cfg.optimization.get("rounds", cfg.optimization.get("nb_steps"))
    if rounds is None:
        raise ValueError("Missing optimization.rounds")

    params: dict[str, Any] = {
        "dataset": cfg.dataset.dataset,
        "model": cfg.dataset.model,
        "nb-workers": nodes,
        "dirichlet-alpha": float(cfg.heterogeneity.alpha),
        "nb-decl-byz": int(cfg.adversary.byzcount),
        "nb-real-byz": int(cfg.adversary.byzcount),
        "nb-neighbors": nb_neighbors,
        "nb-local-steps": int(cfg.optimization.nb_local_steps),
        "neighbor-sampler": sampler,
        "sampler-params": _sampler_params(cfg),
        "sampler-reward": _sampler_reward(cfg),
        "batch-size": int(cfg.optimization.batch_size),
        "loss": cfg.optimization.loss,
        "weight-decay": float(cfg.optimization.weight_decay),
        "momentum-worker": float(cfg.optimization.momentum_worker),
        "rounds": int(rounds),
        "aggregator": cfg.aggregator.aggregator,
        "pre-aggregator": _get(cfg.aggregator, "pre-aggregator", "pre_aggregator"),
        "rag": bool(cfg.aggregator.rag),
        "numb-labels": int(cfg.heterogeneity.numb_labels),
        "evaluation-delta": int(cfg.evaluation.evaluation_delta),
    }

    learning_rate = cfg.optimization.get("learning_rate")
    if learning_rate is not None:
        params["learning-rate"] = float(learning_rate)
    learning_rate_decay = cfg.optimization.get("learning_rate_decay")
    if learning_rate_decay is not None:
        params["learning-rate-decay"] = int(learning_rate_decay)
    learning_rate_decay_delta = _get(
        cfg.optimization, "learning_rate_decay_delta", "learning_rate_decay-delta"
    )
    if learning_rate_decay_delta is not None:
        params["learning-rate-decay-delta"] = int(learning_rate_decay_delta)

    attack = cfg.adversary.get("attack")
    if attack is not None:
        params["attack"] = attack
    mimic_learning_phase = _get(
        cfg.adversary, "mimic_learning_phase", "mimic-learning-phase"
    )
    if mimic_learning_phase is not None:
        params["mimic-learning-phase"] = int(mimic_learning_phase)

    byz_budget_raw = cfg.adversary.get("byzantine_budget")
    byzantine_budget = int(
        cfg.adversary.byzcount if byz_budget_raw is None else byz_budget_raw
    )
    params["b-hat"] = byzantine_budget

    if is_dynamic:
        params["rag"] = True
        params["sampling-ratio"] = float(cfg.topology.sampling)
    else:
        params["method"] = cfg.topology.get("method", sampler)

    return EngineRunConfig(
        params=params,
        run_mode="dynamic" if is_dynamic else "fixed",
        run_name=_run_name(cfg, byzantine_budget, nb_neighbors),
        nb_neighbors=nb_neighbors,
        byzantine_budget=byzantine_budget,
    )


def resolve_device(cfg: DictConfig) -> str:
    configured = str(cfg.device)
    if configured and configured != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
