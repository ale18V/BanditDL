import pytest
from omegaconf import OmegaConf

from banditdl.experiments.config_adapter import build_engine_config
from banditdl.experiments.engine import _build_worker_config


def _base_cfg():
    return OmegaConf.create(
        {
            "dataset": {"dataset": "mnist", "model": "cnn_mnist", "numb_labels": 10},
            "topology": {"nodes": 30, "sampling": 0.2},
            "sampler": {
                "name": "epsilon_greedy",
                "reward": "parameter_distance",
                "params": {"epsilon": 0.1, "initial_value": 0.0},
            },
            "adversary": {"byzcount": 0, "byzantine_budget": 0, "attack": None},
            "aggregator": {"pre_aggregator": "nnm", "aggregator": "average", "rag": True},
            "heterogeneity": {"method": "dirichlet", "alpha": 0.5, "clusters": None},
            "optimization": {
                "batch_size": 25,
                "loss": "NLLLoss",
                "weight_decay": 1e-4,
                "momentum_worker": 0.9,
                "rounds": 200,
                "nb_local_steps": 1,
            },
            "evaluation": {"evaluation_delta": 20},
            "seed": 123,
            "device": "cpu",
        }
    )


def test_build_dynamic_engine_config_uses_sampler_group():
    run_cfg = build_engine_config(_base_cfg())

    assert run_cfg.config.resolved_sampler_name == "epsilon_greedy"
    assert run_cfg.config.sampler["params"] == {"epsilon": 0.1, "initial_value": 0.0}
    assert run_cfg.config.optimization.rounds == 200


def test_worker_config_receives_adversary_attack():
    cfg = _base_cfg()
    cfg.adversary.byzcount = 1
    cfg.adversary.byzantine_budget = 1
    cfg.adversary.attack = "ALIE"

    worker_cfg = _build_worker_config(build_engine_config(cfg).config, "cpu")

    assert worker_cfg.attack == "ALIE"


def test_build_engine_config_rejects_invalid_sampling_ratio():
    cfg = _base_cfg()
    cfg.topology.sampling = 0

    with pytest.raises(ValueError, match="sampling"):
        build_engine_config(cfg)


def test_build_engine_config_rejects_zero_rounds():
    cfg = _base_cfg()
    cfg.optimization.rounds = 0

    with pytest.raises(ValueError, match="rounds"):
        build_engine_config(cfg)


def test_nodes_include_byzantine_participants():
    cfg = _base_cfg()
    cfg.adversary.byzcount = 2

    config = build_engine_config(cfg).config

    assert config.topology.nodes == 30
    assert config.nb_honests == 28


def test_build_engine_config_dirichlet_default_method():
    run_cfg = build_engine_config(_base_cfg())

    assert run_cfg.config.heterogeneity.method == "dirichlet"
    assert run_cfg.config.heterogeneity.alpha == 0.5
    assert "alpha_0.5" in run_cfg.run_name


def test_build_engine_config_pathological_classes_per_worker():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create(
        {
            "method": "pathological",
            "classes_per_group": 2,
            "clusters": None,
        }
    )

    run_cfg = build_engine_config(cfg)

    assert run_cfg.config.heterogeneity.method == "pathological"
    assert run_cfg.config.heterogeneity.classes_per_group == 2
    assert "pathological_c_2" in run_cfg.run_name


def test_build_engine_config_pathological_grouped_classes():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create(
        {
            "method": "pathological",
            "clusters": 5,
            "classes_per_group": 2,
            "group_overlap": 0,
        }
    )

    run_cfg = build_engine_config(cfg)

    assert run_cfg.config.heterogeneity.method == "pathological"
    assert run_cfg.config.heterogeneity.clusters == 5
    assert run_cfg.config.heterogeneity.classes_per_group == 2
    assert run_cfg.config.heterogeneity.group_overlap == 0
    assert "clustered_5_pathological_c_2" in run_cfg.run_name


def test_build_engine_config_pathological_grouped_classes_with_overlap_token():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create(
        {
            "method": "pathological",
            "clusters": 3,
            "classes_per_group": 3,
            "group_overlap": 1,
        }
    )

    run_cfg = build_engine_config(cfg)

    assert run_cfg.config.heterogeneity.group_overlap == 1
    assert "clustered_3_pathological_c_3_ov_1" in run_cfg.run_name


def test_build_engine_config_femnist_writer_mode_bypasses_alpha():
    cfg = _base_cfg()
    cfg.dataset = OmegaConf.create(
        {
            "dataset": "femnist",
            "model": "cnn_femnist",
            "numb_labels": 62,
            "mode": "writer_per_node",
            "nb_writers_limit": None,
        }
    )
    cfg.heterogeneity = OmegaConf.create(
        {
            "method": "dirichlet",
            "alpha": None,
        }
    )

    run_cfg = build_engine_config(cfg)

    assert run_cfg.config.dataset.dataset == "femnist"
    assert run_cfg.config.dataset.mode == "writer_per_node"
    assert run_cfg.config.dataset.numb_labels == 62
    assert "femnist_writers" in run_cfg.run_name


def test_build_engine_config_femnist_pool_mode_uses_heterogeneity():
    cfg = _base_cfg()
    cfg.dataset = OmegaConf.create(
        {
            "dataset": "femnist",
            "model": "cnn_femnist",
            "numb_labels": 62,
            "mode": "pool",
            "nb_writers_limit": None,
        }
    )

    run_cfg = build_engine_config(cfg)

    assert run_cfg.config.dataset.mode == "pool"
    assert run_cfg.config.heterogeneity.alpha == 0.5
    assert "alpha_0.5" in run_cfg.run_name


def test_build_engine_config_writer_mode_rejects_non_femnist():
    cfg = _base_cfg()
    cfg.dataset.mode = "writer_per_node"

    with pytest.raises(ValueError, match="FEMNIST"):
        build_engine_config(cfg)


def test_build_engine_config_pathological_rejects_missing_classes_per_group():
    cfg = _base_cfg()
    cfg.heterogeneity = OmegaConf.create(
        {
            "method": "pathological",
            "alpha": None,
        }
    )

    with pytest.raises(ValueError, match="classes_per_group"):
        build_engine_config(cfg)
