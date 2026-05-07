from omegaconf import OmegaConf

from banditdl.experiments.config_adapter import build_engine_config


def _base_cfg():
    return OmegaConf.create(
        {
            "dataset": {"dataset": "mnist", "model": "cnn_mnist"},
            "topology": {"nodes": 30, "sampling": 0.2},
            "sampler": {
                "name": "epsilon_greedy",
                "reward": "parameter_distance",
                "params": {"epsilon": 0.1, "initial_value": 0.0},
            },
            "adversary": {"byzcount": 0, "byzantine_budget": 0, "attack": None},
            "aggregator": {"pre-aggregator": "nnm", "aggregator": "average", "rag": True},
            "heterogeneity": {"alpha": 0.5, "numb_labels": 10},
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

    assert run_cfg.run_mode == "dynamic"
    assert run_cfg.nb_neighbors == 6
    assert run_cfg.params["neighbor-sampler"] == "epsilon_greedy"
    assert run_cfg.params["sampler-params"] == {"epsilon": 0.1, "initial_value": 0.0}
    assert run_cfg.params["rounds"] == 200


def test_build_fixed_engine_config_uses_topology_method():
    cfg = _base_cfg()
    del cfg.topology.sampling
    cfg.topology.degree = 15
    cfg.topology.method = "cs+"

    run_cfg = build_engine_config(cfg)

    assert run_cfg.run_mode == "fixed"
    assert run_cfg.nb_neighbors == 15
    assert run_cfg.params["method"] == "cs+"
