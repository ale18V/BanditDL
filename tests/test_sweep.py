from pathlib import Path

import numpy as np
import optuna
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from optuna.trial import TrialState, create_trial

from banditdl.experiments.config_adapter import build_engine_config
from banditdl.experiments.sweep import (
    _attempt_counts,
    _categorical_distributions,
    _completed_config_ids,
    _device_assignments,
    _read_final_metric,
    _training_config,
    _trial_directory,
    _validate_grid_manifest,
    _visible_devices,
    _worker_count,
)
from banditdl.utils.plot_sweep_base import (
    build_axis_metadata,
    enumerate_valid_param_dicts,
)

CONF_DIR = Path(__file__).parents[1] / "conf"


def _compose(profile):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="sweep", overrides=[f"optuna={profile}"])


def test_final_metric_uses_last_evaluation_and_averages_nodes(tmp_path):
    path = tmp_path / "validation_accuracy.npy"
    np.save(path, [[0.9, 0.7], [0.4, 0.6]])

    assert _read_final_metric(path) == pytest.approx(0.5)


def test_worker_count_defaults_to_devices_and_allows_oversubscription():
    assert _worker_count(OmegaConf.create({"workers": None}), ["cuda:0", "cuda:1"]) == 2
    assert _worker_count(OmegaConf.create({"workers": 5}), ["cuda:0", "cuda:1"]) == 5


def test_workers_above_gpu_count_share_gpus_round_robin():
    assert _device_assignments(5, ["cuda:0", "cuda:1"]) == [
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
        "cuda:0",
    ]


def test_visible_devices_enumerates_all_cuda_devices(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 3)

    assert _visible_devices(OmegaConf.create({"device": "cuda"})) == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]


def test_trial_directory_uses_single_separators(tmp_path):
    params = {
        "sampler.name": "cts",
        "sampler.reward": "cosine_similarity",
        "heterogeneity.alpha": 0.5,
    }
    axis_lookup = {
        "sampler.name": {"display_name": "sampler"},
        "sampler.reward": {"display_name": "reward"},
        "heterogeneity.alpha": {"display_name": "alpha"},
    }

    path = _trial_directory(tmp_path, 42, params, axis_lookup)

    assert path.name.startswith("config-0042_")
    assert "__" not in path.name
    assert "sampler=cts" in path.name
    assert "reward=cosine_similarity" in path.name
    assert "alpha=0.5" in path.name


@pytest.mark.parametrize(
    "profile",
    ["alpha_grid", "clustering_grid"],
)
def test_grid_profiles_are_exhaustive_and_valid(profile):
    cfg = _compose(profile)
    search_space = OmegaConf.to_container(cfg.optuna.search_space, resolve=True)
    combos = enumerate_valid_param_dicts(cfg, search_space)

    assert combos
    _categorical_distributions(search_space)
    for params in combos:
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        for path, value in params.items():
            OmegaConf.update(trial_cfg, path, value, force_add=True)
        build_engine_config(_training_config(trial_cfg))


def test_completed_and_failed_attempts_are_recoverable_from_study():
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        create_trial(
            value=0.8,
            user_attrs={"config_id": 2, "attempt": 1},
        )
    )
    study.add_trial(
        create_trial(
            state=TrialState.FAIL,
            user_attrs={"config_id": 3, "attempt": 2},
        )
    )

    assert _completed_config_ids(study) == {2}
    assert _attempt_counts(study) == {2: 1, 3: 2}


def test_profile_name_drives_readable_hydra_output_directory():
    cfg = _compose("alpha_grid")

    assert cfg.optuna.name == "alpha-grid"
    _, axis_meta = build_axis_metadata(
        OmegaConf.to_container(cfg.optuna.search_space, resolve=True)
    )
    assert axis_meta["sampler.name"]["display_name"] == "sampler"


def test_grid_manifest_rejects_changed_configuration_ids(tmp_path):
    _validate_grid_manifest(tmp_path, "alpha-grid", [{"x": 1}, {"x": 2}])
    _validate_grid_manifest(tmp_path, "alpha-grid", [{"x": 1}, {"x": 2}])

    with pytest.raises(ValueError, match="grid differs"):
        _validate_grid_manifest(tmp_path, "alpha-grid", [{"x": 2}, {"x": 1}])


def test_non_categorical_search_space_is_rejected():
    with pytest.raises(ValueError, match="must define categorical choices"):
        _categorical_distributions({"x": {"type": "float", "low": 0.0, "high": 1.0}})
