from __future__ import annotations

from pathlib import Path

import hydra
import optuna
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.engine import run_dynamic, run_fixed
from banditdl.utils.plot_sweep_base import (
    build_axis_metadata,
    enumerate_valid_param_dicts,
    normalize_directions,
    normalize_plot_modes,
    plot_sweep,
    trial_folder_name,
)


def _read_metric_file_max(metric_file: Path) -> float:
    if not metric_file.exists():
        raise FileNotFoundError(f"Missing metric file: {metric_file}")

    metric_values = []
    for line in metric_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        if len(fields) >= 2:
            metric_values.append(float(fields[1]))
    if not metric_values:
        raise ValueError(f"No metric values found in: {metric_file}")
    return max(metric_values)


def _resolved_trial_params(trial) -> dict:
    if trial.params:
        return dict(trial.params)
    resolved = trial.user_attrs.get("resolved_params")
    if isinstance(resolved, dict):
        return dict(resolved)
    return {}


def _objective(trial, base_cfg, trials_root: Path, axis_lookup: dict, combos: list) -> float:
    trial_index = int(trial.number)
    if trial_index >= len(combos):
        raise IndexError(
            f"Trial index {trial_index} out of bounds for {len(combos)} combinations"
        )
    trial_params = dict(combos[trial_index])

    trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    for path, value in trial_params.items():
        OmegaConf.update(trial_cfg, path, value, merge=False)

    folder_name = trial_folder_name(trial_params, axis_lookup)
    trial_result_dir = trials_root / folder_name / "results"
    trial_result_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = build_engine_config(trial_cfg)
    seed_value = int(trial_cfg.seed) + trial_index
    device = resolve_device(trial_cfg)

    if run_cfg.run_mode == "dynamic":
        run_dynamic(params=run_cfg.params, result_dir=trial_result_dir, seed=seed_value, device=device)
    else:
        run_fixed(params=run_cfg.params, result_dir=trial_result_dir, seed=seed_value, device=device)

    validation_metric = _read_metric_file_max(trial_result_dir / "validation")
    trial.set_user_attr("validation_accuracy", validation_metric)
    trial.set_user_attr("result_dir", str(trial_result_dir))
    trial.set_user_attr("seed", seed_value)
    trial.set_user_attr("resolved_params", trial_params)
    return validation_metric


def _run_best_trial_test_evaluation(best_trial, base_cfg, output_root: Path) -> float:
    best_params = _resolved_trial_params(best_trial)
    best_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    for param_path, sampled_value in best_params.items():
        OmegaConf.update(best_cfg, param_path, sampled_value, merge=False)

    best_result_dir = output_root / "best_trial_test_eval" / "results"
    best_result_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = build_engine_config(best_cfg)
    run_cfg.params["evaluate-test"] = True
    seed_value = int(best_cfg.seed) + int(best_trial.number)
    device = resolve_device(best_cfg)

    if run_cfg.run_mode == "dynamic":
        run_dynamic(params=run_cfg.params, result_dir=best_result_dir, seed=seed_value, device=device)
    else:
        run_fixed(params=run_cfg.params, result_dir=best_result_dir, seed=seed_value, device=device)

    return _read_metric_file_max(best_result_dir / "test")


def _metrics_list_from_cfg(cfg) -> list[str]:
    raw = cfg.get("plot_metrics")
    if raw is None:
        return []
    return list(OmegaConf.to_container(raw, resolve=True))


@hydra.main(version_base=None, config_path="../../conf", config_name="sweep")
def main(cfg) -> None:
    output_root = Path(HydraConfig.get().runtime.output_dir)
    trials_root = output_root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    if "optuna" not in cfg:
        raise ValueError("Missing 'optuna' section in Hydra config")
    optuna_cfg = cfg.optuna
    if "search_space" not in optuna_cfg:
        raise ValueError("Missing 'optuna.search_space' in Hydra config")

    search_space = OmegaConf.to_container(optuna_cfg.search_space, resolve=True)
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping")

    combos = enumerate_valid_param_dicts(cfg, search_space)
    if not combos:
        raise ValueError(
            "No categorical grid combinations found. Use categorical sweeps or add list-style search_space entries."
        )

    _, axis_meta = build_axis_metadata(search_space)
    axis_lookup = {path: axis_meta.get(path, {}) for path in search_space.keys()}

    direction = str(optuna_cfg.direction)
    study = optuna.create_study(direction=direction)
    total_trials = len(combos)
    print(f"[optuna] grid trials={total_trials} | metric=validation_accuracy | trials_dir={trials_root}")
    study.optimize(lambda trial: _objective(trial, cfg, trials_root, axis_lookup, combos), n_trials=total_trials)

    best = study.best_trial
    best_dir = best.user_attrs.get("result_dir")
    print(f"[optuna] best trial: {best.number}")
    print(f"[optuna] best validation_accuracy: {best.value:.6f}")
    if best_dir:
        print(f"[optuna] best result directory: {best_dir}")
    print("[optuna] best parameters:")
    for name, value in _resolved_trial_params(best).items():
        print(f"  - {name}: {value}")

    final_test_accuracy = _run_best_trial_test_evaluation(best, cfg, output_root)
    print(f"[optuna] best trial final test directory: {output_root / 'best_trial_test_eval' / 'results'}")
    print(f"[optuna] best trial final test_accuracy: {final_test_accuracy:.6f}")

    metrics_list = _metrics_list_from_cfg(cfg)
    plot_modes = normalize_plot_modes(cfg.get("plot_mode"))
    plot_directions = normalize_directions(cfg.get("direction"))
    sweep_plot_root = output_root / "sweep_artifacts"
    plot_sweep(
        plot_modes,
        plot_directions,
        trials_root,
        study,
        search_space,
        metrics_list,
        sweep_plot_root,
    )
    print(
        f"[optuna] sweep plots written to: {sweep_plot_root} | "
        f"modes={plot_modes} directions={plot_directions}"
    )


if __name__ == "__main__":
    main()
