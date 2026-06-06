from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import optuna
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.engine import run_experiment
from banditdl.utils.plot_sweep_base import (
    STUDY_NAME,
    _choices_from_spec,
    _conditions_met,
    _normalize_search_space,
    _strip_meta,
    _when_clause,
    build_axis_metadata,
    enumerate_valid_param_dicts,
    optuna_storage_url,
    plot_sweep_from_cfg,
    trial_folder_name,
)
from banditdl.utils.seed_averaging import run_seed_averaged, seed_result_dir


def _read_metric_file_max(metric_file: Path) -> float:
    if not metric_file.exists():
        raise FileNotFoundError(f"Missing metric file: {metric_file}")

    metric_values = []
    for i, line in enumerate(metric_file.read_text().splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        try:
            if len(fields) >= 2:
                metric_values.append(float(fields[1]))
            else:
                print(f"Warning: Malformed line {i + 1} in {metric_file} (too few fields)")
        except ValueError:
            print(f"Warning: Could not parse metric value on line {i + 1} in {metric_file}")

    if not metric_values:
        raise ValueError(f"No valid metric values found in: {metric_file}")
    return max(metric_values)


def _read_seed_metric_file_max(
    result_dir: Path, seeds: list[int], metric_name: str
) -> tuple[float, list[float]]:
    seed_values = [
        _read_metric_file_max(seed_result_dir(result_dir, seed) / metric_name) for seed in seeds
    ]
    return float(np.mean(seed_values)), seed_values


def _copy_dict_config(cfg: DictConfig) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(copied, DictConfig):
        raise TypeError("Expected a DictConfig copy")
    return copied


def _resolved_trial_params(trial) -> dict:
    if trial.params:
        return dict(trial.params)
    resolved = trial.user_attrs.get("resolved_params")
    if isinstance(resolved, dict):
        return dict(resolved)
    return {}


def _apply_trial_params(trial_cfg: DictConfig, trial_params: dict) -> None:
    for path, value in trial_params.items():
        OmegaConf.update(trial_cfg, path, value, merge=False, force_add=True)


def _objective_from_params(
    trial,
    base_cfg: DictConfig,
    output_root: Path,
    axis_lookup: dict,
    trial_params: dict,
) -> float:
    trial_cfg = _copy_dict_config(base_cfg)
    _apply_trial_params(trial_cfg, trial_params)

    config = build_engine_config(trial_cfg).config

    trials_root = output_root / "trials"
    folder_name = trial_folder_name(trial_params, axis_lookup)
    trial_result_dir = trials_root / folder_name / "results"
    trial_result_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(trial_cfg)

    seeds = run_seed_averaged(
        run_once=run_experiment,
        config=config,
        result_dir=trial_result_dir,
        base_seed=config.seed,
        num_seeds=config.num_seeds,
        device=device,
    )

    validation_metric, validation_by_seed = _read_seed_metric_file_max(
        trial_result_dir, seeds, "validation"
    )
    trial.set_user_attr("validation_accuracy", validation_metric)
    trial.set_user_attr("validation_accuracy_by_seed", validation_by_seed)
    trial.set_user_attr("result_dir", str(trial_result_dir))
    trial.set_user_attr("result_path", str(trial_result_dir.relative_to(output_root)))
    trial.set_user_attr("seeds", seeds)
    trial.set_user_attr("num_seeds", config.num_seeds)
    trial.set_user_attr("resolved_params", trial_params)
    return validation_metric


def _objective_grid(
    trial, base_cfg: DictConfig, output_root: Path, axis_lookup: dict, combos: list
) -> float:
    trial_index = int(trial.number)
    if trial_index >= len(combos):
        raise IndexError(f"Trial index {trial_index} out of bounds for {len(combos)} combinations")
    return _objective_from_params(
        trial, base_cfg, output_root, axis_lookup, dict(combos[trial_index])
    )


def _all_search_axes_categorical(search_space: dict) -> bool:
    ordered_paths, _, _ = _normalize_search_space(search_space)
    return all(_choices_from_spec(_strip_meta(search_space[path])[0]) for path in ordered_paths)


def _suggest_value(trial, path: str, spec):
    choices = _choices_from_spec(spec)
    if choices:
        return trial.suggest_categorical(path, choices)
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported Optuna search-space spec for '{path}': {spec!r}")
    param_type = str(spec.get("type", "")).lower()
    if param_type == "float":
        return trial.suggest_float(
            path,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if param_type == "int":
        return trial.suggest_int(
            path,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    raise ValueError(f"Unsupported Optuna search-space type for '{path}': {param_type!r}")


def _suggest_trial_params(trial, base_cfg: DictConfig, search_space: dict) -> dict:
    ordered_paths, _, _ = _normalize_search_space(search_space)
    trial_cfg = _copy_dict_config(base_cfg)
    trial_params = {}
    for path in ordered_paths:
        raw_spec = search_space[path]
        when_clause = _when_clause(raw_spec)
        if when_clause is not None and not _conditions_met(trial_cfg, when_clause):
            continue
        inner_spec, _ = _strip_meta(raw_spec)
        value = _suggest_value(trial, path, inner_spec)
        trial_params[path] = value
        OmegaConf.update(trial_cfg, path, value, merge=False, force_add=True)
    return trial_params


def _objective_suggested(
    trial, base_cfg: DictConfig, output_root: Path, axis_lookup: dict, search_space: dict
) -> float:
    trial_params = _suggest_trial_params(trial, base_cfg, search_space)
    return _objective_from_params(trial, base_cfg, output_root, axis_lookup, trial_params)


def _run_best_trial_test_evaluation(best_trial, base_cfg: DictConfig, output_root: Path) -> float:
    best_params = _resolved_trial_params(best_trial)
    best_cfg = _copy_dict_config(base_cfg)
    _apply_trial_params(best_cfg, best_params)

    OmegaConf.update(best_cfg, "evaluation.evaluate_test", True, merge=False)
    config = build_engine_config(best_cfg).config

    best_result_dir = output_root / "best_trial_test_eval" / "results"
    best_result_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(best_cfg)

    seeds = run_seed_averaged(
        run_once=run_experiment,
        config=config,
        result_dir=best_result_dir,
        base_seed=config.seed,
        num_seeds=config.num_seeds,
        device=device,
    )

    test_metric, _ = _read_seed_metric_file_max(best_result_dir, seeds, "test")
    return test_metric


def _load_search_space(optuna_cfg) -> dict:
    raw = OmegaConf.to_container(optuna_cfg.search_space, resolve=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("optuna.search_space must be a non-empty mapping")
    return {str(path): spec for path, spec in raw.items()}


@hydra.main(version_base=None, config_path="../../conf", config_name="sweep")
def main(cfg: DictConfig) -> None:
    output_root = Path(HydraConfig.get().runtime.output_dir)
    (output_root / "trials").mkdir(parents=True, exist_ok=True)

    if "optuna" not in cfg:
        raise ValueError("Missing 'optuna' section in Hydra config")
    optuna_cfg = cfg.optuna
    if "search_space" not in optuna_cfg:
        raise ValueError("Missing 'optuna.search_space' in Hydra config")

    search_space = _load_search_space(optuna_cfg)
    _, axis_meta = build_axis_metadata(search_space)
    axis_lookup = {path: axis_meta.get(path, {}) for path in search_space.keys()}

    study = optuna.create_study(
        direction=str(optuna_cfg.direction),
        storage=optuna_storage_url(output_root),
        study_name=STUDY_NAME,
        load_if_exists=True,
    )

    if _all_search_axes_categorical(search_space):
        combos = enumerate_valid_param_dicts(cfg, search_space)
        if not combos:
            raise ValueError("No categorical grid combinations found.")
        total_trials = len(combos)
        print(
            f"[optuna] grid trials={total_trials} | seeds_per_trial={int(cfg.num_seeds)} | "
            f"metric=validation_accuracy | trials_dir={output_root / 'trials'}"
        )
        study.optimize(
            lambda trial: _objective_grid(trial, cfg, output_root, axis_lookup, combos),
            n_trials=total_trials,
        )
    else:
        total_trials = int(optuna_cfg.get("n_trials", 20))
        print(
            f"[optuna] sampled trials={total_trials} | seeds_per_trial={int(cfg.num_seeds)} | "
            f"metric=validation_accuracy | trials_dir={output_root / 'trials'}"
        )
        study.optimize(
            lambda trial: _objective_suggested(trial, cfg, output_root, axis_lookup, search_space),
            n_trials=total_trials,
        )

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
    final_test_dir = output_root / "best_trial_test_eval" / "results"
    print(
        f"[optuna] best trial final test directory: {final_test_dir}"
    )
    print(f"[optuna] best trial final test_accuracy: {final_test_accuracy:.6f}")

    plot_sweep_from_cfg(output_root, cfg, study=study)
    print(f"[optuna] sweep plots written to: {output_root / 'sweep_artifacts'}")


if __name__ == "__main__":
    main()
