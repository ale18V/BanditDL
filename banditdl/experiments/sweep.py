from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import hydra
import numpy as np
import optuna
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial

from banditdl.experiments.config_adapter import build_engine_config, resolve_device
from banditdl.experiments.engine import run_experiment
from banditdl.utils.plot_sweep_base import (
    STUDY_NAME,
    _choices_from_spec,
    _normalize_search_space,
    _strip_meta,
    build_axis_metadata,
    enumerate_valid_param_dicts,
    optuna_storage_url,
    plot_sweep_from_cfg,
    trial_folder_name,
)
from banditdl.utils.seed_averaging import run_seed_averaged, seed_result_dir

_WORKER_DEVICE = "cpu"
_MAX_ATTEMPTS = 2
_PROGRESS_INTERVAL_SECONDS = 60
_QUEUE_PREVIEW = 10


def _copy_dict_config(cfg: DictConfig) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(copied, DictConfig):
        raise TypeError("Expected a DictConfig copy")
    return copied


def _apply_trial_params(trial_cfg: DictConfig, trial_params: dict) -> None:
    for path, value in trial_params.items():
        OmegaConf.update(trial_cfg, path, value, merge=False, force_add=True)


def _training_config(cfg: DictConfig) -> DictConfig:
    copied = _copy_dict_config(cfg)
    for key in ("optuna", "plot"):
        if key in copied:
            del copied[key]
    return copied


def _read_final_metric(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.asarray(np.load(path), dtype=float)
    if values.size == 0:
        raise ValueError(f"{path} is empty")
    final = values[-1] if values.ndim > 1 else values
    return float(np.nanmean(final))


def _read_seed_final_metric(
    result_dir: Path,
    seeds: list[int],
    metric_name: str,
) -> tuple[float, list[float]]:
    seed_values = [
        _read_final_metric(seed_result_dir(result_dir, seed) / metric_name) for seed in seeds
    ]
    return float(np.mean(seed_values)), seed_values


def _resolved_trial_params(trial) -> dict:
    resolved = trial.user_attrs.get("resolved_params")
    return dict(resolved) if isinstance(resolved, dict) else dict(trial.params)


def _worker_initializer(device_queue, threads: int) -> None:
    global _WORKER_DEVICE  # noqa: PLW0603 - process-local worker state
    _WORKER_DEVICE = device_queue.get()
    torch.set_num_threads(threads)
    if _WORKER_DEVICE.startswith("cuda"):
        torch.cuda.set_device(_WORKER_DEVICE)


def _run_configuration(task: dict) -> dict:
    config_id = int(task["config_id"])
    attempt = int(task["attempt"])
    params = dict(task["params"])
    trial_cfg = OmegaConf.create(task["base_cfg"])
    _apply_trial_params(trial_cfg, params)

    trial_dir = Path(task["trial_dir"])
    result_dir = trial_dir / f"attempt-{attempt:02d}" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = build_engine_config(_training_config(trial_cfg)).config
        seeds = run_seed_averaged(
            run_once=run_experiment,
            config=config,
            result_dir=result_dir,
            base_seed=config.seed,
            num_seeds=config.num_seeds,
            device=_WORKER_DEVICE,
        )
        value, seed_values = _read_seed_final_metric(
            result_dir,
            seeds,
            "validation_accuracy.npy",
        )
        return {
            "ok": True,
            "config_id": config_id,
            "attempt": attempt,
            "params": params,
            "value": value,
            "seed_values": seed_values,
            "seeds": seeds,
            "result_dir": str(result_dir),
            "device": _WORKER_DEVICE,
        }
    except Exception:
        return {
            "ok": False,
            "config_id": config_id,
            "attempt": attempt,
            "params": params,
            "result_dir": str(result_dir),
            "device": _WORKER_DEVICE,
            "error": traceback.format_exc(),
        }


def _visible_devices(cfg: DictConfig) -> list[str]:
    configured = str(cfg.device)
    if configured == "cpu":
        return ["cpu"]
    if configured not in {"auto", "cuda"}:
        return [configured]
    count = torch.cuda.device_count()
    if count:
        return [f"cuda:{index}" for index in range(count)]
    if configured == "cuda":
        raise RuntimeError("device=cuda requested but no CUDA devices are visible")
    return ["cpu"]


def _worker_count(optuna_cfg, devices: list[str]) -> int:
    configured = optuna_cfg.get("workers")
    workers = len(devices) if configured is None else int(configured)
    if workers < 1:
        raise ValueError("optuna.workers must be >= 1")
    return workers


def _device_assignments(workers: int, devices: list[str]) -> list[str]:
    return [devices[worker_id % len(devices)] for worker_id in range(workers)]


def _threads_per_worker(workers: int) -> int:
    available = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    return max(1, available // workers)


def _categorical_distributions(search_space: dict) -> dict:
    distributions = {}
    ordered_paths, _, _ = _normalize_search_space(search_space)
    for path in ordered_paths:
        spec, _ = _strip_meta(search_space[path])
        choices = _choices_from_spec(spec)
        if not choices:
            raise ValueError(
                f"Optuna sweeps are exhaustive; '{path}' must define categorical choices"
            )
        distributions[path] = CategoricalDistribution(choices)
    return distributions


def _completed_config_ids(study) -> set[int]:
    return {
        int(trial.user_attrs["config_id"])
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and "config_id" in trial.user_attrs
    }


def _attempt_counts(study) -> dict[int, int]:
    counts: dict[int, int] = {}
    for trial in study.trials:
        config_id = trial.user_attrs.get("config_id")
        if config_id is not None:
            counts[int(config_id)] = max(
                counts.get(int(config_id), 0),
                int(trial.user_attrs.get("attempt", 0)),
            )
    return counts


def _record_result(study, result: dict, distributions: dict, output_root: Path) -> None:
    params = result["params"]
    user_attrs = {
        "config_id": result["config_id"],
        "attempt": result["attempt"],
        "device": result["device"],
        "resolved_params": params,
        "result_dir": result["result_dir"],
        "result_path": str(Path(result["result_dir"]).relative_to(output_root)),
    }
    used_distributions = {path: distributions[path] for path in params}
    if result["ok"]:
        user_attrs.update(
            {
                "validation_accuracy": result["value"],
                "validation_accuracy_by_seed": result["seed_values"],
                "seeds": result["seeds"],
            }
        )
        trial = create_trial(
            params=params,
            distributions=used_distributions,
            value=result["value"],
            user_attrs=user_attrs,
        )
    else:
        user_attrs["error"] = result["error"]
        trial = create_trial(
            params=params,
            distributions=used_distributions,
            state=TrialState.FAIL,
            user_attrs=user_attrs,
        )
    study.add_trial(trial)


def _trial_directory(
    output_root: Path,
    config_id: int,
    params: dict,
    axis_lookup: dict,
) -> Path:
    tokens = trial_folder_name(params, axis_lookup)
    suffix = f"_{tokens}" if tokens else ""
    return output_root / "trials" / f"config-{config_id:04d}{suffix}"


def _run_pending(  # noqa: C901 - scheduling and retry policy belong together
    cfg: DictConfig,
    output_root: Path,
    study,
    combos: list[dict],
    search_space: dict,
    axis_lookup: dict,
) -> None:
    devices = _visible_devices(cfg)
    workers = _worker_count(cfg.optuna, devices)
    threads = _threads_per_worker(workers)
    completed = _completed_config_ids(study)
    attempts = _attempt_counts(study)
    pending = [index for index in range(len(combos)) if index not in completed]
    exhausted = [config_id for config_id in pending if attempts.get(config_id, 0) >= _MAX_ATTEMPTS]
    if exhausted:
        exhausted_text = ", ".join(f"{config_id:04d}" for config_id in exhausted)
        raise RuntimeError(f"Configurations already exhausted both attempts: {exhausted_text}")
    distributions = _categorical_distributions(search_space)

    print(
        f"[optuna] configurations={len(combos)} pending={len(pending)} "
        f"workers={workers} devices={devices} threads_per_worker={threads}"
    )
    if not pending:
        return
    print(f"[optuna] output={output_root}")

    context = get_context("spawn")
    device_queue = context.Queue()
    for device in _device_assignments(workers, devices):
        device_queue.put(device)

    base_cfg = OmegaConf.to_container(cfg, resolve=False)
    failed: list[int] = []
    completed_now = 0
    started_at = time.monotonic()
    last_progress_at = started_at
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(device_queue, threads),
    ) as executor:
        active = {}
        for config_id in pending:
            attempt = attempts.get(config_id, 0) + 1
            task = {
                "base_cfg": base_cfg,
                "config_id": config_id,
                "attempt": attempt,
                "params": combos[config_id],
                "trial_dir": str(
                    _trial_directory(
                        output_root,
                        config_id,
                        combos[config_id],
                        axis_lookup,
                    )
                ),
            }
            active[executor.submit(_run_configuration, task)] = task
            if len(active) <= _QUEUE_PREVIEW:
                print(
                    f"[optuna] queued config={config_id:04d} "
                    f"attempt={attempt}/{_MAX_ATTEMPTS} params={combos[config_id]}"
                )
        if len(active) > _QUEUE_PREVIEW:
            print(f"[optuna] queued preview: showing {_QUEUE_PREVIEW}/{len(active)} configs")

        print(f"[optuna] submitted={len(active)} running<=workers={workers}")
        for future in as_completed(active):
            task = active[future]
            try:
                result = future.result()
            except Exception:
                result = {
                    "ok": False,
                    "config_id": task["config_id"],
                    "attempt": task["attempt"],
                    "params": task["params"],
                    "result_dir": str(
                        Path(task["trial_dir"]) / f"attempt-{task['attempt']:02d}" / "results"
                    ),
                    "device": "worker-process",
                    "error": traceback.format_exc(),
                }
            _record_result(study, result, distributions, output_root)
            completed_now += 1
            if result["ok"]:
                elapsed = time.monotonic() - started_at
                print(
                    f"[optuna] config={result['config_id']:04d} "
                    f"done={completed_now}/{len(pending)} "
                    f"value={result['value']:.6f} device={result['device']} "
                    f"elapsed={elapsed / 60:.1f}m"
                )
            elif task["attempt"] < _MAX_ATTEMPTS:
                print(
                    f"[optuna] config={result['config_id']:04d} "
                    f"attempt={task['attempt']}/{_MAX_ATTEMPTS} failed; retrying"
                )
                retry = dict(task)
                retry["attempt"] = task["attempt"] + 1
                retry_result = executor.submit(_run_configuration, retry).result()
                _record_result(study, retry_result, distributions, output_root)
                if not retry_result["ok"]:
                    failed.append(result["config_id"])
                    print(
                        f"[optuna] config={result['config_id']:04d} "
                        f"failed after retry; see {retry_result['result_dir']}"
                    )
                else:
                    print(
                        f"[optuna] config={retry_result['config_id']:04d} "
                        f"retry ok value={retry_result['value']:.6f} "
                        f"device={retry_result['device']}"
                    )
            else:
                failed.append(result["config_id"])
                print(
                    f"[optuna] config={result['config_id']:04d} failed; see {result['result_dir']}"
                )

            now = time.monotonic()
            if now - last_progress_at >= _PROGRESS_INTERVAL_SECONDS:
                remaining = len(pending) - completed_now
                print(
                    f"[optuna] progress done={completed_now}/{len(pending)} "
                    f"remaining={remaining} elapsed={(now - started_at) / 60:.1f}m"
                )
                last_progress_at = now

    if failed:
        failed_text = ", ".join(f"{config_id:04d}" for config_id in sorted(failed))
        raise RuntimeError(f"Configurations failed after two attempts: {failed_text}")


def _run_best_trial_test_evaluation(
    best_trial,
    base_cfg: DictConfig,
    output_root: Path,
) -> float:
    best_cfg = _copy_dict_config(base_cfg)
    _apply_trial_params(best_cfg, _resolved_trial_params(best_trial))
    OmegaConf.update(best_cfg, "evaluation.evaluate_test", True, merge=False)
    config = build_engine_config(_training_config(best_cfg)).config
    result_dir = output_root / "best_trial_test_eval" / "results"
    aggregate_metric = result_dir / "test_accuracy.npy"
    if aggregate_metric.exists():
        return _read_final_metric(aggregate_metric)
    device = resolve_device(best_cfg)
    seeds = run_seed_averaged(
        run_once=run_experiment,
        config=config,
        result_dir=result_dir,
        base_seed=config.seed,
        num_seeds=config.num_seeds,
        device=device,
    )
    value, _ = _read_seed_final_metric(result_dir, seeds, "test_accuracy.npy")
    return value


def _load_search_space(optuna_cfg) -> dict:
    raw = OmegaConf.to_container(optuna_cfg.search_space, resolve=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("optuna.search_space must be a non-empty mapping")
    return {str(path): spec for path, spec in raw.items()}


def _validate_grid_manifest(output_root: Path, profile: str, combos: list[dict]) -> None:
    path = output_root / "grid_manifest.json"
    manifest = {"profile": profile, "combinations": combos}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError(
                "Sweep grid differs from the existing grid_manifest.json; "
                "use a new output directory"
            )
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@hydra.main(version_base=None, config_path="../../conf", config_name="sweep")
def main(cfg: DictConfig) -> None:
    output_root = Path(HydraConfig.get().runtime.output_dir)
    (output_root / "trials").mkdir(parents=True, exist_ok=True)
    search_space = _load_search_space(cfg.optuna)
    _categorical_distributions(search_space)
    _, axis_meta = build_axis_metadata(search_space)
    axis_lookup = {path: axis_meta.get(path, {}) for path in search_space}
    combos = enumerate_valid_param_dicts(cfg, search_space)
    if not combos:
        raise ValueError("No valid categorical grid combinations found")
    _validate_grid_manifest(output_root, str(cfg.optuna.name), combos)

    study = optuna.create_study(
        direction=str(cfg.optuna.direction),
        storage=optuna_storage_url(output_root),
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    _run_pending(
        cfg,
        output_root,
        study,
        combos,
        search_space,
        axis_lookup,
    )

    best = study.best_trial
    print(f"[optuna] best config: {best.user_attrs['config_id']:04d}")
    print(f"[optuna] best final validation accuracy: {best.value:.6f}")
    final_test_accuracy = _run_best_trial_test_evaluation(best, cfg, output_root)
    print(f"[optuna] best final test accuracy: {final_test_accuracy:.6f}")
    plot_sweep_from_cfg(output_root, cfg, study=study)
    print(f"[optuna] sweep artifacts: {output_root / 'sweep_artifacts'}")


if __name__ == "__main__":
    main()
