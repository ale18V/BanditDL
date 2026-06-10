"""Aggregate repeated seed result directories for experiment trials.

Used by:
    uv run -m banditdl
    uv run python -m banditdl.experiments.sweep

The engine writes one normal result directory per seed. This module combines
those directories into the public trial result directory and keeps stacked
`*_by_seed.npy` arrays so downstream reducers can average seeds last.
"""

from __future__ import annotations

import json
import gc
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from banditdl.experiments.config_schema import BanditDLConfig

NON_AVERAGEABLE_ARRAYS: tuple[str, ...] = (
    "selected_neighbors.npy",
    "oracle_neighbors.npy",
)


def seed_result_dir(result_dir: Path, seed: int) -> Path:
    """Return the isolated result directory for one repeated seed.

    Args:
        result_dir: Path
            Public aggregate result directory for one trial.
        seed: int
            Concrete seed value used for the repeated run.
        return: Path
            Per-seed result directory used by the engine.
    """
    return Path(result_dir) / "seeds" / f"seed_{seed}" / "results"


def run_seed_averaged(
    run_once: Callable[[BanditDLConfig, Path, int, str], None],
    config: BanditDLConfig,
    result_dir: Path,
    base_seed: int,
    num_seeds: int,
    device: str,
) -> list[int]:
    """Run one configuration for consecutive seeds and aggregate its artifacts.

    Args:
        run_once: Callable[[BanditDLConfig, Path, int, str], None]
            Engine function that writes one standard result directory.
        config: BanditDLConfig
            Typed configuration object.
        result_dir: Path
            Public aggregate result directory for this trial.
        base_seed: int
            First seed in the repeated run set.
        num_seeds: int
            Number of consecutive seeds to run; must be positive.
        device: str
            Torch device string passed to the engine.
        return: list[int]
            Concrete seed values that were run and aggregated.
    """
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")

    seeds = [base_seed + offset for offset in range(num_seeds)]
    for seed in seeds:
        per_seed_dir = seed_result_dir(result_dir, seed)
        try:
            run_once(config, per_seed_dir, seed, device)
        finally:
            _cleanup_after_seed(device)
    aggregate_seed_results(result_dir, seeds)
    return seeds


def _cleanup_after_seed(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def aggregate_seed_results(result_dir: Path, seeds: list[int]) -> None:
    """Write seed-averaged public artifacts from per-seed result directories.

    Args:
        result_dir: Path
            Public result directory that receives averaged artifacts.
        seeds: list[int]
            Concrete seed values whose artifacts must exist under `result_dir`.
        return: None
            This function writes files in `result_dir`.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")

    result_dir = Path(result_dir)
    seed_dirs = [seed_result_dir(result_dir, seed) for seed in seeds]
    for seed_dir in seed_dirs:
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"Missing per-seed result directory: {seed_dir}")

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_seed_metadata(result_dir, seeds, seed_dirs)
    _aggregate_numpy_metrics(result_dir, seed_dirs)
    _aggregate_jsonl_metrics(result_dir, seeds, seed_dirs)


def _write_seed_metadata(result_dir: Path, seeds: list[int], seed_dirs: list[Path]) -> None:
    metadata = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "seed_result_dirs": [str(seed_dir) for seed_dir in seed_dirs],
    }
    (result_dir / "seed_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _aggregate_numpy_metrics(result_dir: Path, seed_dirs: list[Path]) -> None:
    filenames = sorted(
        path.name
        for seed_dir in seed_dirs
        for path in seed_dir.glob("*.npy")
        if not path.name.endswith("_by_seed.npy")
    )
    for filename in sorted(set(filenames)):
        paths = [seed_dir / filename for seed_dir in seed_dirs]
        if not all(path.exists() for path in paths):
            missing = [str(path) for path in paths if not path.exists()]
            raise FileNotFoundError(
                f"Array file {filename!r} is missing for some seeds: {missing}"
            )

        arrays = [np.load(path) for path in paths]
        first_shape = arrays[0].shape
        if any(array.shape != first_shape for array in arrays):
            raise ValueError(f"Array shapes differ across seeds for {filename}")

        stacked = np.stack(arrays, axis=0)
        stem = Path(filename).stem
        np.save(result_dir / f"{stem}_by_seed.npy", stacked)
        aggregate_path = result_dir / filename
        if filename in NON_AVERAGEABLE_ARRAYS:
            if len(seed_dirs) == 1:
                np.save(aggregate_path, arrays[0])
            elif aggregate_path.exists():
                aggregate_path.unlink()
            continue
        np.save(aggregate_path, _nanmean(stacked.astype(float), axis=0))


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _aggregate_jsonl_metrics(result_dir: Path, seeds: list[int], seed_dirs: list[Path]) -> None:
    filename = "sampler_states.jsonl"
    paths = [seed_dir / filename for seed_dir in seed_dirs]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    if len(existing) != len(paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"JSONL file {filename!r} is missing for some seeds: {missing}")

    if len(paths) == 1:
        shutil.copyfile(paths[0], result_dir / filename)
        return

    with (result_dir / "sampler_states_by_seed.jsonl").open("w") as out:
        for seed, path in zip(seeds, paths, strict=True):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                row["seed"] = seed
                out.write(json.dumps(row, sort_keys=True) + "\n")
