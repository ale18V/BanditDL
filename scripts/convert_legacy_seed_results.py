#!/usr/bin/env python3
"""Convert legacy seed-averaged sweep results into one-trial-per-seed layout.

The converter is intentionally non-destructive: it writes a new sweep directory.
Legacy runs produced both averaged metrics, stacked ``*_by_seed.npy`` metrics, and
nested ``results/seeds/seed_*/results`` directories. The current layout stores one
trial directory per seed. This script copies the nested seed result directories to
that layout and skips plots by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState, create_trial

STUDY_NAME = "sweep"


def copy_path(src: Path, dst: Path, *, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, hardlink: bool) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            copy_path(path, target, hardlink=hardlink)


def config_id(trial_dir: Path) -> int:
    match = re.match(r"config-(\d+)", trial_dir.name)
    if not match:
        raise ValueError(f"Cannot parse config id from {trial_dir}")
    return int(match.group(1))


def strip_config_prefix(trial_dir: Path) -> str:
    match = re.match(r"config-\d+_?(.*)", trial_dir.name)
    if not match:
        return trial_dir.name
    return match.group(1).strip("_")


def seed_from_dir(seed_dir: Path) -> int:
    match = re.fullmatch(r"seed_(\d+)", seed_dir.name)
    if not match:
        raise ValueError(f"Cannot parse seed from {seed_dir}")
    return int(match.group(1))


def output_trial_name(trial_dir: Path, seed: int) -> str:
    suffix = strip_config_prefix(trial_dir)
    name = f"config-{config_id(trial_dir):04d}_seed={seed}"
    return f"{name}_{suffix}" if suffix else name


def copy_root_metadata(src: Path, dst: Path, *, hardlink: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (".hydra",):
        copy_tree(src / name, dst / name, hardlink=hardlink)
    for name in ("grid_manifest.json", "sweep.log"):
        path = src / name
        if path.exists():
            copy_path(path, dst / name, hardlink=hardlink)


def convert_manifest(src: Path, dst: Path, seeds: list[int]) -> None:
    path = src / "grid_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    manifest["seeds"] = seeds
    (dst / "grid_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def copy_seed_results(src_results: Path, dst_results: Path, *, hardlink: bool) -> None:
    for item in src_results.iterdir():
        if item.is_dir():
            if item.name in {"plots", "seeds"}:
                continue
            copy_tree(item, dst_results / item.name, hardlink=hardlink)
            continue
        if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tex"}:
            continue
        copy_path(item, dst_results / item.name, hardlink=hardlink)


def convert_trial(trial_dir: Path, out_trials: Path, *, hardlink: bool) -> list[tuple[int, Path]]:
    converted = []
    for attempt in sorted(trial_dir.glob("attempt-*")):
        seeds_root = attempt / "results" / "seeds"
        if not seeds_root.is_dir():
            continue
        for seed_dir in sorted(seeds_root.glob("seed_*"), key=seed_from_dir):
            seed = seed_from_dir(seed_dir)
            src_results = seed_dir / "results"
            if not src_results.is_dir():
                continue
            out_trial = out_trials / output_trial_name(trial_dir, seed) / attempt.name
            copy_seed_results(src_results, out_trial / "results", hardlink=hardlink)
            converted.append((seed, out_trial / "results"))
    return converted



def final_validation_accuracy(result_dir: Path) -> float:
    path = result_dir / "validation_accuracy.npy"
    if not path.exists():
        return float("nan")
    values = np.asarray(np.load(path), dtype=float)
    if values.size == 0:
        return float("nan")
    final = values[-1] if values.ndim > 1 else values
    return float(np.nanmean(final))


def load_legacy_trials(src: Path):
    db = src / "optuna.db"
    if not db.exists():
        return {}
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{db}")
    by_config = {}
    for trial in study.trials:
        config = trial.user_attrs.get("config_id")
        if config is None:
            continue
        by_config[int(config)] = trial
    return by_config


def write_converted_study(src: Path, dst: Path, converted: list[tuple[int, int, Path]]) -> None:
    legacy_trials = load_legacy_trials(src)
    if not legacy_trials:
        return
    db = dst / "optuna.db"
    if db.exists():
        db.unlink()
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{db}",
        study_name=STUDY_NAME,
        load_if_exists=False,
    )
    for config, seed, result_dir in converted:
        legacy = legacy_trials.get(config)
        if legacy is None:
            continue
        params = dict(legacy.params)
        validation_accuracy = final_validation_accuracy(result_dir)
        objective_value = validation_accuracy if np.isfinite(validation_accuracy) else 0.0
        user_attrs = {
            "config_id": config,
            "seed": seed,
            "attempt": 1,
            "device": "converted",
            "resolved_params": params,
            "result_dir": str(result_dir),
            "result_path": str(result_dir.relative_to(dst)),
            "validation_accuracy": validation_accuracy,
        }
        trial = create_trial(
            params=params,
            distributions=legacy.distributions,
            value=objective_value,
            state=TrialState.COMPLETE,
            user_attrs=user_attrs,
        )
        study.add_trial(trial)

def convert_run(src: Path, dst: Path, *, hardlink: bool, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    copy_root_metadata(src, dst, hardlink=hardlink)
    out_trials = dst / "trials"
    all_seeds = set()
    converted_records = []
    for trial_dir in sorted((src / "trials").glob("config-*"), key=config_id):
        seed_results = convert_trial(trial_dir, out_trials, hardlink=hardlink)
        for seed, result_dir in seed_results:
            converted_records.append((config_id(trial_dir), seed, result_dir))
            all_seeds.add(seed)

    if not converted_records:
        raise SystemExit(f"No legacy seed results found in {src}")
    convert_manifest(src, dst, sorted(all_seeds))
    write_converted_study(src, dst, converted_records)
    print(f"converted_seed_trials={len(converted_records)}")
    print(f"seeds={sorted(all_seeds)}")
    print(f"output={dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Legacy sweep directory")
    parser.add_argument("output", type=Path, help="Converted sweep directory")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Physically copy files instead of hard-linking them when possible.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output if it exists.")
    args = parser.parse_args()
    convert_run(
        args.input.resolve(),
        args.output.resolve(),
        hardlink=not args.copy,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
