"""Cluster-formation analysis for bandit-sampler runs on grouped pathological partitions.

Reads `selected_neighbors.npy` or `selected_neighbors_by_seed.npy` from one or
more Hydra run folders and computes whether each worker's converged neighbor
selections concentrate inside its own label group.

Usage:
    uv run python scripts/analyze_clustering.py <run_dir> [<run_dir> ...] \
        --partition pathological_5g_2c --tail 200

Outputs a per-run summary table:
    sampling, seed, mean_purity, min_purity, max_purity

Purity for worker i = fraction of (round, slot) selections in the trailing window whose
selected arm shares worker i's label group. 1.0 means perfect cluster formation.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from omegaconf import OmegaConf

PARTITIONS = {
    "pathological_5g_2c": dict(nb_groups=5, classes_per_group=2, overlap=0),
    "pathological_2g_5c": dict(nb_groups=2, classes_per_group=5, overlap=0),
    "pathological_3g_3c_ov1": dict(nb_groups=3, classes_per_group=3, overlap=1),
}


def worker_to_group(nb_workers: int, nb_groups: int) -> np.ndarray:
    # Mirrors banditdl.data.dataset_utils.pathological_grouped_classes worker layout.
    sizes = [nb_workers // nb_groups] * nb_groups
    for i in range(nb_workers % nb_groups):
        sizes[i] += 1
    assignment = np.empty(nb_workers, dtype=int)
    cursor = 0
    for g, size in enumerate(sizes):
        assignment[cursor : cursor + size] = g
        cursor += size
    return assignment


def cluster_purity(selected: np.ndarray, worker_group: np.ndarray, tail: int) -> np.ndarray:
    if selected.ndim == 4:
        per_seed = np.stack(
            [
                _cluster_purity_one_seed(seed_selected, worker_group, tail)
                for seed_selected in selected
            ]
        )
        return np.nanmean(per_seed, axis=0)
    if selected.ndim != 3:
        raise ValueError("selected must have shape (T, N, k) or (S, T, N, k)")
    return _cluster_purity_one_seed(selected, worker_group, tail)


def _cluster_purity_one_seed(
    selected: np.ndarray, worker_group: np.ndarray, tail: int
) -> np.ndarray:
    # selected: (T, N, k) ints; -1 = no pick.
    selected_tail = selected[-tail:]
    purities = np.full(selected_tail.shape[1], np.nan)
    for i in range(selected_tail.shape[1]):
        picks = selected_tail[:, i, :].reshape(-1)
        picks = picks[picks >= 0]
        if picks.size == 0:
            continue
        purities[i] = float(np.mean(worker_group[picks] == worker_group[i]))
    return purities


def baseline_purity(worker_group: np.ndarray) -> float:
    # Probability that a uniformly random other worker shares the picker's group.
    n = len(worker_group)
    total = 0.0
    for i in range(n):
        same = (worker_group == worker_group[i]).sum() - 1
        total += same / (n - 1)
    return total / n


def load_hydra_cfg(run_dir: pathlib.Path):
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return None
    return OmegaConf.load(cfg_path)


def parse_run(run_dir: pathlib.Path, partition: str, tail: int):
    cfg = load_hydra_cfg(run_dir)
    if cfg is None:
        print(f"[skip] {run_dir}: missing .hydra/config.yaml")
        return None
    sampling = float(cfg.topology.sampling)
    seed = int(cfg.seed)
    nb_workers = int(cfg.topology.nodes)
    by_seed_path = run_dir / "results" / "selected_neighbors_by_seed.npy"
    selected_path = (
        by_seed_path if by_seed_path.is_file() else run_dir / "results" / "selected_neighbors.npy"
    )
    if not selected_path.is_file():
        print(f"[skip] {run_dir}: missing {selected_path}")
        return None
    selected = np.load(selected_path)
    nb_groups = PARTITIONS[partition]["nb_groups"]
    worker_group = worker_to_group(nb_workers, nb_groups)
    purities = cluster_purity(selected, worker_group, tail=tail)
    return {
        "run_dir": str(run_dir),
        "sampling": sampling,
        "seed": seed,
        "nb_workers": nb_workers,
        "baseline_purity": baseline_purity(worker_group),
        "mean_purity": float(np.nanmean(purities)),
        "min_purity": float(np.nanmin(purities)),
        "max_purity": float(np.nanmax(purities)),
        "per_worker_purity": purities.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--partition", default="pathological_5g_2c", choices=sorted(PARTITIONS))
    parser.add_argument("--tail", type=int, default=200, help="rounds at the end to average over")
    parser.add_argument(
        "--json", type=pathlib.Path, default=None, help="optional path to dump full results as JSON"
    )
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        row = parse_run(run_dir, args.partition, args.tail)
        if row is not None:
            rows.append(row)

    if not rows:
        print("no usable runs found")
        return

    rows.sort(key=lambda r: (r["sampling"], r["seed"]))
    baseline = rows[0]["baseline_purity"]
    print(f"partition={args.partition} tail={args.tail} baseline_purity={baseline:.4f}\n")
    print(f"{'sampling':>9}  {'seed':>4}  {'mean':>6}  {'min':>6}  {'max':>6}  run")
    for r in rows:
        print(
            f"{r['sampling']:>9.3f}  {r['seed']:>4d}  "
            f"{r['mean_purity']:>6.3f}  {r['min_purity']:>6.3f}  {r['max_purity']:>6.3f}  "
            f"{r['run_dir']}"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
