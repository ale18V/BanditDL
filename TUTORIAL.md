# Tutorial: First Experiment

## Setup

Install the project and development dependencies:

```bash
uv sync
```

Inspect the resolved default configuration:

```bash
uv run -m banditdl --cfg job
```

## Run an Experiment

This example runs 20 honest MNIST nodes with non-IID Dirichlet partitions:

```bash
uv run -m banditdl \
  dataset=mnist \
  adversary=none \
  sampler=uniform \
  topology.nodes=20 \
  topology.sampling=0.2 \
  heterogeneity.alpha=0.1 \
  optimization.rounds=500 \
  evaluation.evaluation_delta=50
```

Important parameters:

- `topology.nodes`: total participants.
- `topology.sampling`: fraction of other participants sampled each round.
- `heterogeneity.alpha`: Dirichlet concentration. Smaller values produce more
  heterogeneous local datasets.
- `optimization.rounds`: decentralized training rounds.
- `evaluation.evaluation_delta`: rounds between evaluations.

## Dataset Partitioning

Dataset profiles select a provider:

- `dataset=mnist`: MNIST with synthetic Dirichlet/pathological partitioning.
- `dataset=cifar10`: CIFAR-10 with synthetic partitioning.
- `dataset=femnist`: one natural FEMNIST writer per honest node.
- `dataset=femnist_pool`: pooled FEMNIST with synthetic partitioning.

For synthetic datasets, the heterogeneity profile is the partition strategy:

```bash
uv run -m banditdl \
  dataset=cifar10 \
  heterogeneity=pathological_5g_2c \
  topology.nodes=20
```

An explicit cluster count must divide the number of honest nodes.

## Results

Hydra creates:

```text
.hydra_runs/<date>/<time>_local/
  .hydra/                 # Resolved config and overrides
  hydra_run.log
  results/                # Seed-aggregated metrics
  plots/                  # Generated figures
```

Each seed also has its own results and partition audit:

```text
results/seeds/seed_<seed>/results/audit.json
```

The audit records the partition strategy, selected parameters, participant
counts, and label distribution of every honest node.

Find the latest run:

```bash
export RUN_DIR=$(find .hydra_runs -mindepth 2 -maxdepth 2 -type d | sort | tail -1)
echo "$RUN_DIR"
```

Common metrics under `$RUN_DIR/results/` include:

- `local_accuracy.npy`
- `global_accuracy.npy`
- `local_loss.npy`
- `train_loss.npy`
- `sampler_probabilities.npy`
- `reward_algorithm.npy`
- `reward_oracle.npy`
- `regret.npy`

Arrays ending in `_by_seed.npy` preserve the seed dimension.

## Inspect Metrics

```bash
uv run python - <<'PY'
import os
from pathlib import Path

import numpy as np

results = Path(os.environ["RUN_DIR"]) / "results"
local = np.load(results / "local_accuracy.npy")
global_ = np.load(results / "global_accuracy.npy")

print("Final local accuracy:", np.nanmean(local[-1]))
print("Final global accuracy:", np.nanmean(global_[-1]))
PY
```

The experiment automatically generates figures in `$RUN_DIR/plots/`.

## Compare Configurations

Use Hydra multirun for a direct Cartesian product:

```bash
uv run -m banditdl -m \
  sampler=uniform,exp3 \
  heterogeneity.alpha=0.1,0.5 \
  topology.sampling=0.1,0.2 \
  optimization.rounds=100 \
  seed=0,1
```

Use Optuna when parameters should be sampled or optimized:

```bash
uv run python -m banditdl.experiments.sweep \
  optuna=sweep \
  optimization.rounds=100 \
  num_seeds=3
```

See [`docs/config.md`](docs/config.md) and [`docs/sweeps.md`](docs/sweeps.md)
for the complete workflows.

## Crash Safety

Runtime metrics and sampler probabilities are progressively flushed to valid
NumPy files. Unwritten preallocated rounds contain `NaN`, allowing partial runs
to be inspected without treating missing rounds as real measurements.
