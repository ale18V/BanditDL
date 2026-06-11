# Running BanditDL on CSCS

This guide is for the CSCS/Clariden setup using EDF environments and the local
scratch-backed UV environment.

## One-time Setup

Create the EDF file:

```bash
cat > ~/.edf/banditdl.toml <<'TOML'
image = "nvcr.io#nvidia/pytorch:25.01-py3"

mounts = [
    "/users/abombarda:/users/abombarda",
    "/capstor/store/cscs/swissai/infra01:/capstor/store/cscs/swissai/infra01",
    "/iopsstor/scratch/cscs/abombarda:/ioscratch",
]

workdir = "/users/abombarda/Projects/apertus"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
SCRATCH = "/ioscratch"
HF_HOME = "/ioscratch/banditdl/hf-home"
HF_DATASETS_CACHE = "/ioscratch/banditdl/hf-cache"
BANDITDL_DATASET_ROOT = "/ioscratch/banditdl/datasets"
UV_CACHE_DIR = "/ioscratch/banditdl/uv-cache"
PYTHONUNBUFFERED = "1"
HYDRA_FULL_ERROR = "1"
TOML
```

Create the local Hydra override file if it does not exist:

```bash
printf '{}\n' > conf/override.yaml
```

`conf/override.yaml` is gitignored and can contain personal defaults. Keep heavy
runtime paths on scratch, not under `/users`.

## Smoke Sweep

The tested smoke script is:

```bash
sbatch slurm/sbatch_banditdl_cscs_smoke.sh
```

It runs `conf/optuna/sanitysweep.yaml` on one debug node with 4 GPUs and 4
workers. It also sets:

```bash
TMPDIR=/tmp
UV_PROJECT_ENVIRONMENT=$SCRATCH/banditdl/venv
UV_CACHE_DIR=$SCRATCH/banditdl/uv-cache
BANDITDL_DATASET_ROOT=$SCRATCH/banditdl/datasets
```

This keeps the Python environment, UV cache, and datasets on scratch.

## Single Run

For a single experiment, use the same CSCS script with an override that bypasses
Optuna only if you create a dedicated single-run script. The simplest current
option is to run the module through `sbatch --wrap` using the same EDF:

```bash
sbatch \
  --job-name=bdl_single \
  --account=infra01 \
  --partition=debug \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=32 \
  --mem=460000 \
  --time=00:30:00 \
  --gpus-per-node=4 \
  --environment=banditdl \
  --export=ALL,TMPDIR=/tmp \
  --output=job_output/%x_%j.out \
  --error=job_output/%x_%j.err \
  --wrap='cd /users/abombarda/Projects/apertus && \
    export PATH=$HOME/.local/bin:$PATH && \
    export UV_PROJECT_ENVIRONMENT=$SCRATCH/banditdl/venv && \
    export UV_CACHE_DIR=$SCRATCH/banditdl/uv-cache && \
    export UV_LINK_MODE=copy && \
    export BANDITDL_DATASET_ROOT=$SCRATCH/banditdl/datasets && \
    uv sync --frozen && \
    uv run -m banditdl dataset=mnist topology.nodes=5 optimization.rounds=3 evaluation.evaluation_delta=1 device=cuda'
```

For a real single run, replace only the Hydra overrides after `uv run -m
banditdl`, for example:

```bash
dataset=cifar10 optimization=opt_cifar10 topology.nodes=100 optimization.rounds=1000 sampler=cts device=cuda
```

## Bigger Optuna Sweep

Use the smoke script as the template and override the sweep/config parameters at
submission time:

```bash
sbatch \
  --partition=normal \
  --time=12:00:00 \
  slurm/sbatch_banditdl_cscs_smoke.sh \
  optuna=alpha_grid \
  optuna.workers=4 \
  dataset=cifar10 \
  optimization=opt_cifar10 \
  topology.nodes=100 \
  optimization.rounds=1000 \
  evaluation.evaluation_delta=20 \
  device=cuda
```

Notes:

- `optuna.workers=4` maps one worker per GPU on a 4-GPU GH200 node.
- Edit the selected `conf/optuna/*.yaml` profile to define the exhaustive grid.
- Use `debug` only for short tests. Use `normal` for overnight sweeps if your
  account accepts it.

## Outputs

Single-run Hydra outputs:

```text
.hydra_runs/<date>/<time>_<slurm-job-id>/
```

Optuna sweep outputs:

```text
.optuna_runs/<sweep-name>/<timestamp>_<slurm-job-id>/
```

Useful subdirectories:

```text
trials/                 # one directory per grid configuration
sweep_artifacts/        # sweep-level plots and summaries
```

Job logs:

```text
job_output/<job-name>_<job-id>.out
job_output/<job-name>_<job-id>.err
```

## Cleanup

The repo-local `.venv` is not needed by CSCS jobs if
`UV_PROJECT_ENVIRONMENT=$SCRATCH/banditdl/venv` is set. If home quota is tight,
it is safe to remove the repo-local venv:

```bash
rm -rf .venv
```

Do not remove scratch caches before a large sweep unless you are willing to pay
the dependency/download cost again.
