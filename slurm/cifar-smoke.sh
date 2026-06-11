#!/bin/bash
#SBATCH --job-name=bdl
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460000
#SBATCH --time=01:30:00
#SBATCH --partition=debug
#SBATCH --gpus-per-node=4
#SBATCH --environment=banditdl
#SBATCH --output=job_output/%j.out
#SBATCH --error=job_output/%j.err

set -euo pipefail


REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO="$(git -C "$REPO" rev-parse --show-toplevel)"
cd "$REPO"

mkdir -p job_output
mkdir -p "${SCRATCH:-/ioscratch}/banditdl"/{datasets,hf-cache,hf-home,uv-cache}

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-${SCRATCH:-/ioscratch}/banditdl/hf-home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH:-/ioscratch}/banditdl/hf-cache}"
export BANDITDL_DATASET_ROOT="${BANDITDL_DATASET_ROOT:-${SCRATCH:-/ioscratch}/banditdl/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-/ioscratch}/banditdl/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${SCRATCH:-/ioscratch}/banditdl/venv"
export UV_LINK_MODE=copy
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID:-none}"
echo "node_list=${SLURM_JOB_NODELIST:-none}"
echo "repo=$REPO"
date
nvidia-smi -L
uv --version

uv sync --frozen

uv run python -m banditdl.experiments.sweep \
  dataset=femnist \
  optimization=opt_femnist \
  topology=dynamic \
  optuna=customsweep \
  hydra.run.dir=${SCRATCH:-/ioscratch}/banditdl/optuna_runs/femnist-pre-sweep/2026-06-09_20-04-54_2509140/ \
  evaluation.evaluation_delta=25 \
  num_seeds=3 \
  identical_initialization=true \
  device=cuda \
  "$@"
