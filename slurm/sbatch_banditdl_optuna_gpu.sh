#!/bin/bash -l
# Submit a BanditDL Optuna sweep on the EPFL Izar GPU cluster.
#
#   sbatch --gres=gpu:4 slurm/sbatch_banditdl_optuna_gpu.sh \
#       optuna=alpha_grid optuna.workers=8
#
# All positional args are forwarded to `banditdl.experiments.sweep`. The script
# auto-adds `device=cuda` unless the caller passes their own `device=` override.
# Workers are assigned to visible GPUs round-robin, so workers may exceed GPUs.

#SBATCH --job-name=banditdl_optuna
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=job_output/banditdl_optuna_%j.txt
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p job_output

module load gcc 2>/dev/null || echo "[sbatch] WARN: 'module load gcc' failed; continuing"
module load cuda 2>/dev/null || echo "[sbatch] WARN: 'module load cuda' failed; continuing"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "[sbatch] ERROR: 'uv' not on PATH. Run 'bash slurm/setup.sh' from a login node first." >&2
    exit 1
fi

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export BANDITDL_DATASET_ROOT="${BANDITDL_DATASET_ROOT:-$HOME/BanditDL/banditdl/datasets/cache}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

HAS_DEVICE=false
for arg in "$@"; do
    if [[ "$arg" == device=* ]]; then HAS_DEVICE=true; break; fi
done
if ! $HAS_DEVICE; then
    set -- device=cuda "$@"
fi

echo "[sbatch] job_id=$SLURM_JOB_ID job_name=$SLURM_JOB_NAME node=$SLURMD_NODENAME"
echo "[sbatch] gpus=${CUDA_VISIBLE_DEVICES:-unset} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
echo "[sbatch] hydra overrides: $*"

srun --cpu-bind=cores uv run python -m banditdl.experiments.sweep "$@"
