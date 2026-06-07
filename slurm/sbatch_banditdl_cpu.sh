#!/bin/bash -l
# Submit a BanditDL CPU-only run on an EPFL CPU cluster (e.g. Jed).
#
#   sbatch slurm/sbatch_banditdl_cpu.sh dataset=cifar10 sampler=epsilon_greedy seed=0
#   sbatch slurm/sbatch_banditdl_cpu.sh dataset=femnist topology.nodes=30
#   sbatch --time=04:00:00 --job-name=cifar_seed3 slurm/sbatch_banditdl_cpu.sh dataset=cifar10 seed=3
#
# All positional args are forwarded as Hydra overrides. The script auto-adds
# `device=cpu` unless the caller passes their own `device=` override.

#SBATCH --job-name=banditdl
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=job_output/banditdl_%j.txt
#SBATCH --mail-type=END,FAIL
# Mail notifications are opt-in: pass `--mail-user=you@example.com` on the
# `sbatch` CLI (or uncomment and edit the line below) to receive them.
# #SBATCH --mail-user=you@example.com

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p job_output

module load gcc 2>/dev/null || echo "[sbatch] WARN: 'module load gcc' failed; continuing"
module load cuda 2>/dev/null || echo "[sbatch] WARN: 'module load cuda' failed; continuing"

# Surface uv if it was installed under $HOME/.local/bin (uv installer default).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "[sbatch] ERROR: 'uv' not on PATH. Run 'bash slurm/setup.sh' from a login node first." >&2
    exit 1
fi

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
# Torchvision MNIST/CIFAR cache. Override BANDITDL_DATASET_ROOT in the calling
# environment (or in this script) when $HOME is quota-limited (e.g. on Jed).
export BANDITDL_DATASET_ROOT="${BANDITDL_DATASET_ROOT:-$HOME/BanditDL/banditdl/datasets/cache}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Inject device=cpu if the caller did not provide their own device override.
HAS_DEVICE=false
for arg in "$@"; do
    if [[ "$arg" == device=* ]]; then HAS_DEVICE=true; break; fi
done
if ! $HAS_DEVICE; then
    set -- device=cpu "$@"
fi

echo "[sbatch] job_id=$SLURM_JOB_ID job_name=$SLURM_JOB_NAME node=$SLURMD_NODENAME"
echo "[sbatch] cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
echo "[sbatch] hf_cache=$HF_DATASETS_CACHE"
echo "[sbatch] hydra overrides: $*"

srun --cpu-bind=cores uv run -m banditdl "$@"
