#!/bin/bash
#SBATCH --job-name=blindtasks_eval
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=4:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch --export=ALL,MODEL_FAMILY=apertus slurm/run_blindtasks.slurm.template

set -euo pipefail
ulimit -n 65535

echo "### Job started at $(date '+%Y-%m-%d %H:%M:%S') ###"

: "${MODEL_FAMILY:=apertus}" # apertus or qwen

unset CC CXX CPP LD CPPFLAGS CFLAGS CXXFLAGS LDFLAGS

BASE="$(pwd)"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH}/cache}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" logs

echo "### Container package auto-install (idempotent) ###"
ensure_pkg() {
    local mod="$1" pip_name="${2:-$1}"
    if ! python3 -c "import $mod" 2>/dev/null; then
        echo "  installing $pip_name (missing: $mod)"
        python3 -m pip install --user --no-cache-dir "$pip_name"
    else
        echo "  ok: $mod"
    fi
}
if [ "$MODEL_FAMILY" == "qwen" ]; then
    ensure_pkg qwen_vl_utils qwen-vl-utils
fi

echo "### Pre-flight Checks ###"
ERROR_COUNT=0
check_path() {
    if [ -e "$2" ]; then echo "  OK: $1"
    else echo "  FAIL: $1 ($2)"; ((ERROR_COUNT++)); fi
}

check_path "reward fn"            "$BASE/rewards/blindtasks_reward.py"
check_path "blindtasks scorer"    "$BASE/evaluation/blindtasks_scorer.py"
check_path "blindtasks package"   "$BASE/blindtasks/__init__.py"

if [ "$MODEL_FAMILY" == "apertus" ]; then
    if [ -z "${APERTUS_CHECKPOINT:-}" ]; then
        echo "  FAIL: APERTUS_CHECKPOINT env var not set"
        ((ERROR_COUNT++))
    else
        check_path "Apertus runner"       "$BASE/inference/run_blindtasks_apertus.py"
        check_path "Apertus checkpoint"   "$APERTUS_CHECKPOINT/config.json"
    fi
else
    check_path "BlindTasks runner"    "$BASE/inference/run_blindtasks.py"
fi

python3 -c "import pyarrow, PIL, numpy, torch, transformers, vllm" 2>/dev/null \
    && echo "  OK: required packages imported successfully" \
    || { echo "  FAIL: required Python deps not importable"; ((ERROR_COUNT++)); }

[ "$ERROR_COUNT" -gt 0 ] && { echo "ERROR: $ERROR_COUNT pre-flight checks failed."; exit 1; }

echo "### Environment ###"
which python3
python3 - <<'PY'
import sys, torch, transformers, vllm
print("python :", sys.version.split()[0], sys.executable)
print("torch  :", torch.__version__, "cuda?", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
print("trf    :", transformers.__version__)
print("vllm   :", vllm.__version__)
PY

echo ""
echo "### GPU Info ###"
nvidia-smi --list-gpus
echo ""

echo "### Starting BlindTasks $MODEL_FAMILY vLLM inference ###"
echo "ARGS=$*"
OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-$BASE/results/blindtasks/${MODEL_FAMILY}_passk}"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--output-directory" && $((i+1)) -lt ${#ARGS[@]} ]]; then
        OUTPUT_DIRECTORY="${ARGS[i+1]}"
    fi
done

if [ "$MODEL_FAMILY" == "apertus" ]; then
    CMD=(
        python3 inference/run_blindtasks_apertus.py
        --model "$APERTUS_CHECKPOINT"
        --output-directory "$OUTPUT_DIRECTORY"
        # Make sure to pass the dtaaset path --dataset-path
    )
else
    CMD=(
        python3 inference/run_blindtasks.py
        --output-directory "$OUTPUT_DIRECTORY"
        --tensor-parallel-size 4
    )
fi
CMD+=("${ARGS[@]}")
"${CMD[@]}"

echo "### Job ended at $(date '+%Y-%m-%d %H:%M:%S') ###"