#!/bin/bash -l
# Submit a parameterized sweep on a CPU cluster (e.g. Jed) by looping sbatch.
#
# Usage:
#   bash slurm/submit_sweep_cpu.sh <sweep_name>
#
# Sweep names defined below: cifar_dirichlet, femnist_pool_dirichlet,
# cifar_grouped, femnist_pool_grouped, femnist_writer.
#
# This is a thin wrapper around `sbatch slurm/sbatch_banditdl_cpu.sh ...`.
# Each combination becomes its own job. Edit the axis arrays below to adjust scope.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

SWEEP="${1:-}"
if [ -z "$SWEEP" ]; then
    echo "usage: $0 <cifar_dirichlet|femnist_pool_dirichlet|cifar_grouped|femnist_pool_grouped|femnist_writer|mnist_grouped_clustering|cifar_grouped_clustering|cifar_grouped_2x5_clustering|femnist_pool_clustering>"
    exit 1
fi

# Shared axes
SAMPLERS=(uniform bandit exp3)
SAMPLINGS=(0.1 0.2)
SEEDS=(0 1 2)

submit_one() {
    local name="$1"; shift
    local override="$1"; shift
    echo "[submit] $name -- $override"
    sbatch --job-name="$name" --time=02:00:00 slurm/sbatch_banditdl_cpu.sh $override
}

count=0
case "$SWEEP" in
  cifar_dirichlet)
    NODES=30
    ROUNDS=500
    for alpha in dirichlet_alpha0.5 dirichlet_alpha1 dirichlet_alpha10; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="cifar_${alpha}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=cifar10 optimization=opt_cifar10 sampler=$sampler heterogeneity=$alpha topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  femnist_pool_dirichlet)
    NODES=30
    ROUNDS=300
    for alpha in dirichlet_alpha0.5 dirichlet_alpha1 dirichlet_alpha10; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="fmpool_${alpha}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=femnist_pool optimization=opt_femnist sampler=$sampler heterogeneity=$alpha topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  cifar_grouped)
    NODES=30
    ROUNDS=500
    for group in pathological_5g_2c pathological_2g_5c; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="cifar_${group}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=cifar10 optimization=opt_cifar10 sampler=$sampler heterogeneity=$group topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  femnist_pool_grouped)
    NODES=30
    ROUNDS=300
    for group in pathological_5g_2c pathological_2g_5c; do
      for sampler in "${SAMPLERS[@]}"; do
        for sampling in "${SAMPLINGS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            name="fmpool_${group}_${sampler}_s${sampling}_seed${seed}"
            submit_one "$name" \
              "dataset=femnist_pool optimization=opt_femnist sampler=$sampler heterogeneity=$group topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
            count=$((count + 1))
          done
        done
      done
    done
    ;;
  femnist_writer)
    NODES=30
    ROUNDS=300
    for sampler in "${SAMPLERS[@]}"; do
      for sampling in "${SAMPLINGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          name="fmwriter_${sampler}_s${sampling}_seed${seed}"
          submit_one "$name" \
            "dataset=femnist optimization=opt_femnist sampler=$sampler topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS seed=$seed"
          count=$((count + 1))
        done
      done
    done
    ;;
  cifar_grouped_clustering)
    # Cluster-formation study on CIFAR-10: 30 nodes, disjoint label groups
    # (pathological_5g_2c -> 5 clusters of 6 nodes, 2 labels each), bandit sampler,
    # no adversaries. Sweeps topology.sampling x seed. Evaluates every 20 rounds.
    # Uses opt_cifar10 (CrossEntropyLoss + lr decay) to match cnn_cifar's logits
    # output -- the default opt_mnist (NLLLoss) diverges to NaN here.
    NODES=30
    ROUNDS=500
    LOCAL_SAMPLINGS=(0.05 0.1 0.2 0.3 0.5)
    LOCAL_SEEDS=(0 1 2)
    for sampling in "${LOCAL_SAMPLINGS[@]}"; do
      for seed in "${LOCAL_SEEDS[@]}"; do
        name="cifargrp_bandit_s${sampling}_seed${seed}"
        submit_one "$name" \
          "dataset=cifar10 optimization=opt_cifar10 sampler=bandit heterogeneity=pathological_5g_2c adversary=none topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS evaluation.evaluation_delta=20 seed=$seed"
        count=$((count + 1))
      done
    done
    ;;
  cifar_grouped_2x5_clustering)
    # Cluster-formation study on CIFAR-10 with coarser clusters:
    # pathological_2g_5c -> 2 disjoint clusters of 15 nodes, 5 labels each, no overlap.
    # Same grid (5 samplings x 3 seeds) as cifar_grouped_clustering for direct
    # comparison against pathological_5g_2c. Uses opt_cifar10 (CrossEntropy + lr decay).
    NODES=30
    ROUNDS=500
    LOCAL_SAMPLINGS=(0.05 0.1 0.2 0.3 0.5)
    LOCAL_SEEDS=(0 1 2)
    for sampling in "${LOCAL_SAMPLINGS[@]}"; do
      for seed in "${LOCAL_SEEDS[@]}"; do
        name="cifar2x5_bandit_s${sampling}_seed${seed}"
        submit_one "$name" \
          "dataset=cifar10 optimization=opt_cifar10 sampler=bandit heterogeneity=pathological_2g_5c adversary=none topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS evaluation.evaluation_delta=20 seed=$seed"
        count=$((count + 1))
      done
    done
    ;;
  femnist_pool_clustering)
    # Cluster-formation study on FEMNIST (pool mode): 30 nodes, disjoint label
    # groups via pathological_5g_2c -> 5 clusters of 6 nodes, 2 labels each.
    # Sweeps topology.sampling x seed with the bandit sampler, no adversaries.
    # Mirrors cifar_grouped_clustering's grid for direct comparison.
    NODES=30
    ROUNDS=500
    LOCAL_SAMPLINGS=(0.05 0.1 0.2 0.3 0.5)
    LOCAL_SEEDS=(0 1 2)
    for sampling in "${LOCAL_SAMPLINGS[@]}"; do
      for seed in "${LOCAL_SEEDS[@]}"; do
        name="fmpoolgrp_bandit_s${sampling}_seed${seed}"
        submit_one "$name" \
          "dataset=femnist_pool optimization=opt_femnist sampler=bandit heterogeneity=pathological_5g_2c adversary=none topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS evaluation.evaluation_delta=20 seed=$seed"
        count=$((count + 1))
      done
    done
    ;;
  mnist_grouped_clustering)
    # Cluster-formation study: 30 MNIST nodes, disjoint label groups (pathological_5g_2c ->
    # 5 clusters of 6 nodes, 2 labels each), bandit sampler, no adversaries.
    # Sweeps topology.sampling x seed. Logs/evaluates every 10 rounds.
    NODES=30
    ROUNDS=800
    LOCAL_SAMPLINGS=(0.05 0.1 0.2 0.3 0.5)
    LOCAL_SEEDS=(0 1 2)
    for sampling in "${LOCAL_SAMPLINGS[@]}"; do
      for seed in "${LOCAL_SEEDS[@]}"; do
        name="mnistgrp_bandit_s${sampling}_seed${seed}"
        submit_one "$name" \
          "dataset=mnist sampler=bandit heterogeneity=pathological_5g_2c adversary=none topology.nodes=$NODES topology.sampling=$sampling optimization.rounds=$ROUNDS evaluation.evaluation_delta=20 seed=$seed"
        count=$((count + 1))
      done
    done
    ;;
  *)
    echo "unknown sweep: $SWEEP"
    exit 1
    ;;
esac

echo "[submit] queued $count jobs"
