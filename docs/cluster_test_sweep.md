# Cluster Test Sweep

This guide explains how to run a small Optuna sweep on one Slurm node with
four NVIDIA GH200 120GB GPUs. The goal is to validate the cluster setup before
spending time on a production sweep.

The recommended execution model is:

- one Slurm job,
- one node,
- four visible GPUs,
- four Optuna worker processes for the first test,
- one worker process assigned to each GPU.

After the small test passes, increase either the sweep size or `optuna.workers`.

## 1. Check The Local Branch

Before submitting, make sure the branch contains the code you want to run:

```bash
git status --short
git log --oneline -5
```

If `git status` shows modified config files, make sure they are intentional.
For example, an edited `conf/optuna/alpha_grid.yaml` changes the sweep grid.

## 2. Bootstrap The Environment

Run this once from the repository root on the login node:

```bash
bash slurm/setup.sh
```

If `$HOME` quota is limited, put datasets and HuggingFace caches on scratch:

```bash
export HF_DATASETS_CACHE=$SCRATCH/banditdl/hf-cache
export BANDITDL_DATASET_ROOT=$SCRATCH/banditdl/datasets
bash slurm/setup.sh
```

If you use scratch paths, add the exports to `~/.bashrc` so Slurm jobs see the
same paths. The sbatch scripts use `bash -l`, so login-shell startup files are
loaded.

## 3. Submit A Tiny 4-GPU Test Sweep

Use the Optuna sweep entry point:

```bash
sbatch \
  --job-name=bdl_test_sweep \
  --gres=gpu:4 \
  --cpus-per-task=32 \
  --mem=128G \
  --time=00:30:00 \
  slurm/sbatch_banditdl_optuna_gpu.sh \
  optuna=sanitysweep \
  optuna.workers=4 \
  dataset=mnist \
  topology.nodes=5 \
  optimization.rounds=3 \
  evaluation.evaluation_delta=1 \
  'optuna.seeds=[123]' \
  identical_initialization=false
```

If your cluster requires a typed GPU request for GH200, replace:

```bash
--gres=gpu:4
```

with the local Slurm syntax, for example:

```bash
--gres=gpu:gh200:4
```

The test uses:

- `optuna=sanitysweep`: small categorical grid.
- `optuna.workers=4`: four spawned training processes.
- `device=cuda`: injected automatically by `slurm/sbatch_banditdl_optuna_gpu.sh`.
- `topology.nodes=5`: very small decentralized system.
- `optimization.rounds=3`: fast run, enough to write metrics.
- `evaluation.evaluation_delta=1`: evaluation at every round.
- `identical_initialization=false`: current default, explicit for reproducibility.

## 4. Monitor The Job

After `sbatch`, Slurm prints a job id:

```text
Submitted batch job 123456
```

Monitor queue state:

```bash
squeue -j 123456
```

Follow the job log:

```bash
tail -f job_output/banditdl_optuna_123456.txt
```

In the log, verify CUDA visibility and worker assignment. You want to see
something like:

```text
[sbatch] gpus=0,1,2,3
[optuna] configurations=...
[optuna] workers=4 devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
```

If only one GPU is visible, check the Slurm `--gres` request and the cluster's
GPU resource syntax.

## 5. Inspect The Output Directory

Optuna sweep outputs are written under:

```text
.optuna_runs/<profile>/<timestamp>_<slurm-job-id>/
```

Find the latest sanity sweep:

```bash
RUN=$(ls -td .optuna_runs/sanity-grid/* | head -1)
echo "$RUN"
```

Check the directory structure:

```bash
find "$RUN" -maxdepth 3 -type d | head -40
```

Check that core metrics were written:

```bash
find "$RUN" -name "validation_accuracy.npy" | head
find "$RUN" -name "validation_loss.npy" | head
find "$RUN" -name "global_accuracy.npy" | head
find "$RUN" -name "global_loss.npy" | head
find "$RUN" -name "sampler_weights.npy" | head
find "$RUN" -name "sampler_probabilities.npy" | head
find "$RUN" -name "sampler_states.jsonl" -o -name "sampler_states_by_seed.jsonl" | head
```

Check that sweep artifacts were generated:

```bash
find "$RUN" -type f -path "*sweep_artifacts*" | head -40
```

For one trial, inspect files:

```bash
TRIAL=$(find "$RUN/trials" -type d -name results | head -1)
echo "$TRIAL"
ls "$TRIAL"
```

Expected files include:

- `validation_accuracy.npy`
- `validation_loss.npy`
- `global_accuracy.npy`
- `global_loss.npy`
- `train_loss.npy`
- `sampler_weights.npy`
- `sampler_probabilities.npy`
- `sampler_states.jsonl` or `sampler_states_by_seed.jsonl`
- `reward_algorithm.npy`
- `reward_oracle.npy`
- `regret.npy`

Some files depend on run settings. For example, `test_accuracy.npy` only exists
when `evaluation.evaluate_test=true`.

## 6. Run A Moderate Sweep

After the tiny test succeeds, run a moderate sweep. Start with one worker per
GPU:

```bash
sbatch \
  --job-name=bdl_alpha_moderate \
  --gres=gpu:4 \
  --cpus-per-task=48 \
  --mem=256G \
  --time=06:00:00 \
  slurm/sbatch_banditdl_optuna_gpu.sh \
  optuna=alpha_grid \
  optuna.workers=4 \
  dataset=cifar10 \
  optimization=opt_cifar10 \
  topology.nodes=15 \
  optimization.rounds=50 \
  evaluation.evaluation_delta=10 \
  'optuna.seeds=[123]' \
  identical_initialization=false
```

This is still a validation run, not the final production run. It checks that:

- CIFAR-10 loads correctly,
- GPU memory is sufficient,
- metrics and plots work for a realistic model,
- trial folders are readable,
- Optuna can complete more than a tiny sanity grid.

## 7. Try GPU Oversubscription Only After The Moderate Test

The sweep runner assigns workers to visible GPUs round-robin. With four GPUs:

```bash
optuna.workers=8
```

means:

```text
cuda:0 -> 2 workers
cuda:1 -> 2 workers
cuda:2 -> 2 workers
cuda:3 -> 2 workers
```

GH200 120GB has a lot of memory, but oversubscription can still hurt throughput
or cause OOM depending on model, batch size, number of nodes, and dataset.

Try:

```bash
sbatch \
  --job-name=bdl_alpha_moderate_w8 \
  --gres=gpu:4 \
  --cpus-per-task=64 \
  --mem=384G \
  --time=06:00:00 \
  slurm/sbatch_banditdl_optuna_gpu.sh \
  optuna=alpha_grid \
  optuna.workers=8 \
  dataset=cifar10 \
  optimization=opt_cifar10 \
  topology.nodes=15 \
  optimization.rounds=50 \
  evaluation.evaluation_delta=10 \
  'optuna.seeds=[123]' \
  identical_initialization=false
```

Use `optuna.workers=8` only if the 4-worker run leaves substantial GPU memory
and the cluster allows enough CPU cores.

## 8. Resume An Interrupted Sweep

To resume, reuse the existing Hydra output directory with `hydra.run.dir`.

Example:

```bash
sbatch \
  --job-name=bdl_resume \
  --gres=gpu:4 \
  --cpus-per-task=48 \
  --mem=256G \
  --time=06:00:00 \
  slurm/sbatch_banditdl_optuna_gpu.sh \
  hydra.run.dir=/absolute/path/to/.optuna_runs/alpha-grid/2026-..._123456 \
  optuna=alpha_grid \
  optuna.workers=4 \
  dataset=cifar10 \
  optimization=opt_cifar10 \
  topology.nodes=15 \
  optimization.rounds=50 \
  evaluation.evaluation_delta=10 \
  'optuna.seeds=[123]' \
  identical_initialization=false
```

The sweep runner reads the existing Optuna study database and skips completed
configuration IDs.

Use the same overrides when resuming. Changing the search grid in the same
output directory can invalidate the manifest and should be avoided.

## 9. Production Run Checklist

Before launching a long sweep:

- Confirm `git status --short`.
- Confirm the exact `conf/optuna/<profile>.yaml` choices.
- Confirm `identical_initialization=false` is intended.
- Confirm `aggregator=mean` if you want unfiltered model averaging.
- Confirm `optuna.seeds`.
- Confirm `optimization.rounds`.
- Confirm `evaluation.evaluation_delta`.
- Confirm `topology.nodes` and `topology.sampling`.
- Confirm the reward strategy, e.g. `sampler.reward=update_cosine_similarity`.
- Run the tiny sanity sweep.
- Run the moderate CIFAR sweep.
- Inspect metrics and plots.
- Only then scale walltime, workers, and grid size.

## 10. Troubleshooting

### `uv` Not Found

Run:

```bash
bash slurm/setup.sh
```

Then resubmit.

### Only One GPU Visible

Check the job log:

```bash
grep gpus job_output/banditdl_optuna_<jobid>.txt
```

If only one GPU is visible, fix the Slurm GPU request:

```bash
--gres=gpu:4
```

or the cluster-specific equivalent:

```bash
--gres=gpu:gh200:4
```

### CUDA Out Of Memory

Reduce one or more of:

- `optuna.workers`
- `topology.nodes`
- `optimization.batch_size`
- model size

Start from:

```bash
optuna.workers=4
```

before trying `8`.

### Sweep Creates Too Many Trials

Check the grid before submitting:

```bash
cat conf/optuna/alpha_grid.yaml
```

The number of trials grows combinatorially with the number of swept axes. Keep
production grids explicit and small enough to reason about.

### Missing `global_loss.npy`

Make sure you are running a commit that includes:

```text
feat: save subsampled global loss
```

Then check:

```bash
find "$RUN" -name "global_loss.npy" | head
```

### No `test_accuracy.npy`

This is expected unless:

```bash
evaluation.evaluate_test=true
```

Final full-test evaluation can be expensive, so it is optional.
