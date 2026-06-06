# Running BanditDL on EPFL Slurm clusters

Scripts:

| File | Where to run | Purpose |
| --- | --- | --- |
| `slurm/setup.sh` | **Login node** | One-time: install `uv`, sync deps, optionally pre-download FEMNIST. |
| `slurm/sbatch_banditdl_gpu.sh` | **Login node** (via `sbatch`) | Submits one GPU training run (e.g. Izar). Takes Hydra overrides as args. |
| `slurm/sbatch_banditdl_cpu.sh` | **Login node** (via `sbatch`) | Submits one CPU-only training run (e.g. Jed). Takes Hydra overrides as args. |
| `slurm/sbatch_banditdl_optuna_gpu.sh` | **Login node** (via `sbatch`) | Submits one GPU Optuna sweep. Takes sweep Hydra overrides as args. |

The single-run `sbatch_banditdl_gpu.sh` and `sbatch_banditdl_cpu.sh` scripts are
identical except for the partition/GPU directives and the auto-injected
`device=` default (`cuda` for GPU, `cpu` for CPU). The examples below use the
GPU script; swap in `sbatch_banditdl_cpu.sh` to run the same command on a CPU
partition.

## 1. One-time bootstrap (login node)

```bash
# CIFAR / MNIST only — no FEMNIST download yet
bash slurm/setup.sh

# CIFAR / MNIST + pre-download FEMNIST (~700 MB into ~/.cache/huggingface/datasets)
bash slurm/setup.sh --femnist
```

The bootstrap loads `gcc` + `cuda`, installs `uv` to `~/.local/bin` if missing, and runs `uv sync` to populate `.venv`. Re-run safely; each step is idempotent.

If your `$HOME` quota is tight, redirect the HuggingFace cache once and re-run:

```bash
export HF_DATASETS_CACHE=$SCRATCH/banditdl/hf-cache
bash slurm/setup.sh --femnist
```

You will also need to `export HF_DATASETS_CACHE=...` in any sbatch invocation — the easiest path is to put that `export` in `~/.bashrc` (jobs use `bash -l`, so login-shell init files are sourced).

## 2. Submitting a single run

```bash
# CIFAR-10, epsilon-greedy bandit, seed 0
sbatch slurm/sbatch_banditdl_gpu.sh dataset=cifar10 sampler=bandit seed=0

# FEMNIST writer-per-node, default sampler, 30 nodes
sbatch slurm/sbatch_banditdl_gpu.sh dataset=femnist topology.nodes=30

# CIFAR-10 with the grouped clustering partition
sbatch slurm/sbatch_banditdl_gpu.sh \
    dataset=cifar10 sampler=bandit heterogeneity=pathological_5g_2c topology.nodes=30 seed=0
```

All positional args are forwarded to `uv run -m banditdl`. `device=cuda` (GPU script) or `device=cpu` (CPU script) is added automatically unless you provide your own `device=` override.

Override sbatch directives via the CLI. Email notifications are **not** baked
into the committed scripts, so add them here if you want them:

```bash
sbatch --time=04:00:00 --job-name=fast_cifar \
    --mail-type=END,FAIL --mail-user=you@epfl.ch \
    slurm/sbatch_banditdl_gpu.sh dataset=cifar10
```

## 3. Sweeps

Two patterns work; pick by sweep size.

**Pattern A — one sbatch per combination (best for >1h runs, parallelizes across GPUs):**

```bash
for seed in 0 1 2; do
  for sampler in uniform bandit exp3; do
    sbatch slurm/sbatch_banditdl_gpu.sh \
        dataset=cifar10 topology=dynamic sampler=$sampler seed=$seed
  done
done
```

Each combination becomes its own job and runs in parallel as cluster capacity allows. Output dirs are independent (Hydra writes `.hydra_runs/<date>/<time>/`).

**Pattern B — single sbatch with Hydra multirun (best for sweeps that fit in one walltime):**

```bash
sbatch --time=24:00:00 slurm/sbatch_banditdl_gpu.sh -m \
    dataset=cifar10 topology=dynamic \
    sampler=uniform,bandit,exp3 seed=0,1,2
```

Hydra's `-m` flag enumerates the Cartesian product **sequentially** inside one job. Output goes to `.hydra_multirun/<date>/<time>/<idx>/`. Use this when total walltime is manageable; otherwise prefer Pattern A.

**Pattern C — Optuna sweep (dedicated sweep entry point):**

```bash
sbatch --time=24:00:00 slurm/sbatch_banditdl_optuna_gpu.sh optuna=sweep
```

Use `slurm/submit_sweep_gpu.sh` only for its hard-coded shell sweeps
(`cifar_dirichlet`, `femnist_pool_dirichlet`, etc.); it is not the Optuna entry
point.

## 4. Output

- **Job logs**: `job_output/banditdl_<jobid>.txt` (created by `--output=job_output/...`).
- **Experiment artifacts**: `.hydra_runs/<date>/<time>/` (single runs) or `.hydra_multirun/<date>/<time>/<idx>/` (Pattern B). Inside each: `results/` (numpy arrays), `plots/` (auto-generated PNGs), `.hydra/` (resolved config).

Both `job_output/` and the Hydra dirs are gitignored.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ERROR: 'uv' not on PATH` | Forgot the bootstrap | `bash slurm/setup.sh` on a login node |
| `CUDA out of memory` | Too-large batch on V100 | Override: `optimization.batch_size=16` |
| `Could not load 'override'` | Missing `conf/override.yaml` | Create one per the root README, or pass overrides via CLI |
| FEMNIST stalls on first run | First call downloads the HF dataset | Pre-download with `bash slurm/setup.sh --femnist` |
| Job dies at 12h walltime | Default time limit | Override: `sbatch --time=24:00:00 slurm/sbatch_banditdl_gpu.sh ...` |

## 6. Defaults at a glance

The sbatch directives in `sbatch_banditdl_gpu.sh`:

- `--partition=gpu --gres=gpu:1` (the CPU script uses `--partition=academic` and no GPU)
- `--nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G`
- `--time=12:00:00` (CIFAR 2000 rounds finishes in well under that on a V100)
- `--mail-type=END,FAIL` — but **no** `--mail-user`; pass your own on the `sbatch` CLI
- `srun --cpu-bind=cores`

All overridable via the `sbatch` CLI.
