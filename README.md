# banditdl

Hydra-multirun experiments for Byzantine-resilient decentralized learning.

## Setup

```bash
uv sync
```

If `uv` cache is not writable in your environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## Run One Experiment

```bash
uv run -m banditdl
```

Example overrides:

```bash
uv run -m banditdl dataset=mnist topology=dynamic sampler=uniform topology.nodes=100 topology.sampling=0.05 seed=0
```

Runs print lightweight progress to stdout: start metadata, result directory, periodic decentralized-learning rounds, evaluation accuracy when available, and completion.

## Local Hydra Override

`conf/config.yaml` intentionally loads `conf/override.yaml`. That file is ignored by Git and is where each person puts machine-specific defaults such as device and output directories.

Create `conf/override.yaml` locally:

```yaml
defaults:
  - override /dataset: mnist
  - override /topology: dynamic
  - override /sampler: uniform
  - override /adversary: none

seed: 0
device: mps

hydra:
  run:
    dir: .hydra_runs_override/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: .hydra_multirun_override/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

## Run Sweeps (Hydra Multirun)

Hydra does orchestration. The custom in-repo scheduler is no longer the main path.

## Run Sweeps (Optuna)

Launch categorical grid sweeps with Hydra output folders:

```bash
uv run python -m banditdl.experiments.sweep
```

Defaults (`conf/sweep.yaml`) compose `conf/config.yaml` plus `conf/optuna/sanitysweep.yaml`.

Behavior:
- Enumerates every valid Cartesian combination of **categorical** `optuna.search_space` entries (respecting optional `when:` guards).
- Queues each combo via Optuna `enqueue_trial`, runs one training job per trial.
- Writes trial artifacts under `<hydra_run>/trials/<param_tokens>/results/` (numpy arrays mirror standard runs).
- Tracks validation accuracy from `results/validation`, selects the best trial, then re-runs it once with test evaluation under `<hydra_run>/best_trial_test_eval/results`.
- After all trials finish, renders default sweep plots under `<hydra_run>/sweep_artifacts/`.

Sweep plotting is intentionally defined in Python, not YAML. The default sweep plot set lives in `banditdl/utils/plot_sweep_base.py` and currently generates:

| Mode | Meaning |
| --- | --- |
| `per_parameter` | WAY 2 style curves (per-axis grids described in `banditdl/utils/plot_sweep_perparam.py`). |
| `all_together` | WAY 1 style overlays (`banditdl/utils/plot_sweep_alltogether.py`). |
| `heatmap` | Pairwise heatmaps (`banditdl/utils/plot_sweep_heatmap.py`). |

Default sweep reductions over timesteps and nodes:
- `avg`: arithmetic mean over all timesteps and all nodes.
- `worse`: worst observed value across timesteps and nodes; uses max for
  metrics where higher is worse (`regret`, `normalized_regret`,
  `neighbor_disagreement`, `consensus_drift`, `validation_losses`,
  `train_losses`) and min otherwise (accuracies, rewards).

Plot outputs are written under:

```text
<hydra_run>/sweep_artifacts/
  per_parameter/direction=avg/...
  per_parameter/direction=worse/...
  all_together/direction=avg/...
  all_together/direction=worse/...
  heatmap/direction=avg/...
  heatmap/direction=worse/...
```

If a particular `(metric, mode, direction, fixed-axes)` combination has no
plottable points (missing arrays or no matching trials), the plot is **skipped
with a warning** instead of writing a blank PNG.

To change sweep plots, edit the Python defaults or run a separate plotting script after the sweep. Experiment parameters and Optuna search spaces remain Hydra-controlled.

```bash
uv run python -m banditdl.experiments.sweep optuna=sweep
```

Note on Hydra composition: `conf/override.yaml` is loaded as the last entry of
`conf/config.yaml`'s defaults list. `conf/sweep.yaml` then composes `config`
first and selects the bundled `optuna` group afterwards. `override.yaml` can override fields owned by `config.yaml` and its sub-groups, while `optuna.*` is controlled by the selected Optuna config or CLI overrides.

### Ad-hoc Sweep From CLI

```bash
uv run -m banditdl -m \
  dataset=mnist \
  topology=dynamic \
  sampler=uniform,bandit \
  seed=0,1 \
  topology.nodes=50,100 \
  topology.sampling=0.03,0.05 \
  sampler.params.epsilon=0.1,0.3 \
  optimization.nb_local_steps=1,3
```

## Existing Config Groups

Dataset:
- `mnist`
- `cifar10`

Topology:
- `dynamic`
- `fixed_cs`

Sampler:
- `uniform`
- `bandit` (epsilon-greedy profile)
- `epsilon_greedy`
- `exp3`

Adversary:
- `none`
- `alie`

## Config Reference

Hydra config lives in `conf/`. The main entry point is `conf/config.yaml`.

Inspect the resolved config before launching a large run:

```bash
uv run -m banditdl --cfg job
```

### Top-Level Config

- `dataset`: dataset/model config group.
- `topology`: decentralized topology config group.
- `sampler`: dynamic neighbor sampler config group.
- `adversary`: Byzantine/adversarial setup config group.
- `aggregator`: robust aggregation config group.
- `heterogeneity`: data heterogeneity config group.
- `optimization`: local optimizer/training schedule config group.
- `evaluation`: evaluation cadence config group.
- `seed`: random seed. Use comma-separated values under `-m` for sweeps.
- `device`: `auto`, `cpu`, or a torch device string such as `cuda`.

### Dataset Config

Dataset configs are in `conf/dataset/`.

- `dataset`: dataset name passed to the loader. Common values: `mnist`, `cifar10`.
- `model`: model constructor from `banditdl/data/models.py`, for example `cnn_mnist` or `cnn_cifar_old`.

### Heterogeneity Config

Heterogeneity configs are in `conf/heterogeneity/`.

- `alpha`: Dirichlet data heterogeneity parameter passed as `dirichlet-alpha`.
- `numb_labels`: number of dataset labels.

### Optimization Config

Optimization configs are in `conf/optimization/`.

- `batch_size`: training batch size.
- `loss`: torch loss class name, for example `NLLLoss`.
- `learning_rate`: optional SGD learning rate. Engine default is `0.5`.
- `learning_rate_decay`: optional worker learning-rate decay scale.
- `learning_rate_decay_delta`: optional step interval for learning-rate decay checks.
- `weight_decay`: SGD weight decay.
- `momentum_worker`: worker momentum.
- `rounds`: number of communication/training rounds. This is the sampler horizon.
- `nb_local_steps`: local SGD steps per communication round.

### Topology Config

Topology configs are in `conf/topology/`.

- `nodes`: total simulated participants, including Byzantine participants.
- `sampling`: dynamic sampling ratio. Dynamic topologies define `sampling`.
- `degree`: fixed-graph degree target. Fixed topologies define `degree`.
- `method`: fixed-graph method, for example `cs+`.

### Sampler Config

Sampler configs are in `conf/sampler/`. They are used by dynamic topologies.

- `name`: sampler implementation, for example `uniform`, `epsilon_greedy`, or `exp3`.
- `reward`: reward strategy for learning samplers. Current value: `parameter_distance`.
- `params`: sampler-specific parameters, for example `epsilon` and `initial_value`.

Shared runtime facts such as `topology.nodes`, sampled-neighbor count, `optimization.rounds`, and seed are passed to samplers through runtime context rather than duplicated in sampler config.

### Adversary Config

Adversary configs are in `conf/adversary/`.

- `byzcount`: number of declared and real Byzantine workers currently instantiated by the Hydra adapter.
- `byzantine_budget`: robustness budget `b_hat`. If unset/null, defaults to `byzcount`.
- `attack`: Byzantine attack name or `null`. Available attacks include `SF`, `LF`, `FOE`, `ALIE`, `mimic`, `auto_ALIE`, `auto_FOE`, `inf`.

### Aggregator Config

Aggregator configs are in `conf/aggregator/`.

- `pre-aggregator`: optional first-stage robust aggregation rule, commonly `nnm`.
- `aggregator`: robust aggregator, commonly `average` or `trmean`.
- `rag`: robust aggregation flag. Dynamic runs force this to `true`.

Available robust aggregators include `average`, `trmean`, `median`, `geometric_median`, `krum`, `multi_krum`, `nnm`, `bucketing`, `pmk`, `cc`, `mda`, `mva`, `monna`, `meamed`.

### Sweep Syntax

Ad-hoc sweep:

```bash
uv run -m banditdl -m \
  dataset=mnist,cifar10 \
  topology=dynamic \
  sampler=uniform,bandit \
  topology.nodes=50,100 \
  topology.sampling=0.03,0.05 \
  adversary=none \
  seed=0,1,2
```

Hydra takes the Cartesian product of comma-separated override values.

## How To Create A New Experiment

1. Add or copy a config in `conf/dataset/`, `conf/topology/`, `conf/sampler/`, `conf/aggregator/`, `conf/heterogeneity/`, `conf/optimization/`, or `conf/adversary/`.
2. Compose them from the CLI with Hydra overrides.

Example:

```yaml
# conf/sampler/my_bandit.yaml
name: epsilon_greedy
reward: parameter_distance
params:
  epsilon: 0.1
  initial_value: 0.0
```

```yaml
# conf/topology/my_dynamic.yaml
nodes: 100
sampling: 0.05
```

Run it:

```bash
uv run -m banditdl dataset=mnist topology=my_dynamic sampler=my_bandit adversary=none
```

## Plot Saved Results

Each Hydra run writes artifacts directly in its run folder:
- `<hydra_run>/results/`: raw metrics and arrays (`validation`, `validation_worst`, `test` (optional), `*.npy`).
- `<hydra_run>/plots/`: auto-generated plots for all supported metrics.

Example run folder:

```text
.hydra_runs_override/2026-05-05/12-26-01/
  .hydra/
  hydra_run.log
  results/
  plots/
```

Plotting logic is code-driven. Metric loading, transforms, and aggregations live in `banditdl/utils/metrics.py`; runtime figures are defined imperatively in `banditdl/utils/plotting.py`. The script `scripts/plot_results.py` remains as a thin offline CLI wrapper around those helpers.

Plot one run:

```bash
uv run python scripts/plot_results.py \
  .hydra_runs_override/<date>/<time>/results \
  -o .hydra_runs_override/<date>/<time>/plots/example.png
```

Compare multiple runs:

```bash
uv run python scripts/plot_results.py \
  .hydra_runs_override/<date>/<time-a>/results \
  .hydra_runs_override/<date>/<time-b>/results \
  --label uniform \
  --label bandit \
  -o comparison.png
```

Aggregate seed runs:

```bash
uv run python scripts/plot_results.py \
  .hydra_runs_override/<date>/*/results \
  --aggregate \
  --label "uniform mean" \
  -o uniform_seed_mean.png
```

Useful options:
- `--metric accuracies`: plot from `accuracies.npy` (default).
- `--metric validation`: plot average accuracy from `validation`.
- `--metric validation_worst`: plot worst-worker accuracy from `validation_worst`.
- `--metric test`: plot held-out test accuracy from `test` (single final point when available).
- `--metric eval|eval_worst`: legacy aliases for older run folders.
- `--metric regret`: plot regret against the best fixed neighbor subset in hindsight.
- `--metric normalized_regret`: plot time-averaged regret, derived from `regret.npy`.
- `--metric reward_algorithm|reward_oracle`: plot cumulative reward curves,
  normalized by sampled-neighbor count.
- `--metric reward_selected_min|reward_selected_max`: plot per-round selected
  neighbor reward extrema.
- `--metric neighbor_disagreement`: plot mean/median/max neighbor disagreement over rounds.
- `--metric consensus_drift`: plot mean/median/max drift from the global average model.
- `--metric sampler_aggressiveness`: plot KL to uniform plus min/max sampler probabilities.
- `--metric sampler_kl_to_uniform`: plot only the KL-to-uniform node aggregates.
- `--stat mean|worst`: choose mean worker or worst worker; for regret, worst means highest regret.
- `--legend outside|best|none`: choose legend placement; default keeps it below the plot.
- `--max-label-length 48`: cap auto-generated labels.

## Runtime Architecture

This section describes runtime execution logic and module interactions.

### Runtime Interaction Diagram

```mermaid
flowchart TD
    A[User: uv run -m banditdl ...] --> B[banditdl.__main__]
    B --> C[experiments.hydra_run]
    C --> D[Hydra config composition
conf/config.yaml + dataset/topology/sampler/...]

    D --> E{Hydra mode}
    E -->|single run| F[One composed config]
    E -->|multirun -m| G[Cartesian expansion from
hydra.sweeper.params + CLI overrides]

    F --> H[config_adapter builds one engine config]
    G --> H
    H --> X[hydra_run dispatches training engine]

    X --> I1[Training engine
experiments.engine::run_dynamic]
    X --> I2[Training engine
    experiments.engine::run_fixed]

    I1 --> J1[data.*
    models + dataset loaders]
    I1 --> K1[core.worker.dynamic
    local updates + neighbor sampling]
    K1 --> L1[core.robustness.*
    attacks + aggregators]

    I2 --> J2[data.*
    models + dataset loaders]
    I2 --> K2[core.worker.fixed
    fixed-graph updates]
    K2 --> L2[core.robustness.*
    attacks + summations]

    I1 --> M[Per-run result directory
    validation, validation_worst, logs]
    I2 --> M
```

### End-to-end Flow

1. You run `uv run -m banditdl ...`.
2. `banditdl.__main__` dispatches to `banditdl.experiments.hydra_run`.
3. Hydra composes config from `conf/`.
4. In multirun mode, Hydra generates one run per parameter combination.
5. For each run, `hydra_run` dispatches the corresponding training engine function.
6. Training engine (`experiments.engine`) executes and writes results.

### Responsibilities By Module

- `banditdl.experiments.hydra_run`
  - Hydra entry point.
  - Dispatches one composed run to the engine.

- `banditdl.experiments.config_adapter`
  - Converts composed Hydra config into legacy engine args.
  - Computes dynamic/fixed run mode, neighbor count, device, and run name.

- `banditdl.experiments.engine`
  - Per-run execution logic for dynamic/fixed paths.
  - Drives training/evaluation loops and persistence.

- `banditdl.core.worker.*`
  - Worker logic for local updates and communication.

- `banditdl.core.robustness.*`
  - Byzantine attacks and robust aggregation/summation rules.

- `banditdl.data.*`
  - Dataset loading/partitioning and model construction.

- `banditdl.core.sampling`
  - Neighbor sampler implementations and reward strategies.


### Terminology: Worker = Node

In this repository, a **worker** is one decentralized learning participant (node/client):
- it owns local train/test data loaders,
- performs local optimization steps,
- communicates with neighbors,
- applies robust aggregation logic under Byzantine settings.

Honest participants are modeled as `DynamicWorker`/`FixedGraphWorker`; Byzantine participants are modeled as explicit attack-only nodes.

### Decentralized Structure Diagram

```mermaid
flowchart LR
    subgraph Topology["Decentralized Topology (N workers)"]
        W0["Worker 0 (possibly Byzantine)"]
        W1["Worker 1"]
        W2["Worker 2"]
        W3["Worker 3"]
        W0 --- W1
        W1 --- W2
        W2 --- W3
        W3 --- W0
        W0 --- W2
    end

    W0 --> S["core.sampling: choose neighbors (dynamic samplers)"]
    W1 --> S
    W2 --> S
    W3 --> S

    S --> U["core.worker.*: local SGD/update + send/receive"]
    U --> A["core.robustness: attack model + robust aggregation"]
    A --> M["Updated model state per worker"]

    D["data.*: local shards + model ctor"] --> U
```

Interpretation:
- Each worker is a simulated node with its own local data and model copy.
- Communication is peer-to-peer, not centralized; each node exchanges updates with selected neighbors.
- Dynamic topologies re-sample neighbors each round through `core.sampling`.
- Received updates pass through Byzantine attack/aggregation logic before updating local state.

## Sampling / Bandit Hook Points

- `banditdl/core/sampling.py`
- `banditdl/experiments/engine.py`
- `banditdl/core/worker/`

Use the epsilon-greedy bandit sampler:

```bash
uv run -m banditdl \
  dataset=mnist \
  topology=dynamic \
  sampler=bandit \
  sampler.params.epsilon=0.1 \
  topology.sampling=0.05 \
  seed=0
```

Use EXP3:

```bash
uv run -m banditdl \
  dataset=mnist \
  topology=dynamic \
  sampler=exp3 \
  sampler.params.gamma=auto \
  topology.sampling=0.05 \
  seed=0
```

Current bandit feedback:
- each neighbor is one arm,
- MABWiser provides epsilon-greedy; EXP3 is implemented locally,
- dynamic workers update selected arms after receiving neighbor weights,
- reward is selected through a strategy object; the default is `parameter_distance`,
- `parameter_distance` uses `1 / (1 + parameter_distance)` against the local model before aggregation.

Dynamic runs also save hindsight diagnostics for every sampler, including uniform:
- `reward_algorithm.npy`: cumulative reward achieved by sampled neighbors,
  normalized by sampled-neighbor count.
- `reward_oracle.npy`: cumulative reward of the best fixed neighbor subset in
  hindsight, normalized by sampled-neighbor count.
- `regret.npy`: `reward_oracle - reward_algorithm`.
- time-averaged regret is derived from `regret.npy` when plotting.
- `reward_selected_min.npy`: per-round, per-node minimum reward among selected neighbors.
- `reward_selected_max.npy`: per-round, per-node maximum reward among selected neighbors.
- `selected_neighbors.npy`: sampled neighbors per round and worker.
- `oracle_neighbors.npy`: best fixed hindsight neighbors per round and worker.
- `sampler_kl_to_uniform.npy`: per-round, per-node KL divergence from the sampler distribution to uniform.
- `sampler_min_probability.npy`: per-round, per-node minimum sampler probability.
- `sampler_max_probability.npy`: per-round, per-node maximum sampler probability.

The automatic plot `plots/sampler_aggressiveness.png` shows:
- KL divergence to uniform aggregated across nodes by average, median, min, and max.
- The global min and max sampler probabilities per round.

This is intentionally small: sampler choice and sampler-specific parameters are Hydra-controlled, shared runtime facts are passed through `SamplerContext`, and reward design remains isolated behind the reward strategy API in `banditdl/core/sampling.py`. For EXP3, `gamma: auto` uses `optimization.rounds` as the known horizon.
