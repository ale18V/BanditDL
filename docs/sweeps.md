# Sweep Guide

This document covers:

- Hydra multirun
- Optuna grid sweeps
- search-space syntax
- sweep plotting config

## Hydra Multirun

Use Hydra multirun when you want a direct Cartesian product from CLI overrides.

Example:

```bash
uv run -m banditdl -m \
  topology=dynamic \
  sampler=uniform,exp3 \
  topology.sampling=0.03,0.05 \
  heterogeneity.alpha=0.1,0.5 \
  seed=0,1
```

Use this when:

- you already know the values you want to compare
- you want the simplest sweep workflow
- you do not need best-trial selection

## Optuna Sweep Runner

Main command:

```bash
uv run python -m banditdl.experiments.sweep
```

This uses `conf/sweep.yaml`, which composes:

- `conf/config.yaml`
- one `conf/optuna/*.yaml` profile

Profiles:

- `optuna=alpha_grid`
- `optuna=clustering_grid`
- `optuna=sanitysweep`
- `optuna=customsweep`

Example:

```bash
uv run python -m banditdl.experiments.sweep optuna=alpha_grid
```

## What the Sweep Runner Does

The runner executes exhaustive categorical grids. Continuous and sampled
search spaces are intentionally unsupported.

Workflow:

1. read `optuna.search_space`
2. enumerate every valid categorical combination
3. respect `when:` guards for conditional parameters
4. expand every configuration across `optuna.seeds`
5. run configurations concurrently and save isolated trial attempts
6. persist the Optuna study to `<hydra_run>/optuna.db`
7. reduce each seed independently and average seeds during plotting
8. generate sweep plots

Sweep outputs live under:

```text
.optuna_runs/<profile>/<timestamp>_<job-id>/
  optuna.db
  trials/
    config-0042_seed=123_sampler=cts_reward=cosine_similarity/
      attempt-01/results/
  sweep_artifacts/
```

Configure repetitions independently from the scientific search space:

```yaml
optuna:
  seeds: [123, 124, 125]
```

Each seed is independently scheduled, retried, and resumed. The plotter groups
trials by configuration, excluding seed, and warns when expected seeds are
missing.

## Search Space Format

Use categorical choices and optional `when:` guards.

```yaml
search_space:
  heterogeneity.alpha:
    name: alpha
    type: categorical
    choices: [0.1, 0.5]

  topology.sampling:
    type: categorical
    choices: [0.1, 0.2]

  sampler.name:
    name: sampler
    type: categorical
    choices: [uniform, epsilon_greedy]

  sampler.params.epsilon:
    name: epsilon
    type: categorical
    choices: [0.1]
    when:
      sampler.name: epsilon_greedy
```

Notes:

- `name` is a display label in sweep outputs
- `choices` defines a categorical axis
- `when:` prevents invalid combinations

## Parallel Workers

`optuna.workers: null` starts one process per visible GPU. Set an explicit
value to change concurrency:

```bash
uv run python -m banditdl.experiments.sweep \
  optuna=alpha_grid optuna.workers=8
```

Workers are assigned to GPUs round-robin. Eight workers with four GPUs means
two independent trials per GPU. Each trial uses one GPU; seeds remain
sequential inside that trial.

The parent process alone writes `optuna.db`. Failed configurations are retried
once. Re-running with the same `hydra.run.dir` resumes the study and skips
completed configuration IDs.

## Sweep Plotting Config

User-facing sweep plotting config lives in `conf/sweep.yaml` under `plot:`.

Current shape:

```yaml
plot:
  enabled: true
  directions: [final, avg, worse]

  heatmaps:
    - x: heterogeneity.alpha
      y: topology.sampling
      split_by:
        - sampler.name
        - [sampler.name, sampler.params.epsilon]
      aggregate_by: avg
      render: [heatmap]
      exclude_metrics: []

  single_runs:
    enabled: false
```

## Plotting Fields

### `plot.enabled`

Enable or disable sweep plotting after the sweep completes.

### `plot.directions`

How a metric is reduced over saved timesteps and nodes.

- `avg`: arithmetic mean
- `worse`: worst value
- `best`: best value
- `final`: mean over the last 2% of saved timesteps, with a minimum one timestep

`worse` uses max for losses/regret/drift and min for accuracies/rewards.

## Heatmaps

`plot.heatmaps` is a list of explicit heatmap specs. This is the important part of the refactor: heatmaps are no longer generated from every possible parameter pair.

Each heatmap spec defines:

- `x`: x-axis parameter
- `y`: y-axis parameter
- `metrics`: metrics to include; defaults to all known metrics
- `split_by`: how to split into multiple figures
- `aggregate_by`: how to collapse unused sweep dimensions
- `render`: output kinds
- `exclude_metrics`: metrics to skip for that spec

### `split_by`

Examples:

```yaml
split_by:
  - sampler.name
  - [sampler.name, sampler.params.epsilon]
```

Meaning:

- a string creates one slice per value
- a list creates one slice per value combination
- omitting `split_by` splits by the Cartesian combinations of every unused
  swept parameter
- `split_by: []` disables splitting and aggregates unused dimensions

### `aggregate_by`

When the sweep has more parameters than the axes, line groups, and active
figure split, the remaining dimensions are collapsed using:

- `avg`
- `min`
- `max`

### `render`

Supported values:

- `heatmap`
- `heatmap3d`

`heatmap3d` is experimental.

Every 2D heatmap also writes a `.tex` table beside its `.png` file.

### `metrics` and `exclude_metrics`

Each specification uses:

```text
(metrics or all known metrics) - exclude_metrics
```

Example:

```yaml
metrics: [validation_accuracy, global_accuracy, train_loss, regret]
exclude_metrics: [train_loss]
```

This plots `validation_accuracy`, `global_accuracy`, and `regret`.

## Line Plots

`plot.lines` uses the same metric selection and aggregation fields as heatmaps:

```yaml
lines:
  - x: topology.sampling
    metrics: [global_accuracy]
    group_by:
      - [sampler.name, sampler.params.discount]
    aggregate_by: avg
    exclude_metrics: []
```

- `x` is the horizontal parameter axis.
- The selected metric is the vertical axis.
- Every observed `group_by` combination becomes a line.
- Parameters outside `x` and `group_by` are reduced with `aggregate_by`.
- Missing conditional values are retained. For example, CUCB is labelled
  `sampler=cucb`, while discounted CUCB includes its discount.

Use `split_by` to generate one independent plot per observed parameter value:

```yaml
split_by:
  - sampler.reward
```

Rows where a conditional split parameter does not apply are included in every
figure. This keeps baselines such as `uniform` in reward-specific comparisons.

Heatmap axes can combine conditional parameters:

```yaml
x: [sampler.name, sampler.params.discount]
```

Missing values are omitted from labels, producing categories such as `cucb`,
`discounted_cucb-0.9`, and `discounted_cucb-0.95`.

## Single-Run Plots

Enable standard runtime plots for every completed trial:

```yaml
single_runs:
  enabled: true
```

They are written to each trial's `attempt-*/plots/` directory. For an existing
sweep, enable them without editing its stored config:

```bash
uv run python scripts/plot_sweep.py <sweep-dir> --single-runs
```

## Offline Plot Configuration

By default, the offline plotter uses `<sweep-dir>/.hydra/config.yaml` and writes
to `<sweep-dir>/sweep_artifacts/`.

To change plotting without modifying the completed sweep, create a YAML file:

```yaml
plot:
  directions: [final]
  heatmaps:
    - x: sweep.partition_profile
      y: sampler.name
      aggregate_by: avg
      render: [heatmap]
      exclude_metrics:
        - train_loss
        - gradient_norms

```

Then run:

```bash
uv run python scripts/plot_sweep.py <sweep-dir> \
  --config plot.yaml \
  --output-dir <sweep-dir>/sweep_artifacts
```

Both options are optional. The external `plot:` section is merged over the
effective plotting configuration stored by Hydra.

## Practical Recommendations

If you want a small sweep artifact set:

- define only the heatmaps you actually care about
- use `exclude_metrics` aggressively

If you want to tune parameters without plotting every dimension:

- keep those parameters in `optuna.search_space`
- only expose the scientifically interesting axes in `plot.heatmaps`

That is the main reason for the explicit `plot:` structure.
