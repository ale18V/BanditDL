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
4. run one seed-averaged training job per trial
5. run configurations concurrently and save isolated trial attempts
6. persist the Optuna study to `<hydra_run>/optuna.db`
7. pick the best trial from validation accuracy
8. rerun the best trial with test evaluation
9. generate sweep plots

Sweep outputs live under:

```text
.optuna_runs/<profile>/<timestamp>_<job-id>/
  optuna.db
  trials/
    config-0042_sampler=cts_reward=cosine_similarity/
      attempt-01/results/
  best_trial_test_eval/results/
  sweep_artifacts/
```

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
  directions: [avg, worse]

  heatmaps:
    - x: heterogeneity.alpha
      y: topology.sampling
      group_by:
        - sampler.name
        - [sampler.name, sampler.params.epsilon]
      aggregate_by: avg
      render: [heatmap]
      exclude_metrics: []

  per_parameter:
    enabled: false
    exclude_metrics: []
```

## Plotting Fields

### `plot.enabled`

Enable or disable sweep plotting after the sweep completes.

### `plot.directions`

How a metric is reduced over saved timesteps and nodes.

- `avg`: arithmetic mean
- `worse`: worst value

`worse` uses max for losses/regret/drift and min for accuracies/rewards.

## Heatmaps

`plot.heatmaps` is a list of explicit heatmap specs. This is the important part of the refactor: heatmaps are no longer generated from every possible parameter pair.

Each heatmap spec defines:

- `x`: x-axis parameter
- `y`: y-axis parameter
- `group_by`: how to split into multiple slices
- `aggregate_by`: how to collapse unused sweep dimensions
- `render`: output kinds
- `exclude_metrics`: metrics to skip for that spec

### `group_by`

Examples:

```yaml
group_by:
  - sampler.name
  - [sampler.name, sampler.params.epsilon]
```

Meaning:

- a string creates one slice per value
- a list creates one slice per value combination

### `aggregate_by`

When the sweep has more parameters than `x`, `y`, and the active `group_by`, the remaining dimensions are collapsed using:

- `avg`
- `min`
- `max`

### `render`

Supported values:

- `heatmap`
- `heatmap3d`

`heatmap3d` is experimental.

### `exclude_metrics`

By default, sweep plots try every known scalar metric. Use `exclude_metrics` to suppress metrics for a specific heatmap spec.

## Per-Parameter Plots

`plot.per_parameter.enabled` controls whether the per-parameter plotter runs.

Example:

```yaml
per_parameter:
  enabled: false
```

`exclude_metrics` works the same way there.

## Practical Recommendations

If you want a small sweep artifact set:

- keep `per_parameter.enabled: false`
- define only the heatmaps you actually care about
- use `exclude_metrics` aggressively

If you want to tune parameters without plotting every dimension:

- keep those parameters in `optuna.search_space`
- only expose the scientifically interesting axes in `plot.heatmaps`

That is the main reason for the explicit `plot:` structure.
