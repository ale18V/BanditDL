# Configuration Guide

Main entrypoint:

```bash
uv run -m banditdl
```

Inspect the resolved config with:

```bash
uv run -m banditdl --cfg job
```

## Composition

`conf/config.yaml` composes the experiment from:

- `dataset`
- `topology`
- `sampler`
- `adversary`
- `aggregator`
- `heterogeneity`
- `optimization`
- `evaluation`
- `override`

The default workflow is: pick config groups, then override a few fields from the CLI.

## Local Override

`conf/override.yaml` is loaded last and is Git-ignored. Use it for per-machine defaults such as device and Hydra output directories.

Example:

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

## Main Experiment Knobs

### Dataset

Group: `conf/dataset/`

- `dataset`: `mnist`, `cifar10`, `femnist`, or `femnist_pool`
- `model`: model constructor name
- `provider`: Hydra target responsible for loading samples and metadata

Use `dataset=femnist` for natural writer-per-node partitions. Use
`dataset=femnist_pool` to apply the selected synthetic `heterogeneity` config
to pooled FEMNIST samples.

### Topology

Group: `conf/topology/`

- `topology=dynamic`: dynamic neighbor sampling

Important fields:

- `topology.nodes`: total workers
- `topology.sampling`: fraction of other participants sampled each round

### Sampler

Group: `conf/sampler/`

Used only by dynamic topology.

- `sampler=uniform`
- `sampler=epsilon_greedy`
- `sampler=exp3`
- `sampler=cucb`
- `sampler=cts`
- `sampler=discounted_cucb`
- `sampler=discounted_cts`
- `sampler=bandit`: epsilon-greedy profile

Important fields:

- `sampler.name`
- `sampler.reward`
- `sampler.params.*`

The sampler horizon is `optimization.rounds`; do not duplicate it in sampler config.
`discounted_cucb` and `discounted_cts` use `sampler.params.gamma` as their
evidence discount. CUCB also exposes `sampler.params.exploration`.

### Optimization

Group: `conf/optimization/`

Important fields:

- `optimization.rounds`
- `optimization.nb_local_steps`
- `optimization.batch_size`
- `optimization.learning_rate`

### Heterogeneity

Group: `conf/heterogeneity/`

Each profile is the Hydra-instantiated synthetic partition strategy.

- `heterogeneity.alpha`: Dirichlet heterogeneity parameter
- `heterogeneity.clusters`: number of clusters; `null` means one cluster per honest node
- `heterogeneity.classes_per_group`: labels per pathological cluster
- `heterogeneity.group_overlap`: overlap between pathological clusters
- `heterogeneity.gamma_similarity`: interpolation toward IID data in `[0, 1]`

The honest-node count must be divisible by an explicit cluster count. If
`clusters` is `null`, every honest node is treated as its own cluster.

### Adversary

Group: `conf/adversary/`

- `adversary.byzcount`
- `adversary.byzantine_budget`
- `adversary.attack`

For honest-only runs, keep `adversary=none`.

### Aggregation

Group: `conf/aggregator/`

- `aggregator.pre_aggregator`
- `aggregator.aggregator`
- `aggregator.rag`

## Common CLI Patterns

Single run:

```bash
uv run -m banditdl \
  dataset=mnist \
  topology=dynamic \
  topology.nodes=100 \
  topology.sampling=0.05 \
  sampler=uniform \
  optimization.rounds=500
```

Hydra multirun:

```bash
uv run -m banditdl -m \
  topology=dynamic \
  sampler=uniform,exp3 \
  topology.sampling=0.03,0.05 \
  seed=0,1
```

Hydra takes the Cartesian product of comma-separated values.

## Sweeps

For actual sweep workflows and plotting controls, see [docs/sweeps.md](/home/ale/Projects/BanditDL/docs/sweeps.md).
