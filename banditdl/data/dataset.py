###
 # @file   dataset.py
 # @author John stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Dataset wrappers/helpers.
###


import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from .dataset_utils import (
  get_default_root,
  partition_hierarchical,
)


def _is_femnist(dataset_name):
  return str(dataset_name).lower() == "femnist"


def _build_underlying_train(dataset_name):
  """Return (torch Dataset, targets tensor) for the training pool.

  Branches on dataset_name so FEMNIST can use its custom loader instead of
  torchvision.datasets. For FEMNIST in pool mode we use the full pooled
  training set (no writer-level holdout): the global-test holdout is now
  carved out uniformly inside `make_train_validation_test_datasets`.
  """
  if _is_femnist(dataset_name):
    from .femnist import build_femnist_pool_dataset
    train_dataset = build_femnist_pool_dataset()
    return train_dataset, train_dataset.targets
  dataset = getattr(torchvision.datasets, dict_names[dataset_name])(
    root=get_default_root(), train=True, download=True, transform=transforms[dataset_name][0]
  )
  targets = dataset.targets
  if isinstance(targets, list):
    targets = torch.FloatTensor(targets)
  return dataset, targets

# ---------------------------------------------------------------------------- #
# Collection of default transforms
transforms_horizontalflip = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
# Transforms from "A Little is Enough" (https://github.com/moranant/attacking_distributed_learning)
transforms_mnist = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
# Transforms from https://github.com/kuangliu/pytorch-cifar
transforms_cifar = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Per-dataset image transformations (automatically completed, see 'Dataset._get_datasets')
transforms = {
  "mnist":        (transforms_mnist, transforms_mnist),
  "fashionmnist": (transforms_horizontalflip, transforms_horizontalflip),
  "cifar10":      (transform_train, transform_test),
  "cifar100":     (transforms_cifar, transforms_cifar),
  "imagenet":     (transforms_horizontalflip, transforms_horizontalflip) }

#JS: Dataset names in pytorch
dict_names = {
  "mnist":        "MNIST",
  "fashionmnist": "FashionMNIST",
  "emnist":       "EMNIST",
  "cifar10":      "CIFAR10",
  "cifar100":     "CIFAR100",
  "imagenet":     "ImageNet"}


def _partition_worker_indices(
    targets,
    *,
    honest_workers,
    numb_labels,
    gamma_similarity=None,
    alpha_dirichlet=None,
    partition_method=None,
    clusters=None,
    classes_per_group=None,
    group_overlap=0,
    rng,
):
  """Return ({worker_id: [indices]}, stats) using matrix-based partitioning."""
  def _compute_stats(partition_map):
    stats = {}
    targets_np = np.asarray(targets)
    for worker_id, idx_list in partition_map.items():
      labels, counts = np.unique(targets_np[idx_list], return_counts=True)
      stats[int(worker_id)] = {
        "total": len(idx_list),
        "labels": {int(label): int(count) for label, count in zip(labels, counts, strict=False)},
      }
    return stats

  config = {
    "method": partition_method,
    "clusters": clusters,
    "alpha": alpha_dirichlet,
    "classes_per_group": classes_per_group,
    "group_overlap": group_overlap,
    "gamma_similarity": gamma_similarity,
  }

  indices = partition_hierarchical(targets, honest_workers, numb_labels, config, rng)
  return indices, _compute_stats(indices)

def _uniform_train_test_split(indices, test_ratio, rng):
  """Uniform shuffle + split `indices` into (train_indices, test_indices).

  Guarantees at least 1 train sample if any indices were provided; if the
  partition is empty, returns two empty lists.
  """
  indices = list(indices)
  if not indices:
    return [], []
  rng.shuffle(indices)
  test_size = round(len(indices) * test_ratio)
  test_size = max(0, min(test_size, len(indices) - 1)) if len(indices) > 1 else 0
  if test_size == 0:
    return indices, []
  return indices[test_size:], indices[:test_size]


# ---------------------------------------------------------------------------- #
def make_train_validation_test_datasets(
    dataset, *, numb_labels=None, gamma_similarity=None, alpha_dirichlet=None,
    honest_workers=None, train_batch=None, test_batch=None,
    global_test_ratio=0.1, local_test_ratio=0.2, split_seed=0,
    partition_method=None,
    clusters=None, classes_per_group=None, group_overlap=0,
    dataset_mode=None, nb_writers_limit=None):
  """Build per-worker training + per-worker local test DataLoaders, plus a shared
  global test DataLoader.

  The official train pool is split in two stages:
    1. Uniformly hold out `global_test_ratio` of it as the global test set.
    2. Partition the remainder among honest workers using the selected
       heterogeneity scheme.
  Each worker's resulting partition is then split into local train / local test
  uniformly using `local_test_ratio`, so the local test mirrors the client's
  (skewed) distribution.

  The official test split is intentionally discarded for non-FEMNIST datasets;
  the global test set comes entirely from the train pool.

  Returns:
    (train_loaders: dict[int, DataLoader],
     local_test_loaders: dict[int, DataLoader],
     global_test_loader: DataLoader,
     distribution_stats: dict)
  """
  if dataset_mode == "writer_per_node":
    if not _is_femnist(dataset):
      raise ValueError(
        f"dataset_mode='writer_per_node' is only supported for FEMNIST (got dataset={dataset!r})"
      )
    from .femnist import load_femnist_writer_loaders
    return load_femnist_writer_loaders(
      nb_honest=honest_workers,
      train_batch=train_batch,
      test_batch=test_batch,
      global_test_ratio=global_test_ratio,
      local_test_ratio=local_test_ratio,
      split_seed=split_seed,
      writers_cap=nb_writers_limit,
      return_stats=True,
    )

  full_train_dataset, full_targets = _build_underlying_train(dataset)

  # Stage 1: carve a uniform global test set out of the train pool.
  total = len(full_train_dataset)
  if total < 2:
    raise ValueError("Need at least 2 training samples to carve a global test split")
  if honest_workers >= total:
    raise ValueError(
      f"honest_workers ({honest_workers}) must be < number of training samples ({total}); "
      "otherwise no samples remain for the client pool after the global-test holdout"
    )
  rng = np.random.default_rng(split_seed)
  permuted = rng.permutation(total)
  global_test_size = round(total * global_test_ratio)
  # total - honest_workers >= 1 here, so the upper clamp stays positive.
  global_test_size = min(max(global_test_size, 1), total - honest_workers)
  global_test_indices = permuted[:global_test_size].tolist()
  client_pool_indices = permuted[global_test_size:].tolist()

  # Stage 2: partition the remaining client pool across honest workers.
  client_pool_indices_arr = np.asarray(client_pool_indices)
  client_pool_targets = full_targets[client_pool_indices_arr] \
    if isinstance(full_targets, torch.Tensor) \
    else torch.as_tensor([full_targets[i] for i in client_pool_indices_arr])

  local_indices_per_worker, distribution_stats = _partition_worker_indices(
    client_pool_targets,
    honest_workers=honest_workers,
    numb_labels=numb_labels,
    gamma_similarity=gamma_similarity,
    alpha_dirichlet=alpha_dirichlet,
    partition_method=partition_method,
    clusters=clusters,
    classes_per_group=classes_per_group,
    group_overlap=group_overlap,
    rng=rng,
  )

  # Stage 3: per-worker uniform local train/test split, then build loaders.
  train_loaders: dict[int, torch.utils.data.DataLoader] = {}
  local_test_loaders: dict[int, torch.utils.data.DataLoader] = {}
  for worker_id in range(honest_workers):
    local_pool = local_indices_per_worker.get(worker_id, [])
    # Map partition-local indices back to absolute indices in the full dataset.
    absolute_indices = [int(client_pool_indices_arr[i]) for i in local_pool]
    worker_rng = np.random.default_rng(split_seed + 1 + worker_id)
    train_idx, local_test_idx = _uniform_train_test_split(
      absolute_indices, local_test_ratio, worker_rng
    )
    train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
    train_loaders[worker_id] = torch.utils.data.DataLoader(
      train_subset, batch_size=train_batch, shuffle=True
    )
    local_test_subset = torch.utils.data.Subset(full_train_dataset, local_test_idx)
    local_test_loaders[worker_id] = torch.utils.data.DataLoader(
      local_test_subset, batch_size=test_batch, shuffle=False
    )

  global_test_subset = torch.utils.data.Subset(full_train_dataset, global_test_indices)
  global_test_loader = torch.utils.data.DataLoader(
    global_test_subset, batch_size=test_batch, shuffle=False
  )

  return train_loaders, local_test_loaders, global_test_loader, distribution_stats
