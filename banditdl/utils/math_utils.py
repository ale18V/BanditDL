"""Mathematical utility functions for robust aggregation and optimization."""

import torch
import math
from itertools import combinations


def clip_vector(vector, clip_threshold):
    """Clip the vector if its L2 norm is greater than clip_threshold."""
    vector_norm = vector.norm().item()
    if vector_norm > clip_threshold:
        vector.mul_(clip_threshold / vector_norm)
    return vector


def line_maximize(scape, evals=16, start=0., delta=1., ratio=0.8):
    """Best-effort arg-maximize a scape: ℝ⁺⟶ ℝ, by mere exploration.
    
    Args:
        scape: Function to best-effort arg-maximize
        evals: Maximum number of evaluations, must be a positive integer
        start: Initial x evaluated, must be a non-negative float
        delta: Initial step delta, must be a positive float
        ratio: Contraction ratio, must be between 0.5 and 1. (both excluded)
    Returns:
        Best-effort maximizer x under the evaluation budget
    """
    # Variable setup
    best_x = start
    best_y = scape(best_x)
    evals -= 1
    # Expansion phase
    while evals > 0:
        prop_x = best_x + delta
        prop_y = scape(prop_x)
        evals -= 1
        # Check if best
        if prop_y > best_y:
            best_y = prop_y
            best_x = prop_x
            delta *= 2
        else:
            delta *= ratio
            break
    # Contraction phase
    while evals > 0:
        if prop_x < best_x:
            prop_x += delta
        else:
            x = prop_x - delta
            while x < 0:
                x = (x + prop_x) / 2
            prop_x = x
        prop_y = scape(prop_x)
        evals -= 1
        # Check if best
        if prop_y > best_y:
            best_y = prop_y
            best_x = prop_x
        # Reduce delta
        delta *= ratio
    # Return found maximizer
    return best_x


def smoothed_weiszfeld(nb_vectors, vectors, nu=0.1, T=3):
    """Approximation algorithm used for geometric median."""
    z = torch.zeros_like(vectors[0])
    # Calculate mask to exclude vectors containing infinite values
    mask = ~torch.any(torch.isinf(torch.stack(vectors)), dim=-1)
    filtered_vectors = [v for v, m in zip(vectors, mask) if m]
    alphas = [1 / nb_vectors] * len(filtered_vectors)
    for _ in range(T):
        betas = list()
        for i, vector in enumerate(filtered_vectors):
            distance = z.sub(vector).norm().item()
            if math.isnan(distance):
                # Distance is infinite or NaN
                betas.append(0)
            else:
                betas.append(alphas[i] / max(distance, nu))
        z.zero_()
        for vector, beta in zip(filtered_vectors, betas):
            z.add_(vector, alpha=beta)
        z.div_(sum(betas))
    return z


def smoothed_weiszfeld2(nb_vectors, vectors, nu=0.1, T=3):
    """Smoothed Weiszfeld algorithm.
    
    Args:
        nb_vectors: Number of vectors
        vectors: Non-empty list of vectors to aggregate
        nu: RFA parameter
        T: Number of iterations to run the smoothed Weiszfeld algorithm
    Returns:
        Aggregated vector
    """
    z = torch.zeros_like(vectors[0])
    vectors = torch.stack(vectors)
    # Exclude vectors that contain any infinite values
    mask = ~torch.any(torch.isinf(vectors), dim=-1)
    filtered_vectors = vectors[mask]
    alphas = torch.tensor([1 / nb_vectors] * len(filtered_vectors)).to(vectors[0].device)
    for _ in range(T):
        distances = torch.linalg.vector_norm(z - filtered_vectors, dim=-1)
        betas = torch.div(alphas, torch.clamp(distances, min=nu))
        # Update z using the betas and filtered vectors
        z = torch.sum(betas[:, None] * filtered_vectors, dim=0).div(betas.sum())
    return z


def compute_distances(vectors):
    """Compute all pairwise distances between vectors.
    
    Args:
        vectors: List or tensor of vectors
    Returns:
        Distance matrix (n x n)
    """
    if type(vectors) != torch.Tensor:
       vectors = torch.stack(vectors)
    distances = torch.cdist(vectors, vectors)
    # set non-finite values to inf
    distances[~torch.isfinite(distances)] = float('inf')
    return distances


def get_vector_best_score(vectors, nb_byz, distances):
    """Get the vector with the smallest score for Krum aggregator."""
    vectors = torch.stack(vectors)
    n_vectors = vectors.size(0)
    min_score, min_index = torch.tensor(math.inf), 0
    for worker_id in range(n_vectors):
        # Create a mask for selecting all vectors except the current one
        mask = torch.ones(n_vectors, dtype=torch.bool)
        mask[worker_id] = 0
        # Select all distances to the current vector
        distances_to_vector = distances[worker_id, mask]
        # Square and sort the distances
        distances_squared_to_vector = distances_to_vector.pow(2).sort()[0]
        # Compute the score
        score = distances_squared_to_vector[:n_vectors - nb_byz - 1].sum()
        # Update min score and min index
        if score < min_score:
            min_score, min_index = score, worker_id

    return vectors[min_index]


def get_vector_scores(vectors, nb_byz, distances):
    """Get scores of all vectors for Multi-Krum aggregator.
    
    Returns:
        List of (score, worker_id) tuples sorted by score
    """
    vectors = torch.stack(vectors)
    n_vectors = vectors.size(0)
    scores = []
    
    for worker_id in range(n_vectors):
        # Create a mask for selecting all vectors except the current one
        mask = torch.ones(n_vectors, dtype=torch.bool)
        mask[worker_id] = 0
        # Select all distances to the current vector
        distances_to_vector = distances[worker_id, mask]
        # Square and sort the distances
        distances_squared_to_vector = distances_to_vector.pow(2).sort()[0]
        # Compute the score
        score = distances_squared_to_vector[:n_vectors - nb_byz - 1].sum()
        scores.append((score.item(), worker_id))
    
    return sorted(scores)


def average_nearest_neighbors(vectors, f, pivot=None):
    """Compute the average of the n-f closest vectors to pivot."""
    if isinstance(vectors, torch.Tensor):
        vectors = list(torch.unbind(vectors))
    if pivot is None:
        return torch.stack(
            [average_nearest_neighbors(vectors, f, vector) for vector in vectors]
        )
    vector_scores = list()
    
    for i in range(len(vectors)):
        #JS: compute distance to pivot
        distance = vectors[i].sub(pivot).norm().item()
        vector_scores.append((i, distance))
    
    #JS: sort vector_scores by increasing distance to pivot
    vector_scores.sort(key=lambda x: x[1])
    
    #JS: Return the average of the n-f closest vectors to pivot
    closest_vectors = [vectors[vector_scores[j][0]] for j in range(len(vectors) -f)]
    return torch.stack(closest_vectors).mean(dim=0)


def compute_min_diameter_subset(vectors, nb_workers, nb_byz):
    """Compute the subset with minimum diameter for MDA aggregator."""
    nb_vectors = nb_workers
    #JS: compute all pairwise distances
    distances = dict()
    all_pairs = list(combinations(range(nb_vectors), 2))
    for (x,y) in all_pairs:
        dist = vectors[x].sub(vectors[y]).norm().item()
        if math.isnan(dist):
            dist = float('inf')
        distances[(x,y)] = dist
    
    min_diameter = float('inf')
    #JS: Get all subsets of size n - f
    all_subsets = list(combinations(range(nb_vectors), nb_vectors - nb_byz))
    for subset in all_subsets:
        subset_diameter = 0
        #JS: Compute diameter of subset
        for i, vector1 in enumerate(subset):
            for vector2 in subset[i+1:]:
                distance = distances.get((vector1, vector2), 0)
                subset_diameter = distance if distance > subset_diameter else subset_diameter

        #JS: Update min diameter (if needed)
        if min_diameter > subset_diameter:
            min_diameter = subset_diameter
            min_subset = subset

    return min_subset


def compute_min_variance_subset(vectors, nb_workers, nb_byz):
    """Compute the subset with minimum variance for MVA aggregator."""
    nb_vectors = nb_workers
    #JS: compute all pairwise distances
    distances = dict()
    all_pairs = list(combinations(range(nb_vectors), 2))
    for (x,y) in all_pairs:
        dist = vectors[x].sub(vectors[y]).norm().item()
        if math.isnan(dist):
            dist = float('inf')
        distances[(x,y)] = dist

    #JS: Get all subsets of size n - f
    all_subsets = list(combinations(range(nb_vectors), nb_vectors - nb_byz))
    min_variance = float('inf')

    for subset in all_subsets:
        current_variance = 0
        #JS: Compute diameter of subset
        for i, vector1 in enumerate(subset):
            for vector2 in subset[i+1:]:
                distance = distances.get((vector1, vector2), 0)
                current_variance += distance**2
        
        if min_variance > current_variance:
            min_variance = current_variance
            min_subset = subset

    return min_subset


def compute_closest_vectors_and_mean(vectors, nb_workers, nb_byz):
    """Compute and return the mean of n-f closest vectors (MONNA aggregator)."""
    # Convert vectors from a list of 1D tensors to a 2D tensor
    vectors = torch.stack(vectors)
    pivot_vector = vectors[-1]
    # Calculate distances using vectorized operations
    distances = torch.norm(vectors - pivot_vector, dim=1)
    # Get the indices of the smallest n-f distances
    _, indices = torch.topk(distances, k=nb_workers-nb_byz, largest=False)
    # Use advanced indexing to select the closest vectors and compute their mean
    return vectors[indices].mean(dim=0)


def _stack_vectors(vectors: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(vectors, torch.Tensor):
        if vectors.dim() == 1:
            return vectors.unsqueeze(0)
        if vectors.dim() == 2:
            return vectors
        raise ValueError("vectors must be 1D or 2D tensor")
    return torch.stack(list(vectors))


def consensus_drift(vectors: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Return per-node squared distance to the global mean model."""
    stacked = _stack_vectors(vectors)
    mean_params = stacked.mean(dim=0)
    deltas = stacked - mean_params
    return (deltas * deltas).sum(dim=1)


def neighbor_disagreement(
    vectors: torch.Tensor | list[torch.Tensor],
    *,
    neighbor_indices: torch.Tensor | list[list[int]] | None = None,
    adjacency: torch.Tensor | list[list[float]] | None = None,
) -> torch.Tensor:
    """Return per-node mean squared distance to neighbors.

    Pass either neighbor_indices (shape: N x K with -1 padding) or adjacency
    (shape: N x N, non-zero indicates neighbor). Self edges are ignored.
    """
    if (neighbor_indices is None) == (adjacency is None):
        raise ValueError("pass exactly one of neighbor_indices or adjacency")

    stacked = _stack_vectors(vectors)
    device = stacked.device

    if adjacency is not None:
        adj = adjacency
        if not isinstance(adj, torch.Tensor):
            adj = torch.as_tensor(adj, device=device)
        else:
            adj = adj.to(device)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if adj.shape[0] != stacked.shape[0]:
            raise ValueError("adjacency size must match number of vectors")
        if torch.any(torch.diagonal(adj) != 0):
            adj = adj.clone()
            adj.fill_diagonal_(0)
        mask = adj != 0
        if mask.sum() == 0:
            return torch.zeros(stacked.shape[0], device=device)
        dist_sq = torch.cdist(stacked, stacked).pow(2)
        masked = dist_sq * mask
        counts = mask.sum(dim=1).clamp(min=1).to(dist_sq.dtype)
        return masked.sum(dim=1) / counts

    neighbors = neighbor_indices
    if not isinstance(neighbors, torch.Tensor):
        neighbors = torch.as_tensor(neighbors, device=device)
    else:
        neighbors = neighbors.to(device)
    if neighbors.ndim == 1:
        neighbors = neighbors.unsqueeze(0)
    if neighbors.numel() == 0:
        return torch.zeros(stacked.shape[0], device=device)

    valid_mask = neighbors >= 0
    if valid_mask.sum() == 0:
        return torch.zeros(stacked.shape[0], device=device)

    safe_neighbors = neighbors.clamp(min=0).long()
    neighbor_weights = stacked[safe_neighbors]
    diffs = neighbor_weights - stacked.unsqueeze(1)
    dist_sq = (diffs * diffs).sum(dim=2)
    dist_sq = dist_sq * valid_mask
    counts = valid_mask.sum(dim=1).clamp(min=1).to(dist_sq.dtype)
    return dist_sq.sum(dim=1) / counts
