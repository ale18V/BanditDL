"""Render a 2D weighted graph of the worker network from a saved run.

Two edge-weight modes:

- `sampler_probability`: uses the final round of `sampler_probabilities.npy`
  or legacy `sampler_probabilities_final.npy`, the per-worker sampler
  distribution. This is **directional**: entry `P[i, j]` is worker `i`'s
  converged bandit probability of sampling worker `j`. The graph is therefore
  drawn as a directed graph with two opposite edges between each pair — edge
  `i -> j` carries `P[i, j]` and edge `j -> i` carries `P[j, i]`, one per node's
  own bandit.
- `neighbor_disagreement`: uses `pairwise_model_distance_final.npy` or
  `pairwise_model_distance_final_by_seed.npy`. Edge weight is `1 / (1 + dist)`
  so closer (more agreeing) workers get heavier edges — matching the bandit
  reward semantics. Model distance is symmetric, so this mode stays an
  undirected graph.

Pass `threshold` to keep only edges whose weight exceeds it (e.g. drop the
near-uniform exploration edges of an epsilon-greedy sampler), and/or
`top_edges_per_node` to keep only each node's strongest outgoing edges.

For clustered pathological partitions, nodes are colored by cluster and laid
out on concentric clusters; otherwise a spring layout is used.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from matplotlib import cm
from omegaconf import OmegaConf

from banditdl.utils.metrics import trim_unwritten_probability_rounds

WeightSource = Literal["sampler_probability", "neighbor_disagreement"]


def _hydra_cfg(run_dir: pathlib.Path):
    for candidate in (
        run_dir / ".hydra" / "config.yaml",
        run_dir.parent / ".hydra" / "config.yaml",
    ):
        if candidate.is_file():
            return OmegaConf.load(candidate)
    return None


def _worker_groups(cfg, n_honest: int) -> np.ndarray | None:
    """Return per-worker cluster IDs for clustered pathological partitions."""
    if cfg is None:
        return None
    het = cfg.get("heterogeneity", {})
    if str(het.get("method", "")) != "pathological":
        return None
    nb_groups = int(het.get("clusters") or n_honest)
    if nb_groups == n_honest:
        return None
    return np.repeat(np.arange(nb_groups), n_honest // nb_groups)


def _load_weights(
    run_dir: pathlib.Path, weight_source: WeightSource, n_honest: int | None
) -> tuple[np.ndarray, bool]:
    """Return ``((N, N) honest-worker weight matrix, is_directed)``.

    For ``sampler_probability`` the matrix is kept asymmetric — entry ``[i, j]``
    is worker ``i``'s converged probability of sampling worker ``j`` (its own
    bandit) — and is rendered as a directed graph. For ``neighbor_disagreement``
    the matrix is a symmetric model-distance similarity (undirected).
    """
    if weight_source == "sampler_probability":
        full_by_seed_path = run_dir / "sampler_probabilities_by_seed.npy"
        full_path = run_dir / "sampler_probabilities.npy"
        by_seed_path = run_dir / "sampler_probabilities_final_by_seed.npy"
        path = run_dir / "sampler_probabilities_final.npy"
        if full_by_seed_path.is_file():
            history = trim_unwritten_probability_rounds(np.load(full_by_seed_path))
            if history.shape[1] == 0:
                raise ValueError(f"{full_by_seed_path} has no completed rounds")
            prob = np.nanmean(history[:, -1], axis=0)
        elif full_path.is_file():
            history = trim_unwritten_probability_rounds(np.load(full_path))
            if history.shape[0] == 0:
                raise ValueError(f"{full_path} has no completed rounds")
            prob = history[-1]
        elif by_seed_path.is_file():
            prob = np.nanmean(np.load(by_seed_path), axis=0)
        elif path.is_file():
            prob = np.load(path)
        else:
            raise FileNotFoundError(f"Missing {full_path}")
        n = prob.shape[0]
        honest_block = prob[:, :n]  # restrict to honest-honest edges
        return np.asarray(honest_block, dtype=float), True
    if weight_source == "neighbor_disagreement":
        by_seed_path = run_dir / "pairwise_model_distance_final_by_seed.npy"
        path = run_dir / "pairwise_model_distance_final.npy"
        if by_seed_path.is_file():
            dist_by_seed = np.load(by_seed_path)
            if n_honest is not None:
                dist_by_seed = dist_by_seed[:, :n_honest, :n_honest]
            return np.nanmean(1.0 / (1.0 + dist_by_seed), axis=0), False
        if path.is_file():
            dist = np.load(path)
            if n_honest is not None:
                dist = dist[:n_honest, :n_honest]
            return 1.0 / (1.0 + dist), False
        raise FileNotFoundError(f"Missing {path}")
    raise ValueError(f"Unknown weight_source: {weight_source!r}")


def _filter_edges(
    weights: np.ndarray,
    *,
    directed: bool,
    threshold: float | None = None,
    top_edges_per_node: int | None = None,
) -> np.ndarray:
    """Zero out edges that fail the threshold / top-k filters.

    - ``threshold``: keep only edges with weight strictly greater than it.
    - ``top_edges_per_node``: keep only each node's ``k`` strongest *outgoing*
      edges. For undirected graphs the kept mask is symmetrized so both
      endpoints agree on the edge; for directed graphs each node keeps its own
      outgoing edges independently.

    Self-loops (the diagonal) are always removed. Returns a new matrix.
    """
    weights = np.array(weights, dtype=float)
    np.fill_diagonal(weights, 0.0)
    n = weights.shape[0]

    if top_edges_per_node is not None and top_edges_per_node < n - 1:
        keep = np.zeros_like(weights, dtype=bool)
        for i in range(n):
            order = np.argsort(weights[i])[::-1]
            keep[i, order[:top_edges_per_node]] = True
        if not directed:
            keep = keep | keep.T
        weights = weights * keep

    if threshold is not None:
        weights = np.where(weights > threshold, weights, 0.0)

    return weights


def _grouped_layout(groups: np.ndarray) -> dict[int, tuple[float, float]]:
    """Place each group on its own ring; workers within a group spread evenly."""
    unique_groups = sorted(set(int(g) for g in groups))
    nb_groups = len(unique_groups)
    pos: dict[int, tuple[float, float]] = {}
    for gi, g in enumerate(unique_groups):
        members = [int(i) for i, gg in enumerate(groups) if int(gg) == g]
        cx = np.cos(2 * np.pi * gi / nb_groups)
        cy = np.sin(2 * np.pi * gi / nb_groups)
        if len(members) == 1:
            pos[members[0]] = (cx, cy)
            continue
        for mi, worker_id in enumerate(members):
            theta = 2 * np.pi * mi / len(members)
            r = 0.32
            pos[worker_id] = (cx + r * np.cos(theta), cy + r * np.sin(theta))
    return pos


def _node_colors(groups: np.ndarray | None, n: int):
    if groups is None:
        return ["tab:blue"] * n
    cmap = cm.get_cmap("tab10")
    return [cmap(int(g) % 10) for g in groups]


def plot_clustering_graph(
    run_dir: pathlib.Path,
    output_path: pathlib.Path,
    *,
    weight_source: WeightSource = "sampler_probability",
    threshold: float | None = None,
    top_edges_per_node: int | None = None,
    layout: Literal["auto", "spring", "group"] = "auto",
    title: str | None = None,
) -> pathlib.Path:
    """Render and save the weighted clustering graph for `run_dir`.

    `threshold` keeps only edges whose weight exceeds it (e.g. drop near-uniform
    exploration edges). `top_edges_per_node` keeps only the k strongest outgoing
    edges per node so the plot stays readable for dense N. Both default to None
    (keep all edges) and compose when both are set.
    """
    run_dir = pathlib.Path(run_dir)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _hydra_cfg(run_dir)
    n_honest = None
    if cfg is not None:
        nb_workers = int(OmegaConf.select(cfg, "topology.nodes"))
        nb_byz = int(OmegaConf.select(cfg, "adversary.byzcount") or 0)
        n_honest = nb_workers - nb_byz

    weights, directed = _load_weights(run_dir, weight_source, n_honest)
    n = weights.shape[0]
    weights = _filter_edges(
        weights,
        directed=directed,
        threshold=threshold,
        top_edges_per_node=top_edges_per_node,
    )

    create_using = nx.DiGraph if directed else nx.Graph
    graph = nx.from_numpy_array(weights, create_using=create_using)
    groups = _worker_groups(cfg, n) if cfg is not None else None
    if layout == "group" or (layout == "auto" and groups is not None):
        if groups is None:
            pos = nx.spring_layout(graph, seed=0, weight="weight")
        else:
            pos = _grouped_layout(groups)
    else:
        pos = nx.spring_layout(graph, seed=0, weight="weight")

    edges = list(graph.edges(data="weight"))
    edge_weights = np.array([w for _, _, w in edges]) if edges else np.array([])
    edge_cmap = plt.get_cmap("viridis")
    if edges and edge_weights.max() > 0:
        norm = mcolors.Normalize(vmin=0.0, vmax=float(edge_weights.max()))
        edge_colors = edge_cmap(norm(edge_weights))
        edge_widths = 0.4 + 3.5 * edge_weights / edge_weights.max()
    else:
        edge_colors = "lightgray"
        edge_widths = 0.5
        norm = None

    node_size = 260
    fig, ax = plt.subplots(figsize=(8, 8))
    # For a directed graph draw arrowheads and curve the edges so the two
    # opposite-direction edges between a pair of nodes don't overlap.
    directed_edge_kwargs = (
        dict(
            arrows=True,
            arrowstyle="-|>",
            arrowsize=11,
            connectionstyle="arc3,rad=0.12",
            node_size=node_size,
        )
        if directed
        else {}
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.85,
        ax=ax,
        **directed_edge_kwargs,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=_node_colors(groups, n),
        node_size=node_size,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)

    if norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=edge_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(_edge_label(weight_source))

    if title is None:
        title = f"{run_dir.name} — {weight_source}"
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _edge_label(weight_source: WeightSource) -> str:
    if weight_source == "sampler_probability":
        return "Sampler probability P(i → j), final round"
    return "1 / (1 + final model L2 distance)"
