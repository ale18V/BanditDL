"""Render graph and collaboration views of saved sampler behavior.

Three edge-weight modes:

- `sampler_weight`: uses the mean sampler weights over the final rounds.
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

`relative_threshold=z` keeps outgoing edges above the node-specific
`mean + z * standard deviation`, removing low-signal edges without requiring
an absolute threshold.

For clustered partitions, nodes are colored by cluster and laid out on
concentric clusters; otherwise a spring layout is used.
"""

from __future__ import annotations

import json
import pathlib
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from matplotlib import cm
from omegaconf import OmegaConf

from banditdl.utils.metrics import trim_unwritten_rounds

WeightSource = Literal[
    "sampler_weight",
    "sampler_probability",
    "neighbor_disagreement",
]


def _hydra_cfg(run_dir: pathlib.Path):
    for candidate in (
        run_dir / ".hydra" / "config.yaml",
        run_dir.parent / ".hydra" / "config.yaml",
    ):
        if candidate.is_file():
            return OmegaConf.load(candidate)
    return None


def _participant_counts(run_dir: pathlib.Path) -> tuple[int, int] | None:
    for path in (run_dir / "audit.json", run_dir.parent / "audit.json"):
        if not path.is_file():
            continue
        participants = json.loads(path.read_text()).get("participants", {})
        total = participants.get("total")
        honest = participants.get("honest")
        if total is not None and honest is not None:
            return int(total), int(honest)
    return None


def _worker_groups(run_dir: pathlib.Path, cfg, n_honest: int) -> np.ndarray | None:
    """Return cluster IDs from the partition audit, with config as fallback."""
    audits = [run_dir / "audit.json", *sorted(run_dir.glob("seeds/*/results/audit.json"))]
    nb_groups = None
    for path in audits:
        if path.is_file():
            partition = json.loads(path.read_text()).get("partition", {})
            nb_groups = partition.get("resolved_clusters") or partition.get("clusters")
            break
    if nb_groups is None and cfg is not None:
        nb_groups = cfg.get("heterogeneity", {}).get("clusters")
    if nb_groups is None:
        return None
    nb_groups = int(nb_groups)
    if nb_groups == n_honest:
        return None
    if n_honest % nb_groups:
        return None
    return np.repeat(np.arange(nb_groups), n_honest // nb_groups)


def _load_weights(
    run_dir: pathlib.Path,
    weight_source: WeightSource,
    n_honest: int | None,
    *,
    include_byzantine: bool = True,
) -> tuple[np.ndarray, bool]:
    """Return ``((N, N) honest-worker weight matrix, is_directed)``.

    For ``sampler_probability`` the matrix is kept asymmetric — entry ``[i, j]``
    is worker ``i``'s converged probability of sampling worker ``j`` (its own
    bandit) — and is rendered as a directed graph. For ``neighbor_disagreement``
    the matrix is a symmetric model-distance similarity (undirected).
    """
    if weight_source == "sampler_weight":
        weights = _load_sampler_history(run_dir, "sampler_weights.npy", tail_fraction=0.1)
        if include_byzantine and weights.shape[1] > weights.shape[0]:
            full = np.zeros((weights.shape[1], weights.shape[1]), dtype=float)
            full[: weights.shape[0], :] = weights
            return full, True
        return np.asarray(weights[:, : weights.shape[0]], dtype=float), True
    if weight_source == "sampler_probability":
        full_by_seed_path = run_dir / "sampler_probabilities_by_seed.npy"
        full_path = run_dir / "sampler_probabilities.npy"
        by_seed_path = run_dir / "sampler_probabilities_final_by_seed.npy"
        path = run_dir / "sampler_probabilities_final.npy"
        if full_by_seed_path.is_file():
            history = trim_unwritten_rounds(np.load(full_by_seed_path))
            if history.shape[1] == 0:
                raise ValueError(f"{full_by_seed_path} has no completed rounds")
            prob = np.nanmean(history[:, -1], axis=0)
        elif full_path.is_file():
            history = trim_unwritten_rounds(np.load(full_path))
            if history.shape[0] == 0:
                raise ValueError(f"{full_path} has no completed rounds")
            prob = history[-1]
        elif by_seed_path.is_file():
            prob = np.nanmean(np.load(by_seed_path), axis=0)
        elif path.is_file():
            prob = np.load(path)
        else:
            raise FileNotFoundError(f"Missing {full_path}")
        if include_byzantine and prob.shape[1] > prob.shape[0]:
            full = np.zeros((prob.shape[1], prob.shape[1]), dtype=float)
            full[: prob.shape[0], :] = prob
            return full, True
        return np.asarray(prob[:, : prob.shape[0]], dtype=float), True
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


def _load_sampler_history(
    run_dir: pathlib.Path,
    filename: str,
    *,
    tail_fraction: float,
) -> np.ndarray:
    stem = pathlib.Path(filename).stem
    by_seed_path = run_dir / f"{stem}_by_seed.npy"
    path = run_dir / filename
    source = by_seed_path if by_seed_path.is_file() else path
    if not source.is_file():
        raise FileNotFoundError(path)

    history = trim_unwritten_rounds(np.load(source))
    time_axis = 1 if history.ndim == 4 else 0
    rounds = history.shape[time_axis]
    if rounds == 0:
        raise ValueError(f"{source} has no completed rounds")
    tail = max(1, int(np.ceil(rounds * tail_fraction)))
    history = np.take(history, np.arange(rounds - tail, rounds), axis=time_axis)
    return np.nanmean(history, axis=tuple(range(time_axis + 1)))


def _filter_edges(
    weights: np.ndarray,
    *,
    directed: bool,
    threshold: float | None = None,
    relative_threshold: float | None = None,
    top_edges_per_node: int | None = None,
) -> np.ndarray:
    """Zero out edges that fail the threshold / top-k filters.

    - ``threshold``: keep only edges with weight strictly greater than it.
    - ``relative_threshold``: keep row values above ``mean + z * std``.
    - ``top_edges_per_node``: keep only each node's ``k`` strongest *outgoing*
      edges. For undirected graphs the kept mask is symmetrized so both
      endpoints agree on the edge; for directed graphs each node keeps its own
      outgoing edges independently.

    Self-loops (the diagonal) are always removed. Returns a new matrix.
    """
    weights = np.array(weights, dtype=float)
    np.fill_diagonal(weights, 0.0)
    n = weights.shape[0]

    if relative_threshold is not None:
        if relative_threshold < 0:
            raise ValueError("relative_threshold must be non-negative")
        mask = ~np.eye(n, dtype=bool)
        rows = weights[mask].reshape(n, n - 1)
        cutoffs = rows.mean(axis=1) + relative_threshold * rows.std(axis=1)
        keep = weights > cutoffs[:, None]
        if not directed:
            keep = keep & keep.T
        weights = weights * keep

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


def _node_colors(groups: np.ndarray | None, n: int, n_honest: int | None = None):
    if n_honest is not None and n > n_honest:
        honest_colors = _node_colors(groups, n_honest)
        return [*honest_colors, *(["tab:red"] * (n - n_honest))]
    if groups is None:
        return ["tab:blue"] * n
    cmap = cm.get_cmap("tab10")
    return [cmap(int(g) % 10) for g in groups]


def _append_byzantine_positions(
    honest_pos: dict[int, tuple[float, float]],
    n_honest: int,
    n: int,
) -> dict[int, tuple[float, float]]:
    pos = dict(honest_pos)
    if n <= n_honest:
        return pos

    xs = [xy[0] for xy in honest_pos.values()] or [0.0]
    ys = [xy[1] for xy in honest_pos.values()] or [0.0]
    center = np.array([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2], dtype=float)
    radius = max(
        float(np.linalg.norm(np.array(xy, dtype=float) - center))
        for xy in honest_pos.values()
    )
    radius = max(radius, 1.0)
    byz_count = n - n_honest
    for offset in range(byz_count):
        theta = 2 * np.pi * offset / byz_count + np.pi / byz_count
        xy = center + radius * np.array([np.cos(theta), np.sin(theta)])
        pos[n_honest + offset] = (float(xy[0]), float(xy[1]))
    return pos


def plot_clustering_graph(
    run_dir: pathlib.Path,
    output_path: pathlib.Path,
    *,
    weight_source: WeightSource = "sampler_probability",
    threshold: float | None = None,
    relative_threshold: float | None = None,
    top_edges_per_node: int | None = None,
    layout: Literal["auto", "spring", "group"] = "auto",
    title: str | None = None,
    include_byzantine: bool = True,
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
    participants = _participant_counts(run_dir)
    if participants is not None:
        _, n_honest = participants
    elif cfg is not None:
        nb_workers = int(OmegaConf.select(cfg, "topology.nodes"))
        nb_byz = int(OmegaConf.select(cfg, "adversary.byzcount") or 0)
        n_honest = nb_workers - nb_byz

    include_byzantine = include_byzantine and weight_source in {
        "sampler_weight",
        "sampler_probability",
    }
    weights, directed = _load_weights(
        run_dir,
        weight_source,
        n_honest,
        include_byzantine=include_byzantine,
    )
    n = weights.shape[0]
    weights = _filter_edges(
        weights,
        directed=directed,
        threshold=threshold,
        relative_threshold=relative_threshold,
        top_edges_per_node=top_edges_per_node,
    )

    create_using = nx.DiGraph if directed else nx.Graph
    graph = nx.from_numpy_array(weights, create_using=create_using)
    groups = _worker_groups(run_dir, cfg, n_honest or n)
    if include_byzantine and n_honest is not None and n > n_honest:
        if layout == "group" or (layout == "auto" and groups is not None):
            honest_pos = (
                _grouped_layout(groups)
                if groups is not None
                else nx.spring_layout(graph.subgraph(range(n_honest)), seed=0, weight="weight")
            )
        else:
            honest_pos = nx.spring_layout(graph.subgraph(range(n_honest)), seed=0, weight="weight")
        pos = _append_byzantine_positions(honest_pos, n_honest, n)
    elif layout == "group" or (layout == "auto" and groups is not None):
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
        node_color=_node_colors(groups, n, n_honest if include_byzantine else None),
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
        title = f"{run_dir.name} - {weight_source}"
    ax.set_title(title, fontsize=11)
    if relative_threshold is not None:
        ax.text(
            0.5,
            -0.02,
            f"Edges exceed each source node's mean + {relative_threshold:g} std.",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def mutual_collaboration(weights: np.ndarray) -> np.ndarray:
    """Return strong only when both nodes assign strong weight to each other."""
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    collaboration = np.sqrt(weights * weights.T)
    np.fill_diagonal(collaboration, 0.0)
    return collaboration


def spectral_embedding(collaboration: np.ndarray) -> np.ndarray:
    """Project a symmetric collaboration matrix to two spectral coordinates."""
    collaboration = np.asarray(collaboration, dtype=float)
    degree = collaboration.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = inv_sqrt[:, None] * collaboration * inv_sqrt[None, :]
    laplacian = np.eye(len(collaboration)) - normalized
    _, vectors = np.linalg.eigh(laplacian)
    coordinates = vectors[:, 1:3]
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    return coordinates


def _embedding_fidelity(collaboration: np.ndarray, coordinates: np.ndarray) -> float:
    upper = np.triu_indices_from(collaboration, k=1)
    strengths = collaboration[upper]
    distances = np.linalg.norm(
        coordinates[upper[0]] - coordinates[upper[1]],
        axis=1,
    )
    if strengths.std() == 0 or distances.std() == 0:
        return float("nan")
    return float(np.corrcoef(strengths, -distances)[0, 1])


def plot_collaboration_embedding(
    run_dir: pathlib.Path,
    output_path: pathlib.Path,
    *,
    relative_threshold: float | None = None,
    tail_fraction: float = 0.1,
) -> pathlib.Path:
    """Plot nodes close together when their sampler weights are mutually strong."""
    run_dir, output_path = pathlib.Path(run_dir), pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _hydra_cfg(run_dir)
    n_honest = None
    if cfg is not None:
        n_honest = int(OmegaConf.select(cfg, "topology.nodes")) - int(
            OmegaConf.select(cfg, "adversary.byzcount") or 0
        )

    weights = _load_sampler_history(
        run_dir,
        "sampler_weights.npy",
        tail_fraction=tail_fraction,
    )
    n = n_honest or weights.shape[0]
    weights = weights[:n, :n]
    collaboration = mutual_collaboration(weights)
    coordinates = spectral_embedding(collaboration)

    if relative_threshold is None:
        visible = collaboration
    else:
        strong = _filter_edges(
            weights,
            directed=True,
            relative_threshold=relative_threshold,
        )
        visible = collaboration * ((strong > 0) & (strong.T > 0))
    graph = nx.from_numpy_array(visible)
    pos = {node: coordinates[node] for node in range(n)}
    groups = _worker_groups(run_dir, cfg, n)

    edges = list(graph.edges(data="weight"))
    strengths = np.array([weight for _, _, weight in edges])
    widths = 0.5 + 4 * strengths / strengths.max() if len(strengths) else 0.5

    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        edge_color="tab:blue",
        alpha=0.35,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=_node_colors(groups, n),
        node_size=260,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)

    fidelity = _embedding_fidelity(collaboration, coordinates)
    fidelity_text = "undefined" if np.isnan(fidelity) else f"{fidelity:.2f}"
    ax.set_title("Mutual Sampler Collaboration")
    caption = f"Positions and edges use all mutual weights. Fidelity r={fidelity_text}."
    if relative_threshold is not None:
        caption = (
            f"Positions use all mutual weights; visible edges exceed both nodes' "
            f"mean + {relative_threshold:g} std. Fidelity r={fidelity_text}."
        )
    ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _edge_label(weight_source: WeightSource) -> str:
    if weight_source == "sampler_weight":
        return "Sampler preference weight, final 10% of rounds"
    if weight_source == "sampler_probability":
        return "Sampler probability P(i → j), final round"
    return "1 / (1 + final model L2 distance)"
