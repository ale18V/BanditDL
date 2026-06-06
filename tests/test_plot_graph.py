"""Tests for the clustering-graph plotting helpers.

These exercise the pure edge-matrix logic (directed vs. undirected handling,
threshold and top-k filtering) plus one end-to-end render, without asserting on
pixels.
"""

from __future__ import annotations

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")  # headless backend for CI

from banditdl.utils.plot_graph import (  # noqa: E402
    _filter_edges,
    _load_weights,
    plot_clustering_graph,
)


def test_filter_edges_removes_diagonal_and_below_threshold():
    weights = np.array(
        [
            [0.9, 0.6, 0.1],
            [0.2, 0.9, 0.7],
            [0.05, 0.8, 0.9],
        ]
    )
    filtered = _filter_edges(weights, directed=True, threshold=0.5)

    # Self-loops gone regardless of value.
    assert np.allclose(np.diag(filtered), 0.0)
    # Strictly-greater-than threshold survives; <= threshold is zeroed.
    assert filtered[0, 1] == 0.6
    assert filtered[1, 2] == 0.7
    assert filtered[0, 2] == 0.0  # 0.1 dropped
    assert filtered[1, 0] == 0.0  # 0.2 dropped


def test_filter_edges_directed_topk_keeps_outgoing_only():
    # Row 0 strongly picks 1; row 1 strongly picks 2; asymmetric on purpose.
    weights = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.1, 0.0, 0.9],
            [0.5, 0.4, 0.0],
        ]
    )
    directed = _filter_edges(weights, directed=True, top_edges_per_node=1)
    # Each row keeps exactly its single strongest outgoing edge.
    assert directed[0, 1] == 0.9 and directed[0, 2] == 0.0
    assert directed[1, 2] == 0.9 and directed[1, 0] == 0.0
    assert directed[2, 0] == 0.5 and directed[2, 1] == 0.0
    # Directed filtering must NOT mirror the kept mask: 1->0 stays dropped even
    # though 0->1 is kept.
    assert directed[1, 0] == 0.0

    undirected = _filter_edges(weights, directed=False, top_edges_per_node=1)
    # Undirected symmetrizes the kept mask, so the 0<->1 edge survives both ways.
    assert undirected[0, 1] != 0.0
    assert undirected[1, 0] != 0.0


def test_load_weights_sampler_probability_is_directed_and_unsymmetrized(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    # (n_honest=3, n_total=3) asymmetric row-stochastic probabilities.
    prob = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.3, 0.0, 0.7],
            [0.6, 0.4, 0.0],
        ]
    )
    np.save(results / "sampler_probabilities_final.npy", prob)

    weights, directed = _load_weights(results, "sampler_probability", n_honest=3)
    assert directed is True
    # Must preserve the raw asymmetry, not average opposite directions.
    assert weights[0, 1] == 0.8
    assert weights[1, 0] == 0.3
    assert weights[0, 1] != weights[1, 0]


def test_load_weights_sampler_probability_uses_last_probability_round(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    probabilities = np.array(
        [
            [[0.0, 0.5, 0.5], [0.2, 0.0, 0.8], [0.7, 0.3, 0.0]],
            [[0.0, 0.7, 0.3], [0.4, 0.0, 0.6], [0.1, 0.9, 0.0]],
        ]
    )
    np.save(results / "sampler_probabilities.npy", probabilities)

    weights, directed = _load_weights(results, "sampler_probability", n_honest=3)

    assert directed is True
    np.testing.assert_allclose(weights, probabilities[-1])


def test_load_weights_ignores_unwritten_probability_rounds(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    probabilities = np.full((3, 2, 2), np.nan)
    probabilities[0] = [[0.0, 1.0], [1.0, 0.0]]
    np.save(results / "sampler_probabilities.npy", probabilities)

    weights, _ = _load_weights(results, "sampler_probability", n_honest=2)

    np.testing.assert_allclose(weights, probabilities[0])


def test_load_weights_sampler_probability_averages_seed_probabilities(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    prob_by_seed = np.array(
        [
            [[0.0, 0.8], [0.2, 0.0]],
            [[0.0, 0.4], [0.6, 0.0]],
        ]
    )
    np.save(results / "sampler_probabilities_final_by_seed.npy", prob_by_seed)

    weights, directed = _load_weights(results, "sampler_probability", n_honest=2)

    assert directed is True
    np.testing.assert_allclose(weights, np.array([[0.0, 0.6], [0.4, 0.0]]))


def test_load_weights_neighbor_disagreement_averages_seed_similarities(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    dist_by_seed = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 3.0], [3.0, 0.0]],
        ]
    )
    np.save(results / "pairwise_model_distance_final_by_seed.npy", dist_by_seed)

    weights, directed = _load_weights(results, "neighbor_disagreement", n_honest=2)

    assert directed is False
    np.testing.assert_allclose(weights, np.array([[1.0, 0.375], [0.375, 1.0]]))


def test_plot_clustering_graph_directed_render(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    prob = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.3, 0.0, 0.7],
            [0.6, 0.4, 0.0],
        ]
    )
    np.save(results / "sampler_probabilities_final.npy", prob)

    out = plot_clustering_graph(
        results,
        tmp_path / "plots" / "clustering.png",
        weight_source="sampler_probability",
        threshold=0.25,
    )
    assert out.is_file()
    assert out.stat().st_size > 0


def test_directed_graph_edge_weights_match_probability_matrix(tmp_path):
    # Confirms the DiGraph carries P[i, j] on edge i -> j (the per-node bandit).
    prob = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.3, 0.0, 0.7],
            [0.6, 0.4, 0.0],
        ]
    )
    graph = nx.from_numpy_array(prob, create_using=nx.DiGraph)
    assert graph.is_directed()
    assert graph[0][1]["weight"] == 0.8
    assert graph[1][0]["weight"] == 0.3
