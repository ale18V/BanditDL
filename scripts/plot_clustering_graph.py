"""Render the clustering-graph plot for one or more Hydra run folders.

Usage:
    uv run python scripts/plot_clustering_graph.py <run_dir> [<run_dir> ...] \
        --weight sampler_weight --relative-threshold 1

Looks for seed-averaged final arrays or their `*_by_seed.npy` companions
written by the engine; writes a PNG under each run's `plots/` directory.
"""

from __future__ import annotations

import argparse
import pathlib

from banditdl.utils.plot_graph import (
    plot_clustering_graph,
    plot_collaboration_embedding,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--weight",
        choices=("sampler_weight", "sampler_probability", "neighbor_disagreement"),
        default="sampler_weight",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="keep only edges with weight strictly above this value "
        "(e.g. drop near-uniform exploration edges; None = keep all)",
    )
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=None,
        help="keep outgoing edges above row mean + z standard deviations (default: all edges)",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=None,
        help="keep only the k strongest outgoing edges per node (None = keep all)",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "spring", "group"),
        default="auto",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="plot the mutual sampler-weight spectral embedding instead of the graph",
    )
    parser.add_argument(
        "--hide-byzantine",
        action="store_true",
        help="hide Byzantine target nodes in sampler graphs",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="output filename inside each run's plots/ directory",
    )
    args = parser.parse_args()

    out_name = args.name or (
        "collaboration_embedding.png" if args.embedding else f"clustering_{args.weight}.png"
    )
    for run_dir in args.run_dirs:
        run_dir = pathlib.Path(run_dir)
        # Accept either the Hydra run root or its results directory.
        if (run_dir / "results").is_dir():
            results_dir = run_dir / "results"
            plots_dir = run_dir / "plots"
        else:
            results_dir = run_dir
            plots_dir = run_dir.parent / "plots"
        try:
            if args.embedding:
                out = plot_collaboration_embedding(
                    results_dir,
                    plots_dir / out_name,
                    relative_threshold=args.relative_threshold,
                )
            else:
                out = plot_clustering_graph(
                    results_dir,
                    plots_dir / out_name,
                    weight_source=args.weight,
                    threshold=args.threshold,
                    relative_threshold=args.relative_threshold,
                    top_edges_per_node=args.top_edges,
                    layout=args.layout,
                    include_byzantine=not args.hide_byzantine,
                )
        except FileNotFoundError as exc:
            print(f"[skip] {run_dir}: {exc}")
            continue
        print(f"[ok]  {out}")


if __name__ == "__main__":
    main()
