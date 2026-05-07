#!/usr/bin/env python3
"""Plot saved experiment results from one or more run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from banditdl.utils.plotting import plot_runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a saved banditdl result directory."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Directories containing val_accuracy/validation_loss and other run metrics",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("plot.png"), help="Output image path"
    )
    parser.add_argument(
        "--metric",
        choices=[
            "accuracies",
            "val_accuracy",
            "validation_loss",
            "train_loss",
            "validation",
            "validation_worst",
            "test",
            "eval",
            "eval_worst",
            "reward_algorithm",
            "reward_oracle",
            "regret",
            "normalized_regret",
            "neighbor_disagreement",
            "consensus_drift",
            "sampler_kl_to_uniform",
            "sampler_aggressiveness",
        ],
        default="val_accuracy",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "worst"],
        default="mean",
        help="Worker statistic. For regret metrics, worst means highest regret.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Plot mean +/- std across the given run directories",
    )
    parser.add_argument(
        "--legend", choices=["outside", "best", "none"], default="outside"
    )
    parser.add_argument("--max-label-length", type=int, default=48)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Line label. Repeat once per run, or once with --aggregate",
    )
    args = parser.parse_args()

    plot_runs(
        args.run_dirs,
        args.output,
        args.metric,
        args.stat,
        args.title,
        args.label,
        args.aggregate,
        args.legend,
        args.max_label_length,
    )


if __name__ == "__main__":
    main()
