#!/usr/bin/env python3
"""Plot saved experiment results from one or more run directories."""
import argparse
from collections.abc import Sequence
from pathlib import Path
from textwrap import shorten

from banditdl.utils.metrics import MetricLoader, max_, mean
from banditdl.utils.plotting import (
    StandardPlotter,
    _mark_first_nonfinite,
    _sampler_aggressiveness_panels,
    np,
    plt,
)

def plot_runs(
    run_dirs: Sequence[Path],
    output: Path,
    metric: str,
    stat: str,
    title: str | None,
    labels: Sequence[str] | None,
    aggregate: bool,
    legend: str,
    max_label_length: int,
) -> None:
    """Small CLI-compatible helper for ad hoc single-metric plots."""
    if metric == "sampler_aggressiveness":
        if len(run_dirs) != 1:
            raise ValueError("sampler_aggressiveness expects exactly one run directory")
        StandardPlotter(run_dirs[0], output, title).plot(
            "sampler_aggressiveness.png", _sampler_aggressiveness_panels()
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if labels and len(labels) != len(run_dirs):
        raise SystemExit("--label must be passed once per run directory")

    series = []
    for idx, run_dir in enumerate(run_dirs):
        loader = MetricLoader(run_dir)
        data = loader.load(
            metric,
            interpolate_eval=metric
            in {"accuracies", "val_accuracy", "validation_accuracies"},
        )
        values = data.values
        if values.ndim > 1:
            reducer = max_ if stat == "worst" else mean
            y = reducer.fn(values)
        else:
            y = values
        label = labels[idx] if labels else Path(run_dir).name
        label = shorten(label, width=max_label_length, placeholder="...")
        series.append((data.x, y, label))

    if aggregate:
        length = min(len(y) for _, y, _ in series)
        x = series[0][0][:length]
        stacked = np.stack([y[:length] for _, y, _ in series])
        y = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        label = labels[0] if labels else f"{metric} {stat}"
        ax.plot(x, y, marker="o", linewidth=1.7, label=label)
        ax.fill_between(x, y - std, y + std, alpha=0.2)
    else:
        for x, y, label in series:
            ax.plot(x, y, marker="o", linewidth=1.7, label=label)
            _mark_first_nonfinite(ax, x, y)

    ax.set_title(title or str(metric).replace("_", " ").title())
    ax.set_xlabel("Round")
    ax.grid(True, alpha=0.25)
    if legend != "none":
        ax.legend(
            loc="best" if legend == "best" else "upper center",
            frameon=False,
            fontsize=8,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
            "reward_selected_min",
            "reward_selected_max",
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
