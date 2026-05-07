from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np

ARRAY_METRICS = {
    "accuracies",
    "val_accuracy",
    "reward_algorithm",
    "reward_oracle",
    "regret",
    "normalized_regret",
    "neighbor_disagreement",
    "consensus_drift",
    "sampler_kl_to_uniform",
}
REGRET_METRICS = {"regret", "normalized_regret"}
REWARD_METRICS = {"reward_algorithm", "reward_oracle"}
DISTANCE_METRICS = {"neighbor_disagreement", "consensus_drift"}
MAX_METRICS = REGRET_METRICS | DISTANCE_METRICS
PER_NODE_METRICS = (
    REGRET_METRICS
    | REWARD_METRICS
    | DISTANCE_METRICS
    | {"accuracies", "val_accuracy", "sampler_kl_to_uniform"}
)
ALL_PLOT_METRICS = (
    "val_accuracy",
    "validation_loss",
    "train_loss",
    "reward_algorithm",
    "regret",
    "normalized_regret",
    "neighbor_disagreement",
    "consensus_drift",
    "sampler_aggressiveness",
)
NODE_CURVE_COLORS = {
    "average": "tab:blue",
    "max": "red",
    "min": "orange",
    "median": "green",
}
NODE_LINESTYLE = {"average": "-", "median": "--", "max": "-", "min": "-"}


def _read_eval(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            step, value, *_ = line.split()
            steps.append(float(step))
            values.append(float(value))
    return np.asarray(steps), np.asarray(values)


def _load_raw_array(run_dir: Path, metric: str) -> np.ndarray:
    metric_aliases = {
        "val_accuracy": ("validation_accuracies", "accuracies"),
        "accuracies": ("validation_accuracies", "accuracies"),
    }
    candidate_names = metric_aliases.get(metric, (metric,))
    path = None
    for metric_name in candidate_names:
        candidate = run_dir / f"{metric_name}.npy"
        if candidate.exists():
            path = candidate
            break
    if path is None:
        path = run_dir / f"{candidate_names[0]}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    values_by_worker = np.load(path)
    if values_by_worker.size == 0:
        raise ValueError(f"{path} is empty")
    return values_by_worker


def _load_series(run_dir: Path, metric: str, stat: str) -> tuple[np.ndarray, np.ndarray]:
    if metric in ARRAY_METRICS:
        values_by_worker = _load_raw_array(run_dir, metric)
        if stat == "mean":
            values = values_by_worker.mean(axis=1)
        elif stat == "worst":
            reducer = np.max if metric in MAX_METRICS else np.min
            values = reducer(values_by_worker, axis=1)
        else:
            raise ValueError(f"unsupported stat for {metric}: {stat}")
        return np.arange(len(values)), values

    metric_aliases = {
        "eval": ("validation", "eval"),
        "eval_worst": ("validation_worst", "eval_worst"),
        "validation": ("validation", "eval"),
        "validation_worst": ("validation_worst", "eval_worst"),
        "loss": ("validation_loss",),
        "train_loss": ("train_loss",),
    }
    candidate_names = metric_aliases.get(metric, (metric,))
    metric_path = None
    for metric_name in candidate_names:
        candidate = run_dir / metric_name
        if candidate.exists():
            metric_path = candidate
            break
    if metric_path is None:
        metric_path = run_dir / candidate_names[0]
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    steps, values = _read_eval(metric_path)
    if values.size == 0:
        raise ValueError(f"{metric_path} is empty")
    return steps, values


def _node_curves(raw: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("average", raw.mean(axis=1)),
        ("max", raw.max(axis=1)),
        ("min", raw.min(axis=1)),
        ("median", np.median(raw, axis=1)),
    ]


def _ylabel(metric: str) -> str:
    if metric in {"accuracies", "val_accuracy", "validation", "validation_worst", "test", "eval", "eval_worst"}:
        return "Accuracy"
    if metric in {"validation_loss", "loss", "train_loss"}:
        return "Loss"
    if metric in REGRET_METRICS:
        return "Regret"
    if metric == "neighbor_disagreement":
        return "Neighbor disagreement"
    if metric == "consensus_drift":
        return "Consensus drift"
    if metric == "sampler_kl_to_uniform":
        return "KL divergence"
    return "Reward"


def _display_title(metric: str) -> str:
    titles = {
        "val_accuracy": "Validation Accuracy",
        "accuracies": "Validation Accuracy",
        "validation_loss": "Validation Loss",
        "train_loss": "Training Loss",
        "reward_algorithm": "Bandit Reward vs Oracle",
        "reward_oracle": "Oracle Reward",
        "regret": "Regret",
        "normalized_regret": "Normalized Regret",
        "neighbor_disagreement": "Neighbor Disagreement",
        "consensus_drift": "Consensus Drift",
        "sampler_kl_to_uniform": "Sampler KL Divergence to Uniform",
    }
    return titles.get(metric, metric.replace("_", " ").title())


def _extract_run_hparams(label: str) -> str | None:
    alpha_match = re.search(r"(?:^|-)alpha_([^-\s]+)", label)
    nodes_match = re.search(r"(?:^|-)n_(\d+)", label)
    sampling_match = re.search(r"(?:^|-)sampling_([^-\s]+)", label)
    if not (alpha_match and nodes_match):
        return None
    alpha = alpha_match.group(1)
    nodes = nodes_match.group(1)
    sampling = sampling_match.group(1) if sampling_match else "NA"
    return rf"$\alpha$={alpha}, n={nodes}, s={sampling}"


def _color_for(index: int) -> str:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[index % len(cycle)]


def _plot_curve(ax, steps: np.ndarray, values: np.ndarray, label: str, color: str | None, linestyle: str, marker: bool) -> None:
    ax.plot(
        steps,
        values,
        marker="o" if marker else None,
        linewidth=1.7,
        color=color,
        linestyle=linestyle,
        label=label,
    )


def _default_label(run_dir: Path, max_length: int) -> str:
    tokens = run_dir.name.split("-")
    keep_prefixes = ("sampling_", "degree_", "sampler_", "eps_", "init_", "seed_")
    label_parts = [token for token in tokens if token.startswith(keep_prefixes) or token in {"cs+", "cs_he", "gts"}]
    label = ", ".join(label_parts) if label_parts else run_dir.name
    return shorten(label, width=max_length, placeholder="...")


def _aggregate_series(series: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not series:
        raise ValueError("no series to aggregate")
    length = min(len(values) for _, values in series)
    if length == 0:
        raise ValueError("cannot aggregate empty series")
    steps = series[0][0][:length]
    stacked = np.stack([values[:length] for _, values in series])
    return steps, stacked.mean(axis=0), stacked.std(axis=0)


def _add_legend(ax, legend: str, fig) -> None:
    if legend == "outside":
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.0, -0.21, 1.0, 0.1),
                mode="expand",
                ncols=len(handles),
                frameon=False,
                fontsize=8,
                borderaxespad=0.0,
            )
        fig.subplots_adjust(bottom=0.27)
        return
    if legend == "best":
        ax.legend(fontsize=8)
        fig.tight_layout()
        return
    if legend == "none":
        fig.tight_layout()
        return
    raise ValueError(f"Unknown legend placement: {legend}")


def plot_runs(run_dirs: Sequence[Path], output: Path, metric: str, stat: str, title: str | None, labels: Sequence[str] | None, aggregate: bool, legend: str, max_label_length: int) -> None:
    if metric == "sampler_aggressiveness":
        if len(run_dirs) != 1:
            raise ValueError("sampler_aggressiveness expects exactly one run directory")
        plot_sampler_aggressiveness(run_dirs[0], output, title)
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xlabel = "Evaluation index" if metric in {"accuracies", "val_accuracy"} else "Step"
    is_per_node = metric in PER_NODE_METRICS

    if aggregate:
        if is_per_node:
            raws = [_load_raw_array(run_dir, metric) for run_dir in run_dirs]
            length = min(len(r) for r in raws)
            if length == 0:
                raise ValueError("cannot aggregate empty per-node series")
            steps = np.arange(length)
            template = _node_curves(raws[0][:length])
            for kind_idx, (kind, _) in enumerate(template):
                stacked = np.stack([_node_curves(r[:length])[kind_idx][1] for r in raws])
                mean = stacked.mean(axis=0)
                _plot_curve(ax, steps, mean, kind, NODE_CURVE_COLORS[kind], NODE_LINESTYLE[kind], kind == "average")
            if metric == "reward_algorithm":
                oracle_means = []
                for run_dir in run_dirs:
                    try:
                        oracle_raw = _load_raw_array(run_dir, "reward_oracle")
                        oracle_means.append(oracle_raw[:length].mean(axis=1))
                    except FileNotFoundError:
                        continue
                if oracle_means:
                    oracle_mean = np.stack(oracle_means).mean(axis=0)
                    _plot_curve(ax, steps, oracle_mean, "oracle average", "black", "--", False)
        else:
            series = [_load_series(run_dir, metric, stat) for run_dir in run_dirs]
            steps, mean, std = _aggregate_series(series)
            label = labels[0] if labels else f"{metric} {stat}"
            label = shorten(label, width=max_label_length, placeholder="...")
            _plot_curve(ax, steps, mean, label, None, "-", True)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    else:
        if labels and len(labels) != len(run_dirs):
            raise SystemExit("--label must be passed once per run directory")
        for index, run_dir in enumerate(run_dirs):
            base_label = labels[index] if labels else _default_label(run_dir, max_label_length)
            base_label = shorten(base_label, width=max_label_length, placeholder="...")

            if is_per_node:
                raw = _load_raw_array(run_dir, metric)
                steps = np.arange(len(raw))
                for kind, values in _node_curves(raw):
                    _plot_curve(ax, steps, values, kind, NODE_CURVE_COLORS[kind], NODE_LINESTYLE[kind], kind == "average")
                if metric == "reward_algorithm":
                    try:
                        oracle_raw = _load_raw_array(run_dir, "reward_oracle")
                    except FileNotFoundError:
                        oracle_raw = None
                    if oracle_raw is not None:
                        oracle_steps = np.arange(len(oracle_raw))
                        oracle_values = oracle_raw.mean(axis=1)
                        _plot_curve(
                            ax,
                            oracle_steps,
                            oracle_values,
                            "oracle average",
                            "black",
                            "--",
                            False,
                        )
            else:
                steps, values = _load_series(run_dir, metric, stat)
                _plot_curve(ax, steps, values, base_label, None, "-", True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_ylabel(metric))
    if metric in {"accuracies", "val_accuracy", "validation", "validation_worst", "test", "eval", "eval_worst"}:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    inferred_caption = _extract_run_hparams(title or "")
    if title and not inferred_caption:
        plot_title = title
    elif title:
        plot_title = _display_title(metric)
    elif aggregate:
        plot_title = f"Aggregate {_display_title(metric)}"
    else:
        plot_title = _display_title(metric)
    ax.set_title(plot_title)

    if inferred_caption:
        fig.text(0.5, 0.985, inferred_caption, ha="center", va="top", fontsize=9)
    if is_per_node:
        fig.text(
            0.5,
            0.01,
            "Node-wise metrics are aggregated each round across nodes: average, median, max, min.",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    _add_legend(ax, legend, fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sampler_aggressiveness(run_dir: Path, output: Path, title: str | None = None) -> None:
    kl = _load_raw_array(run_dir, "sampler_kl_to_uniform")
    min_probability = _load_raw_array(run_dir, "sampler_min_probability")
    max_probability = _load_raw_array(run_dir, "sampler_max_probability")
    steps = np.arange(len(kl))

    fig, (ax_kl, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for kind, values in _node_curves(kl):
        _plot_curve(
            ax_kl,
            steps,
            values,
            kind,
            NODE_CURVE_COLORS[kind],
            NODE_LINESTYLE[kind],
            kind == "average",
        )
    ax_kl.set_ylabel("KL(bandit || uniform)")
    ax_kl.set_title(title or "Sampler aggressiveness")
    ax_kl.grid(True, alpha=0.25)
    ax_kl.legend(ncols=4, frameon=False)

    ax_prob.plot(
        steps,
        max_probability.max(axis=1),
        color="tab:red",
        linewidth=1.7,
        label="max probability",
    )
    ax_prob.plot(
        steps,
        min_probability.min(axis=1),
        color="tab:blue",
        linewidth=1.7,
        label="min probability",
    )
    ax_prob.set_xlabel("Round")
    ax_prob.set_ylabel("Probability")
    ax_prob.grid(True, alpha=0.25)
    ax_prob.legend(ncols=2, frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_all(run_dir: Path, plots_dir: Path, run_label: str) -> list[Path]:
    """Generate all supported plots for a single run directory.

    Args:
        run_dir: Path
            Directory containing run artifacts (validation/validation_worst and .npy files).
        plots_dir: Path
            Output directory for generated plots.
        run_label: str
            Label used in plot titles and legends.
        return: list[Path]
            Paths of all generated plot files.
    """
    written_paths: list[Path] = []
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_node_metrics = set(PER_NODE_METRICS) | {"accuracies"}

    for metric in ALL_PLOT_METRICS:
        stats = ("mean",)
        for stat in stats:
            filename = "val_accuracy.png" if metric in {"accuracies", "val_accuracy"} else f"{metric}.png"
            output = plots_dir / filename
            try:
                plot_runs(
                    run_dirs=[run_dir],
                    output=output,
                    metric=metric,
                    stat=stat,
                    title=run_label,
                    labels=[run_label],
                    aggregate=False,
                    legend="outside",
                    max_label_length=48,
                )
                written_paths.append(output)
            except FileNotFoundError:
                continue
    return written_paths
