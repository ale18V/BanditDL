import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.metrics import (
    Aggregation,
    MetricKey,
    MetricLoader,
    TimeAverage,
    Transform,
    max_,
    mean,
    median,
    min_,
)

NODE_CURVE_COLORS = {
    "average": "tab:blue",
    "max": "red",
    "min": "orange",
    "median": "green",
}
NODE_LINESTYLE = {"average": "-", "median": "--", "max": "-", "min": "-"}
NODE_MARKERS = {"average": "o", "median": None, "max": None, "min": None}
NODE_LINEWIDTH = {"average": 1.7, "median": 1.7, "max": 1.3, "min": 1.3}


@dataclass(frozen=True)
class Series:
    metric: MetricKey | str
    label: str
    aggregate: Aggregation | None = None
    transform: Transform | None = None
    color: str | None = None
    linestyle: str = "-"
    marker: bool = True
    interpolate_eval: bool = False


@dataclass(frozen=True)
class Panel:
    title: str
    ylabel: str
    series: Sequence[Series]
    ylim: tuple[float, float] | None = None
    xlabel: str = "Round"
    xscale: str = "linear"
    yscale: str = "linear"
    x_offset: float = 0.0


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


def _mark_first_nonfinite(ax, x: np.ndarray, y: np.ndarray) -> None:
    finite = np.isfinite(y)
    if finite.all():
        return
    first_bad = int(np.flatnonzero(~finite)[0])
    ax.axvline(x[first_bad], color="black", linestyle=":", linewidth=1)
    ax.text(
        x[first_bad],
        0.98,
        "NaN",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
    )


class StandardPlotter:
    def __init__(self, run_dirs: list[Path] | Path, output_dir: Path, labels: list[str] | None = None):
        self.run_dirs = [run_dirs] if isinstance(run_dirs, Path) else run_dirs
        self.output_dir = Path(output_dir)
        self.labels = labels or [d.name for d in self.run_dirs]
        self.loaders = [MetricLoader(d) for d in self.run_dirs]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, name: str, panels: Sequence[Panel]) -> Path:
        if not panels:
            raise ValueError("At least one panel is required")

        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(8, 4.6 if len(panels) == 1 else 3.6 * len(panels)),
            sharex=len(panels) > 1,
        )
        axes = np.atleast_1d(axes)

        for idx, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
            self._draw_panel(ax, panel)
            ax.set_title(panel.title, pad=8)
            ax.set_ylabel(panel.ylabel)
            if panel.ylim is not None:
                ax.set_ylim(*panel.ylim)
            ax.set_xscale(panel.xscale)
            ax.set_yscale(panel.yscale)
            if idx == len(panels) - 1:
                ax.set_xlabel(panel.xlabel)

            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8, frameon=False)

        output = self.output_dir / name
        fig.tight_layout()
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_panel(self, ax, panel: Panel) -> None:
        for series in panel.series:
            for loader_idx, loader in enumerate(self.loaders):
                try:
                    data = loader.load(series.metric, interpolate_eval=series.interpolate_eval)
                except FileNotFoundError:
                    continue

                values = data.values
                if series.transform is not None:
                    values = series.transform(values)

                y = self._aggregate(values, series)
                x = data.x.astype(float) + panel.x_offset

                label = series.label
                if len(self.loaders) > 1:
                    label = f"{self.labels[loader_idx]} - {label}"

                color = series.color or NODE_CURVE_COLORS.get(series.label)
                # If we have multiple loaders, we should probably vary colors/styles
                if len(self.loaders) > 1:
                    color = None # Let matplotlib cycle

                ax.plot(
                    x, y,
                    marker=NODE_MARKERS.get(series.label, "o" if series.marker else None),
                    linewidth=NODE_LINEWIDTH.get(series.label, 1.7),
                    color=color,
                    linestyle=NODE_LINESTYLE.get(series.label, series.linestyle),
                    label=label,
                )
                _mark_first_nonfinite(ax, x, y)

    @staticmethod
    def _aggregate(values: np.ndarray, series: Series) -> np.ndarray:
        if values.ndim == 1:
            return values
        if series.aggregate is None:
            raise ValueError(
                f"Series '{series.label}' for {series.metric} needs an aggregation"
            )
        return series.aggregate.fn(values)


def _node_series(
    metric: MetricKey | str,
    *,
    transform: Transform | None = None,
    interpolate_eval: bool = False,
) -> list[Series]:
    return [
        Series(metric, "average", mean, transform, interpolate_eval=interpolate_eval),
        Series(metric, "max", max_, transform, interpolate_eval=interpolate_eval),
        Series(metric, "min", min_, transform, interpolate_eval=interpolate_eval),
        Series(metric, "median", median, transform, interpolate_eval=interpolate_eval),
    ]


def _sampler_aggressiveness_panels() -> list[Panel]:
    return [
        Panel(
            "Sampler Aggressiveness",
            "KL(sampler || uniform)",
            _node_series(MetricKey.SAMPLER_KL_TO_UNIFORM),
        ),
        Panel(
            "Sampler Probability Entropy",
            "Entropy",
            _node_series(MetricKey.SAMPLER_ENTROPY),
        ),
        Panel(
            "Sampler Probability Range",
            "Probability",
            [
                Series(
                    MetricKey.SAMPLER_MAX_PROBABILITY,
                    "max probability",
                    max_,
                    color="tab:red",
                    marker=False,
                ),
                Series(
                    MetricKey.SAMPLER_MIN_PROBABILITY,
                    "min probability",
                    min_,
                    color="tab:blue",
                    marker=False,
                ),
            ],
        ),
    ]


def _sampler_weight_panels() -> list[Panel]:
    return [
        Panel(
            "Sampler Preference Concentration",
            "KL(weights || uniform)",
            _node_series(MetricKey.SAMPLER_WEIGHT_KL_TO_UNIFORM),
        ),
        Panel(
            "Sampler Preference Entropy",
            "Entropy",
            _node_series(MetricKey.SAMPLER_WEIGHT_ENTROPY),
        ),
        Panel(
            "Sampler Preference Range",
            "Normalized weight",
            [
                Series(
                    MetricKey.SAMPLER_MAX_WEIGHT,
                    "max weight",
                    max_,
                    color="tab:red",
                    marker=False,
                ),
                Series(
                    MetricKey.SAMPLER_MIN_WEIGHT,
                    "min weight",
                    min_,
                    color="tab:blue",
                    marker=False,
                ),
            ],
        ),
    ]


def _gradient_norm_loglog_panel() -> Panel:
    return Panel(
        "Gradient Norm Decay",
        "Gradient norm",
        [
            Series(MetricKey.GRADIENT_NORMS, "average", mean),
            Series(MetricKey.GRADIENT_NORMS, "worst", max_, color="red", marker=False),
            Series(MetricKey.GRADIENT_NORMS, "best", min_, color="green", marker=False),
        ],
        xlabel="Round + 1",
        xscale="log",
        yscale="log",
        x_offset=1.0,
    )


def plot_all(run_dir: Path, plots_dir: Path, run_label: str | None = None) -> None:
    labels = [run_label] if run_label else None
    plotter = StandardPlotter(run_dir, plots_dir, labels=labels)

    plotter.plot(
        "validation_accuracy.png",
        [
            Panel(
                "Validation Accuracy",
                "Accuracy",
                _node_series(MetricKey.VALIDATION_ACCURACY, interpolate_eval=True),
                ylim=(0, 1),
            )
        ],
    )

    plotter.plot(
        "global_accuracy.png",
        [
            Panel(
                "Subsampled Global Test Accuracy",
                "Accuracy",
                _node_series(MetricKey.GLOBAL_ACCURACY, interpolate_eval=True),
                ylim=(0, 1),
            )
        ],
    )

    plotter.plot(
        "validation_loss.png",
        [
            Panel(
                "Validation Loss",
                "Loss",
                _node_series(MetricKey.VALIDATION_LOSS, interpolate_eval=True),
            )
        ],
    )

    plotter.plot(
        "train_loss.png",
        [
            Panel(
                "Training Loss",
                "Loss",
                _node_series(MetricKey.TRAIN_LOSS),
            )
        ],
    )

    plotter.plot(
        "neighbor_disagreement.png",
        [
            Panel(
                "Neighbor Disagreement",
                "Neighbor disagreement",
                _node_series(MetricKey.NEIGHBOR_DISAGREEMENT),
            )
        ],
    )

    plotter.plot(
        "consensus_drift.png",
        [
            Panel(
                "Consensus Drift",
                "Consensus drift",
                _node_series(MetricKey.CONSENSUS_DRIFT),
            )
        ],
    )

    plotter.plot("gradient_norm_loglog.png", [_gradient_norm_loglog_panel()])

    plotter.plot(
        "sampler_aggressiveness.png",
        _sampler_aggressiveness_panels(),
    )
    plotter.plot("sampler_weights.png", _sampler_weight_panels())

    plotter.plot(
        "reward.png",
        [
            Panel(
                "Reward",
                "Reward",
                [*_node_series(MetricKey.REWARD_ALGORITHM), Series(MetricKey.REWARD_ORACLE, "oracle average", mean, color="black", linestyle="--", marker=False)],
            ),
            Panel(
                "Time-normalized reward",
                "Reward",
                [*_node_series(MetricKey.REWARD_ALGORITHM, transform=TimeAverage()), Series(MetricKey.REWARD_ORACLE, "oracle average", mean, TimeAverage(), color="black", linestyle="--", marker=False)],
            ),
        ],
    )

    plotter.plot(
        "reward_extrema.png",
        [
            Panel(
                "Selected Neighbor Max Reward",
                "Reward",
                _node_series(MetricKey.REWARD_SELECTED_MAX),
            ),
            Panel(
                "Selected Neighbor Min Reward",
                "Reward",
                _node_series(MetricKey.REWARD_SELECTED_MIN),
            ),
        ],
    )

    plotter.plot(
        "regret.png",
        [
            Panel("Regret", "Regret", _node_series(MetricKey.REGRET)),
            Panel(
                "Normalized regret",
                "Normalized regret",
                _node_series(MetricKey.REGRET, transform=TimeAverage()),
            ),
        ],
    )

    from banditdl.utils.plot_graph import plot_clustering_graph

    for weight_source in ("sampler_probability", "neighbor_disagreement"):
        try:
            plot_clustering_graph(
                run_dir,
                Path(plots_dir) / f"clustering_{weight_source}.png",
                weight_source=weight_source,
            )
        except FileNotFoundError:
            pass
