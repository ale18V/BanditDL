from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

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
    def __init__(self, run_dir: Path, output_dir: Path, run_label: str | None = None):
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir)
        self.run_label = run_label or ""
        self.loader = MetricLoader(self.run_dir)

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
            ax.set_title(panel.title, pad=18 if idx == 0 else 8)
            ax.set_ylabel(panel.ylabel)
            if panel.ylim is not None:
                ax.set_ylim(*panel.ylim)
            if idx == len(panels) - 1:
                ax.set_xlabel("Round")
            if idx == 0:
                caption = _extract_run_hparams(self.run_label)
                if caption:
                    ax.text(
                        0.5,
                        1.01,
                        caption,
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            ax.grid(True, alpha=0.25)
            ax.legend(
                loc="best", ncols=min(4, len(panel.series)), frameon=False, fontsize=8
            )

        if any(
            series.aggregate is not None for panel in panels for series in panel.series
        ):
            fig.text(
                0.5,
                0.01,
                "Node-wise metrics are aggregated each round across nodes.",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            rect = (0.0, 0.06, 1.0, 0.97)
        else:
            rect = (0.0, 0.03, 1.0, 0.97)

        output = self.output_dir / name
        fig.tight_layout(rect=rect)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_panel(self, ax, panel: Panel) -> None:
        for series in panel.series:
            data = self.loader.load(
                series.metric, interpolate_eval=series.interpolate_eval
            )
            values = data.values
            if series.transform is not None:
                values = series.transform(values)
            y = self._aggregate(values, series)
            color = series.color or NODE_CURVE_COLORS.get(series.label)
            linestyle = NODE_LINESTYLE.get(series.label, series.linestyle)
            marker = NODE_MARKERS.get(series.label, "o" if series.marker else None)
            linewidth = NODE_LINEWIDTH.get(series.label, 1.7)
            ax.plot(
                data.x,
                y,
                marker=marker,
                linewidth=linewidth,
                color=color,
                linestyle=linestyle,
                label=series.label,
            )
            _mark_first_nonfinite(ax, data.x, y)

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


def plot_all(run_dir: Path, plots_dir: Path, run_label: str) -> None:
    plotter = StandardPlotter(run_dir, plots_dir, run_label)

    plotter.plot(
        "val_accuracy.png",
        [
            Panel(
                "Validation Accuracy",
                "Accuracy",
                _node_series(MetricKey.VALIDATION_ACCURACIES, interpolate_eval=True),
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
                [Series(MetricKey.VALIDATION_LOSS, "validation loss")],
            )
        ],
    )

    plotter.plot(
        "train_loss.png",
        [Panel("Training Loss", "Loss", [Series(MetricKey.TRAIN_LOSS, "train loss")])],
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

    plotter.plot(
        "sampler_aggressiveness.png",
        _sampler_aggressiveness_panels(),
    )

    plotter.plot(
        "reward.png",
        [
            Panel(
                "Reward",
                "Reward",
                _node_series(MetricKey.REWARD_ALGORITHM)
                + [
                    Series(
                        MetricKey.REWARD_ORACLE,
                        "oracle average",
                        mean,
                        color="black",
                        linestyle="--",
                        marker=False,
                    )
                ],
            ),
            Panel(
                "Time-normalized reward",
                "Reward",
                _node_series(MetricKey.REWARD_ALGORITHM, transform=TimeAverage())
                + [
                    Series(
                        MetricKey.REWARD_ORACLE,
                        "oracle average",
                        mean,
                        TimeAverage(),
                        color="black",
                        linestyle="--",
                        marker=False,
                    )
                ],
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
