from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.experiment_table import ExperimentTable
from banditdl.utils.plotting_utils import (
    aggregate,
    axis_label,
    axis_name,
    axis_path,
    axis_value,
    cycle_color,
    display_name,
    expand_fixed,
    filter_label,
    filter_path,
    group_filters,
    group_label,
    metric_column,
    normalize_axis,
    normalize_groups,
    normalize_render,
    sanitize_label,
    save_figure,
    sort_key,
)


@dataclass(frozen=True)
class _Heatmap:
    values: np.ndarray
    metric: str
    direction: str
    x_paths: tuple[str, ...]
    y_paths: tuple[str, ...]
    x_values: list
    y_values: list
    fixed: dict
    aggregate_by: str


class SweepPlotter:
    """Render sweep metrics from an ExperimentTable."""

    def __init__(self, table: ExperimentTable, output_dir: Path):
        self.table = table
        self.output_dir = output_dir

    def plot_line_spec(self, spec: dict, metrics: list[str], direction: str) -> None:
        if not (x_path := spec.get("x")):
            return
        fixed = dict(spec.get("fixed") or {})
        mode = str(spec.get("aggregate_by", "avg"))
        for filters in expand_fixed(fixed):
            for paths in normalize_groups(spec.get("group_by")):
                for metric in metrics:
                    self._plot_lines(metric, direction, x_path, paths, filters, mode)

    def _plot_lines(
        self,
        metric: str,
        direction: str,
        x_path: str,
        group_paths: tuple[str, ...],
        fixed: dict,
        mode: str,
    ) -> None:
        column = metric_column(metric, direction)
        curves: dict[tuple, dict[object, list[float]]] = {}
        for row in self.table.filter(fixed).rows:
            if x_path not in row.params or column not in row.metrics:
                continue
            group = tuple(row.params.get(path) for path in group_paths)
            curves.setdefault(group, {}).setdefault(row.params[x_path], []).append(
                row.metrics[column]
            )
        if not curves:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        for index, (group, points) in enumerate(sorted(curves.items(), key=lambda x: str(x[0]))):
            xs = sorted(points, key=sort_key)
            ax.plot(
                xs,
                [aggregate(points[x], mode) for x in xs],
                marker="o",
                color=cycle_color(index),
                label=group_label(self.table, group_paths, group),
            )
        title = f"{metric} ({direction})"
        if fixed:
            title += f"\nfixed: {filter_label(self.table, fixed)}"
        ax.set(
            xlabel=display_name(self.table, x_path),
            ylabel=f"{metric} ({direction})",
            title=title,
        )
        if group_paths:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

        group = "_".join(display_name(self.table, path) for path in group_paths) or "all"
        directory = (
            self.output_dir
            / "line"
            / f"direction={direction}"
            / f"x={sanitize_label(display_name(self.table, x_path))}"
            / f"group={sanitize_label(group)}"
        )
        if fixed:
            directory /= f"fixed={filter_path(fixed)}"
        save_figure(fig, directory / f"{metric}.png")

    # Declarative 2D and 3D heatmaps.
    def plot_heatmap_spec(self, spec: dict, metrics: list[str], direction: str) -> None:
        if not spec.get("x") or not spec.get("y"):
            return
        x_paths, y_paths = normalize_axis(spec["x"]), normalize_axis(spec["y"])
        fixed = dict(spec.get("fixed") or {})
        mode = str(spec.get("aggregate_by", "avg"))
        render = normalize_render(spec.get("render"))
        for fixed_filter in expand_fixed(fixed):
            for paths in normalize_groups(spec.get("group_by")):
                for filters in group_filters(self.table.rows, paths, fixed_filter):
                    for metric in metrics:
                        self.plot_heatmap(
                            metric, direction, x_paths, y_paths, filters, mode, render
                        )

    def plot_heatmap(  # noqa: PLR0913 - preserved public plotting API
        self,
        metric: str,
        direction: str,
        x_path: str | tuple[str, ...],
        y_path: str | tuple[str, ...],
        fixed_params: dict | None = None,
        aggregate_by: str = "avg",
        render: list[str] | None = None,
    ) -> None:
        fixed = fixed_params or {}
        x_paths = (x_path,) if isinstance(x_path, str) else x_path
        y_paths = (y_path,) if isinstance(y_path, str) else y_path
        column = metric_column(metric, direction)
        rows = self.table.filter(fixed).rows
        xs = sorted({axis_value(row.params, x_paths) for row in rows}, key=sort_key)
        ys = sorted({axis_value(row.params, y_paths) for row in rows}, key=sort_key)
        matrix = np.full((len(ys), len(xs)), np.nan)
        for y_index, y_value in enumerate(ys):
            for x_index, x_value in enumerate(xs):
                values = [
                    row.metrics[column]
                    for row in rows
                    if axis_value(row.params, x_paths) == x_value
                    and axis_value(row.params, y_paths) == y_value
                    and column in row.metrics
                ]
                if values:
                    matrix[y_index, x_index] = aggregate(values, aggregate_by)
        if np.isnan(matrix).all():
            return

        data = _Heatmap(
            matrix, metric, direction, x_paths, y_paths, xs, ys, fixed, aggregate_by
        )
        for kind in render or ["heatmap"]:
            if kind == "heatmap":
                self._save_heatmap(data)
            elif kind == "heatmap3d":
                self._save_heatmap3d(data)

    def _save_heatmap(self, data: _Heatmap) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad(color="0.85")
        image = ax.imshow(data.values, origin="lower", aspect="auto", cmap=cmap)
        fig.colorbar(image, ax=ax, label=f"{data.metric} ({data.direction})")
        self._format_heatmap_axes(ax, data)
        for y, x in np.ndindex(data.values.shape):
            value = data.values[y, x]
            ax.text(
                x,
                y,
                "" if np.isnan(value) else f"{value:.3g}",
                ha="center",
                va="center",
                fontsize=8,
            )
        save_figure(fig, self._heatmap_path("heatmap", data))

    def _save_heatmap3d(self, data: _Heatmap) -> None:
        x_grid, y_grid = np.meshgrid(
            np.arange(len(data.x_values)), np.arange(len(data.y_values))
        )
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(
            x_grid, y_grid, np.ma.masked_invalid(data.values), cmap="viridis"
        )
        self._format_heatmap_axes(ax, data, is_3d=True)
        ax.set_zlabel(data.metric)
        fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1)
        save_figure(fig, self._heatmap_path("heatmap3d", data))

    def _format_heatmap_axes(self, ax, data: _Heatmap, is_3d: bool = False) -> None:
        ax.set_xticks(range(len(data.x_values)))
        labels = [axis_label(value) for value in data.x_values]
        if is_3d:
            ax.set_xticklabels(labels, rotation=35, ha="right")
        else:
            ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(len(data.y_values)))
        ax.set_yticklabels([axis_label(value) for value in data.y_values])
        ax.set_xlabel(axis_name(self.table, data.x_paths))
        ax.set_ylabel(axis_name(self.table, data.y_paths))
        details = [f"extra dims={data.aggregate_by}"]
        if data.fixed:
            details.append(
                ", ".join(
                    f"{key.split('.')[-1]}={value}" for key, value in data.fixed.items()
                )
            )
        ax.set_title(f"{data.metric} ({data.direction})\n{'; '.join(details)}")

    def _heatmap_path(self, kind: str, data: _Heatmap) -> Path:
        axes = f"{axis_path(self.table, data.x_paths)}_{axis_path(self.table, data.y_paths)}"
        group = filter_path(data.fixed)
        return (
            self.output_dir
            / kind
            / f"direction={data.direction}"
            / f"axes={axes}"
            / group
            / f"{data.metric}.png"
        )
