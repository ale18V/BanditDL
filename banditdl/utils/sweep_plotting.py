from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.experiment_table import ExperimentTable
from banditdl.utils.plot_sweep_base import column_key_for


def _sanitize_label(text):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    return cleaned.strip("_") or "axis"


def _cycle_color(index):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[index % len(cycle)]


def _sort_key(value):
    if isinstance(value, (int, float, np.floating)):
        return (0, float(value))
    return (1, str(value))


def _axis_label(axis, fallback: str) -> str:
    return getattr(axis, "display_name", fallback.rsplit(".", maxsplit=1)[-1])


def _aggregate(values: list[float], mode: str) -> float:
    values = np.asarray(values, dtype=float)
    if mode in {"avg", "mean", "average"}:
        return float(np.nanmean(values))
    if mode == "min":
        return float(np.nanmin(values))
    if mode == "max":
        return float(np.nanmax(values))
    raise ValueError("aggregate_by must be one of: avg, mean, min, max")


def _normalize_groups(raw) -> list[tuple[str, ...]]:
    if raw is None:
        return [()]
    groups = raw if isinstance(raw, list) else [raw]
    result = []
    for group in groups:
        if isinstance(group, str):
            result.append((group,))
        elif isinstance(group, list):
            result.append(tuple(str(item) for item in group))
        else:
            raise ValueError(f"invalid group_by entry: {group!r}")
    return result or [()]


def _normalize_render(raw) -> list[str]:
    render = raw if isinstance(raw, list) else [raw or "heatmap"]
    result = []
    for raw_item in render:
        item = str(raw_item)
        if item not in {"heatmap", "heatmap3d"}:
            raise ValueError("render entries must be heatmap or heatmap3d")
        if item not in result:
            result.append(item)
    return result


def _row_matches(params: dict, filters: dict) -> bool:
    return all(params.get(path) == value for path, value in filters.items())


class SweepPlotter:
    def __init__(self, table: ExperimentTable, output_dir: Path):
        self.table = table
        self.output_dir = output_dir

    def plot_per_parameter(self, metric: str, direction: str):
        """WAY 2: sweep line plots per secondary axis with remaining dims fixed."""
        axes = getattr(self.table, "axes_meta", [])
        if not axes:
            return

        column_key = column_key_for(metric, direction)

        for x_axis in axes:
            other_axes = [ax for ax in axes if ax.path != x_axis.path]

            if not other_axes:
                # One-dimensional sweep case
                self._draw_line_plot(metric, direction, column_key, x_axis, None, {})
                continue

            for curve_axis in other_axes:
                fixed_axes = [ax for ax in other_axes if ax.path != curve_axis.path]
                fixed_combos = self.table.get_combinations([ax.path for ax in fixed_axes]) or [{}]

                for fixed_params in fixed_combos:
                    self._draw_line_plot(
                        metric, direction, column_key, x_axis, curve_axis, fixed_params
                    )

    def _draw_line_plot(self, metric, direction, column_key, x_axis, curve_axis, fixed_params):
        # Filter table by fixed params
        subset = self.table.filter(fixed_params)
        pivoted = subset.pivot(x_axis.path, curve_axis.path if curve_axis else None)

        if not pivoted:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (curve_val, points) in enumerate(pivoted.items()):
            xs = [p[0] for p in points]
            ys = [p[1].get(column_key, np.nan) for p in points]
            label = f"{curve_axis.display_name}={curve_val}" if curve_axis else "Cross-seed average"
            ax.plot(xs, ys, marker="o", label=label, color=_cycle_color(i))

        ax.set_xlabel(x_axis.display_name)
        ax.set_ylabel(f"{metric} ({direction})")

        fixed_desc = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in fixed_params.items())
        title = f"{metric} | {x_axis.display_name}"
        if fixed_desc:
            title += f"\nfixed: {fixed_desc}"
        ax.set_title(title)
        if curve_axis:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

        # Path management
        subdir = self.output_dir / "per_parameter" / f"x_{_sanitize_label(x_axis.display_name)}"
        if fixed_params:
            fixed_str = "_".join(
                f"{_sanitize_label(k)}={_sanitize_label(v)}" for k, v in fixed_params.items()
            )
            subdir = subdir / fixed_str

        subdir.mkdir(parents=True, exist_ok=True)
        curve_label = _sanitize_label(curve_axis.display_name) if curve_axis else "single_axis"
        save_path = subdir / f"{metric}_{direction}_{curve_label}.png"
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def plot_heatmap_spec(self, spec: dict, metrics: list[str], direction: str) -> None:
        x_path, y_path = spec.get("x"), spec.get("y")
        if not x_path or not y_path:
            return
        aggregate_by = str(spec.get("aggregate_by", "avg"))
        render = _normalize_render(spec.get("render"))
        fixed = dict(spec.get("fixed") or {})
        for group_paths in _normalize_groups(spec.get("group_by")):
            for group_filter in self._group_filters(group_paths, fixed):
                for metric in metrics:
                    self.plot_heatmap(
                        metric,
                        direction,
                        x_path,
                        y_path,
                        fixed_params=group_filter,
                        aggregate_by=aggregate_by,
                        render=render,
                    )

    def plot_line_spec(self, spec: dict, metrics: list[str], direction: str) -> None:
        x_path = spec.get("x")
        if not x_path:
            return
        aggregate_by = str(spec.get("aggregate_by", "avg"))
        fixed = dict(spec.get("fixed") or {})
        for group_paths in _normalize_groups(spec.get("group_by")):
            for metric in metrics:
                self._save_lines(
                    metric,
                    direction,
                    x_path,
                    group_paths,
                    fixed,
                    aggregate_by,
                )

    def _save_lines(
        self,
        metric: str,
        direction: str,
        x_path: str,
        group_paths: tuple[str, ...],
        fixed: dict,
        aggregate_by: str,
    ) -> None:
        column_key = column_key_for(metric, direction)
        grouped = {}
        for row in self.table.filter(fixed).rows:
            if x_path not in row.params or column_key not in row.metrics:
                continue
            group = tuple(row.params.get(path) for path in group_paths)
            grouped.setdefault(group, {}).setdefault(row.params[x_path], []).append(
                row.metrics[column_key]
            )
        if not grouped:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        for index, (group, points) in enumerate(
            sorted(grouped.items(), key=lambda item: str(item[0]))
        ):
            xs = sorted(points, key=_sort_key)
            ys = [_aggregate(points[x], aggregate_by) for x in xs]
            ax.plot(
                xs,
                ys,
                marker="o",
                color=_cycle_color(index),
                label=self._line_label(group_paths, group),
            )

        ax.set_xlabel(self._display_name(x_path))
        ax.set_ylabel(f"{metric} ({direction})")
        ax.set_title(f"{metric} ({direction})")
        if group_paths:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

        group_name = "_".join(self._display_name(path) for path in group_paths) or "all"
        output = (
            self.output_dir
            / "line"
            / f"direction={direction}"
            / f"x={_sanitize_label(self._display_name(x_path))}"
            / f"group={_sanitize_label(group_name)}"
            / f"{metric}.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _line_label(self, paths: tuple[str, ...], values: tuple) -> str:
        parts = [
            f"{self._display_name(path)}={value}"
            for path, value in zip(paths, values, strict=True)
            if value is not None
        ]
        return ", ".join(parts) or "all"

    def plot_heatmap(
        self,
        metric: str,
        direction: str,
        x_axis_path: str,
        y_axis_path: str,
        fixed_params: dict | None = None,
        aggregate_by: str = "avg",
        render: list[str] | None = None,
    ):
        """Plot a heatmap for two parameters."""
        fixed_params = fixed_params or {}
        subset = self.table.filter(fixed_params)

        x_vals = self.table.get_unique_values(x_axis_path)
        y_vals = self.table.get_unique_values(y_axis_path)

        z = np.full((len(y_vals), len(x_vals)), np.nan)
        column_key = column_key_for(metric, direction)

        for yi, yv in enumerate(y_vals):
            for xi, xv in enumerate(x_vals):
                values = [
                    row.metrics[column_key]
                    for row in subset.rows
                    if row.params.get(x_axis_path) == xv
                    and row.params.get(y_axis_path) == yv
                    and column_key in row.metrics
                ]
                if values:
                    z[yi, xi] = _aggregate(values, aggregate_by)

        if np.isnan(z).all():
            return

        render = render or ["heatmap"]
        if "heatmap" in render:
            self._save_heatmap(
                z,
                metric,
                direction,
                x_axis_path,
                y_axis_path,
                x_vals,
                y_vals,
                fixed_params,
                aggregate_by,
            )
        if "heatmap3d" in render:
            self._save_heatmap3d(
                z,
                metric,
                direction,
                x_axis_path,
                y_axis_path,
                x_vals,
                y_vals,
                fixed_params,
                aggregate_by,
            )

    def _save_heatmap(
        self,
        z,
        metric,
        direction,
        x_axis_path,
        y_axis_path,
        x_vals,
        y_vals,
        fixed_params,
        aggregate_by,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad(color="0.85")
        im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, label=f"{metric} ({direction})")

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals], rotation=45)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        for yi in range(z.shape[0]):
            for xi in range(z.shape[1]):
                value = z[yi, xi]
                ax.text(
                    xi,
                    yi,
                    "" if np.isnan(value) else f"{value:.3g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.set_xlabel(self._display_name(x_axis_path))
        ax.set_ylabel(self._display_name(y_axis_path))
        ax.set_title(self._heatmap_title(metric, direction, fixed_params, aggregate_by))

        output_path = self._heatmap_path(
            "heatmap", metric, direction, x_axis_path, y_axis_path, fixed_params
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _save_heatmap3d(
        self,
        z,
        metric,
        direction,
        x_axis_path,
        y_axis_path,
        x_vals,
        y_vals,
        fixed_params,
        aggregate_by,
    ) -> None:
        x_grid, y_grid = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(x_grid, y_grid, np.ma.masked_invalid(z), cmap="viridis")
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals], rotation=35, ha="right")
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_xlabel(self._display_name(x_axis_path))
        ax.set_ylabel(self._display_name(y_axis_path))
        ax.set_zlabel(metric)
        ax.set_title(self._heatmap_title(metric, direction, fixed_params, aggregate_by))
        fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1)
        output_path = self._heatmap_path(
            "heatmap3d", metric, direction, x_axis_path, y_axis_path, fixed_params
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _group_filters(self, group_paths: tuple[str, ...], fixed: dict) -> list[dict]:
        if not group_paths:
            return [fixed]
        value_lists = [
            sorted(
                {
                    row.params[path]
                    for row in self.table.rows
                    if path in row.params and _row_matches(row.params, fixed)
                },
                key=_sort_key,
            )
            for path in group_paths
        ]
        return [
            {**fixed, **dict(zip(group_paths, values, strict=True))}
            for values in product(*value_lists)
        ]

    def _display_name(self, path: str) -> str:
        for axis in getattr(self.table, "axes_meta", []):
            if axis.path == path:
                return _axis_label(axis, path)
        return path.rsplit(".", maxsplit=1)[-1]

    def _heatmap_title(self, metric, direction, fixed_params, aggregate_by) -> str:
        title = f"{metric} ({direction})"
        subtitle = [f"extra dims={aggregate_by}"]
        if fixed_params:
            subtitle.append(", ".join(f"{k.split('.')[-1]}={v}" for k, v in fixed_params.items()))
        return f"{title}\n{'; '.join(subtitle)}"

    def _heatmap_path(self, kind, metric, direction, x_path, y_path, fixed_params) -> Path:
        axes = (
            f"axes={_sanitize_label(self._display_name(x_path))}_"
            f"{_sanitize_label(self._display_name(y_path))}"
        )
        group = (
            "__".join(
                f"{_sanitize_label(k.split('.')[-1])}={_sanitize_label(v)}"
                for k, v in fixed_params.items()
            )
            or "all"
        )
        return self.output_dir / kind / f"direction={direction}" / axes / group / f"{metric}.png"
