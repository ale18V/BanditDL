from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from banditdl.utils.plot_sweep_base import BaseSweepPlotter, column_key_for


class ExplicitHeatmapPlotter(BaseSweepPlotter):
    """Heatmaps declared explicitly by x/y axes and optional grouping slices."""

    def plot_spec(self, spec: dict, metric_names: Sequence[str], direction: str) -> None:
        x_axis = self._axis(str(spec["x"]))
        y_axis = self._axis(str(spec["y"]))
        aggregate_by = str(spec.get("aggregate_by", "avg"))
        render = _normalize_render(spec.get("render"))
        groups = _normalize_groups(spec.get("group_by"))

        group_filters = []
        for group_paths in groups:
            for group_values in self._group_values(group_paths):
                group_filters.append(dict(zip(group_paths, group_values, strict=True)))

        for metric in metric_names:
            column = column_key_for(metric, direction)
            matrices = []
            for group_filter in group_filters:
                matrix = self._matrix(x_axis, y_axis, group_filter, column, aggregate_by)
                if not np.all(np.isnan(matrix)):
                    matrices.append((group_filter, matrix))
            if not matrices:
                continue
            vmin, vmax = _shared_limits([matrix for _, matrix in matrices])
            for group_filter, matrix in matrices:
                group_label = self._group_label(group_filter)
                if "heatmap" in render:
                    self._save_heatmap(
                        matrix, x_axis, y_axis, metric, direction, aggregate_by, group_label, vmin, vmax
                    )
                if "heatmap3d" in render:
                    self._save_heatmap3d(
                        matrix, x_axis, y_axis, metric, direction, aggregate_by, group_label, vmin, vmax
                    )

    def _plot_metric_column(self, metric, direction, column_key):
        raise NotImplementedError("Use plot_spec() for explicit heatmaps")

    def _axis(self, path: str):
        for axis in self.axes:
            if axis.path == path:
                return axis
        raise ValueError(f"Unknown heatmap axis: {path}")

    def _group_values(self, group_paths: tuple[str, ...]) -> list[tuple]:
        if not group_paths:
            return [()]
        values = []
        for path in group_paths:
            values.append(
                sorted(
                    {row.params[path] for row in self.table.rows if path in row.params},
                    key=_sort_key,
                )
            )
        return [
            combo
            for combo in product(*values)
            if self._has_rows(dict(zip(group_paths, combo, strict=True)))
        ]

    def _has_rows(self, filters: dict) -> bool:
        return any(_row_matches(row.params, filters) for row in self.table.rows)

    def _matrix(self, x_axis, y_axis, filters: dict, column: str, aggregate_by: str) -> np.ndarray:
        matrix = np.full((len(y_axis.values), len(x_axis.values)), np.nan, dtype=float)
        for yi, y_val in enumerate(y_axis.values):
            for xi, x_val in enumerate(x_axis.values):
                cell_filter = {**filters, x_axis.path: x_val, y_axis.path: y_val}
                values = [
                    row.metrics[column]
                    for row in self.table.rows
                    if column in row.metrics and _row_matches(row.params, cell_filter)
                ]
                if values:
                    matrix[yi, xi] = _aggregate(values, aggregate_by)
        return matrix

    def _save_heatmap(
        self, matrix, x_axis, y_axis, metric, direction, aggregate_by, group_label, vmin, vmax
    ) -> None:
        subdir = self._subdir(self.output_dir, x_axis, y_axis, group_label)
        outfile = subdir / f"{metric}__{direction}.png"

        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad(color="0.85")
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        self._decorate_axes(ax, x_axis, y_axis, metric, direction, aggregate_by, group_label)
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                val = matrix[yi, xi]
                ax.text(
                    xi,
                    yi,
                    "" if np.isnan(val) else f"{val:.3g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outfile, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _save_heatmap3d(
        self, matrix, x_axis, y_axis, metric, direction, aggregate_by, group_label, vmin, vmax
    ) -> None:
        output_dir = self.output_dir.parent.parent / "heatmap3d" / self.output_dir.name
        subdir = self._subdir(output_dir, x_axis, y_axis, group_label)
        outfile = subdir / f"{metric}__{direction}.png"

        x_grid, y_grid = np.meshgrid(np.arange(len(x_axis.values)), np.arange(len(y_axis.values)))
        z = np.ma.masked_invalid(matrix)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(
            x_grid,
            y_grid,
            z,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=True,
        )
        ax.set_xticks(range(len(x_axis.values)))
        ax.set_xticklabels([str(v) for v in x_axis.values], rotation=35, ha="right")
        ax.set_yticks(range(len(y_axis.values)))
        ax.set_yticklabels([str(v) for v in y_axis.values])
        ax.set_xlabel(x_axis.display_name)
        ax.set_ylabel(y_axis.display_name)
        ax.set_zlabel(metric)
        ax.set_title(self._title(metric, direction, aggregate_by, group_label))
        fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1)
        fig.savefig(outfile, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _subdir(self, root, x_axis, y_axis, group_label):
        subdir = root / f"axes={_sanitize_label(x_axis.display_name)}_{_sanitize_label(y_axis.display_name)}"
        subdir = subdir / (group_label or "all")
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def _decorate_axes(self, ax, x_axis, y_axis, metric, direction, aggregate_by, group_label) -> None:
        ax.set_xticks(range(len(x_axis.values)))
        ax.set_xticklabels([str(v) for v in x_axis.values], rotation=35, ha="right")
        ax.set_yticks(range(len(y_axis.values)))
        ax.set_yticklabels([str(v) for v in y_axis.values])
        ax.set_xlabel(x_axis.display_name)
        ax.set_ylabel(y_axis.display_name)
        ax.set_title(self._title(metric, direction, aggregate_by, group_label))

    @staticmethod
    def _title(metric, direction, aggregate_by, group_label) -> str:
        caption = f"extra dims={aggregate_by}"
        if group_label:
            caption += f", {group_label.replace('__', ', ')}"
        return f"{metric} ({direction})\n{caption}"

    @staticmethod
    def _group_label(filters: dict) -> str:
        if not filters:
            return ""
        return "__".join(
            f"{_sanitize_label(path.split('.')[-1])}={_sanitize_label(value)}"
            for path, value in filters.items()
        )


class HeatmapPlotter(ExplicitHeatmapPlotter):
    """Compatibility name; automatic all-pairs heatmaps are intentionally disabled."""


def _normalize_groups(raw) -> list[tuple[str, ...]]:
    if raw is None:
        return [()]
    groups = raw if isinstance(raw, list) else [raw]
    normalized = []
    for group in groups:
        if isinstance(group, str):
            normalized.append((group,))
        elif isinstance(group, list):
            normalized.append(tuple(str(item) for item in group))
        else:
            raise ValueError(f"Invalid heatmap group_by entry: {group!r}")
    return normalized or [()]


def _normalize_render(raw) -> list[str]:
    if raw is None:
        return ["heatmap"]
    render = raw if isinstance(raw, list) else [raw]
    normalized = []
    for item in render:
        item = str(item)
        if item not in {"heatmap", "heatmap3d"}:
            raise ValueError("render entries must be 'heatmap' or 'heatmap3d'")
        if item not in normalized:
            normalized.append(item)
    return normalized or ["heatmap"]


def _row_matches(params: dict, filters: dict) -> bool:
    return all(path in params and params[path] == value for path, value in filters.items())


def _aggregate(values: list[float], mode: str) -> float:
    arr = np.asarray(values, dtype=float)
    if mode == "avg":
        return float(np.nanmean(arr))
    if mode == "min":
        return float(np.nanmin(arr))
    if mode == "max":
        return float(np.nanmax(arr))
    raise ValueError("aggregate_by must be one of: avg, min, max")


def _shared_limits(matrices: Sequence[np.ndarray]) -> tuple[float | None, float | None]:
    finite_values = [matrix[np.isfinite(matrix)] for matrix in matrices]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        return None, None
    values = np.concatenate(finite_values)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        delta = 1.0 if vmin == 0 else abs(vmin) * 0.05
        return vmin - delta, vmax + delta
    return vmin, vmax


def _sanitize_label(text) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ".-+" else "_" for ch in str(text))
    return cleaned.strip("_") or "value"


def _sort_key(value):
    if isinstance(value, (int, float, np.floating)):
        return (0, float(value))
    return (1, str(value))
