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
        groups = _normalize_groups(spec.get("group_by"))
        for group_paths in groups:
            for group_values in self._group_values(group_paths):
                group_filter = dict(zip(group_paths, group_values, strict=True))
                group_label = self._group_label(group_filter)
                for metric in metric_names:
                    column = column_key_for(metric, direction)
                    matrix = self._matrix(x_axis, y_axis, group_filter, column, aggregate_by)
                    if np.all(np.isnan(matrix)):
                        continue
                    self._save(matrix, x_axis, y_axis, metric, direction, aggregate_by, group_label)

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
            values.append(sorted({row.params[path] for row in self.table.rows if path in row.params}, key=_sort_key))
        return [combo for combo in product(*values) if self._has_rows(dict(zip(group_paths, combo, strict=True)))]

    def _has_rows(self, filters: dict) -> bool:
        return any(_row_matches(row.params, filters) for row in self.table.rows)

    def _matrix(self, x_axis, y_axis, filters: dict, column: str, aggregate_by: str) -> np.ndarray:
        matrix = np.full((len(y_axis.values), len(x_axis.values)), np.nan, dtype=float)
        for yi, y_val in enumerate(y_axis.values):
            for xi, x_val in enumerate(x_axis.values):
                cell_filter = {**filters, x_axis.path: x_val, y_axis.path: y_val}
                values = [row.metrics[column] for row in self.table.rows if column in row.metrics and _row_matches(row.params, cell_filter)]
                if values:
                    matrix[yi, xi] = _aggregate(values, aggregate_by)
        return matrix

    def _save(self, matrix, x_axis, y_axis, metric, direction, aggregate_by, group_label) -> None:
        subdir = self.output_dir / f"axes={_sanitize_label(x_axis.display_name)}_{_sanitize_label(y_axis.display_name)}"
        if group_label:
            subdir = subdir / group_label
        else:
            subdir = subdir / "all"
        subdir.mkdir(parents=True, exist_ok=True)
        outfile = subdir / f"{metric}__{direction}.png"

        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad(color="0.85")
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(x_axis.values)))
        ax.set_xticklabels([str(v) for v in x_axis.values], rotation=35, ha="right")
        ax.set_yticks(range(len(y_axis.values)))
        ax.set_yticklabels([str(v) for v in y_axis.values])
        ax.set_xlabel(x_axis.display_name)
        ax.set_ylabel(y_axis.display_name)
        title = f"{metric} ({direction})"
        caption = f"x={x_axis.display_name}, y={y_axis.display_name}, extra dims={aggregate_by}"
        if group_label:
            caption += f", {group_label.replace('__', ', ')}"
        ax.set_title(f"{title}\n{caption}")
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                val = matrix[yi, xi]
                ax.text(xi, yi, "" if np.isnan(val) else f"{val:.3g}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outfile, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _group_label(filters: dict) -> str:
        if not filters:
            return ""
        return "__".join(f"{_sanitize_label(path.split('.')[-1])}={_sanitize_label(value)}" for path, value in filters.items())


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


def _sanitize_label(text) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ".-+" else "_" for ch in str(text))
    return cleaned.strip("_") or "value"


def _sort_key(value):
    if isinstance(value, (int, float, np.floating)):
        return (0, float(value))
    return (1, str(value))
