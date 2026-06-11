from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def metric_column(metric: str, direction: str) -> str:
    return f"{metric}__{direction}"


def sanitize_label(value) -> str:
    text = "".join(char if char.isalnum() else "_" for char in str(value))
    return text.strip("_") or "axis"


def sort_key(value):
    if isinstance(value, (int, float, np.floating)):
        return (0, float(value))
    return (1, str(value))


def aggregate(values, mode: str) -> float:
    values = np.asarray(values, dtype=float)
    reducers = {
        "avg": np.nanmean,
        "mean": np.nanmean,
        "average": np.nanmean,
        "min": np.nanmin,
        "max": np.nanmax,
    }
    try:
        return float(reducers[mode](values))
    except KeyError:
        raise ValueError("aggregate_by must be one of: avg, mean, average, min, max") from None


def normalize_groups(raw) -> list[tuple[str, ...]]:
    groups = raw if isinstance(raw, list) else [raw] if raw is not None else []
    normalized = []
    for group in groups:
        if isinstance(group, str):
            normalized.append((group,))
        elif isinstance(group, list):
            normalized.append(tuple(map(str, group)))
        else:
            raise ValueError(f"invalid group_by entry: {group!r}")
    return normalized or [()]


def normalize_render(raw) -> list[str]:
    values = raw if isinstance(raw, list) else [raw or "heatmap"]
    render = list(dict.fromkeys(map(str, values)))
    if set(render) - {"heatmap", "heatmap3d"}:
        raise ValueError("render entries must be heatmap or heatmap3d")
    return render


def display_name(table, path: str) -> str:
    for axis in getattr(table, "axes_meta", []):
        if axis.path == path:
            return getattr(axis, "display_name", path.rsplit(".", 1)[-1])
    return path.rsplit(".", 1)[-1]


def normalize_axis(raw) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list) and raw:
        return tuple(map(str, raw))
    raise ValueError(f"plot axis must be a parameter path or non-empty list: {raw!r}")


def axis_value(params: dict, paths: tuple[str, ...]):
    values = tuple(params.get(path) for path in paths)
    return values[0] if len(values) == 1 else values


def axis_label(value) -> str:
    if not isinstance(value, tuple):
        return str(value)
    return "-".join(str(item) for item in value if item is not None)


def axis_name(table, paths: tuple[str, ...]) -> str:
    return " / ".join(display_name(table, path) for path in paths)


def axis_path(table, paths: tuple[str, ...]) -> str:
    return "_".join(sanitize_label(display_name(table, path)) for path in paths)


def expand_fixed(fixed: dict) -> list[dict]:
    keys = list(fixed)
    values = [value if isinstance(value, list) else [value] for value in fixed.values()]
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def group_filters(rows, paths: tuple[str, ...], fixed: dict) -> list[dict]:
    if not paths:
        return [fixed]
    values = [
        sorted(
            {
                row.params[path]
                for row in rows
                if path in row.params
                and all(row.params.get(key) == value for key, value in fixed.items())
            },
            key=sort_key,
        )
        for path in paths
    ]
    return [{**fixed, **dict(zip(paths, items, strict=True))} for items in product(*values)]


def group_label(table, paths: tuple[str, ...], values: tuple) -> str:
    parts = [
        f"{display_name(table, path)}={value}"
        for path, value in zip(paths, values, strict=True)
        if value is not None
    ]
    return ", ".join(parts) or "all"


def filter_label(table, fixed: dict) -> str:
    return ", ".join(f"{display_name(table, path)}={value}" for path, value in fixed.items())


def filter_path(fixed: dict) -> str:
    return (
        "__".join(
            f"{sanitize_label(path.rsplit('.', 1)[-1])}={sanitize_label(value)}"
            for path, value in fixed.items()
        )
        or "all"
    )


def cycle_color(index: int):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[index % len(colors)]


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
