from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from omegaconf import OmegaConf

from banditdl.utils.experiment_table import ExperimentTable, SweepRow
from banditdl.utils.metrics import MetricLoader, scalar_reduce_seed_outer

DEFAULT_PLOT_METRICS: tuple[str, ...] = (
    "validation_accuracy",
    "validation_loss",
    "global_accuracy",
    "train_loss",
    "reward_algorithm",
    "reward_oracle",
    "regret",
    "normalized_regret",
    "neighbor_disagreement",
    "consensus_drift",
    "gradient_norms",
    "sampler_kl_to_uniform",
    "sampler_entropy",
    "sampler_min_probability",
    "sampler_max_probability",
    "sampler_weight_kl_to_uniform",
    "sampler_weight_entropy",
    "sampler_min_weight",
    "sampler_max_weight",
)

DEFAULT_DIRECTIONS: tuple[str, ...] = ("avg", "worse", "best")
DEFAULT_PLOT_MODES: tuple[str, ...] = ("per_parameter", "heatmap")
STUDY_NAME = "sweep"
OPTUNA_DB_NAME = "optuna.db"

_DIRECTION_ALIASES = {
    "avg": "avg",
    "mean": "avg",
    "average": "avg",
    "worse": "worse",
    "worst": "worse",
    "best": "best",
}


def normalize_direction(value):
    token = str(value).lower().strip()
    if token not in _DIRECTION_ALIASES:
        raise ValueError(
            f"Unsupported direction '{value}'. Allowed: avg, mean, average, worse, worst, best."
        )
    return _DIRECTION_ALIASES[token]


def normalize_directions(value):
    if value is None:
        return list(DEFAULT_DIRECTIONS)
    raw = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raw = [raw]
    directions = []
    for item in raw:
        direction = normalize_direction(item)
        if direction not in directions:
            directions.append(direction)
    return directions or list(DEFAULT_DIRECTIONS)


def normalize_plot_modes(value):
    if value is None:
        return list(DEFAULT_PLOT_MODES)
    raw = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raw = [raw]
    valid = set(DEFAULT_PLOT_MODES) | {"all_together"}
    modes = []
    for item in raw:
        mode = str(item).lower().strip()
        if mode not in valid:
            raise ValueError(f"Unsupported plot_mode '{item}'. Allowed: {', '.join(sorted(valid))}.")
        if mode not in modes:
            modes.append(mode)
    return modes or list(DEFAULT_PLOT_MODES)


def _strip_meta(spec):
    if not isinstance(spec, dict):
        return spec, None
    inner = {k: v for k, v in spec.items() if k not in ("when", "name")}
    display_name = spec.get("name")
    return inner, str(display_name) if display_name is not None else None


def _choices_from_spec(inner_spec):
    if isinstance(inner_spec, list):
        return list(inner_spec)
    if not isinstance(inner_spec, dict):
        return []
    if str(inner_spec.get("type", "")).lower() != "categorical":
        return []
    choices = inner_spec.get("choices")
    return list(choices) if isinstance(choices, list) else []


def _when_clause(spec):
    if isinstance(spec, dict) and isinstance(spec.get("when"), dict):
        return spec["when"]
    return None


def _conditions_met(cfg, conditions):
    if conditions is None:
        return True
    if not isinstance(conditions, dict) or not conditions:
        return False
    for path, expected in conditions.items():
        actual = OmegaConf.select(cfg, path)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _normalize_search_space(search_space):
    paths = sorted(search_space.keys())
    normalized = {}
    display_names = {}
    for path in paths:
        raw = search_space[path]
        inner, display_name = _strip_meta(raw)
        normalized[path] = inner if inner is not None else raw
        if display_name:
            display_names[path] = display_name
    return paths, normalized, display_names


def enumerate_valid_param_dicts(base_cfg, search_space):
    ordered_paths, _, _ = _normalize_search_space(search_space)
    outcomes = []

    def walk(acc_params, idx):
        if idx >= len(ordered_paths):
            outcomes.append(dict(acc_params))
            return
        path = ordered_paths[idx]
        raw_spec = search_space[path]
        inner_spec, _ = _strip_meta(raw_spec)
        when_clause = _when_clause(raw_spec)
        trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        for key, value in acc_params.items():
            OmegaConf.update(trial_cfg, key, value, merge=False)
        if when_clause is not None and not _conditions_met(trial_cfg, when_clause):
            walk(acc_params, idx + 1)
            return
        choices = _choices_from_spec(inner_spec)
        if not choices:
            walk(acc_params, idx + 1)
            return
        for val in choices:
            next_params = dict(acc_params)
            next_params[path] = val
            walk(next_params, idx + 1)

    walk({}, 0)
    return outcomes


def build_grid_search_space(search_space):
    ordered_paths, _, _ = _normalize_search_space(search_space)
    grid = {}
    for path in ordered_paths:
        inner_spec, _ = _strip_meta(search_space[path])
        choices = _choices_from_spec(inner_spec)
        if choices:
            grid[path] = choices
    return grid


def trial_folder_name(params, axis_meta):
    parts = []
    for path in sorted(params.keys()):
        label = axis_meta.get(path, {}).get("display_name", path.split(".")[-1])
        parts.append(f"{label}={_sanitize_token(params[path])}")
    return "_".join(parts)


def _sanitize_token(value):
    text = str(value)
    allowed = string.ascii_letters + string.digits + ".-+"
    cleaned = "".join(ch if ch in allowed else "_" for ch in text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "value"


def _sort_key(value):
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


def scalar_for_direction(metric_name, array_values, direction):
    return scalar_reduce_seed_outer(metric_name, array_values, direction)


def column_key_for(metric_name, direction):
    return f"{metric_name}__{direction}"


def build_axis_metadata(search_space):
    ordered_paths, _, _display_names = _normalize_search_space(search_space)
    axes = []
    meta = {}
    for path in ordered_paths:
        raw_spec = search_space[path]
        inner_spec, alias = _strip_meta(raw_spec)
        choices = _choices_from_spec(inner_spec)
        display = alias if alias else path.split(".")[-1]
        axes.append(
            SweepAxis(
                path=path,
                display_name=display,
                values=list(choices),
                when=_when_clause(raw_spec),
                is_conditional=_when_clause(raw_spec) is not None,
            )
        )
        meta[path] = {"display_name": display}
    return axes, meta


def _axis_values_from_rows(axis: SweepAxis, rows: list[SweepRow]) -> SweepAxis:
    if axis.values:
        return axis
    values = sorted({row.params[axis.path] for row in rows if axis.path in row.params}, key=_sort_key)
    return SweepAxis(
        path=axis.path,
        display_name=axis.display_name,
        values=values,
        when=axis.when,
        is_conditional=axis.is_conditional,
    )


@dataclass(frozen=True)
class SweepAxis:
    path: str
    display_name: str
    values: list[Any]
    when: dict | None
    is_conditional: bool


def trial_params_for_table(trial):
    if trial.params:
        return dict(trial.params)
    resolved = trial.user_attrs.get("resolved_params")
    return dict(resolved) if isinstance(resolved, dict) else {}


def trial_result_dir(trials_root: Path, trial, trial_params, axis_meta) -> Path:
    result_path = trial.user_attrs.get("result_path")
    if isinstance(result_path, str) and result_path:
        path = Path(result_path)
        return path if path.is_absolute() else trials_root.parent / path
    result_dir = trial.user_attrs.get("result_dir")
    if isinstance(result_dir, str) and result_dir:
        return Path(result_dir)
    folder = trial_folder_name(trial_params, {p: axis_meta.get(p, {}) for p in trial_params})
    return trials_root / folder / "results"


def sweep_table_from_study(trials_root, study, search_space, metric_names, directions) -> ExperimentTable:
    axes, axis_meta = build_axis_metadata(search_space)
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_params = trial_params_for_table(trial)
        if not trial_params:
            continue
        result_dir = trial_result_dir(Path(trials_root), trial, trial_params, axis_meta)
        metrics_flat = {}
        for metric in metric_names:
            try:
                raw = MetricLoader(result_dir).load_seed_values(metric)
            except (FileNotFoundError, ValueError, OSError) as exc:
                print(f"[plot_sweep] WARNING: skip metric '{metric}' for trial {result_dir}: {exc}")
                continue
            for direction in directions:
                metrics_flat[column_key_for(metric, direction)] = scalar_for_direction(metric, raw, direction)
        rows.append(SweepRow(params=trial_params, metrics=metrics_flat))
    table = ExperimentTable(rows)
    table.axes_meta = [_axis_values_from_rows(axis, rows) for axis in axes]
    table.axis_labels = {ax.path: ax.display_name for ax in table.axes_meta}
    return table


def optuna_storage_url(output_root: Path) -> str:
    return f"sqlite:///{output_root / OPTUNA_DB_NAME}"


def load_sweep_study(output_root: Path):
    return optuna.load_study(study_name=STUDY_NAME, storage=optuna_storage_url(output_root))


def plot_config_from_cfg(cfg):
    plot_cfg = cfg.get("plot")
    if plot_cfg is not None:
        return OmegaConf.to_container(plot_cfg, resolve=True)
    # Legacy shape from older sweep.yaml revisions.
    return {
        "enabled": True,
        "directions": cfg.get("direction"),
        "modes": cfg.get("plot_mode"),
        "metrics": cfg.get("plot_metrics"),
        "heatmaps": [],
        "per_parameter": {"enabled": "per_parameter" in normalize_plot_modes(cfg.get("plot_mode"))},
    }


def metrics_for_plot(plot_cfg: dict, exclude=()) -> list[str]:
    metrics = list(plot_cfg.get("metrics") or DEFAULT_PLOT_METRICS)
    excluded = set(exclude or ())
    return [metric for metric in metrics if metric not in excluded]


def plot_sweep_from_cfg(output_root: Path, cfg, study=None, output_dir: Path | None = None) -> None:
    if "optuna" not in cfg or "search_space" not in cfg.optuna:
        raise ValueError("Missing optuna.search_space in sweep config")
    search_space = OmegaConf.to_container(cfg.optuna.search_space, resolve=True)
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping")
    if study is None:
        study = load_sweep_study(output_root)
    plot_cfg = plot_config_from_cfg(cfg)
    if not bool(plot_cfg.get("enabled", True)):
        return
    plot_sweep(
        plot_cfg=plot_cfg,
        trials_root=output_root / "trials",
        study=study,
        search_space=search_space,
        output_dir=output_dir or output_root / "sweep_artifacts",
    )


def plot_sweep(plot_cfg, trials_root, study, search_space, output_dir):
    directions = normalize_directions(plot_cfg.get("directions"))
    all_metrics = metrics_for_plot(plot_cfg)
    table = sweep_table_from_study(trials_root, study, search_space, all_metrics, directions)

    from banditdl.utils.sweep_plotting import SweepPlotter
    plotter = SweepPlotter(table, output_dir)

    for direction in directions:
        # Per parameter plots
        per_param_cfg = plot_cfg.get("per_parameter") or {}
        if bool(per_param_cfg.get("enabled", False)):
            metrics = metrics_for_plot(plot_cfg, per_param_cfg.get("exclude_metrics"))
            for metric in metrics:
                plotter.plot_per_parameter(metric, direction)

        # Heatmap plots
        heatmap_specs = list(plot_cfg.get("heatmaps") or [])
        for spec in heatmap_specs:
            metrics = metrics_for_plot(plot_cfg, spec.get("exclude_metrics"))
            x_path = spec.get("x")
            y_path = spec.get("y")
            fixed = spec.get("fixed") or {}
            if x_path and y_path:
                for metric in metrics:
                    plotter.plot_heatmap(metric, direction, x_path, y_path, fixed)
