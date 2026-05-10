from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import optuna
from omegaconf import OmegaConf

from banditdl.utils.metrics import MetricLoader, scalar_reduce


DEFAULT_PLOT_METRICS: tuple[str, ...] = (
    "validation_accuracies",
    "validation_losses",
    "train_losses",
    "reward_algorithm",
    "reward_oracle",
    "regret",
    "normalized_regret",
    "neighbor_disagreement",
    "consensus_drift",
)

DEFAULT_DIRECTIONS: tuple[str, ...] = ("avg", "worse")
DEFAULT_PLOT_MODES: tuple[str, ...] = ("per_parameter", "all_together", "heatmap")

_DIRECTION_ALIASES = {
    "avg": "avg",
    "mean": "avg",
    "average": "avg",
    "worse": "worse",
    "worst": "worse",
}


def normalize_direction(value):
    token = str(value).lower().strip()
    if token not in _DIRECTION_ALIASES:
        raise ValueError(
            f"Unsupported direction '{value}'. Allowed: avg, mean, average, worse, worst."
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
    modes = []
    for item in raw:
        mode = str(item).lower().strip()
        if mode not in DEFAULT_PLOT_MODES:
            raise ValueError(
                f"Unsupported plot_mode '{item}'. Allowed: {', '.join(DEFAULT_PLOT_MODES)}."
            )
        if mode not in modes:
            modes.append(mode)
    return modes or list(DEFAULT_PLOT_MODES)


def _strip_meta(spec):
    """
    Remove Optuna metadata keys from a search-space entry.

    Args:
      spec: Any
        Raw YAML entry or dict.

    return: tuple
      (inner_spec, display_name_or_none)
    """
    if not isinstance(spec, dict):
        return spec, None
    inner = {k: v for k, v in spec.items() if k not in ("when", "name")}
    display_name = spec.get("name")
    if display_name is not None:
        display_name = str(display_name)
    return inner, display_name


def _choices_from_spec(inner_spec):
    """
    Build the ordered choice list for a categorical sweep axis.

    Args:
      inner_spec: Any
        Search-space entry without metadata.

    return: list
      Choice values (may be empty if not categorical).
    """
    if isinstance(inner_spec, list):
        return list(inner_spec)
    if not isinstance(inner_spec, dict):
        return []
    param_type = str(inner_spec.get("type", "")).lower()
    if param_type == "categorical":
        choices = inner_spec.get("choices")
        if isinstance(choices, list):
            return list(choices)
    return []


def _when_clause(spec):
    """
    Extract optional conditional clause from a search-space entry.

    Args:
      spec: Any
        Raw YAML entry.

    return: dict | None
      When-clause mapping or None.
    """
    if isinstance(spec, dict) and "when" in spec:
        when_val = spec.get("when")
        if isinstance(when_val, dict):
            return when_val
    return None


def _conditions_met(cfg, conditions):
    """
    Check whether composed config satisfies all expected values.

    Args:
      cfg: Any
        OmegaConf-compatible config object.
      conditions: dict
        Path -> expected scalar or list of allowed values.

    return: bool
      True when every condition holds.
    """
    if conditions is None:
        return True
    if not isinstance(conditions, dict) or not conditions:
        return False
    for path, expected in conditions.items():
        actual = OmegaConf.select(cfg, path)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def _normalize_search_space(search_space):
    """
    Convert sweep YAML mapping into ordered path keys with cleaned specs.

    Args:
      search_space: dict
        Mapping path -> spec.

    return: tuple
      (ordered_paths, normalized_specs, display_names)
    """
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
    """
    Enumerate every valid Optuna param dict for a categorical sweep with optional when clauses.

    Args:
      base_cfg: Any
        OmegaConf base configuration used to evaluate when clauses.
      search_space: dict
        Full optuna.search_space mapping.

    return: list
      List of dict path -> sampled value (only active params included).
    """
    ordered_paths, normalized_specs, _ = _normalize_search_space(search_space)
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
    """
    Build Optuna GridSampler search_space mapping from categorical YAML entries.

    Args:
      search_space: dict
        Full optuna.search_space mapping.

    return: dict
      Optuna GridSampler-compatible mapping path -> list of values.
    """
    ordered_paths, normalized_specs, _ = _normalize_search_space(search_space)
    grid = {}
    for path in ordered_paths:
        raw_spec = search_space[path]
        inner_spec, _ = _strip_meta(raw_spec)
        choices = _choices_from_spec(inner_spec)
        if choices:
            grid[path] = choices
    return grid


def trial_folder_name(params, axis_meta):
    """
    Build a filesystem-safe folder label from trial params.

    Args:
      params: dict
        Flat Optuna param mapping path -> value.
      axis_meta: dict
        Mapping path -> dict with key display_name.

    return: str
      Folder name segments joined by underscores.
    """
    parts = []
    for path in sorted(params.keys()):
        label = axis_meta.get(path, {}).get("display_name", path.split(".")[-1])
        val = params[path]
        token = _sanitize_token(val)
        parts.append(f"{label}={token}")
    return "_".join(parts)


def _sanitize_token(value):
    """
    Convert a sweep value into a short filesystem-safe token.

    Args:
      value: Any
        Parameter value.

    return: str
      Safe token string.
    """
    text = str(value)
    allowed = string.ascii_letters + string.digits + ".-+"
    cleaned = "".join(ch if ch in allowed else "_" for ch in text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "value"
    return cleaned


def scalar_for_direction(metric_name, array_values, direction):
    """
    Reduce a metric array to a single scalar according to a direction.

    avg averages over all timesteps and nodes. worse takes the worst value
    over timesteps and nodes, where worst means max for metrics in
    HIGHER_IS_WORSE and min otherwise.

    Args:
      metric_name: str
        Metric stem matching plotting conventions.
      array_values: ndarray
        Loaded numpy array (any shape).
      direction: str
        Canonical direction name (avg or worse).

    return: float
      Reduced scalar value.
    """
    return scalar_reduce(metric_name, array_values, direction)


def column_key_for(metric_name, direction):
    """
    Build the SweepRow.metrics column key for a metric/direction pair.

    Args:
      metric_name: str
        Metric stem.
      direction: str
        Canonical direction name.

    return: str
      Column key string.
    """
    return f"{metric_name}__{direction}"


def build_axis_metadata(search_space):
    """
    Collect sweep-axis metadata for plotting labels and ordering.

    Args:
      search_space: dict
        Full optuna.search_space mapping.

    return: tuple
      (axes_list, axis_meta_lookup)
    """
    ordered_paths, _, display_names = _normalize_search_space(search_space)
    axes = []
    meta = {}
    for path in ordered_paths:
        raw_spec = search_space[path]
        inner_spec, alias = _strip_meta(raw_spec)
        choices = _choices_from_spec(inner_spec)
        if not choices:
            continue
        display = alias if alias else path.split(".")[-1]
        when_clause = _when_clause(raw_spec)
        axes.append(
            SweepAxis(
                path=path,
                display_name=display,
                values=list(choices),
                when=when_clause,
                is_conditional=when_clause is not None,
            )
        )
        meta[path] = {"display_name": display}
    return axes, meta


@dataclass
class SweepAxis:
    """One categorical sweep dimension."""

    path: str
    display_name: str
    values: list[Any]
    when: dict | None
    is_conditional: bool


@dataclass
class SweepRow:
    """Scalar metrics for one trial."""

    params: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)


class SweepTable:
    """Tabular collection of metric scalars keyed by trial params."""

    def __init__(self, rows):
        """
        Initialize sweep table.

        Args:
          rows: list
            Sequence of SweepRow instances.
        """
        self.rows = rows

    def filtered_rows(self, axis_paths):
        """
        Restrict rows to axes whose when clauses are satisfied jointly.

        Args:
          axis_paths: list
            Paths participating in the current plot.

        return: list
          Filtered SweepRow objects.
        """
        paths_set = set(axis_paths)
        kept = []
        for row in self.rows:
            trial_cfg = OmegaConf.create({})
            for key, value in row.params.items():
                OmegaConf.update(trial_cfg, key, value, merge=False)
            ok = True
            for path in paths_set:
                spec = self._spec_for_path(path)
                if spec is None:
                    continue
                when_clause = _when_clause(spec)
                if when_clause is None:
                    continue
                if not _conditions_met(trial_cfg, when_clause):
                    ok = False
                    break
            if ok:
                kept.append(row)
        return kept

    def attach_spec(self, search_space):
        """
        Attach search_space for filtering lookups.

        Args:
          search_space: dict
            Original sweep YAML mapping.
        """
        self._search_space = search_space

    def _spec_for_path(self, path):
        if not hasattr(self, "_search_space"):
            return None
        return self._search_space.get(path)


def sweep_table_from_study(trials_root, study, search_space, metric_names, directions):
    """
    Build a SweepTable by reading numpy artifacts for completed trials and
    reducing each metric to one scalar per requested direction.

    Args:
      trials_root: Path
        Directory containing per-trial folders.
      study: optuna.study.Study
        Finished Optuna study.
      search_space: dict
        Full sweep mapping for metadata attachment.
      metric_names: list
        Metric stems to load from each trial results folder.
      directions: list
        Canonical direction names (avg, worse) to compute per metric.

    return: SweepTable
      Populated table instance.
    """
    def trial_params_for_table(trial):
        if trial.params:
            return dict(trial.params)
        resolved = trial.user_attrs.get("resolved_params")
        if isinstance(resolved, dict):
            return dict(resolved)
        return {}

    axes, axis_meta = build_axis_metadata(search_space)
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_params = trial_params_for_table(trial)
        if not trial_params:
            continue
        folder = trial_folder_name(trial_params, {p: axis_meta.get(p, {}) for p in trial_params})
        result_dir = trials_root / folder / "results"
        metrics_flat = {}
        for metric in metric_names:
            try:
                raw = MetricLoader(result_dir).load_values(metric)
            except (FileNotFoundError, ValueError, OSError) as exc:
                print(f"[plot_sweep] WARNING: skip metric '{metric}' for trial {result_dir}: {exc}")
                continue
            for direction in directions:
                metrics_flat[column_key_for(metric, direction)] = scalar_for_direction(metric, raw, direction)
        row = SweepRow(params=trial_params, metrics=metrics_flat)
        rows.append(row)
    table = SweepTable(rows)
    table.attach_spec(search_space)
    table.axes_meta = axes
    table.axis_labels = {ax.path: ax.display_name for ax in axes}
    return table


class BaseSweepPlotter(ABC):
    """Shared plotting utilities for sweep visualizations."""

    def __init__(self, table, axes, output_dir):
        """
        Initialize plotter.

        Args:
          table: SweepTable
            Metric table for all trials.
          axes: list
            SweepAxis objects in sweep order.
          output_dir: Path
            Root directory for PNG outputs.
        """
        self.table = table
        self.axes = axes
        self.output_dir = output_dir

    def plot(self, metric_names, direction):
        """
        Generate all plots for requested metrics under a single direction.

        Args:
          metric_names: list
            Metric stems (numpy file stems).
          direction: str
            Canonical direction name (avg or worse) used as the scalar
            reduction and as column-key/file-suffix tag.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for metric in metric_names:
            column = column_key_for(metric, direction)
            self._plot_metric_column(metric, direction, column)

    @abstractmethod
    def _plot_metric_column(self, metric, direction, column_key):
        """
        Plot all figures for one metric scalar column.

        Args:
          metric: str
            Metric stem.
          direction: str
            Canonical direction name (avg or worse).
          column_key: str
            Column name in SweepRow.metrics.
        """

    def _axis_by_path(self):
        lookup = {}
        for axis in self.axes:
            lookup[axis.path] = axis
        return lookup

    def _value_for_row(self, row, path):
        if path not in row.params:
            return None
        return row.params[path]

    def auto_subfolder_name(self, parts):
        """
        Join labeled path fragments into a filesystem-safe folder name.

        Args:
          parts: list
            Tuple-like entries (label, value_token).

        return: str
          Folder name.
        """
        segments = []
        for label, value in parts:
            segments.append(f"{_sanitize_token(label)}={_sanitize_token(value)}")
        return "__".join(segments)


def plot_sweep(
    plot_modes,
    directions,
    trials_root,
    study,
    search_space,
    metric_names,
    output_dir,
):
    """
    Dispatch configured sweep plots.

    For each default plot mode and each default direction the plots are
    written under output_dir/<mode>/direction=<direction>/...

    Args:
      trials_root: Path
        Trial artifact root (contains per-trial folders).
      study: optuna.study.Study
        Completed study with trials.
      search_space: dict
        Original sweep mapping from YAML.
      output_dir: Path
        Destination root folder for PNG outputs.
    """
    resolved_modes = normalize_plot_modes(plot_modes)
    resolved_directions = normalize_directions(directions)
    resolved_metrics = list(metric_names or DEFAULT_PLOT_METRICS)

    table = sweep_table_from_study(
        trials_root, study, search_space, resolved_metrics, resolved_directions
    )
    axes, _ = build_axis_metadata(search_space)

    for mode in resolved_modes:
        for direction in resolved_directions:
            mode_dir = output_dir / mode / f"direction={direction}"
            plotter = make_sweep_plotter(mode, table, axes, mode_dir)
            plotter.plot(resolved_metrics, direction)


def make_sweep_plotter(mode, table, axes, output_dir):
    """
    Instantiate a sweep plotter for the requested mode.

    Args:
      mode: str
        Plot mode key.
      table: SweepTable
        Prepared sweep table.
      axes: list
        SweepAxis descriptors.
      output_dir: Path
        Output directory.

    return: BaseSweepPlotter
      Concrete plotter instance.
    """
    from banditdl.utils.plot_sweep_alltogether import AllTogetherPlotter
    from banditdl.utils.plot_sweep_heatmap import HeatmapPlotter
    from banditdl.utils.plot_sweep_perparam import PerParamPlotter

    normalized = str(mode).lower()
    if normalized == "per_parameter":
        return PerParamPlotter(table, axes, output_dir)
    if normalized == "all_together":
        return AllTogetherPlotter(table, axes, output_dir)
    if normalized == "heatmap":
        return HeatmapPlotter(table, axes, output_dir)
    raise ValueError(f"Unsupported plot_mode: {mode}")
