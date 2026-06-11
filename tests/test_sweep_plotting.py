from pathlib import Path

import numpy as np
import optuna

from banditdl.utils.experiment_table import ExperimentTable, SweepRow
from banditdl.utils.metrics import scalar_reduce_seed_outer
from banditdl.utils.plot_sweep_base import (
    DEFAULT_PLOT_METRICS,
    STUDY_NAME,
    load_sweep_study,
    metrics_for_plot,
    normalize_direction,
    optuna_storage_url,
)
from banditdl.utils.sweep_plotting import SweepPlotter


def test_heatmap_renders_file(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 10, "g": "a"}, {"metric__avg": 1.0}),
        SweepRow({"x": 2, "y": 20, "g": "a"}, {"metric__avg": 5.0}),
    ]
    table = ExperimentTable(rows)
    plotter = SweepPlotter(table, tmp_path)

    plotter.plot_heatmap("metric", "avg", "x", "y", {"g": "a"})

    assert (tmp_path / "heatmap" / "direction=avg" / "axes=x_y" / "g=a" / "metric.png").exists()


def test_heatmap_spec_groups_and_aggregates_free_dimensions(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 10, "g": "a", "free": 0}, {"metric__avg": 1.0}),
        SweepRow({"x": 1, "y": 10, "g": "a", "free": 1}, {"metric__avg": 3.0}),
        SweepRow({"x": 2, "y": 10, "g": "a", "free": 0}, {"metric__avg": 5.0}),
        SweepRow({"x": 1, "y": 20, "g": "b", "free": 0}, {"metric__avg": 7.0}),
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec(
        {"x": "x", "y": "y", "group_by": ["g"], "aggregate_by": "avg"},
        ["metric"],
        "avg",
    )

    assert (tmp_path / "heatmap" / "direction=avg" / "axes=x_y" / "g=a" / "metric.png").exists()
    assert (tmp_path / "heatmap" / "direction=avg" / "axes=x_y" / "g=b" / "metric.png").exists()


def test_heatmap_spec_writes_3d_plot(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 10}, {"metric__avg": 1.0}),
        SweepRow({"x": 2, "y": 10}, {"metric__avg": 2.0}),
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec(
        {"x": "x", "y": "y", "render": ["heatmap3d"]},
        ["metric"],
        "avg",
    )

    assert (tmp_path / "heatmap3d" / "direction=avg" / "axes=x_y" / "all" / "metric.png").exists()


def test_heatmap_supports_composite_conditional_axes(tmp_path: Path):
    rows = [
        SweepRow({"sampler": "cucb", "sampling": 0.1}, {"accuracy__final": 0.5}),
        SweepRow(
            {"sampler": "discounted_cucb", "discount": 0.9, "sampling": 0.1},
            {"accuracy__final": 0.7},
        ),
        SweepRow(
            {"sampler": "discounted_cucb", "discount": 0.95, "sampling": 0.1},
            {"accuracy__final": 0.8},
        ),
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec(
        {"x": ["sampler", "discount"], "y": "sampling"},
        ["accuracy"],
        "final",
    )

    assert (
        tmp_path
        / "heatmap"
        / "direction=final"
        / "axes=sampler_discount_sampling"
        / "all"
        / "accuracy.png"
    ).exists()


def test_line_spec_groups_conditional_parameters_into_lines(tmp_path: Path):
    rows = [
        SweepRow({"sampling": 0.1, "sampler": "cucb"}, {"accuracy__final": 0.5}),
        SweepRow({"sampling": 0.2, "sampler": "cucb"}, {"accuracy__final": 0.6}),
        SweepRow(
            {"sampling": 0.1, "sampler": "discounted_cucb", "discount": 0.9},
            {"accuracy__final": 0.7},
        ),
        SweepRow(
            {"sampling": 0.2, "sampler": "discounted_cucb", "discount": 0.9},
            {"accuracy__final": 0.8},
        ),
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_line_spec(
        {
            "x": "sampling",
            "group_by": [["sampler", "discount"]],
            "aggregate_by": "avg",
        },
        ["accuracy"],
        "final",
    )

    assert (
        tmp_path
        / "line"
        / "direction=final"
        / "x=sampling"
        / "group=sampler_discount"
        / "accuracy.png"
    ).exists()


def test_fixed_lists_generate_separate_line_plots(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "reward": reward}, {"accuracy__final": value})
        for reward, value in [("distance", 0.5), ("cosine", 0.7)]
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_line_spec(
        {"x": "x", "fixed": {"reward": ["distance", "cosine"]}},
        ["accuracy"],
        "final",
    )

    root = tmp_path / "line" / "direction=final" / "x=x" / "group=all"
    assert (root / "fixed=reward=distance" / "accuracy.png").exists()
    assert (root / "fixed=reward=cosine" / "accuracy.png").exists()


def test_fixed_lists_generate_separate_heatmaps(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 1, "reward": reward}, {"accuracy__final": value})
        for reward, value in [("distance", 0.5), ("cosine", 0.7)]
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec(
        {"x": "x", "y": "y", "fixed": {"reward": ["distance", "cosine"]}},
        ["accuracy"],
        "final",
    )

    root = tmp_path / "heatmap" / "direction=final" / "axes=x_y"
    assert (root / "reward=distance" / "accuracy.png").exists()
    assert (root / "reward=cosine" / "accuracy.png").exists()


def test_final_direction_uses_last_two_percent_with_seed_outer_average():
    values = np.arange(2 * 100 * 3, dtype=float).reshape(2, 100, 3)

    got = scalar_reduce_seed_outer("validation_accuracy", values, "final")
    expected = np.mean([values[0, -2:].mean(), values[1, -2:].mean()])

    assert got == expected
    assert normalize_direction("final") == "final"


def test_plot_metrics_are_included_then_excluded():
    assert metrics_for_plot(
        {
            "metrics": ["validation_accuracy", "train_loss", "regret"],
            "exclude_metrics": ["train_loss"],
        }
    ) == ["validation_accuracy", "regret"]
    assert metrics_for_plot({"exclude_metrics": ["train_loss"]}) == [
        metric for metric in DEFAULT_PLOT_METRICS if metric != "train_loss"
    ]


def test_optuna_storage_url_is_loadable(tmp_path: Path):
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=optuna_storage_url(tmp_path),
        direction="maximize",
    )
    study.set_user_attr("smoke", True)

    loaded = load_sweep_study(tmp_path)

    assert loaded.study_name == STUDY_NAME
    assert loaded.user_attrs["smoke"] is True
