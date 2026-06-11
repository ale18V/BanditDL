from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pytest

from banditdl.utils.experiment_table import ExperimentTable, SweepRow
from banditdl.utils.metrics import scalar_reduce_seed_outer
from banditdl.utils.plot_sweep_base import (
    DEFAULT_PLOT_METRICS,
    STUDY_NAME,
    load_sweep_study,
    metrics_for_plot,
    normalize_direction,
    optuna_storage_url,
    sweep_table_from_study,
)
from banditdl.utils.plotting_utils import (
    axis_values,
    matches_axis,
    matches_split,
    split_filters,
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

    root = tmp_path / "heatmap" / "direction=avg" / "axes=x_y" / "g=a"
    assert (root / "metric.png").exists()
    assert r"\begin{tabular}" in (root / "metric.tex").read_text()


def test_heatmap_spec_groups_and_aggregates_free_dimensions(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 10, "g": "a", "free": 0}, {"metric__avg": 1.0}),
        SweepRow({"x": 1, "y": 10, "g": "a", "free": 1}, {"metric__avg": 3.0}),
        SweepRow({"x": 2, "y": 10, "g": "a", "free": 0}, {"metric__avg": 5.0}),
        SweepRow({"x": 1, "y": 20, "g": "b", "free": 0}, {"metric__avg": 7.0}),
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec(
        {"x": "x", "y": "y", "split_by": ["g"], "aggregate_by": "avg"},
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


def test_missing_conditional_axis_value_is_reused_without_none_category():
    rows = [
        SweepRow({"sampler": "uniform"}, {}),
        SweepRow({"sampler": "cucb", "reward": "distance"}, {}),
        SweepRow({"sampler": "cucb", "reward": "cosine"}, {}),
    ]

    assert axis_values(rows, ("reward",)) == ["cosine", "distance"]
    assert matches_axis(rows[0].params, ("reward",), "distance")
    assert matches_axis(rows[0].params, ("reward",), "cosine")


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


def test_split_by_generates_separate_line_plots(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "reward": reward}, {"accuracy__final": value})
        for reward, value in [("distance", 0.5), ("cosine", 0.7)]
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_line_spec(
        {"x": "x", "split_by": "reward"},
        ["accuracy"],
        "final",
    )

    root = tmp_path / "line" / "direction=final" / "x=x" / "group=all"
    assert (root / "split=reward=distance" / "accuracy.png").exists()
    assert (root / "split=reward=cosine" / "accuracy.png").exists()


def test_missing_conditional_parameter_matches_every_split():
    assert matches_split({"sampler": "uniform"}, {"reward": "distance"})
    assert matches_split({"reward": "distance"}, {"reward": "distance"})
    assert not matches_split({"reward": "cosine"}, {"reward": "distance"})


def test_missing_conditional_parameter_does_not_create_its_own_split():
    rows = [
        SweepRow({"sampler": "uniform", "alpha": 1}, {}),
        SweepRow({"sampler": "cucb", "reward": "distance", "alpha": 1}, {}),
        SweepRow({"sampler": "cucb", "reward": "cosine", "alpha": 1}, {}),
    ]

    assert split_filters(rows, ("reward", "alpha")) == [
        {"reward": "cosine", "alpha": 1},
        {"reward": "distance", "alpha": 1},
    ]


def test_split_by_generates_separate_heatmaps(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 1, "reward": reward}, {"accuracy__final": value})
        for reward, value in [("distance", 0.5), ("cosine", 0.7)]
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec(
        {"x": "x", "y": "y", "split_by": "reward"},
        ["accuracy"],
        "final",
    )

    root = tmp_path / "heatmap" / "direction=final" / "axes=x_y"
    assert (root / "reward=distance" / "accuracy.png").exists()
    assert (root / "reward=cosine" / "accuracy.png").exists()


def test_missing_split_by_uses_all_unused_dimensions(tmp_path: Path):
    table = ExperimentTable(
        [
            SweepRow(
                {"x": 1, "y": 1, "reward": reward, "sampling": sampling},
                {"m__avg": 1},
            )
            for reward, sampling in [("distance", 0.1), ("cosine", 0.2)]
        ]
    )
    plotter = SweepPlotter(table, tmp_path)

    plotter.plot_heatmap_spec({"x": "x", "y": "y"}, ["m"], "avg")

    root = tmp_path / "heatmap" / "direction=avg" / "axes=x_y"
    assert (root / "reward=distance__sampling=0_1" / "m.png").exists()
    assert (root / "reward=cosine__sampling=0_2" / "m.png").exists()


def test_default_split_ignores_single_valued_parameters(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 1, "reward": reward, "exploration": 1.0}, {"m__avg": 1})
        for reward in ("distance", "cosine")
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec({"x": "x", "y": "y"}, ["m"], "avg")

    root = tmp_path / "heatmap" / "direction=avg" / "axes=x_y"
    assert (root / "reward=distance" / "m.png").exists()
    assert not any("exploration" in str(path) for path in root.rglob("*"))


def test_default_split_ignores_tuple_derived_parameters(tmp_path: Path):
    table = ExperimentTable(
        [
            SweepRow(
                {
                    "x": 1,
                    "y": 1,
                    "partition": partition,
                    "clusters": clusters,
                    "sampling": sampling,
                },
                {"m__avg": 1},
            )
            for partition, clusters, sampling in [("p3", 3, 0.3), ("p5", 5, 0.2)]
        ]
    )
    table.axes_meta = [
        SimpleNamespace(path=path, display_name=path)
        for path in ("x", "y", "partition")
    ]

    SweepPlotter(table, tmp_path).plot_heatmap_spec({"x": "x", "y": "y"}, ["m"], "avg")

    root = tmp_path / "heatmap" / "direction=avg" / "axes=x_y"
    assert (root / "partition=p3" / "m.png").exists()
    assert not any("clusters" in str(path) or "sampling" in str(path) for path in root.rglob("*"))


def test_invalid_heatmap_axis_fails_loudly(tmp_path: Path):
    plotter = SweepPlotter(
        ExperimentTable([SweepRow({"x": 1, "y": 1}, {"m__avg": 1})]),
        tmp_path,
    )

    with pytest.raises(ValueError, match=r"Unknown heatmap y-axis.*clusters"):
        plotter.plot_heatmap_spec({"x": "x", "y": "partition.clusters"}, ["m"], "avg")


def test_empty_split_by_aggregates_unused_dimensions(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 1, "reward": reward}, {"m__avg": value})
        for reward, value in [("distance", 1), ("cosine", 3)]
    ]
    plotter = SweepPlotter(ExperimentTable(rows), tmp_path)

    plotter.plot_heatmap_spec(
        {"x": "x", "y": "y", "split_by": [], "aggregate_by": "avg"},
        ["m"],
        "avg",
    )

    assert (
        tmp_path / "heatmap" / "direction=avg" / "axes=x_y" / "all" / "m.png"
    ).exists()


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


def test_sweep_table_averages_independent_seed_trials(tmp_path: Path):
    study = optuna.create_study(direction="maximize")
    params = {"sampler.name": "cucb"}
    for seed, value in ((10, 0.2), (11, 0.8)):
        result_dir = tmp_path / f"seed-{seed}"
        result_dir.mkdir()
        np.save(result_dir / "validation_accuracy.npy", np.array([[value], [value]]))
        study.add_trial(
            optuna.trial.create_trial(
                value=value,
                params={"sampler.name": "cucb"},
                distributions={
                    "sampler.name": optuna.distributions.CategoricalDistribution(["cucb"])
                },
                user_attrs={
                    "config_id": 0,
                    "seed": seed,
                    "attempt": 1,
                    "resolved_params": params,
                    "result_dir": str(result_dir),
                },
            )
        )

    table = sweep_table_from_study(
        tmp_path,
        study,
        {"sampler.name": {"type": "categorical", "choices": ["cucb"]}},
        ["validation_accuracy"],
        ["final"],
        [10, 11],
    )

    assert len(table.rows) == 1
    assert table.rows[0].metrics["validation_accuracy__final"] == pytest.approx(0.5)
