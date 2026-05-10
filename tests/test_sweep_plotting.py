from pathlib import Path

import optuna

from banditdl.utils.plot_sweep_base import (
    STUDY_NAME,
    SweepAxis,
    SweepRow,
    SweepTable,
    load_sweep_study,
    optuna_storage_url,
)
from banditdl.utils.plot_sweep_heatmap import ExplicitHeatmapPlotter


def test_heatmap_groups_and_aggregates_free_dimensions(tmp_path: Path):
    rows = [
        SweepRow({"x": 1, "y": 10, "g": "a", "free": 0}, {"metric__avg": 1.0}),
        SweepRow({"x": 1, "y": 10, "g": "a", "free": 1}, {"metric__avg": 3.0}),
        SweepRow({"x": 2, "y": 10, "g": "a", "free": 0}, {"metric__avg": 5.0}),
        SweepRow({"x": 1, "y": 20, "g": "b", "free": 0}, {"metric__avg": 7.0}),
    ]
    axes = [
        SweepAxis("x", "x", [1, 2], None, False),
        SweepAxis("y", "y", [10, 20], None, False),
        SweepAxis("g", "g", ["a", "b"], None, False),
    ]
    plotter = ExplicitHeatmapPlotter(SweepTable(rows), axes, tmp_path)

    matrix = plotter._matrix(axes[0], axes[1], {"g": "a"}, "metric__avg", "avg")

    assert matrix[0, 0] == 2.0
    assert matrix[0, 1] == 5.0
    assert matrix[1, 0] != matrix[1, 0]  # NaN: no row for y=20,g=a


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
