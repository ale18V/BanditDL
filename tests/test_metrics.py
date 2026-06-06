import numpy as np
import pytest

from banditdl.utils.metrics import (
    MetricKey,
    MetricLoader,
    TimeAverage,
    min_,
    scalar_reduce_seed_outer,
)
from banditdl.utils.plot_sweep_base import normalize_directions


def test_normalized_regret_is_derived_from_regret(tmp_path):
    regret = np.array([[2.0, 4.0], [6.0, 8.0], [9.0, 12.0]])
    np.save(tmp_path / "regret.npy", regret)

    values = MetricLoader(tmp_path).load_values(MetricKey.NORMALIZED_REGRET)

    np.testing.assert_allclose(
        values,
        np.array([[2.0, 4.0], [3.0, 4.0], [3.0, 4.0]]),
    )


def test_validation_accuracy_can_interpolate_to_full_round_axis(tmp_path):
    (tmp_path / "validation").write_text("0\t0.1\n2\t0.5\n4\t0.9\n")
    np.save(
        tmp_path / "validation_accuracies.npy",
        np.array([[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]]),
    )
    np.save(tmp_path / "regret.npy", np.zeros((5, 2)))

    data = MetricLoader(tmp_path).load(
        MetricKey.VALIDATION_ACCURACIES,
        interpolate_eval=True,
    )

    np.testing.assert_allclose(data.x, np.arange(5))
    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [0.0, 0.2],
                [0.2, 0.4],
                [0.4, 0.6],
                [0.6, 0.8],
                [0.8, 1.0],
            ]
        ),
    )


def test_seed_stacked_validation_accuracy_interpolates_on_time_axis(tmp_path):
    (tmp_path / "validation").write_text("0\t0.1\n2\t0.5\n4\t0.9\n")
    np.save(
        tmp_path / "validation_accuracies_by_seed.npy",
        np.array(
            [
                [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]],
                [[0.1, 0.3], [0.5, 0.7], [0.9, 1.1]],
            ]
        ),
    )
    np.save(tmp_path / "regret.npy", np.zeros((5, 2)))

    data = MetricLoader(tmp_path).load(
        MetricKey.VALIDATION_ACCURACIES,
        interpolate_eval=True,
    )

    np.testing.assert_allclose(data.x, np.arange(5))
    assert data.values.shape == (2, 5, 2)
    np.testing.assert_allclose(data.values[0, :, 0], np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
    np.testing.assert_allclose(data.values[1, :, 1], np.array([0.3, 0.5, 0.7, 0.9, 1.1]))


def test_node_reduction_averages_seeds_outermost():
    values = np.array(
        [
            [[0.0, 100.0], [5.0, 10.0]],
            [[100.0, 0.0], [20.0, 1.0]],
        ]
    )

    np.testing.assert_allclose(min_.fn(values), np.array([0.0, 3.0]))


def test_scalar_worse_reduction_averages_per_seed_worst_values():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[100.0, 0.0], [0.0, 0.0]],
        ]
    )

    reduced = scalar_reduce_seed_outer(MetricKey.VALIDATION_LOSSES, values, "worse")

    assert reduced == pytest.approx(52.0)


def test_scalar_best_reduction_uses_opposite_extreme_for_loss_metrics():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[100.0, 0.0], [0.0, 0.0]],
        ]
    )

    reduced = scalar_reduce_seed_outer(MetricKey.VALIDATION_LOSSES, values, "best")

    assert reduced == pytest.approx(0.5)


def test_scalar_best_reduction_uses_opposite_extreme_for_accuracy_metrics():
    values = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.9, 0.0], [0.2, 0.1]],
        ]
    )

    reduced = scalar_reduce_seed_outer(MetricKey.VALIDATION_ACCURACIES, values, "best")

    assert reduced == pytest.approx(0.65)


def test_sweep_direction_normalization_accepts_best_and_keeps_order():
    directions = normalize_directions(["worst", "best", "avg", "best"])

    assert directions == ["worse", "best", "avg"]


def test_time_average_uses_time_axis_after_seed_axis():
    values = np.array([[[2.0], [6.0], [12.0]], [[4.0], [10.0], [18.0]]])

    averaged = TimeAverage()(values)

    np.testing.assert_allclose(
        averaged,
        np.array([[[2.0], [3.0], [4.0]], [[4.0], [5.0], [6.0]]]),
    )


def test_metric_loader_loads_selected_reward_extrema(tmp_path):
    values = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.save(tmp_path / "reward_selected_min.npy", values)

    loaded = MetricLoader(tmp_path).load_values(MetricKey.REWARD_SELECTED_MIN)

    np.testing.assert_allclose(loaded, values)


def test_sampler_probability_summaries_are_derived_from_raw_probabilities(tmp_path):
    probabilities = np.array(
        [
            [[0.0, 0.25, 0.75], [0.5, 0.0, 0.5]],
            [[0.0, 0.5, 0.5], [0.9, 0.0, 0.1]],
        ]
    )
    np.save(tmp_path / "sampler_probabilities.npy", probabilities)

    loader = MetricLoader(tmp_path)

    np.testing.assert_allclose(
        loader.load_values(MetricKey.SAMPLER_MIN_PROBABILITY),
        np.array([[0.25, 0.5], [0.5, 0.1]]),
    )
    np.testing.assert_allclose(
        loader.load_values(MetricKey.SAMPLER_MAX_PROBABILITY),
        np.array([[0.75, 0.5], [0.5, 0.9]]),
    )
    expected_kl = np.array(
        [
            [
                0.25 * np.log(0.25 / 0.5) + 0.75 * np.log(0.75 / 0.5),
                0.0,
            ],
            [
                0.0,
                0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5),
            ],
        ]
    )
    np.testing.assert_allclose(loader.load_values(MetricKey.SAMPLER_KL_TO_UNIFORM), expected_kl)


def test_seed_stacked_sampler_probability_summaries_are_derived_on_inner_axes(tmp_path):
    probabilities = np.array(
        [
            [[[0.0, 0.25, 0.75], [0.5, 0.0, 0.5]]],
            [[[0.0, 0.5, 0.5], [0.9, 0.0, 0.1]]],
        ]
    )
    np.save(tmp_path / "sampler_probabilities_by_seed.npy", probabilities)

    values = MetricLoader(tmp_path).load_seed_values(MetricKey.SAMPLER_MAX_PROBABILITY)

    np.testing.assert_allclose(values, np.array([[[0.75, 0.5]], [[0.5, 0.9]]]))


def test_sampler_probability_loader_trims_unwritten_rounds(tmp_path):
    probabilities = np.full((4, 2, 3), np.nan)
    probabilities[0] = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]
    probabilities[1] = [[0.0, 0.2, 0.8], [0.7, 0.0, 0.3]]
    np.save(tmp_path / "sampler_probabilities.npy", probabilities)

    loaded = MetricLoader(tmp_path).load_values(MetricKey.SAMPLER_PROBABILITIES)

    assert loaded.shape == (2, 2, 3)
    np.testing.assert_allclose(loaded, probabilities[:2])
