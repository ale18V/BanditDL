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
    np.save(tmp_path / "evaluation_steps.npy", np.array([0, 2, 4]))
    np.save(
        tmp_path / "validation_accuracy.npy",
        np.array([[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]]),
    )
    np.save(tmp_path / "regret.npy", np.zeros((5, 2)))

    data = MetricLoader(tmp_path).load(
        MetricKey.VALIDATION_ACCURACY,
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


def test_validation_interpolation_keeps_final_round_after_round_metrics(tmp_path):
    np.save(tmp_path / "evaluation_steps.npy", np.array([0, 2, 4]))
    np.save(tmp_path / "validation_loss.npy", np.array([[4.0], [2.0], [1.0]]))
    np.save(tmp_path / "sampler_probabilities.npy", np.zeros((4, 1, 2)))

    data = MetricLoader(tmp_path).load(
        MetricKey.VALIDATION_LOSS,
        interpolate_eval=True,
    )

    np.testing.assert_allclose(data.x, np.arange(5))
    np.testing.assert_allclose(data.values[:, 0], np.array([4.0, 3.0, 2.0, 1.5, 1.0]))


def test_metric_loader_reads_current_engine_metric_names(tmp_path):
    validation_accuracy = np.array([[0.2, 0.4], [0.6, 0.8]])
    validation_loss = np.array([[2.0, 1.0], [0.8, 0.4]])
    global_accuracy = np.array([[0.1, 0.3], [0.5, 0.7]])
    train_loss = np.array([[1.8, 0.9], [0.7, 0.3]])
    np.save(tmp_path / "validation_accuracy.npy", validation_accuracy)
    np.save(tmp_path / "validation_loss.npy", validation_loss)
    np.save(tmp_path / "global_accuracy.npy", global_accuracy)
    np.save(tmp_path / "train_loss.npy", train_loss)

    loader = MetricLoader(tmp_path)

    np.testing.assert_allclose(
        loader.load_values(MetricKey.VALIDATION_ACCURACY),
        validation_accuracy,
    )
    np.testing.assert_allclose(
        loader.load_values(MetricKey.VALIDATION_LOSS),
        validation_loss,
    )
    np.testing.assert_allclose(
        loader.load_values(MetricKey.GLOBAL_ACCURACY),
        global_accuracy,
    )
    np.testing.assert_allclose(loader.load_values(MetricKey.TRAIN_LOSS), train_loss)


def test_seed_stacked_validation_accuracy_interpolates_on_time_axis(tmp_path):
    np.save(tmp_path / "evaluation_steps.npy", np.array([0, 2, 4]))
    np.save(
        tmp_path / "validation_accuracy_by_seed.npy",
        np.array(
            [
                [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]],
                [[0.1, 0.3], [0.5, 0.7], [0.9, 1.1]],
            ]
        ),
    )
    np.save(tmp_path / "regret.npy", np.zeros((5, 2)))

    data = MetricLoader(tmp_path).load(
        MetricKey.VALIDATION_ACCURACY,
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

    reduced = scalar_reduce_seed_outer(MetricKey.VALIDATION_LOSS, values, "worse")

    assert reduced == pytest.approx(52.0)


def test_scalar_best_reduction_uses_opposite_extreme_for_loss_metrics():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[100.0, 0.0], [0.0, 0.0]],
        ]
    )

    reduced = scalar_reduce_seed_outer(MetricKey.VALIDATION_LOSS, values, "best")

    assert reduced == pytest.approx(0.5)


def test_scalar_best_reduction_uses_opposite_extreme_for_accuracy_metrics():
    values = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.9, 0.0], [0.2, 0.1]],
        ]
    )

    reduced = scalar_reduce_seed_outer(MetricKey.VALIDATION_ACCURACY, values, "best")

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


def test_uniform_sampler_kl_is_zero_after_excluding_self(tmp_path):
    probabilities = np.array(
        [
            [
                [0.0, 1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 0.0, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 0.0, 1 / 3],
            ]
        ]
    )
    np.save(tmp_path / "sampler_probabilities.npy", probabilities)

    values = MetricLoader(tmp_path).load_values(MetricKey.SAMPLER_KL_TO_UNIFORM)

    np.testing.assert_allclose(values, np.zeros((1, 3)), atol=1e-12)


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


def test_sampler_weight_summaries_use_generic_distribution_logic(tmp_path):
    weights = np.array(
        [
            [[0.0, 0.25, 0.75], [0.5, 0.0, 0.5]],
            [[0.0, 0.5, 0.5], [0.9, 0.0, 0.1]],
        ]
    )
    np.save(tmp_path / "sampler_weights.npy", weights)

    loader = MetricLoader(tmp_path)

    np.testing.assert_allclose(
        loader.load_values(MetricKey.SAMPLER_MAX_WEIGHT),
        np.array([[0.75, 0.5], [0.5, 0.9]]),
    )
    np.testing.assert_allclose(
        loader.load_values(MetricKey.SAMPLER_WEIGHT_ENTROPY),
        -np.sum(np.where(weights > 0, weights * np.log(np.maximum(weights, 1e-12)), 0), axis=2),
    )
