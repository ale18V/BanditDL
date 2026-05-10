import numpy as np

from banditdl.utils.metrics import MetricKey, MetricLoader


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


def test_metric_loader_loads_selected_reward_extrema(tmp_path):
    values = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.save(tmp_path / "reward_selected_min.npy", values)

    loaded = MetricLoader(tmp_path).load_values(MetricKey.REWARD_SELECTED_MIN)

    np.testing.assert_allclose(loaded, values)
