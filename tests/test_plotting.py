import matplotlib
import numpy as np

matplotlib.use("Agg")

from banditdl.utils.plotting import plot_all


def _write_common_plot_metrics(run_dir):
    np.save(run_dir / "validation_accuracy.npy", np.ones((2, 2)) * 0.5)
    np.save(run_dir / "validation_loss.npy", np.ones((2, 2)))
    np.save(run_dir / "global_accuracy.npy", np.ones((2, 2)) * 0.4)
    np.save(run_dir / "train_loss.npy", np.ones((2, 2)))
    np.save(run_dir / "neighbor_disagreement.npy", np.ones((2, 2)))
    np.save(run_dir / "consensus_drift.npy", np.ones((2, 2)))
    np.save(run_dir / "sampler_kl_to_uniform.npy", np.ones((2, 2)))
    np.save(run_dir / "sampler_min_probability.npy", np.ones((2, 2)) * 0.1)
    np.save(run_dir / "sampler_max_probability.npy", np.ones((2, 2)) * 0.9)
    np.save(
        run_dir / "sampler_weights.npy",
        np.array(
            [
                [[0.0, 0.25, 0.75], [0.5, 0.0, 0.5]],
                [[0.0, 0.5, 0.5], [0.9, 0.0, 0.1]],
            ]
        ),
    )
    np.save(run_dir / "reward_algorithm.npy", np.ones((2, 2)))
    np.save(run_dir / "reward_oracle.npy", np.ones((2, 2)) * 2.0)
    np.save(run_dir / "reward_selected_min.npy", np.ones((2, 2)) * 0.1)
    np.save(run_dir / "reward_selected_max.npy", np.ones((2, 2)) * 0.9)
    np.save(run_dir / "regret.npy", np.ones((2, 2)))
    np.save(run_dir / "gradient_norms.npy", np.array([[4.0, 2.0], [1.0, 0.5]]))


def test_plot_all_writes_gradient_norm_loglog_plot(tmp_path):
    run_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    run_dir.mkdir()
    _write_common_plot_metrics(run_dir)

    plot_all(run_dir, plots_dir, "test-run")

    output = plots_dir / "gradient_norm_loglog.png"
    assert output.is_file()
    assert output.stat().st_size > 0
    assert (plots_dir / "global_accuracy.png").is_file()
    assert (plots_dir / "sampler_aggressiveness.png").is_file()
    assert (plots_dir / "sampler_weights.png").is_file()
