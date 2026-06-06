import numpy as np
import pytest

from banditdl.experiments.config_schema import BanditDLConfig
from banditdl.experiments.engine import (
    ResultTracker,
    _best_fixed_subset,
    _mean_selected_reward,
)


def test_best_fixed_subset_reward_is_cardinality_normalized():
    selected, reward = _best_fixed_subset([0.9, 0.5, 0.8, 0.1], worker_id=0, k=2)

    assert selected.tolist() == [2, 1]
    assert reward == pytest.approx(0.65)


def test_mean_selected_reward_is_cardinality_normalized():
    assert _mean_selected_reward([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert _mean_selected_reward([]) == 0.0


def test_probability_file_is_preallocated_for_honest_workers_with_nan(tmp_path):
    cfg = BanditDLConfig()
    cfg.topology.nodes = 4
    cfg.adversary.byzcount = 1
    cfg.optimization.rounds = 2

    with ResultTracker(cfg, tmp_path):
        pass

    probabilities = np.load(tmp_path / "sampler_probabilities.npy")
    assert probabilities.shape == (2, 3, 4)
    assert np.isnan(probabilities).all()
