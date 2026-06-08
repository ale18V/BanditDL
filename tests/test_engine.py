import numpy as np
import pytest
import torch

from banditdl.experiments.config_schema import BanditDLConfig
from banditdl.experiments.engine import (
    ResultTracker,
    _best_fixed_subset,
    _mean_selected_reward,
)


class _FakeWorker:
    last_gradient_norm = 0.0

    def __init__(self):
        self.worker_id = 0
        self.train_loss_calls = 0

    def compute_validation_accuracy(self):
        return 0.5

    def compute_validation_loss(self):
        return 1.5

    def compute_train_loss(self):
        self.train_loss_calls += 1
        return float(self.train_loss_calls)


def test_best_fixed_subset_reward_is_cardinality_normalized():
    selected, reward = _best_fixed_subset([0.9, 0.5, 0.8, 0.1], worker_id=0, k=2)

    assert selected.tolist() == [2, 1]
    assert reward == pytest.approx(0.65)


def test_mean_selected_reward_is_cardinality_normalized():
    assert _mean_selected_reward([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert _mean_selected_reward([]) == 0.0


def test_sampler_diagnostics_are_progressively_written(tmp_path):
    cfg = BanditDLConfig()
    cfg.topology.nodes = 4
    cfg.adversary.byzcount = 1
    cfg.optimization.rounds = 2

    with ResultTracker(cfg, tmp_path) as tracker:
        tracker.record_sampler_diagnostics(
            0,
            np.full((3, 4), 0.25),
            np.full((3, 4), 0.25),
        )

    probabilities = np.load(tmp_path / "sampler_probabilities.npy")
    weights = np.load(tmp_path / "sampler_weights.npy")
    assert probabilities.shape == (2, 3, 4)
    np.testing.assert_allclose(probabilities[0], 0.25)
    assert np.isnan(probabilities[1]).all()
    assert weights.shape == (2, 3, 4)
    np.testing.assert_allclose(weights[0], 0.25)
    assert np.isnan(weights[1]).all()

def test_tracker_records_validation_checkpoints_and_roundwise_train_loss(tmp_path):
    cfg = BanditDLConfig()
    cfg.topology.nodes = 1
    cfg.optimization.rounds = 5
    cfg.evaluation.evaluation_delta = 2
    worker = _FakeWorker()

    with ResultTracker(cfg, tmp_path) as tracker:
        for step in range(cfg.effective_rounds + 1):
            tracker.evaluate_step(step, [worker])
            tracker.record_train_loss(step, [worker])

    np.testing.assert_allclose(np.load(tmp_path / "evaluation_steps.npy"), [0, 2, 4, 5])
    assert np.load(tmp_path / "validation_accuracy.npy").shape == (4, 1)
    assert np.load(tmp_path / "validation_loss.npy").shape == (4, 1)
    np.testing.assert_allclose(
        np.load(tmp_path / "train_loss.npy")[:, 0],
        np.arange(1, 7, dtype=float),
    )


class _EvaluationWorker:
    worker_id = 0

    def compute_validation_accuracy(self):
        return 0.75

    def compute_validation_loss(self):
        return 0.5

    def compute_train_loss(self):
        return 0.4

    def compute_accuracy_on_loader(self, loader):
        return 0.8

    def pull(self, context):
        return torch.tensor([1.0])


def test_final_evaluation_is_saved_when_rounds_are_not_divisible_by_delta(tmp_path):
    cfg = BanditDLConfig()
    cfg.topology.nodes = 2
    cfg.optimization.rounds = 25
    cfg.evaluation.evaluation_delta = 20
    worker = _EvaluationWorker()

    with ResultTracker(cfg, tmp_path) as tracker:
        tracker.evaluate_step(0, [worker, worker])
        tracker.evaluate_step(20, [worker, worker])
        tracker.finalize([worker, worker])

    accuracy = np.load(tmp_path / "validation_accuracy.npy")
    assert tracker.validation_steps == [0, 20, 25]
    assert accuracy.shape == (3, 2)
    np.testing.assert_allclose(accuracy[-1], 0.75)


def test_final_test_accuracy_has_a_dedicated_metric_file(tmp_path):
    cfg = BanditDLConfig()
    cfg.topology.nodes = 2
    cfg.optimization.rounds = 1
    cfg.evaluation.evaluation_delta = 1
    cfg.evaluation.evaluate_test = True
    worker = _EvaluationWorker()

    with ResultTracker(cfg, tmp_path, test_loader=object()) as tracker:
        tracker.finalize([worker, worker])

    np.testing.assert_allclose(np.load(tmp_path / "test_accuracy.npy"), [0.8, 0.8])
