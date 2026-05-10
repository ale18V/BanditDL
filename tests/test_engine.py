from io import StringIO
from types import SimpleNamespace

import pytest

from banditdl.experiments.engine import (
    _best_fixed_subset,
    _mean_selected_reward,
    _record_final_evaluation_if_needed,
)


class _FakeWorker:
    def __init__(self, value):
        self.value = value

    def compute_validation_accuracy(self):
        return self.value

    def compute_validation_loss(self):
        return 1.0 - self.value

    def compute_train_loss(self):
        return 2.0 - self.value


def test_final_evaluation_is_recorded_at_rounds():
    args = SimpleNamespace(evaluation_delta=10, rounds=2)
    validation_steps = [0]
    validation_accuracies = [[0.1, 0.2]]
    validation_losses = [[0.9, 0.8]]
    train_losses = [[1.9, 1.8]]

    _record_final_evaluation_if_needed(
        args,
        [_FakeWorker(0.3), _FakeWorker(0.5)],
        StringIO(),
        StringIO(),
        StringIO(),
        validation_steps,
        validation_accuracies,
        validation_losses,
        train_losses,
    )

    assert validation_steps == [0, 2]
    assert validation_accuracies[-1] == [0.3, 0.5]


def test_best_fixed_subset_reward_is_cardinality_normalized():
    selected, reward = _best_fixed_subset([0.9, 0.5, 0.8, 0.1], worker_id=0, k=2)

    assert selected.tolist() == [2, 1]
    assert reward == pytest.approx(0.65)


def test_mean_selected_reward_is_cardinality_normalized():
    assert _mean_selected_reward([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert _mean_selected_reward([]) == 0.0
