import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from banditdl.core.local_training import (
    BatchedEvaluator,
    BatchedLocalTrainer,
)
from banditdl.core.worker.config import WorkerConfig
from banditdl.core.worker.dynamic import DynamicWorker


def _loader(offset=0.0, batch_size=2):
    x = torch.arange(8 * 28 * 28, dtype=torch.float32).reshape(8, 1, 28, 28) / 255.0
    x = x + offset
    y = torch.arange(8) % 10
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True)


def _worker(worker_id=0, offset=0.0):
    cfg = WorkerConfig(
        model="fc_mnist",
        learning_rate=0.1,
        learning_rate_decay=0,
        learning_rate_decay_delta=1,
        weight_decay=0.0,
        loss="NLLLoss",
        momentum=0.0,
        device="cpu",
        nb_local_steps=1,
        nb_workers=2,
        nb_byz=0,
        nb_real_byz=0,
        b_hat=0,
        sampling_ratio=0.5,
    )
    return DynamicWorker(worker_id, _loader(offset), _loader(offset), cfg)


def test_batched_training_single_client_matches_sequential_step():
    torch.manual_seed(7)
    sequential = _worker(0)
    torch.manual_seed(7)
    batched = _worker(0)

    sequential.train()
    BatchedLocalTrainer(clients_per_batch=1).train_workers([batched])

    torch.testing.assert_close(batched.pull(None), sequential.pull(None), rtol=1e-5, atol=1e-6)
    assert batched.last_train_loss == sequential.last_train_loss
    assert batched.last_gradient_norm == pytest.approx(sequential.last_gradient_norm)


def test_batched_shared_loader_evaluation_matches_worker_evaluation():
    torch.manual_seed(9)
    workers = [_worker(0), _worker(1, offset=0.1)]
    loader = _loader(batch_size=4)

    expected = [worker.compute_metrics_on_loader(loader) for worker in workers]
    got = BatchedEvaluator(clients_per_batch=2).evaluate_workers(workers, loader)

    for (got_acc, got_loss), (exp_acc, exp_loss) in zip(got, expected, strict=True):
        assert got_acc == exp_acc
        assert got_loss == pytest.approx(exp_loss)


def test_batched_per_worker_loader_evaluation_matches_worker_evaluation():
    torch.manual_seed(11)
    workers = [_worker(0), _worker(1, offset=0.1)]
    workers[0].loaders["validation"] = _loader(batch_size=3)
    workers[1].loaders["validation"] = _loader(offset=0.1, batch_size=2)

    expected = [
        worker.compute_metrics_on_loader(worker.loaders["validation"])
        for worker in workers
    ]
    got = BatchedEvaluator(clients_per_batch=2).evaluate_worker_loaders(
        workers,
        lambda worker: worker.loaders["validation"],
    )

    for (got_acc, got_loss), (exp_acc, exp_loss) in zip(got, expected, strict=True):
        assert got_acc == exp_acc
        assert got_loss == pytest.approx(exp_loss)
