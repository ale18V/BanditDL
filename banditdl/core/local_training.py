from __future__ import annotations

from math import prod
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap


def has_stateful_training_layers(model: nn.Module) -> bool:
    stateful = (
        nn.modules.batchnorm._BatchNorm,
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
    )
    return any(isinstance(module, stateful) for module in model.modules())


def chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class BatchedLocalTrainer:
    def __init__(self, clients_per_batch: int | str = "auto"):
        self.clients_per_batch = clients_per_batch

    def can_train(self, workers: list) -> bool:
        return (
            bool(workers)
            and not any(getattr(worker, "labelflipping", False) for worker in workers)
            and not has_stateful_training_layers(workers[0].model)
        )

    def train_workers(self, workers: list) -> None:
        if not workers:
            return
        group_size = self._group_size(len(workers))
        for group in chunks(workers, group_size):
            self._train_group(group)

    def _group_size(self, workers: int) -> int:
        if isinstance(self.clients_per_batch, str):
            if self.clients_per_batch != "auto":
                raise ValueError("runtime.clients_per_batch must be 'auto' or an integer")
            return workers
        return max(1, min(workers, int(self.clients_per_batch)))

    def _train_group(self, workers: list) -> None:
        reference = workers[0].model
        reference.train()
        for worker in workers:
            worker.model.train()
        loss_fn = workers[0].loss
        device = workers[0].device
        params, buffers = _stack_state(workers, device)
        momentum = _stack_momentum(workers, params)
        last_losses = None
        last_norms = None

        def loss_one(one_params, one_buffers, x, y, mask):
            logits = functional_call(reference, (one_params, one_buffers), (x,))
            losses = _per_sample_loss(loss_fn, logits, y)
            return (losses * mask).sum() / mask.sum().clamp_min(1)

        grad_fn = vmap(grad(loss_one), in_dims=(0, 0, 0, 0, 0))
        loss_values_fn = vmap(loss_one, in_dims=(0, 0, 0, 0, 0))

        for _ in range(workers[0].nb_local_steps):
            x, y, mask = _stack_batches(workers, device)
            grads = grad_fn(params, buffers, x, y, mask)
            last_losses = loss_values_fn(params, buffers, x, y, mask)
            momentum = _update_momentum(momentum, grads, workers[0].momentum)
            step_grad = _clip_grads(momentum, workers[0].gradient_clip)
            last_norms = _grad_norms(step_grad)
            lr = _learning_rate_for_step(workers[0], workers[0]._current_step)
            params = _sgd_step(params, step_grad, lr, workers[0].config.weight_decay)

        _write_back(workers, params, momentum, last_losses, last_norms)


def _stack_state(workers: list, device: str):
    names = [name for name, _ in workers[0].model.named_parameters()]
    buffer_names = [name for name, _ in workers[0].model.named_buffers()]
    params = {
        name: torch.stack(
            [dict(worker.model.named_parameters())[name].detach().clone() for worker in workers]
        )
        .to(device)
        .requires_grad_(True)
        for name in names
    }
    buffers = {
        name: torch.stack(
            [dict(worker.model.named_buffers())[name].detach().clone() for worker in workers]
        ).to(device)
        for name in buffer_names
    }
    return params, buffers


def _stack_momentum(workers: list, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    shapes = [param.shape for param in workers[0].model.parameters()]
    sizes = [prod(shape) for shape in shapes]
    out = {name: [] for name in params}
    for worker in workers:
        pieces = torch.split(worker.momentum_gradient, sizes)
        for name, shape, piece in zip(params, shapes, pieces, strict=True):
            out[name].append(piece.reshape(shape))
    return {name: torch.stack(values).to(params[name].device) for name, values in out.items()}


def _stack_batches(workers: list, device: str):
    xs, ys = [], []
    for worker in workers:
        x, y = worker.next_train_batch()
        xs.append(x.to(device, non_blocking=True))
        ys.append(y.to(device, non_blocking=True))
    batch_size = max(x.shape[0] for x in xs)
    if batch_size <= 0:
        raise RuntimeError("cannot stack empty client batches for batched local training")
    masks = []
    padded_xs, padded_ys = [], []
    for x, y in zip(xs, ys, strict=True):
        real_size = x.shape[0]
        if real_size < batch_size:
            pad_indices = torch.arange(batch_size - real_size, device=device) % real_size
            x = torch.cat([x, x.index_select(0, pad_indices)], dim=0)
            y = torch.cat([y, y.index_select(0, pad_indices)], dim=0)
        mask = torch.zeros(batch_size, dtype=torch.float32, device=device)
        mask[:real_size] = 1.0
        padded_xs.append(x)
        padded_ys.append(y)
        masks.append(mask)
    return torch.stack(padded_xs), torch.stack(padded_ys), torch.stack(masks)


def _per_sample_loss(loss_fn, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if isinstance(loss_fn, nn.NLLLoss):
        return F.nll_loss(
            logits,
            targets,
            weight=loss_fn.weight,
            ignore_index=loss_fn.ignore_index,
            reduction="none",
        )
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return F.cross_entropy(
            logits,
            targets,
            weight=loss_fn.weight,
            ignore_index=loss_fn.ignore_index,
            reduction="none",
            label_smoothing=loss_fn.label_smoothing,
        )
    losses = loss_fn(logits, targets)
    return losses if losses.ndim > 0 else losses.expand(targets.shape[0])


def _update_momentum(momentum: dict[str, torch.Tensor], grads: dict[str, torch.Tensor], beta: float):
    return {name: momentum[name] * beta + grads[name] * (1 - beta) for name in grads}


def _clip_grads(grads: dict[str, torch.Tensor], max_norm: float | None):
    if max_norm is None:
        return grads
    norms = _grad_norms(grads).clamp_min(1e-12)
    scale = torch.clamp(torch.as_tensor(max_norm, device=norms.device) / norms, max=1.0)
    return {
        name: grad * scale.reshape((-1,) + (1,) * (grad.ndim - 1))
        for name, grad in grads.items()
    }


def _grad_norms(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for grad_tensor in grads.values():
        values = grad_tensor.flatten(start_dim=1).pow(2).sum(dim=1)
        total = values if total is None else total + values
    return total.sqrt()


def _learning_rate_for_step(worker, step: int) -> float:
    if worker.learning_rate_decay > 0 and step % worker.learning_rate_decay_delta == 0:
        return worker.initial_learning_rate / (step / worker.learning_rate_decay + 1)
    return worker.current_learning_rate


def _sgd_step(
    params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
    lr: float,
    weight_decay: float,
):
    updated = {}
    for name, values in params.items():
        step = grads[name]
        if weight_decay:
            step = step + weight_decay * values
        updated[name] = (values - lr * step).detach().requires_grad_(True)
    return updated


def _write_back(workers: list, params, momentum, losses, norms) -> None:
    names = list(params)
    flat_momentum = torch.cat([momentum[name].flatten(start_dim=1) for name in names], dim=1)
    with torch.no_grad():
        for i, worker in enumerate(workers):
            for name, param in worker.model.named_parameters():
                param.copy_(params[name][i])
            worker.model.zero_grad(set_to_none=True)
            worker.momentum_gradient.copy_(flat_momentum[i])
            if losses is not None:
                worker.last_train_loss = float(losses[i].detach().cpu().item())
            if norms is not None:
                worker.last_gradient_norm = float(norms[i].detach().cpu().item())
            worker.current_learning_rate = _learning_rate_for_step(worker, worker._current_step)
            for group in worker.optimizer.param_groups:
                group["lr"] = worker.current_learning_rate
            worker._refresh_flat_params()
            worker._current_step += 1


class BatchedEvaluator:
    def __init__(self, clients_per_batch: int | str = "auto"):
        self.clients_per_batch = clients_per_batch

    def can_evaluate(self, workers: list) -> bool:
        return bool(workers) and not has_stateful_training_layers(workers[0].model)

    @torch.no_grad()
    def evaluate_workers(self, workers: list, loader) -> list[tuple[float, float]]:
        if not workers:
            return []
        group_size = self._group_size(len(workers))
        out = []
        for group in chunks(workers, group_size):
            out.extend(self._evaluate_group_shared_loader(group, loader))
        return out

    @torch.no_grad()
    def evaluate_worker_loaders(self, workers: list, loader_fn) -> list[tuple[float, float]]:
        if not workers:
            return []
        group_size = self._group_size(len(workers))
        out = []
        for group in chunks(workers, group_size):
            loaders = [loader_fn(worker) for worker in group]
            out.extend(self._evaluate_group_worker_loaders(group, loaders))
        return out

    def _group_size(self, workers: int) -> int:
        if isinstance(self.clients_per_batch, str):
            return workers if self.clients_per_batch == "auto" else int(self.clients_per_batch)
        return max(1, min(workers, int(self.clients_per_batch)))

    @torch.no_grad()
    def _evaluate_group_shared_loader(self, workers: list, loader) -> list[tuple[float, float]]:
        context = _EvaluationContext(workers)
        for inputs, targets in loader:
            x = inputs.to(context.device, non_blocking=True)
            y = targets.to(context.device, non_blocking=True)
            x = x.unsqueeze(0).expand(len(workers), *x.shape)
            y = y.unsqueeze(0).expand(len(workers), *y.shape)
            mask = torch.ones((len(workers), y.shape[1]), device=context.device)
            context.update(x, y, mask)
        return context.results()

    @torch.no_grad()
    def _evaluate_group_worker_loaders(self, workers: list, loaders: list) -> list[tuple[float, float]]:
        context = _EvaluationContext(workers)
        iterators = [iter(loader) for loader in loaders]
        active = [True] * len(iterators)
        while any(active):
            xs, ys, masks = [], [], []
            max_batch = 0
            for i, iterator in enumerate(iterators):
                if not active[i]:
                    xs.append(None)
                    ys.append(None)
                    continue
                try:
                    x, y = next(iterator)
                except StopIteration:
                    active[i] = False
                    xs.append(None)
                    ys.append(None)
                    continue
                x = x.to(context.device, non_blocking=True)
                y = y.to(context.device, non_blocking=True)
                xs.append(x)
                ys.append(y)
                max_batch = max(max_batch, y.shape[0])
            if max_batch == 0:
                continue
            for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
                if x is None or y is None:
                    sample_x, sample_y = _empty_like_batch(xs, ys, max_batch, context.device)
                    xs[i], ys[i] = sample_x, sample_y
                    masks.append(torch.zeros(max_batch, device=context.device))
                    continue
                real_size = y.shape[0]
                if real_size < max_batch:
                    pad_indices = torch.arange(max_batch - real_size, device=context.device) % real_size
                    x = torch.cat([x, x.index_select(0, pad_indices)], dim=0)
                    y = torch.cat([y, y.index_select(0, pad_indices)], dim=0)
                mask = torch.zeros(max_batch, device=context.device)
                mask[:real_size] = 1.0
                xs[i], ys[i] = x, y
                masks.append(mask)
            context.update(torch.stack(xs), torch.stack(ys), torch.stack(masks))
        return context.results()


class _EvaluationContext:
    def __init__(self, workers: list):
        self.workers = workers
        self.reference = workers[0].model
        self.reference.eval()
        for worker in workers:
            worker.model.eval()
        self.loss_fn = workers[0].loss
        self.device = workers[0].device
        self.params, self.buffers = _stack_state_no_grad(workers, self.device)
        self.total = torch.zeros(len(workers), device=self.device)
        self.correct = torch.zeros(len(workers), device=self.device)
        self.loss_total = torch.zeros(len(workers), device=self.device)

        def forward_one(one_params, one_buffers, x):
            return functional_call(self.reference, (one_params, one_buffers), (x,))

        self.forward_fn = vmap(forward_one, in_dims=(0, 0, 0))

    def update(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        outputs = self.forward_fn(self.params, self.buffers, x)
        predictions = outputs.argmax(dim=2)
        batch_losses = vmap(lambda logits, targets: _per_sample_loss(self.loss_fn, logits, targets))(
            outputs, y
        )
        self.total += mask.sum(dim=1)
        self.correct += ((predictions == y) * mask.bool()).sum(dim=1)
        self.loss_total += (batch_losses * mask).sum(dim=1)

    def results(self) -> list[tuple[float, float]]:
        acc = torch.where(self.total > 0, self.correct / self.total.clamp_min(1), torch.nan)
        losses = torch.where(
            self.total > 0,
            self.loss_total / self.total.clamp_min(1),
            torch.nan,
        )
        return list(zip(acc.detach().cpu().tolist(), losses.detach().cpu().tolist(), strict=True))


def _empty_like_batch(xs: list, ys: list, batch_size: int, device: str):
    sample_x = next(x for x in xs if x is not None)
    sample_y = next(y for y in ys if y is not None)
    return (
        torch.zeros((batch_size, *sample_x.shape[1:]), dtype=sample_x.dtype, device=device),
        torch.zeros((batch_size, *sample_y.shape[1:]), dtype=sample_y.dtype, device=device),
    )


def _stack_state_no_grad(workers: list, device: str):
    params = {
        name: torch.stack(
            [dict(worker.model.named_parameters())[name].detach() for worker in workers]
        ).to(device)
        for name, _ in workers[0].model.named_parameters()
    }
    buffers = {
        name: torch.stack(
            [dict(worker.model.named_buffers())[name].detach() for worker in workers]
        ).to(device)
        for name, _ in workers[0].model.named_buffers()
    }
    return params, buffers
