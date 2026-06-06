from abc import ABC, abstractmethod

import torch

from banditdl.core.worker.config import WorkerConfig
from banditdl.data import models
from banditdl.utils.math_utils import clip_vector
from banditdl.utils.tensor_utils import flatten, unflatten


class BaseWorker(ABC):
    """Worker API for all participants (honest or Byzantine)."""

    def __init__(self, worker_id, is_byzantine=False):
        self.worker_id = worker_id
        self.is_byzantine = is_byzantine

    @abstractmethod
    def train(self) -> None:
        """Execute local local-training step."""

    @abstractmethod
    def aggregate(self, weights) -> None:
        """Consume received weights and update local state."""

    @abstractmethod
    def pull(self) -> torch.Tensor:
        """Return weights/message to neighbors, optionally using context."""


class ByzantineWorker(BaseWorker):
    """Shared API for Byzantine workers"""

    @abstractmethod
    def inform(self, context):
        """Receive information from the engine to inform Byzantine behavior."""

class HonestWorker(BaseWorker):
    """Shared training logic for honest decentralized workers."""

    def __init__(
        self,
        worker_id,
        data_loader,
        data_loader_validation,
        config: WorkerConfig,
    ):
        super().__init__(worker_id=worker_id, is_byzantine=False)
        self.config = config
        self.nb_byz = config.nb_byz
        self.nb_real_byz = config.nb_real_byz
        self.nb_honest = config.nb_workers - config.nb_byz
        self.rag = config.rag
        self.b_hat = config.b_hat

        self.loaders = {"train": data_loader, "validation": data_loader_validation}
        self.iterators = {"train": iter(data_loader), "validation": iter(data_loader_validation)}

        self.initial_learning_rate = self.current_learning_rate = config.learning_rate
        self.learning_rate_decay = config.learning_rate_decay
        self.learning_rate_decay_delta = config.learning_rate_decay_delta

        self.device = config.device
        self.loss = getattr(torch.nn, config.loss)()
        self.model = getattr(models, config.model)()
        self.model.to(self.device)
        self.model_shapes = [param.shape for param in self.model.parameters()]
        self.model_size = len(flatten(self.model.parameters()))

        if self.device == "cuda":
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.initial_learning_rate, weight_decay=config.weight_decay
        )
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.momentum = config.momentum
        self.gradient_clip = config.gradient_clip

        self.labelflipping = config.labelflipping
        self.numb_labels = config.numb_labels
        self.nb_local_steps = config.nb_local_steps
        self.num_selected_byz = []
        self._current_step = 0
        self.last_gradient_norm = float("nan")

    def sample_batch(self, mode):
        try:
            return next(self.iterators[mode])
        except Exception:
            self.iterators[mode] = iter(self.loaders[mode])
            return next(self.iterators[mode])

    def backward_pass(self, inputs, targets):
        self.model.zero_grad()
        loss = self.loss(self.model(inputs), targets)
        loss.backward()
        return flatten([param.grad for param in self.model.parameters()])

    def compute_gradients(self):
        self.model.train()
        inputs, targets = self.sample_batch("train")
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            targets_flipped = targets.sub(self.numb_labels - 1).mul(-1)
            self.gradient_labelflipping = self.backward_pass(inputs, targets_flipped)
            self.model.train()

        return self.backward_pass(inputs, targets)

    def compute_momentum(self):
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(self.compute_gradients(), alpha=1 - self.momentum)

        if self.gradient_clip is not None:
            return clip_vector(self.momentum_gradient, self.gradient_clip)

        return self.momentum_gradient

    def local_model_update(self, current_step):
        def update_learning_rate(step):
            if self.learning_rate_decay > 0 and step % self.learning_rate_decay_delta == 0:
                return self.initial_learning_rate / (step / self.learning_rate_decay + 1)
            return self.current_learning_rate

        new_learning_rate = update_learning_rate(current_step)
        if self.current_learning_rate != new_learning_rate:
            self.current_learning_rate = new_learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_learning_rate

        self.optimizer.step()

    def perform_local_step(self, current_step):
        gradient_norms = []
        for _ in range(self.nb_local_steps):
            gradient = self.compute_momentum()
            gradient_norms.append(float(gradient.norm().detach().cpu().item()))
            self.set_gradient(gradient)
            self.local_model_update(current_step)
        if gradient_norms:
            self.last_gradient_norm = sum(gradient_norms) / len(gradient_norms)
        return flatten(self.model.parameters())

    def train(self) -> None:
        self.perform_local_step(self._current_step)
        self._current_step += 1

    def pull(self, context=None) -> torch.Tensor:
        return flatten(self.model.parameters())

    @torch.no_grad()
    def compute_accuracy_on_loader(self, data_loader):
        self.model.eval()
        total = 0
        correct = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct / total

    @torch.no_grad()
    def compute_validation_accuracy(self):
        return self.compute_accuracy_on_loader(self.loaders["validation"])

    @torch.no_grad()
    def compute_loss_on_loader(self, data_loader):
        self.model.eval()
        total = 0
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss_value = self.loss(outputs, targets)
            batch_size = targets.size(0)
            total += batch_size
            total_loss += float(loss_value.item()) * batch_size
        return total_loss / total

    @torch.no_grad()
    def compute_validation_loss(self):
        return self.compute_loss_on_loader(self.loaders["validation"])

    @torch.no_grad()
    def compute_train_loss(self):
        return self.compute_loss_on_loader(self.loaders["train"])

    def set_gradient(self, gradient):
        gradient = unflatten(gradient, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.grad = gradient[j].detach().clone()

    def set_model_parameters(self, params):
        params = unflatten(params, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = params[j].data.detach().clone()
