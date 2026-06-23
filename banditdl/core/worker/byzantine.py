"""Byzantine worker implementations for distributed training."""

import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.robustness.attacks import ByzantineAttack
from banditdl.core.worker.base import BaseWorker
from banditdl.core.worker.config import WorkerConfig


class ByzantineWorker(BaseWorker):
    """Byzantine participant implementing the worker API."""

    def __init__(
        self,
        worker_id,
        model_size,
        config: WorkerConfig,
    ):
        super().__init__(worker_id=worker_id, is_byzantine=True)
        robust_aggregator = RobustAggregator(
            config.aggregator,
            config.pre_aggregator,
            config.server_clip,
            config.nb_workers,
            config.nb_byz,
            config.bucket_size,
            model_size,
            config.device,
        )
        self.byzantine_attack = ByzantineAttack(
            config.attack if config.attack else "SF",
            config.nb_real_byz,
            model_size,
            config.device,
            config.mimic_learning_phase,
            config.gradient_clip,
            robust_aggregator,
        )
        self.cached_vector = None
        self.previous_vector = None
        self.cached_delta = None

    def train(self) -> None:
        return None

    def aggregate(self, weights) -> None:
        return None

    def inform(self, honest_weights, step):
        """Compute and cache the byzantine vector once per round."""
        vectors = self.byzantine_attack.generate_byzantine_vectors(
            honest_weights, None, step
        )
        next_vector = vectors[0] if len(vectors) > 0 else None
        if next_vector is None:
            self.cached_vector = None
            self.cached_delta = None
            return

        if self.previous_vector is None:
            self.cached_delta = torch.zeros_like(next_vector)
        else:
            self.cached_delta = next_vector - self.previous_vector
        self.previous_vector = next_vector.detach().clone()
        self.cached_vector = next_vector

    def pull(self, context=None):
        return self.cached_vector

    def delta(self):
        return self.cached_delta

    def compute_validation_accuracy(self):
        return None

    def compute_validation_loss(self):
        return None

    def compute_train_loss(self):
        return None
