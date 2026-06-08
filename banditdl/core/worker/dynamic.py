import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.sampling import make_neighbor_sampler, make_reward_strategy
from banditdl.core.worker.base import HonestWorker
from banditdl.core.worker.config import WorkerConfig


class DynamicWorker(HonestWorker):
    def __init__(
        self,
        worker_id,
        data_loader,
        data_loader_validation,
        config: WorkerConfig,
    ):
        super().__init__(
            worker_id,
            data_loader,
            data_loader_validation,
            config,
        )
        self.sampling_ratio = config.sampling_ratio
        self.nb_neighbors = max(
            1,
            min(
                self.nb_honest + self.nb_byz - 1,
                round((self.nb_honest + self.nb_byz - 1) * config.sampling_ratio),
            ),
        )
        self.neighbor_sampler = config.neighbor_sampler or make_neighbor_sampler("uniform")
        self.reward_strategy = config.reward_strategy or make_reward_strategy("parameter_distance")
        self.robust_aggregator = RobustAggregator(
            config.aggregator,
            config.pre_aggregator,
            config.server_clip,
            self.nb_neighbors + 1 - self.b_hat,
            self.b_hat,
            config.bucket_size,
            self.model_size,
            self.device,
        )

    def aggregate(self, weights) -> None:
        if len(weights) == 0:
            return None
        pivot_params = self.pull(None)
        if self.rag:
            self._aggregate_with_rag(pivot_params, weights)
        else:
            self._aggregate_cgplus(pivot_params, weights, max(self._current_step - 1, 0))
        return None

    def _sample_neighbors(self):
        indices_list = list(range(self.nb_honest + self.nb_byz))
        indices_list.remove(self.worker_id)
        return self.neighbor_sampler.sample(indices_list, self.nb_neighbors)

    def observe_neighbors(self, neighbor_indices, rewards) -> None:
        if not hasattr(self.neighbor_sampler, "update"):
            return None
        self.neighbor_sampler.update(neighbor_indices, rewards)
        return None

    def _aggregate_cgplus(self, pivot_params, worker_params, current_step):
        worker_params = torch.stack(worker_params)
        differences = worker_params - pivot_params
        distances = differences.norm(dim=1)
        clipping_threshold = (
            torch.topk(distances, 2 * self.b_hat).values[-1] if self.b_hat > 0 else torch.inf
        )
        mask = distances[:, None].broadcast_to(differences.shape) > clipping_threshold
        clipped_differences = torch.where(
            mask, differences * (clipping_threshold / distances[:, None]), differences
        )

        communication_lr = 1 / (current_step // 250 + 1)
        aggregate_params = pivot_params + communication_lr * clipped_differences.sum(dim=0) * (
            1 / self.nb_neighbors
        )
        self.set_model_parameters(aggregate_params)

    def _aggregate_with_rag(self, pivot_params, worker_params):
        worker_params = list(worker_params)
        worker_params.append(pivot_params)
        aggregate_params = self.robust_aggregator.aggregate(worker_params)
        self.set_model_parameters(aggregate_params)


P2PWorker = DynamicWorker
