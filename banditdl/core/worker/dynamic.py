import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.sampling import make_neighbor_sampler, make_reward_strategy
from banditdl.core.worker.base import HonestWorker


class DynamicWorker(HonestWorker):
    def __init__(
        self,
        worker_id,
        data_loader,
        data_loader_validation,
        nb_workers,
        nb_byz,
        nb_real_byz,
        aggregator,
        pre_aggregator,
        server_clip,
        bucket_size,
        model,
        learning_rate,
        learning_rate_decay,
        learning_rate_decay_delta,
        weight_decay,
        loss,
        momentum,
        device,
        labelflipping,
        gradient_clip,
        numb_labels,
        nb_neighbors,
        sampling_ratio,
        rag,
        b_hat,
        nb_local_steps,
        neighbor_sampler=None,
        reward_strategy=None,
    ):
        super().__init__(
            worker_id,
            data_loader,
            data_loader_validation,
            nb_workers,
            nb_byz,
            nb_real_byz,
            model,
            learning_rate,
            learning_rate_decay,
            learning_rate_decay_delta,
            weight_decay,
            loss,
            momentum,
            device,
            labelflipping,
            gradient_clip,
            numb_labels,
            nb_local_steps,
            rag,
            b_hat,
        )
        self.sampling_ratio = sampling_ratio
        self.nb_neighbors = (
            max(
                1,
                min(
                    self.nb_honest + self.nb_byz - 1,
                    int(round((self.nb_honest + self.nb_byz - 1) * sampling_ratio)),
                ),
            )
            if sampling_ratio is not None
            else nb_neighbors
        )
        self.neighbor_sampler = neighbor_sampler or make_neighbor_sampler("uniform")
        self.reward_strategy = reward_strategy or make_reward_strategy(
            "parameter_distance"
        )
        self.robust_aggregator = RobustAggregator(
            aggregator,
            pre_aggregator,
            server_clip,
            self.nb_neighbors + 1 - b_hat,
            b_hat,
            bucket_size,
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
            self._aggregate_cgplus(
                pivot_params, weights, max(self._current_step - 1, 0)
            )
        return None

    def _sample_neighbors(self):
        indices_list = list(range(self.nb_honest + self.nb_byz))
        indices_list.remove(self.worker_id)
        return self.neighbor_sampler.sample(indices_list, self.nb_neighbors)

    def observe_neighbors(self, neighbor_indices, neighbor_weights) -> None:
        if not hasattr(self.neighbor_sampler, "update"):
            return None
        pivot_params = self.pull(None)
        rewards = self.reward_strategy.score(pivot_params, neighbor_weights)
        self.neighbor_sampler.update(neighbor_indices, rewards)
        return None

    def _aggregate_cgplus(self, pivot_params, worker_params, current_step):
        worker_params = torch.stack(worker_params)
        differences = worker_params - pivot_params
        distances = differences.norm(dim=1)
        clipping_threshold = (
            torch.topk(distances, 2 * self.b_hat).values[-1]
            if self.b_hat > 0
            else torch.inf
        )
        mask = distances[:, None].broadcast_to(differences.shape) > clipping_threshold
        clipped_differences = torch.where(
            mask, differences * (clipping_threshold / distances[:, None]), differences
        )

        communication_lr = 1 / (current_step // 250 + 1)
        aggregate_params = pivot_params + communication_lr * clipped_differences.sum(
            dim=0
        ) * (1 / self.nb_neighbors)
        self.set_model_parameters(aggregate_params)

    def _aggregate_with_rag(self, pivot_params, worker_params):
        worker_params = list(worker_params)
        worker_params.append(pivot_params)
        aggregate_params = self.robust_aggregator.aggregate(worker_params)
        self.set_model_parameters(aggregate_params)


P2PWorker = DynamicWorker
