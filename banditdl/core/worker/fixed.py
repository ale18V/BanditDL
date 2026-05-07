import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.robustness.summations import cs_plus, gts, cs_he
from banditdl.core.topology.gossip import LaplacianGossipMatrix
from banditdl.core.worker.base import HonestWorker


_METHODS = {"cs+": cs_plus, "cs_he": cs_he, "gts": gts}


class FixedGraphWorker(HonestWorker):
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
        rag,
        b_hat,
        nb_local_steps,
        method,
        comm_graph,
        dissensus,
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
        self.comm_graph = comm_graph
        self.method = method
        self.dissensus = dissensus
        self.rho = 1.0
        self.W = torch.tensor(LaplacianGossipMatrix(self.comm_graph)).to(device)
        self.nb_neighbors = len(list(self.comm_graph.neighbors(self.worker_id)))

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

        metropolis = True
        if metropolis:
            self.byz_weights = 0
            neighbors = list(self.comm_graph.neighbors(self.worker_id))
            neighbors_degrees = sorted([comm_graph.degree(i) for i in neighbors])
            pivot_degree = comm_graph.degree(self.worker_id)
            for i in range(b_hat):
                try:
                    self.byz_weights += 1 / (neighbors_degrees[i] + pivot_degree + 1)
                except Exception:
                    print(
                        f"Warning: b_hat = {b_hat} is too large compared to the number of neighbors of worker {self.worker_id}, "
                        f"which is {len(neighbors)}"
                    )
        else:
            self.byz_weights = b_hat

        self.num_clipped = []

    def aggregate(self, weights) -> None:
        if len(weights) == 0:
            return None
        pivot_params = self.pull(None)
        neighbor_indices = list(self.comm_graph.neighbors(self.worker_id))
        neighbor_indices.append(self.worker_id)
        with torch.no_grad():
            worker_params = torch.stack(weights)
            differences = worker_params - pivot_params

            robust_aggregate, num_clipped = _METHODS[self.method](
                weights=self.comm_graph.weights(self.worker_id)[neighbor_indices].clone().detach().requires_grad_(False),
                gradients=differences,
                byz_weights=self.byz_weights,
            )
            self.num_clipped.append(num_clipped)
            aggregate_params = pivot_params + self.rho * robust_aggregate
            self.set_model_parameters(aggregate_params)
        return None


FixedGraphP2PWorker = FixedGraphWorker
