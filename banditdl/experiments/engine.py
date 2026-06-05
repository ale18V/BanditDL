from __future__ import annotations

import copy
import pathlib
import random
from dataclasses import replace

import numpy as np
import torch

from banditdl.core.sampling import (
    SamplerContext,
    make_neighbor_sampler,
    make_reward_strategy,
)
from banditdl.core.topology.fxgraph import generate_connected_graph
from banditdl.core.topology.graph import CommunicationNetwork
from banditdl.core.worker.byzantine import ByzantineWorker, DecByzantineWorker
from banditdl.core.worker.config import WorkerConfig
from banditdl.core.worker.dynamic import DynamicWorker
from banditdl.core.worker.fixed import FixedGraphWorker
from banditdl.data import dataset
from banditdl.experiments.config_schema import BanditDLConfig
from banditdl.utils.math_utils import consensus_drift, neighbor_disagreement
from banditdl.utils.results import make_result_file, store_result


def _setup_seed(seed: int) -> None:
    reproducible = seed >= 0
    if reproducible:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark = not reproducible


def _progress_interval(rounds: int) -> int:
    return max(1, rounds // 20)


def _should_log_step(current_step: int, rounds: int) -> bool:
    return (
        current_step in (0, rounds) or current_step % _progress_interval(rounds) == 0
    )


def _log_start(mode: str, cfg: BanditDLConfig, result_dir: pathlib.Path) -> None:
    print(
        f"[banditdl] starting {mode} run: "
        f"dataset={cfg.dataset.dataset}, model={cfg.dataset.model}, nodes={cfg.topology.nodes}, "
        f"honest={cfg.nb_honests}, byzantine={cfg.adversary.byzcount}, "
        f"rounds={cfg.effective_rounds}, seed={cfg.seed}, device={cfg.device}",
        flush=True,
    )
    print(f"[banditdl] results: {result_dir}", flush=True)


def _log_progress(mode: str, current_step: int, cfg: BanditDLConfig, accuracy=None, validation_loss=None, train_loss=None) -> None:
    message = f"[banditdl] {mode} round {current_step}/{cfg.effective_rounds}"
    if accuracy is not None:
        message += f" | mean_accuracy={accuracy:.4f}"
    if validation_loss is not None:
        message += f" | val_loss={validation_loss:.4f}"
    if train_loss is not None:
        message += f" | train_loss={train_loss:.4f}"
    print(message, flush=True)


def _log_done(mode: str) -> None:
    print(f"[banditdl] finished {mode} run", flush=True)


def _raise_if_nonfinite_weights(workers, current_step: int, mode: str) -> None:
    for worker in workers:
        weights = worker.pull(None)
        if not torch.isfinite(weights).all():
            raise FloatingPointError(
                f"{mode} produced non-finite weights at round {current_step} "
                f"for worker {worker.worker_id}"
            )


def _record_evaluation(
    workers,
    fd_validation,
    fd_validation_loss,
    fd_train_loss,
    current_step,
    validation_steps,
    validation_accuracies,
    validation_losses,
    train_losses,
):
    accs = [w.compute_validation_accuracy() for w in workers]
    val_losses_round = [w.compute_validation_loss() for w in workers]
    train_losses_round = [w.compute_train_loss() for w in workers]
    mean_val_acc = sum(accs) / len(accs)
    mean_val_loss = sum(val_losses_round) / len(val_losses_round)
    mean_train_loss = sum(train_losses_round) / len(train_losses_round)

    validation_steps.append(current_step)
    validation_accuracies.append(accs)
    validation_losses.append(val_losses_round)
    train_losses.append(train_losses_round)
    store_result(fd_validation, current_step, mean_val_acc)
    store_result(fd_validation_loss, current_step, mean_val_loss)
    store_result(fd_train_loss, current_step, mean_train_loss)

    return mean_val_acc, mean_val_loss, mean_train_loss


def _record_final_evaluation_if_needed(
    cfg: BanditDLConfig,
    workers,
    fd_validation,
    fd_validation_loss,
    fd_train_loss,
    validation_steps,
    validation_accuracies,
    validation_losses,
    train_losses,
):
    evaluation_delta = getattr(getattr(cfg, "evaluation", cfg), "evaluation_delta")
    rounds = cfg.effective_rounds if hasattr(cfg, "effective_rounds") else cfg.rounds
    if evaluation_delta <= 0:
        return None, None, None
    if validation_steps and validation_steps[-1] == rounds:
        return None, None, None
    return _record_evaluation(
        workers,
        fd_validation,
        fd_validation_loss,
        fd_train_loss,
        rounds,
        validation_steps,
        validation_accuracies,
        validation_losses,
        train_losses,
    )


class ResultTracker:
    """Consolidates metrics tracking, evaluation, and saving results."""

    def __init__(self, cfg: BanditDLConfig, result_dir: pathlib.Path, test_loader=None):
        self.cfg = cfg
        self.result_dir = result_dir
        self.test_loader = test_loader
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.fd_validation = (result_dir / "validation").open("w")
        self.fd_validation_worst = (result_dir / "validation_worst").open("w")
        self.fd_validation_loss = (result_dir / "validation_loss").open("w")
        self.fd_train_loss = (result_dir / "train_loss").open("w")

        make_result_file(self.fd_validation, ["Step number", "Cross-accuracy"])
        make_result_file(self.fd_validation_worst, ["Step number", "Cross-accuracy"])
        make_result_file(self.fd_validation_loss, ["Step number", "Cross-loss"])
        make_result_file(self.fd_train_loss, ["Step number", "Cross-loss"])

        self.validation_steps = []
        self.validation_accuracies = []
        self.validation_losses = []
        self.train_losses = []
        self.neighbor_disagreement_history = []
        self.consensus_drift_history = []
        self.gradient_norm_history = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fd_validation.close()
        self.fd_validation_worst.close()
        self.fd_validation_loss.close()
        self.fd_train_loss.close()

    def evaluate_step(self, current_step, workers, mode):
        mean_acc = None
        mean_val_loss = None
        mean_train_loss = None

        if self.cfg.evaluation.evaluation_delta > 0 and current_step % self.cfg.evaluation.evaluation_delta == 0:
            (
                mean_acc,
                mean_val_loss,
                mean_train_loss,
            ) = _record_evaluation(
                workers,
                self.fd_validation,
                self.fd_validation_loss,
                self.fd_train_loss,
                current_step,
                self.validation_steps,
                self.validation_accuracies,
                self.validation_losses,
                self.train_losses,
            )

        if _should_log_step(current_step, self.cfg.effective_rounds):
            _log_progress(
                mode,
                current_step,
                self.cfg,
                mean_acc,
                mean_val_loss,
                mean_train_loss,
            )
        return mean_acc, mean_val_loss, mean_train_loss

    def record_gradient_norms(self, workers):
        self.gradient_norm_history.append(
            np.array([w.last_gradient_norm for w in workers], dtype=float)
        )

    def record_drift(self, disagreement, consensus):
        self.neighbor_disagreement_history.append(disagreement.cpu().numpy())
        self.consensus_drift_history.append(consensus.cpu().numpy())

    def save_snapshot(self):
        np.save(self.result_dir / "validation_accuracies.npy", np.array(self.validation_accuracies))
        np.save(self.result_dir / "validation_losses.npy", np.array(self.validation_losses))
        np.save(self.result_dir / "train_losses.npy", np.array(self.train_losses))
        np.save(self.result_dir / "neighbor_disagreement.npy", np.array(self.neighbor_disagreement_history))
        np.save(self.result_dir / "consensus_drift.npy", np.array(self.consensus_drift_history))
        np.save(self.result_dir / "gradient_norms.npy", np.array(self.gradient_norm_history))

    def finalize(self, workers, mode):
        final_accuracy, final_val_loss, final_train_loss = _record_final_evaluation_if_needed(
            self.cfg,
            workers,
            self.fd_validation,
            self.fd_validation_loss,
            self.fd_train_loss,
            self.validation_steps,
            self.validation_accuracies,
            self.validation_losses,
            self.train_losses,
        )
        if _should_log_step(self.cfg.effective_rounds, self.cfg.effective_rounds) and final_accuracy is not None:
            _log_progress(
                mode,
                self.cfg.effective_rounds,
                self.cfg,
                final_accuracy,
                final_val_loss,
                final_train_loss,
            )

        if self.validation_accuracies:
            worst_idx = min(range(len(workers)), key=lambda i: self.validation_accuracies[-1][i])
            for step, accs in zip(self.validation_steps, self.validation_accuracies, strict=True):
                store_result(self.fd_validation_worst, step, accs[worst_idx])

        if self.cfg.evaluation.evaluate_test and self.test_loader:
            fd_test = (self.result_dir / "test").open("w")
            make_result_file(fd_test, ["Step number", "Cross-accuracy"])
            test_accuracies = [w.compute_accuracy_on_loader(self.test_loader) for w in workers]
            store_result(fd_test, self.cfg.effective_rounds, sum(test_accuracies) / len(test_accuracies))
            fd_test.close()

        self.save_snapshot()

        with torch.no_grad():
            final_weights = torch.stack([w.pull(None) for w in workers])
            pairwise_distance_final = torch.cdist(final_weights, final_weights).cpu().numpy()
        np.save(self.result_dir / "pairwise_model_distance_final.npy", pairwise_distance_final)
        _log_done(mode)


def _build_worker_config(cfg: BanditDLConfig, device: str) -> WorkerConfig:
    return WorkerConfig(
        model=cfg.dataset.model,
        learning_rate=cfg.optimization.learning_rate,
        learning_rate_decay=cfg.optimization.learning_rate_decay,
        learning_rate_decay_delta=cfg.optimization.learning_rate_decay_delta,
        weight_decay=cfg.optimization.weight_decay,
        loss=cfg.optimization.loss,
        momentum=cfg.optimization.momentum_worker,
        device=device,
        nb_local_steps=cfg.optimization.nb_local_steps,
        nb_workers=cfg.topology.nodes,
        nb_byz=cfg.adversary.byzcount,
        nb_real_byz=cfg.adversary.byzcount,
        b_hat=cfg.adversary.byzantine_budget if cfg.adversary.byzantine_budget is not None else cfg.adversary.byzcount,
        attack=cfg.adversary.attack,
        rag=cfg.aggregator.rag or cfg.topology.sampling is not None,
        numb_labels=cfg.dataset.numb_labels,
        labelflipping=cfg.adversary.attack == "LF",
        gradient_clip=None,
        server_clip=cfg.aggregator.server_clip,
        bucket_size=cfg.aggregator.bucket_size,
        aggregator=cfg.aggregator.aggregator,
        pre_aggregator=cfg.aggregator.pre_aggregator,
        nb_neighbors=cfg.topology.degree,
        sampling_ratio=cfg.topology.sampling,
        mimic_learning_phase=cfg.adversary.mimic_learning_phase,
        method=cfg.topology.method or cfg.topology.neighbor_sampler,
        epsilon=1.0,
    )


def _init_workers(cfg: BanditDLConfig, train_loader_dict, local_test_loader_dict, device: str, comm_graph=None, dissensus=False):
    workers = []
    base_config = _build_worker_config(cfg, device)

    if comm_graph is not None:
        for worker_id in range(cfg.nb_honests):
            config = replace(base_config, comm_graph=comm_graph, dissensus=dissensus)
            w = FixedGraphWorker(worker_id, train_loader_dict[worker_id], local_test_loader_dict[worker_id], config)
            if worker_id > 0:
                w.model.load_state_dict(workers[0].model.state_dict())
            workers.append(w)
    else:
        for worker_id in range(cfg.nb_honests):
            sampler_params = dict(cfg.sampler.get("params", {}) if cfg.sampler else {})
            sampler_context = SamplerContext(
                worker_id=worker_id,
                nodes=cfg.topology.nodes,
                k=base_config.nb_neighbors or 1,
                horizon=cfg.effective_rounds + 1,
                seed=cfg.seed + worker_id,
            )
            neighbor_sampler = make_neighbor_sampler(cfg.resolved_sampler_name, context=sampler_context, params=sampler_params)
            reward_strategy = make_reward_strategy(cfg.topology.bandit_reward)

            config = replace(base_config, neighbor_sampler=neighbor_sampler, reward_strategy=reward_strategy)
            w = DynamicWorker(worker_id, train_loader_dict[worker_id], local_test_loader_dict[worker_id], config)
            if worker_id > 0:
                w.model.load_state_dict(workers[0].model.state_dict())
            workers.append(w)

    return workers


def _best_fixed_subset(scores, worker_id: int, k: int):
    scores = np.asarray(scores, dtype=float)
    candidates = [i for i in range(len(scores)) if i != worker_id]
    selected = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
    reward = 0.0 if not selected else float(scores[selected].sum() / len(selected))
    return np.array(selected, dtype=int), reward


def _mean_selected_reward(rewards) -> float:
    rewards = list(rewards)
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


def _save_dynamic_metrics(
    result_dir: pathlib.Path,
    *,
    algorithm_reward_history,
    oracle_reward_history,
    reward_min_history,
    reward_max_history,
    selected_neighbor_history,
    oracle_neighbor_history,
    sampler_probability_history,
) -> None:
    algorithm_rewards = np.array(algorithm_reward_history)
    oracle_rewards = np.array(oracle_reward_history)
    np.save(result_dir / "reward_algorithm.npy", algorithm_rewards)
    np.save(result_dir / "reward_oracle.npy", oracle_rewards)
    np.save(result_dir / "regret.npy", oracle_rewards - algorithm_rewards)
    np.save(result_dir / "reward_selected_min.npy", np.array(reward_min_history))
    np.save(result_dir / "reward_selected_max.npy", np.array(reward_max_history))
    np.save(result_dir / "selected_neighbors.npy", np.array(selected_neighbor_history, dtype=int))
    np.save(result_dir / "oracle_neighbors.npy", np.array(oracle_neighbor_history, dtype=int))
    np.save(result_dir / "sampler_probabilities.npy", np.array(sampler_probability_history))


def _dynamic_candidate_weights(w, honest_weights, byz_workers, current_step):
    candidate_weights = {
        worker_id: weight
        for worker_id, weight in enumerate(honest_weights)
        if worker_id != w.worker_id
    }
    context = {"honest_weights": honest_weights, "step": current_step}
    for byz_worker in byz_workers:
        weight = copy.deepcopy(byz_worker).pull(context)
        if weight is not None:
            candidate_weights[byz_worker.worker_id] = weight
    return candidate_weights


def _full_sampler_probability_vector(worker, nb_total: int) -> np.ndarray:
    population = [i for i in range(nb_total) if i != worker.worker_id]
    probabilities_by_arm = worker.neighbor_sampler.probabilities(
        population,
        worker.nb_neighbors,
    )
    row = np.zeros(nb_total, dtype=float)
    for arm in population:
        row[arm] = float(probabilities_by_arm[arm])
    return row


def _run_experiment(cfg: BanditDLConfig, result_dir: pathlib.Path, seed: int, device: str, mode: str) -> None:
    _setup_seed(seed)
    _log_start(mode, cfg, result_dir)

    train_loader_dict, local_test_loader_dict, test_loader = dataset.make_train_validation_test_datasets(
        cfg.dataset.dataset,
        heterogeneity=cfg.heterogeneity.method != "iid", # Simplified hetero check
        numb_labels=cfg.dataset.numb_labels,
        alpha_dirichlet=cfg.heterogeneity.alpha,
        distinct_datasets=cfg.dataset.mode == "writer_per_node",
        nb_datapoints=None, # Could be added to config if needed
        honest_workers=cfg.nb_honests,
        train_batch=cfg.optimization.batch_size,
        test_batch=100, # Batch size test
        global_test_ratio=cfg.evaluation.global_test_ratio,
        local_test_ratio=cfg.evaluation.local_test_ratio,
        split_seed=cfg.evaluation.split_seed,
        partition_method=cfg.heterogeneity.method,
        partition_style=cfg.heterogeneity.partition,
        classes_per_worker=cfg.heterogeneity.classes_per_worker,
        nb_shards=cfg.heterogeneity.nb_shards,
        shards_per_worker=cfg.heterogeneity.shards_per_worker,
        nb_groups=cfg.heterogeneity.nb_groups,
        classes_per_group=cfg.heterogeneity.classes_per_group,
        group_overlap=cfg.heterogeneity.group_overlap,
        dataset_mode=cfg.dataset.mode,
        nb_writers_limit=cfg.dataset.nb_writers_limit,
    )

    comm_graph = None
    adjacency_honest = None
    if mode == "fixed":
        nb_edges = cfg.topology.nodes * cfg.topology.degree // 2
        g = generate_connected_graph(cfg.topology.nodes, nb_edges, seed=seed)
        comm_graph = CommunicationNetwork(g, weights_method="metropolis", device=device if device != "auto" else "cpu")
        adjacency_honest = torch.as_tensor(np.asarray(comm_graph.adjacency_matrix[: cfg.nb_honests, : cfg.nb_honests]), dtype=torch.float32, device=device)

    workers = _init_workers(cfg, train_loader_dict, local_test_loader_dict, device, comm_graph=comm_graph, dissensus=(cfg.adversary.attack == "dissensus"))
    base_worker_config = _build_worker_config(cfg, device)
    # Correctly inject graph for fixed mode workers
    if mode == "fixed":
        base_worker_config = replace(base_worker_config, comm_graph=comm_graph)

    byz_workers = [ByzantineWorker(i, workers[0].model_size, base_worker_config) for i in range(cfg.nb_honests, cfg.topology.nodes)]

    byz_workers_by_id = {byz.worker_id: byz for byz in byz_workers}
    dec_byz_workers = {i: DecByzantineWorker(i, cfg.nb_honests, base_worker_config) for i in range(cfg.nb_honests, cfg.topology.nodes)} if mode == "fixed" else {}

    with ResultTracker(cfg, result_dir, test_loader) as tracker:
        cumulative_arm_rewards = np.zeros((cfg.nb_honests, cfg.topology.nodes))
        cumulative_algorithm_rewards = np.zeros(cfg.nb_honests)
        algorithm_reward_history, oracle_reward_history, selected_neighbor_history, oracle_neighbor_history = [], [], [], []
        sampler_probability_history = []
        reward_min_history, reward_max_history = [], []

        for current_step in range(cfg.effective_rounds + 1):
            tracker.evaluate_step(current_step, workers, mode)

            if current_step < cfg.effective_rounds:
                for w in workers:
                    w.train()
                tracker.record_gradient_norms(workers)
                honest_weights = [w.pull(None) for w in workers]

                if mode == "dynamic":
                    selected_round = np.full((cfg.nb_honests, workers[0].nb_neighbors), -1, dtype=int)
                    probability_round = np.stack(
                        [_full_sampler_probability_vector(w, cfg.topology.nodes) for w in workers],
                    )
                    reward_min_round, reward_max_round = np.full(cfg.nb_honests, np.nan), np.full(cfg.nb_honests, np.nan)

                    for w in workers:
                        neighbor_indices = w._sample_neighbors()
                        candidate_weights = _dynamic_candidate_weights(w, honest_weights, byz_workers, current_step)
                        selected_ids = [i for i in neighbor_indices if i in candidate_weights]

                        for nid in selected_ids:
                            if nid >= cfg.nb_honests:
                                weight = byz_workers_by_id[nid].pull({"honest_weights": honest_weights, "step": current_step})
                                if weight is not None:
                                    candidate_weights[nid] = weight

                        candidate_ids = list(candidate_weights)
                        candidate_rewards = w.reward_strategy.score(w.pull(None), [candidate_weights[i] for i in candidate_ids])
                        rewards_by_id = dict(zip(candidate_ids, candidate_rewards, strict=True))
                        cumulative_arm_rewards[w.worker_id, candidate_ids] += candidate_rewards

                        neighbor_weights = [candidate_weights[i] for i in selected_ids]
                        selected_round[w.worker_id, :len(selected_ids)] = selected_ids
                        selected_rewards = [rewards_by_id[i] for i in selected_ids]
                        if selected_rewards:
                            reward_min_round[w.worker_id], reward_max_round[w.worker_id] = min(selected_rewards), max(selected_rewards)
                            cumulative_algorithm_rewards[w.worker_id] += _mean_selected_reward(selected_rewards)

                        w.num_selected_byz.append(len([i for i in selected_ids if i >= cfg.nb_honests]))
                        w.observe_neighbors(selected_ids, neighbor_weights)
                        w.aggregate(neighbor_weights)

                    sampler_probability_history.append(probability_round)
                    reward_min_history.append(reward_min_round)
                    reward_max_history.append(reward_max_round)

                    oracle_neighbors_round, oracle_rewards_round = [], []
                    for w in workers:
                        oids, oreward = _best_fixed_subset(cumulative_arm_rewards[w.worker_id], worker_id=w.worker_id, k=w.nb_neighbors)
                        oracle_neighbors_round.append(oids)
                        oracle_rewards_round.append(oreward)
                    algorithm_reward_history.append(cumulative_algorithm_rewards.copy())
                    oracle_reward_history.append(np.array(oracle_rewards_round))
                    selected_neighbor_history.append(selected_round)
                    oracle_neighbor_history.append(np.stack(oracle_neighbors_round))

                    with torch.no_grad():
                        updated = [w.pull(None) for w in workers]
                        neighbor_matrix = selected_round.copy()
                        neighbor_matrix[neighbor_matrix >= cfg.nb_honests] = -1
                        tracker.record_drift(neighbor_disagreement(updated, neighbor_indices=neighbor_matrix.tolist()), consensus_drift(updated))
                    _save_dynamic_metrics(
                        result_dir,
                        algorithm_reward_history=algorithm_reward_history,
                        oracle_reward_history=oracle_reward_history,
                        reward_min_history=reward_min_history,
                        reward_max_history=reward_max_history,
                        selected_neighbor_history=selected_neighbor_history,
                        oracle_neighbor_history=oracle_neighbor_history,
                        sampler_probability_history=sampler_probability_history,
                    )
                else:
                    # Fixed mode
                    for w in workers:
                        neighbors = [*list(w.comm_graph.neighbors(w.worker_id)), w.worker_id]
                        honest_nids = [i for i in neighbors if i < cfg.nb_honests]
                        byz_nids = [i for i in neighbors if i >= cfg.nb_honests]
                        w.num_selected_byz.append(len(byz_nids))
                        h_weights = [honest_weights[i] for i in honest_nids]
                        if cfg.adversary.attack == "dissensus":
                            b_weights = [dec_byz_workers[i].pull({"target": w.worker_id, "honest_neighbors": honest_nids, "pivot_params": w.pull(None), "honest_local_params": h_weights}) for i in byz_nids]
                        else:
                            b_weights = [byz_workers_by_id[i].pull({"honest_weights": honest_weights, "step": current_step}) for i in byz_nids]
                        w.aggregate(h_weights + b_weights)

                    with torch.no_grad():
                        updated = [w.pull(None) for w in workers]
                        tracker.record_drift(neighbor_disagreement(updated, adjacency=adjacency_honest), consensus_drift(updated))

                tracker.save_snapshot()
                _raise_if_nonfinite_weights(workers, current_step, mode)

        tracker.finalize(workers, mode)
    if mode == "dynamic":
        _save_dynamic_metrics(
            result_dir,
            algorithm_reward_history=algorithm_reward_history,
            oracle_reward_history=oracle_reward_history,
            reward_min_history=reward_min_history,
            reward_max_history=reward_max_history,
            selected_neighbor_history=selected_neighbor_history,
            oracle_neighbor_history=oracle_neighbor_history,
            sampler_probability_history=sampler_probability_history,
        )


def run_dynamic(cfg: BanditDLConfig, result_dir: pathlib.Path, seed: int, device: str) -> None:
    _run_experiment(cfg, result_dir, seed, device, "dynamic")


def run_fixed(cfg: BanditDLConfig, result_dir: pathlib.Path, seed: int, device: str) -> None:
    _run_experiment(cfg, result_dir, seed, device, "fixed")
