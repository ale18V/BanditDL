from __future__ import annotations

import copy
import os
import pathlib
import random
from types import SimpleNamespace

import numpy as np
import torch

from banditdl.utils.math_utils import consensus_drift, neighbor_disagreement
from banditdl.utils.results import make_result_file, store_result
from banditdl.core.sampling import (
    SamplerContext,
    make_neighbor_sampler,
    make_reward_strategy,
)
from banditdl.core.topology.fxgraph import generate_connected_graph
from banditdl.core.topology.graph import CommunicationNetwork
from banditdl.core.worker.byzantine import ByzantineWorker, DecByzantineWorker
from banditdl.core.worker.dynamic import DynamicWorker
from banditdl.core.worker.fixed import FixedGraphWorker
from banditdl.data import dataset


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
        current_step == 0
        or current_step == rounds
        or current_step % _progress_interval(rounds) == 0
    )


def _log_start(mode: str, args, result_dir: pathlib.Path) -> None:
    print(
        f"[banditdl] starting {mode} run: "
        f"dataset={args.dataset}, model={args.model}, nodes={args.nb_workers}, "
        f"honest={args.nb_honests}, byzantine={args.nb_real_byz}, "
        f"rounds={args.rounds}, seed={args.seed}, device={args.device}",
        flush=True,
    )
    print(f"[banditdl] results: {result_dir}", flush=True)


def _log_progress(mode: str, current_step: int, args, accuracy=None, validation_loss=None, train_loss=None) -> None:
    message = f"[banditdl] {mode} round {current_step}/{args.rounds}"
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
    validation_losses_round = [w.compute_validation_loss() for w in workers]
    train_losses_round = [w.compute_train_loss() for w in workers]
    mean_validation_accuracy = sum(accs) / len(accs)
    mean_validation_loss = sum(validation_losses_round) / len(validation_losses_round)
    mean_train_loss = sum(train_losses_round) / len(train_losses_round)

    validation_steps.append(current_step)
    validation_accuracies.append(accs)
    validation_losses.append(validation_losses_round)
    train_losses.append(train_losses_round)
    store_result(fd_validation, current_step, mean_validation_accuracy)
    store_result(fd_validation_loss, current_step, mean_validation_loss)
    store_result(fd_train_loss, current_step, mean_train_loss)

    return mean_validation_accuracy, mean_validation_loss, mean_train_loss


def _record_final_evaluation_if_needed(
    args,
    workers,
    fd_validation,
    fd_validation_loss,
    fd_train_loss,
    validation_steps,
    validation_accuracies,
    validation_losses,
    train_losses,
):
    if args.evaluation_delta <= 0:
        return None, None, None
    if validation_steps and validation_steps[-1] == args.rounds:
        return None, None, None
    return _record_evaluation(
        workers,
        fd_validation,
        fd_validation_loss,
        fd_train_loss,
        args.rounds,
        validation_steps,
        validation_accuracies,
        validation_losses,
        train_losses,
    )


def _make_args(
    params: dict, result_dir: pathlib.Path, seed: int, device: str
) -> SimpleNamespace:
    args = dict(params)
    args.setdefault("hetero", False)
    args.setdefault("distinct-data", False)
    args.setdefault("dirichlet-alpha", None)
    args.setdefault("nb-datapoints", None)
    args.setdefault("numb-labels", None)
    args.setdefault("batch-size-test", 100)
    args.setdefault("loss", "NLLLoss")
    args.setdefault("weight-decay", 0)
    args.setdefault("momentum-worker", 0.99)
    args.setdefault("bucket-size", 1)
    args.setdefault("pre-aggregator", None)
    args.setdefault("aggregator", "average")
    args.setdefault("learning-rate", 0.5)
    args.setdefault("learning-rate-decay", 5000)
    args.setdefault("learning-rate-decay-delta", 1)
    args.setdefault("mimic-learning-phase", None)
    args.setdefault("gradient-clip", None)
    args.setdefault("server-clip", False)
    args.setdefault("rag", False)
    args.setdefault("method", "cs+")
    args.setdefault("attack", None)
    args.setdefault("neighbor-sampler", "uniform")
    args.setdefault("sampler-params", {})
    args.setdefault("sampler-reward", "parameter_distance")
    args.setdefault("bandit-epsilon", 0.1)
    args.setdefault("bandit-initial-value", 0.0)
    args.setdefault("bandit-reward", "parameter_distance")
    args.setdefault("validation-ratio", 0.5)
    args.setdefault("eval-split-seed", 0)
    args.setdefault("evaluate-test", False)
    args["result-directory"] = str(result_dir)
    args["seed"] = seed
    args["device"] = device
    # normalize dashed keys for existing code style
    normalized = {k.replace("-", "_"): v for k, v in args.items()}
    if "rounds" not in normalized and "nb_steps" in normalized:
        normalized["rounds"] = normalized["nb_steps"]
    normalized["nb_honests"] = normalized["nb_workers"] - normalized["nb_real_byz"]
    return SimpleNamespace(**normalized)


def _init_workers_dynamic(args, train_loader_dict, validation_loader):
    workers = []
    for worker_id in range(args.nb_honests):
        sampler_params = dict(args.sampler_params or {})
        if args.neighbor_sampler in {"bandit", "epsilon_greedy"}:
            sampler_params.setdefault("epsilon", args.bandit_epsilon)
            sampler_params.setdefault("initial_value", args.bandit_initial_value)
        sampler_context = SamplerContext(
            worker_id=worker_id,
            nodes=args.nb_workers,
            k=args.nb_neighbors,
            horizon=args.rounds + 1,
            seed=args.seed + worker_id,
        )
        neighbor_sampler = make_neighbor_sampler(
            args.neighbor_sampler,
            context=sampler_context,
            params=sampler_params,
        )
        reward_strategy = make_reward_strategy(args.sampler_reward)
        w = DynamicWorker(
            worker_id,
            train_loader_dict[worker_id],
            validation_loader,
            args.nb_workers,
            args.nb_decl_byz,
            args.nb_real_byz,
            args.aggregator,
            args.pre_aggregator,
            args.server_clip,
            args.bucket_size,
            args.model,
            args.learning_rate,
            args.learning_rate_decay,
            args.learning_rate_decay_delta,
            args.weight_decay,
            args.loss,
            args.momentum_worker,
            args.device,
            args.attack == "LF",
            args.gradient_clip,
            args.numb_labels,
            args.nb_neighbors,
            getattr(args, "sampling_ratio", None),
            args.rag,
            args.b_hat,
            args.nb_local_steps,
            neighbor_sampler=neighbor_sampler,
            reward_strategy=reward_strategy,
        )
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


def _sampler_probability_stats(worker) -> tuple[float, float, float]:
    population = list(range(worker.nb_honest + worker.nb_byz))
    population.remove(worker.worker_id)
    probabilities_by_arm = worker.neighbor_sampler.probabilities(
        population,
        worker.nb_neighbors,
    )
    probabilities = np.array(
        [probabilities_by_arm[arm] for arm in population],
        dtype=float,
    )
    uniform_probability = 1.0 / len(population)
    kl_to_uniform = float(
        np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12) / uniform_probability))
    )
    return kl_to_uniform, float(probabilities.min()), float(probabilities.max())


def run_dynamic(params: dict, result_dir: pathlib.Path, seed: int, device: str) -> None:
    args = _make_args(params, result_dir, seed, device)
    _setup_seed(args.seed)
    _log_start("dynamic", args, result_dir)

    train_loader_dict, validation_loader, test_loader = dataset.make_train_validation_test_datasets(
        args.dataset,
        heterogeneity=args.hetero,
        numb_labels=args.numb_labels,
        alpha_dirichlet=args.dirichlet_alpha,
        distinct_datasets=args.distinct_data,
        nb_datapoints=args.nb_datapoints,
        honest_workers=args.nb_honests,
        train_batch=args.batch_size,
        test_batch=args.batch_size_test,
        validation_ratio=args.validation_ratio,
        split_seed=args.eval_split_seed,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    workers = _init_workers_dynamic(args, train_loader_dict, validation_loader)

    byz_workers = [
        ByzantineWorker(
            worker_id=i,
            nb_workers=args.nb_workers,
            nb_decl_byz=args.nb_decl_byz,
            nb_real_byz=args.nb_real_byz,
            attack=args.attack,
            aggregator=args.aggregator,
            second_aggregator=args.pre_aggregator,
            server_clip=args.server_clip,
            bucket_size=args.bucket_size,
            model_size=workers[0].model_size,
            mimic_learning_phase=args.mimic_learning_phase,
            gradient_clip=args.gradient_clip,
            device=args.device,
        )
        for i in range(args.nb_honests, args.nb_workers)
    ]
    byz_workers_by_id = {byz.worker_id: byz for byz in byz_workers}

    fd_validation = (result_dir / "validation").open("w")
    fd_validation_worst = (result_dir / "validation_worst").open("w")
    fd_validation_loss = (result_dir / "validation_loss").open("w")
    fd_train_loss = (result_dir / "train_loss").open("w")
    make_result_file(fd_validation, ["Step number", "Cross-accuracy"])
    make_result_file(fd_validation_worst, ["Step number", "Cross-accuracy"])
    make_result_file(fd_validation_loss, ["Step number", "Cross-loss"])
    make_result_file(fd_train_loss, ["Step number", "Cross-loss"])

    validation_steps = []
    validation_accuracies = []
    validation_losses = []
    train_losses = []
    cumulative_arm_rewards = np.zeros((args.nb_honests, args.nb_workers))
    cumulative_algorithm_rewards = np.zeros(args.nb_honests)
    algorithm_reward_history = []
    oracle_reward_history = []
    selected_neighbor_history = []
    oracle_neighbor_history = []
    neighbor_disagreement_history = []
    consensus_drift_history = []
    sampler_kl_history = []
    sampler_min_probability_history = []
    sampler_max_probability_history = []
    selected_reward_min_history = []
    selected_reward_max_history = []

    for current_step in range(args.rounds + 1):
        mean_validation_accuracy = None
        mean_validation_loss = None
        mean_train_loss = None
        if args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0:
            (
                mean_validation_accuracy,
                mean_validation_loss,
                mean_train_loss,
            ) = _record_evaluation(
                workers,
                fd_validation,
                fd_validation_loss,
                fd_train_loss,
                current_step,
                validation_steps,
                validation_accuracies,
                validation_losses,
                train_losses,
            )

        if _should_log_step(current_step, args.rounds):
            _log_progress(
                "dynamic",
                current_step,
                args,
                mean_validation_accuracy,
                mean_validation_loss,
                mean_train_loss,
            )

        for w in workers:
            w.train()
        honest_weights = [w.pull(None) for w in workers]

        selected_round = np.full(
            (args.nb_honests, workers[0].nb_neighbors), -1, dtype=int
        )
        sampler_kl_round = np.zeros(args.nb_honests)
        sampler_min_probability_round = np.zeros(args.nb_honests)
        sampler_max_probability_round = np.zeros(args.nb_honests)
        selected_reward_min_round = np.full(args.nb_honests, np.nan)
        selected_reward_max_round = np.full(args.nb_honests, np.nan)
        for w in workers:
            (
                sampler_kl_round[w.worker_id],
                sampler_min_probability_round[w.worker_id],
                sampler_max_probability_round[w.worker_id],
            ) = _sampler_probability_stats(w)
            neighbor_indices = w._sample_neighbors()
            candidate_weights = _dynamic_candidate_weights(
                w, honest_weights, byz_workers, current_step
            )
            selected_neighbor_ids = [
                i for i in neighbor_indices if i in candidate_weights
            ]
            for neighbor_id in selected_neighbor_ids:
                if neighbor_id >= args.nb_honests:
                    weight = byz_workers_by_id[neighbor_id].pull(
                        {"honest_weights": honest_weights, "step": current_step}
                    )
                    if weight is not None:
                        candidate_weights[neighbor_id] = weight
            candidate_ids = list(candidate_weights)
            candidate_values = [candidate_weights[i] for i in candidate_ids]
            candidate_rewards = w.reward_strategy.score(w.pull(None), candidate_values)
            rewards_by_id = dict(zip(candidate_ids, candidate_rewards, strict=True))
            cumulative_arm_rewards[w.worker_id, candidate_ids] += candidate_rewards

            neighbor_weights = [candidate_weights[i] for i in selected_neighbor_ids]
            byz_neighbor_ids = [
                i for i in selected_neighbor_ids if i >= args.nb_honests
            ]
            selected_round[w.worker_id, : len(selected_neighbor_ids)] = (
                selected_neighbor_ids
            )
            selected_rewards = [rewards_by_id[i] for i in selected_neighbor_ids]
            if selected_rewards:
                selected_reward_min_round[w.worker_id] = min(selected_rewards)
                selected_reward_max_round[w.worker_id] = max(selected_rewards)
                cumulative_algorithm_rewards[w.worker_id] += _mean_selected_reward(
                    selected_rewards
                )
            w.num_selected_byz.append(len(byz_neighbor_ids))
            w.observe_neighbors(selected_neighbor_ids, neighbor_weights)
            w.aggregate(neighbor_weights)

        with torch.no_grad():
            updated_weights = [w.pull(None) for w in workers]
            _raise_if_nonfinite_weights(workers, current_step, "dynamic")
            neighbor_matrix = selected_round.copy()
            neighbor_matrix[neighbor_matrix >= args.nb_honests] = -1
            disagreement = neighbor_disagreement(
                updated_weights, neighbor_indices=neighbor_matrix
            )
            consensus = consensus_drift(updated_weights)
        neighbor_disagreement_history.append(disagreement.cpu().numpy())
        consensus_drift_history.append(consensus.cpu().numpy())
        sampler_kl_history.append(sampler_kl_round)
        sampler_min_probability_history.append(sampler_min_probability_round)
        sampler_max_probability_history.append(sampler_max_probability_round)
        selected_reward_min_history.append(selected_reward_min_round)
        selected_reward_max_history.append(selected_reward_max_round)

        oracle_neighbors = []
        oracle_rewards_round = []
        for w in workers:
            oracle_ids, oracle_reward = _best_fixed_subset(
                cumulative_arm_rewards[w.worker_id],
                worker_id=w.worker_id,
                k=w.nb_neighbors,
            )
            oracle_neighbors.append(oracle_ids)
            oracle_rewards_round.append(oracle_reward)

        algorithm_reward_history.append(cumulative_algorithm_rewards.copy())
        oracle_reward_history.append(np.array(oracle_rewards_round))
        selected_neighbor_history.append(selected_round)
        oracle_neighbor_history.append(np.stack(oracle_neighbors))

    final_accuracy, final_validation_loss, final_train_loss = (
        _record_final_evaluation_if_needed(
            args,
            workers,
            fd_validation,
            fd_validation_loss,
            fd_train_loss,
            validation_steps,
            validation_accuracies,
            validation_losses,
            train_losses,
        )
    )
    if _should_log_step(args.rounds, args.rounds):
        _log_progress(
            "dynamic",
            args.rounds,
            args,
            final_accuracy,
            final_validation_loss,
            final_train_loss,
        )

    if validation_accuracies:
        worst_idx = min(range(len(workers)), key=lambda i: validation_accuracies[-1][i])
        for step, accs in zip(validation_steps, validation_accuracies, strict=True):
            store_result(fd_validation_worst, step, accs[worst_idx])

    if args.evaluate_test:
        fd_test = (result_dir / "test").open("w")
        make_result_file(fd_test, ["Step number", "Cross-accuracy"])
        test_accuracies = [w.compute_accuracy_on_loader(test_loader) for w in workers]
        store_result(fd_test, args.rounds, sum(test_accuracies) / len(test_accuracies))
        fd_test.close()

    fd_validation.close()
    fd_validation_worst.close()
    fd_validation_loss.close()
    fd_train_loss.close()

    algorithm_rewards = np.array(algorithm_reward_history)
    oracle_rewards = np.array(oracle_reward_history)
    regret = oracle_rewards - algorithm_rewards

    np.save(os.path.join(result_dir, "validation_accuracies.npy"), np.array(validation_accuracies))
    np.save(os.path.join(result_dir, "accuracies.npy"), np.array(validation_accuracies))
    np.save(os.path.join(result_dir, "validation_losses.npy"), np.array(validation_losses))
    np.save(os.path.join(result_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(result_dir, "reward_algorithm.npy"), algorithm_rewards)
    np.save(os.path.join(result_dir, "reward_oracle.npy"), oracle_rewards)
    np.save(os.path.join(result_dir, "regret.npy"), regret)
    np.save(
        os.path.join(result_dir, "reward_selected_min.npy"),
        np.array(selected_reward_min_history),
    )
    np.save(
        os.path.join(result_dir, "reward_selected_max.npy"),
        np.array(selected_reward_max_history),
    )
    np.save(
        os.path.join(result_dir, "selected_neighbors.npy"),
        np.array(selected_neighbor_history, dtype=int),
    )
    np.save(
        os.path.join(result_dir, "oracle_neighbors.npy"),
        np.array(oracle_neighbor_history, dtype=int),
    )
    np.save(
        os.path.join(result_dir, "neighbor_disagreement.npy"),
        np.array(neighbor_disagreement_history),
    )
    np.save(
        os.path.join(result_dir, "consensus_drift.npy"),
        np.array(consensus_drift_history),
    )
    np.save(
        os.path.join(result_dir, "sampler_kl_to_uniform.npy"),
        np.array(sampler_kl_history),
    )
    np.save(
        os.path.join(result_dir, "sampler_min_probability.npy"),
        np.array(sampler_min_probability_history),
    )
    np.save(
        os.path.join(result_dir, "sampler_max_probability.npy"),
        np.array(sampler_max_probability_history),
    )
    _log_done("dynamic")


def run_fixed(params: dict, result_dir: pathlib.Path, seed: int, device: str) -> None:
    args = _make_args(params, result_dir, seed, device)
    _setup_seed(args.seed)
    _log_start("fixed", args, result_dir)

    train_loader_dict, validation_loader, test_loader = dataset.make_train_validation_test_datasets(
        args.dataset,
        heterogeneity=args.hetero,
        numb_labels=args.numb_labels,
        alpha_dirichlet=args.dirichlet_alpha,
        distinct_datasets=args.distinct_data,
        nb_datapoints=args.nb_datapoints,
        honest_workers=args.nb_honests,
        train_batch=args.batch_size,
        test_batch=args.batch_size_test,
        validation_ratio=args.validation_ratio,
        split_seed=args.eval_split_seed,
    )

    nb_edges = args.nb_workers * args.nb_neighbors // 2
    g = generate_connected_graph(args.nb_workers, nb_edges, seed=args.seed)
    comm_graph = CommunicationNetwork(
        g,
        weights_method="metropolis",
        device=args.device if args.device != "auto" else "cpu",
    )
    dissensus = args.attack == "dissensus"
    adjacency_honest = torch.as_tensor(
        np.asarray(
            comm_graph.adjacency_matrix[: args.nb_honests, : args.nb_honests]
        ),
        dtype=torch.float32,
        device=args.device,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    workers: list[FixedGraphWorker] = []
    for worker_id in range(args.nb_honests):
        w = FixedGraphWorker(
            worker_id,
            train_loader_dict[worker_id],
            validation_loader,
            args.nb_workers,
            args.nb_decl_byz,
            args.nb_real_byz,
            args.aggregator,
            args.pre_aggregator,
            args.server_clip,
            args.bucket_size,
            args.model,
            args.learning_rate,
            args.learning_rate_decay,
            args.learning_rate_decay_delta,
            args.weight_decay,
            args.loss,
            args.momentum_worker,
            args.device,
            args.attack == "LF",
            args.gradient_clip,
            args.numb_labels,
            args.nb_neighbors,
            args.rag,
            args.b_hat,
            args.nb_local_steps,
            args.method,
            comm_graph,
            dissensus,
        )
        if worker_id > 0:
            w.model.load_state_dict(workers[0].model.state_dict())
        workers.append(w)

    byz_workers = {
        i: ByzantineWorker(
            i,
            args.nb_workers,
            args.nb_decl_byz,
            args.nb_real_byz,
            args.attack,
            args.aggregator,
            args.pre_aggregator,
            args.server_clip,
            args.bucket_size,
            workers[0].model_size,
            args.mimic_learning_phase,
            args.gradient_clip,
            args.device,
        )
        for i in range(args.nb_honests, args.nb_workers)
    }
    dec_byz_workers = {
        i: DecByzantineWorker(i, args.nb_honests, comm_graph, args.device)
        for i in range(args.nb_honests, args.nb_workers)
    }

    fd_validation = (result_dir / "validation").open("w")
    fd_validation_worst = (result_dir / "validation_worst").open("w")
    fd_validation_loss = (result_dir / "validation_loss").open("w")
    fd_train_loss = (result_dir / "train_loss").open("w")
    make_result_file(fd_validation, ["Step number", "Cross-accuracy"])
    make_result_file(fd_validation_worst, ["Step number", "Cross-accuracy"])
    make_result_file(fd_validation_loss, ["Step number", "Cross-loss"])
    make_result_file(fd_train_loss, ["Step number", "Cross-loss"])

    validation_steps = []
    validation_accuracies = []
    validation_losses = []
    train_losses = []
    neighbor_disagreement_history = []
    consensus_drift_history = []
    for current_step in range(args.rounds + 1):
        mean_validation_accuracy = None
        mean_validation_loss = None
        mean_train_loss = None
        if args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0:
            (
                mean_validation_accuracy,
                mean_validation_loss,
                mean_train_loss,
            ) = _record_evaluation(
                workers,
                fd_validation,
                fd_validation_loss,
                fd_train_loss,
                current_step,
                validation_steps,
                validation_accuracies,
                validation_losses,
                train_losses,
            )

        if _should_log_step(current_step, args.rounds):
            _log_progress(
                "fixed",
                current_step,
                args,
                mean_validation_accuracy,
                mean_validation_loss,
                mean_train_loss,
            )

        for w in workers:
            w.train()
        honest_weights = [w.pull(None) for w in workers]

        for w in workers:
            neighbors = list(w.comm_graph.neighbors(w.worker_id)) + [w.worker_id]
            honest_neighbors = [i for i in neighbors if i < args.nb_honests]
            byz_neighbors = [i for i in neighbors if i >= args.nb_honests]
            w.num_selected_byz.append(len(byz_neighbors))
            honest_neighbor_weights = [honest_weights[i] for i in honest_neighbors]
            if dissensus:
                byz_weights = [
                    dec_byz_workers[i].pull(
                        {
                            "target": w.worker_id,
                            "honest_neighbors": honest_neighbors,
                            "pivot_params": w.pull(None),
                            "honest_local_params": honest_neighbor_weights,
                        }
                    )
                    for i in byz_neighbors
                ]
            else:
                byz_weights = (
                    [
                        byz_workers[byz_neighbor].pull(
                            {"honest_weights": honest_weights, "step": current_step}
                        )
                        for byz_neighbor in byz_neighbors
                    ]
                    if byz_neighbors
                    else []
                )
            w.aggregate(honest_neighbor_weights + byz_weights)

        with torch.no_grad():
            updated_weights = [w.pull(None) for w in workers]
            _raise_if_nonfinite_weights(workers, current_step, "fixed")
            disagreement = neighbor_disagreement(
                updated_weights, adjacency=adjacency_honest
            )
            consensus = consensus_drift(updated_weights)
        neighbor_disagreement_history.append(disagreement.cpu().numpy())
        consensus_drift_history.append(consensus.cpu().numpy())

    final_accuracy, final_validation_loss, final_train_loss = (
        _record_final_evaluation_if_needed(
            args,
            workers,
            fd_validation,
            fd_validation_loss,
            fd_train_loss,
            validation_steps,
            validation_accuracies,
            validation_losses,
            train_losses,
        )
    )
    if _should_log_step(args.rounds, args.rounds):
        _log_progress(
            "fixed",
            args.rounds,
            args,
            final_accuracy,
            final_validation_loss,
            final_train_loss,
        )

    if validation_accuracies:
        worst_idx = min(range(len(workers)), key=lambda i: validation_accuracies[-1][i])
        for step, accs in zip(validation_steps, validation_accuracies, strict=True):
            store_result(fd_validation_worst, step, accs[worst_idx])

    if args.evaluate_test:
        fd_test = (result_dir / "test").open("w")
        make_result_file(fd_test, ["Step number", "Cross-accuracy"])
        test_accuracies = [w.compute_accuracy_on_loader(test_loader) for w in workers]
        store_result(fd_test, args.rounds, sum(test_accuracies) / len(test_accuracies))
        fd_test.close()

    fd_validation.close()
    fd_validation_worst.close()
    fd_validation_loss.close()
    fd_train_loss.close()

    np.save(os.path.join(result_dir, "validation_accuracies.npy"), np.array(validation_accuracies))
    np.save(os.path.join(result_dir, "accuracies.npy"), np.array(validation_accuracies))
    np.save(os.path.join(result_dir, "validation_losses.npy"), np.array(validation_losses))
    np.save(os.path.join(result_dir, "train_losses.npy"), np.array(train_losses))
    np.save(
        os.path.join(result_dir, "neighbor_disagreement.npy"),
        np.array(neighbor_disagreement_history),
    )
    np.save(
        os.path.join(result_dir, "consensus_drift.npy"),
        np.array(consensus_drift_history),
    )
    _log_done("fixed")
