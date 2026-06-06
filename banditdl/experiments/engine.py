from __future__ import annotations

import copy
import json
import os
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
from banditdl.core.worker.byzantine import ByzantineWorker
from banditdl.core.worker.config import WorkerConfig
from banditdl.core.worker.dynamic import DynamicWorker
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
    return current_step in (0, rounds) or current_step % _progress_interval(rounds) == 0


def _log_start(cfg: BanditDLConfig, result_dir: pathlib.Path) -> None:
    print(
        "[banditdl] starting run: "
        f"dataset={cfg.dataset.dataset}, model={cfg.dataset.model}, nodes={cfg.topology.nodes}, "
        f"honest={cfg.nb_honests}, byzantine={cfg.adversary.byzcount}, "
        f"rounds={cfg.effective_rounds}, seed={cfg.seed}, device={cfg.device}",
        flush=True,
    )
    print(f"[banditdl] results: {result_dir}", flush=True)


def _log_progress(
    current_step: int,
    cfg: BanditDLConfig,
    accuracy=None,
    validation_loss=None,
    train_loss=None,
) -> None:
    message = f"[banditdl] round {current_step}/{cfg.effective_rounds}"
    if accuracy is not None:
        message += f" | mean_accuracy={accuracy:.4f}"
    if validation_loss is not None:
        message += f" | val_loss={validation_loss:.4f}"
    if train_loss is not None:
        message += f" | train_loss={train_loss:.4f}"
    print(message, flush=True)


def _log_done() -> None:
    print("[banditdl] finished run", flush=True)


def _raise_if_nonfinite_weights(workers, current_step: int) -> None:
    for worker in workers:
        weights = worker.pull(None)
        if not torch.isfinite(weights).all():
            raise FloatingPointError(
                f"produced non-finite weights at round {current_step} for worker {worker.worker_id}"
            )


def _record_evaluation(
    workers,
    fd_val,
    fd_val_loss,
    fd_train_loss,
    step,
    val_steps,
    val_accs,
    val_losses,
    train_losses,
):
    accs = [w.compute_validation_accuracy() for w in workers]
    v_losses = [w.compute_validation_loss() for w in workers]
    t_losses = [w.compute_train_loss() for w in workers]
    mean_acc = sum(accs) / len(accs)
    mean_v = sum(v_losses) / len(v_losses)
    mean_t = sum(t_losses) / len(t_losses)

    val_steps.append(step)
    val_accs.append(accs)
    val_losses.append(v_losses)
    train_losses.append(t_losses)
    store_result(fd_val, step, mean_acc)
    store_result(fd_val_loss, step, mean_v)
    store_result(fd_train_loss, step, mean_t)

    return mean_acc, mean_v, mean_t


def _record_final_evaluation_if_needed(
    cfg: BanditDLConfig,
    workers,
    fd_val,
    fd_v_loss,
    fd_t_loss,
    v_steps,
    v_accs,
    v_losses,
    t_losses,
):
    evaluation = getattr(cfg, "evaluation", cfg)
    rounds = cfg.effective_rounds if hasattr(cfg, "effective_rounds") else cfg.rounds
    if evaluation.evaluation_delta <= 0:
        return None, None, None
    if v_steps and v_steps[-1] == rounds:
        return None, None, None
    return _record_evaluation(
        workers,
        fd_val,
        fd_v_loss,
        fd_t_loss,
        rounds,
        v_steps,
        v_accs,
        v_losses,
        t_losses,
    )


class ResultTracker:
    """Consolidates metrics tracking, evaluation, and saving results."""

    def __init__(self, cfg: BanditDLConfig, result_dir: pathlib.Path, test_loader=None):
        self.cfg, self.result_dir, self.test_loader = cfg, result_dir, test_loader
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.fd_validation = (result_dir / "validation").open("w")
        self.fd_validation_worst = (result_dir / "validation_worst").open("w")
        self.fd_validation_loss = (result_dir / "validation_loss").open("w")
        self.fd_train_loss = (result_dir / "train_loss").open("w")

        for fd, fields in [
            (self.fd_validation, ["Step number", "Cross-accuracy"]),
            (self.fd_validation_worst, ["Step number", "Cross-accuracy"]),
            (self.fd_validation_loss, ["Step number", "Cross-loss"]),
            (self.fd_train_loss, ["Step number", "Cross-loss"]),
        ]:
            make_result_file(fd, fields)

        self.validation_steps, self.validation_accuracies = [], []
        self.validation_losses, self.train_losses = [], []
        self.neighbor_disagreement_history, self.consensus_drift_history = [], []
        self.gradient_norm_history = []

        self.algorithm_reward_history, self.oracle_reward_history = [], []
        self.selected_neighbor_history, self.oracle_neighbor_history = [], []
        self.reward_min_history, self.reward_max_history = [], []

        self.prob_file = result_dir / "sampler_probabilities.npy"
        self.probs_mmap = None
        if cfg.effective_rounds > 0:
            import numpy.lib.format

            self.probs_mmap = numpy.lib.format.open_memmap(
                self.prob_file,
                dtype="float32",
                mode="w+",
                shape=(cfg.effective_rounds, cfg.nb_honests, cfg.topology.nodes),
            )
            self.probs_mmap[:] = np.nan
            self.probs_mmap.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.save_snapshot()
        for fd in [
            self.fd_validation,
            self.fd_validation_worst,
            self.fd_validation_loss,
            self.fd_train_loss,
        ]:
            fd.close()
        if self.probs_mmap is not None:
            self.probs_mmap.flush()

    def save_audit(self, audit_data: dict):
        with (self.result_dir / "audit.json").open("w") as f:
            json.dump(audit_data, f, indent=2)

    def evaluate_step(self, step, workers, mode):
        mean_acc, mean_v, mean_t = None, None, None
        if (
            self.cfg.evaluation.evaluation_delta > 0
            and step % self.cfg.evaluation.evaluation_delta == 0
        ):
            mean_acc, mean_v, mean_t = _record_evaluation(
                workers,
                self.fd_validation,
                self.fd_validation_loss,
                self.fd_train_loss,
                step,
                self.validation_steps,
                self.validation_accuracies,
                self.validation_losses,
                self.train_losses,
            )
        if _should_log_step(step, self.cfg.effective_rounds):
            _log_progress(step, self.cfg, mean_acc, mean_v, mean_t)
        return mean_acc, mean_v, mean_t

    def record_gradient_norms(self, workers):
        """Record the gradient norm for each worker."""
        self.gradient_norm_history.append(
            np.array([w.last_gradient_norm for w in workers], dtype=float)
        )

    def record_drift(self, disagreement, consensus):
        self.neighbor_disagreement_history.append(disagreement.cpu().numpy())
        self.consensus_drift_history.append(consensus.cpu().numpy())

    def record_probabilities(self, step, workers):
        if self.probs_mmap is not None and step < self.cfg.effective_rounds:
            probs = np.stack(
                [_full_sampler_probability_vector(w, self.cfg.topology.nodes) for w in workers]
            )
            self.probs_mmap[step] = probs.astype("float32")
            if step % 10 == 0:
                self.probs_mmap.flush()

    def record_rewards(self, alg, ora, sel, ora_n, r_min, r_max):
        self.algorithm_reward_history.append(alg.copy())
        self.oracle_reward_history.append(np.array(ora))
        self.selected_neighbor_history.append(sel.copy())
        self.oracle_neighbor_history.append(np.stack(ora_n))
        self.reward_min_history.append(r_min.copy())
        self.reward_max_history.append(r_max.copy())

    def save_snapshot(self):
        d = self.result_dir
        _atomic_save(d / "validation_accuracies.npy", self.validation_accuracies)
        _atomic_save(d / "validation_losses.npy", self.validation_losses)
        _atomic_save(d / "train_losses.npy", self.train_losses)
        _atomic_save(d / "neighbor_disagreement.npy", self.neighbor_disagreement_history)
        _atomic_save(d / "consensus_drift.npy", self.consensus_drift_history)
        _atomic_save(d / "gradient_norms.npy", self.gradient_norm_history)
        if self.algorithm_reward_history:
            _atomic_save(d / "reward_algorithm.npy", self.algorithm_reward_history)
            _atomic_save(d / "reward_oracle.npy", self.oracle_reward_history)
            _atomic_save(d / "reward_selected_min.npy", self.reward_min_history)
            _atomic_save(d / "reward_selected_max.npy", self.reward_max_history)
            _atomic_save(
                d / "selected_neighbors.npy",
                self.selected_neighbor_history,
                dtype=int,
            )
            _atomic_save(
                d / "oracle_neighbors.npy",
                self.oracle_neighbor_history,
                dtype=int,
            )
            regret = np.array(self.oracle_reward_history) - np.array(self.algorithm_reward_history)
            _atomic_save(d / "regret.npy", regret)

    def finalize(self, workers):
        _record_final_evaluation_if_needed(
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
        if self.validation_accuracies:
            worst_idx = min(range(len(workers)), key=lambda i: self.validation_accuracies[-1][i])
            for s, a in zip(self.validation_steps, self.validation_accuracies, strict=True):
                store_result(self.fd_validation_worst, s, a[worst_idx])
        if self.cfg.evaluation.evaluate_test and self.test_loader:
            fd_test = (self.result_dir / "test").open("w")
            make_result_file(fd_test, ["Step number", "Cross-accuracy"])
            accs = [w.compute_accuracy_on_loader(self.test_loader) for w in workers]
            store_result(fd_test, self.cfg.effective_rounds, sum(accs) / len(accs))
            fd_test.close()
        self.save_snapshot()
        with torch.no_grad():
            weights = torch.stack([w.pull(None) for w in workers])
            dist = torch.cdist(weights, weights).cpu().numpy()
        np.save(self.result_dir / "pairwise_model_distance_final.npy", dist)
        _log_done()


def _atomic_save(path: pathlib.Path, values, dtype=None) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fd:
        np.save(fd, np.asarray(values, dtype=dtype))
    os.replace(tmp, path)


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
        b_hat=cfg.adversary.byzantine_budget
        if cfg.adversary.byzantine_budget is not None
        else cfg.adversary.byzcount,
        attack=cfg.adversary.attack,
        rag=cfg.aggregator.rag or cfg.topology.sampling is not None,
        numb_labels=cfg.dataset.numb_labels,
        labelflipping=cfg.adversary.attack == "LF",
        gradient_clip=None,
        server_clip=cfg.aggregator.server_clip,
        bucket_size=cfg.aggregator.bucket_size,
        aggregator=cfg.aggregator.aggregator,
        pre_aggregator=cfg.aggregator.pre_aggregator,
        sampling_ratio=cfg.topology.sampling,
        mimic_learning_phase=cfg.adversary.mimic_learning_phase,
    )


def _init_workers(cfg: BanditDLConfig, train_dict, test_dict, device: str):
    workers = []
    base_config = _build_worker_config(cfg, device)
    for worker_id in range(cfg.nb_honests):
        s_params = dict(cfg.sampler.get("params", {}) if cfg.sampler else {})
        s_ctx = SamplerContext(
            worker_id,
            cfg.topology.nodes,
            1,
            cfg.effective_rounds + 1,
            cfg.seed + worker_id,
        )
        config = replace(
            base_config,
            neighbor_sampler=make_neighbor_sampler(
                cfg.resolved_sampler_name,
                context=s_ctx,
                params=s_params,
            ),
            reward_strategy=make_reward_strategy(cfg.resolved_reward_name),
        )
        w = DynamicWorker(worker_id, train_dict[worker_id], test_dict[worker_id], config)
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
    return float(sum(rewards) / len(rewards)) if rewards else 0.0


def _dynamic_candidate_weights(w, honest_weights, byz_workers, step):
    weights = {i: weight for i, weight in enumerate(honest_weights) if i != w.worker_id}
    for byz in byz_workers:
        weight = copy.deepcopy(byz).pull({"honest_weights": honest_weights, "step": step})
        if weight is not None:
            weights[byz.worker_id] = weight
    return weights


def _full_sampler_probability_vector(worker, nb_total: int) -> np.ndarray:
    pop = [i for i in range(nb_total) if i != worker.worker_id]
    probs_by_arm = worker.neighbor_sampler.probabilities(pop, worker.nb_neighbors)
    row = np.zeros(nb_total, dtype=float)
    for arm in pop:
        row[arm] = float(probs_by_arm[arm])
    return row


def _step_dynamic(
    step, cfg, workers, byz_workers, byz_by_id, honest_weights, cum_arm_r, cum_alg_r, tracker
):
    selected_round = np.full((cfg.nb_honests, workers[0].nb_neighbors), -1, dtype=int)
    r_min_round, r_max_round = np.full(cfg.nb_honests, np.nan), np.full(cfg.nb_honests, np.nan)

    for w in workers:
        neighbor_indices = w._sample_neighbors()
        c_weights = _dynamic_candidate_weights(w, honest_weights, byz_workers, step)
        sel_ids = [i for i in neighbor_indices if i in c_weights]
        for nid in sel_ids:
            if nid >= cfg.nb_honests:
                weight = byz_by_id[nid].pull({"honest_weights": honest_weights, "step": step})
                if weight is not None:
                    c_weights[nid] = weight

        candidate_ids = list(c_weights)
        c_rewards = w.reward_strategy.score(w.pull(None), [c_weights[i] for i in candidate_ids])
        rewards_by_id = dict(zip(candidate_ids, c_rewards, strict=True))
        cum_arm_r[w.worker_id, candidate_ids] += c_rewards

        n_weights = [c_weights[i] for i in sel_ids]
        selected_round[w.worker_id, : len(sel_ids)] = sel_ids
        s_rewards = [rewards_by_id[i] for i in sel_ids]
        if s_rewards:
            r_min_round[w.worker_id], r_max_round[w.worker_id] = min(s_rewards), max(s_rewards)
            cum_alg_r[w.worker_id] += _mean_selected_reward(s_rewards)

        w.num_selected_byz.append(len([i for i in sel_ids if i >= cfg.nb_honests]))
        w.observe_neighbors(sel_ids, n_weights)
        w.aggregate(n_weights)

    ora_n_round, ora_r_round = [], []
    for w in workers:
        oids, oreward = _best_fixed_subset(cum_arm_r[w.worker_id], w.worker_id, w.nb_neighbors)
        ora_n_round.append(oids)
        ora_r_round.append(oreward)

    tracker.record_rewards(
        cum_alg_r, ora_r_round, selected_round, ora_n_round, r_min_round, r_max_round
    )
    with torch.no_grad():
        updated = [w.pull(None) for w in workers]
        n_matrix = selected_round.copy()
        n_matrix[n_matrix >= cfg.nb_honests] = -1
        tracker.record_drift(
            neighbor_disagreement(updated, neighbor_indices=n_matrix.tolist()),
            consensus_drift(updated),
        )


def run_experiment(
    cfg: BanditDLConfig,
    result_dir: pathlib.Path,
    seed: int,
    device: str,
) -> None:
    _setup_seed(seed)
    _log_start(cfg, result_dir)
    train_dict, local_test_dict, test_loader, dist_stats = (
        dataset.make_train_validation_test_datasets(
            cfg.dataset.dataset,
            numb_labels=cfg.dataset.numb_labels,
            alpha_dirichlet=cfg.heterogeneity.alpha,
            honest_workers=cfg.nb_honests,
            train_batch=cfg.optimization.batch_size,
            test_batch=100,
            global_test_ratio=cfg.evaluation.global_test_ratio,
            local_test_ratio=cfg.evaluation.local_test_ratio,
            split_seed=cfg.evaluation.split_seed,
            partition_method=cfg.heterogeneity.method,
            clusters=cfg.heterogeneity.clusters,
            classes_per_group=cfg.heterogeneity.classes_per_group,
            group_overlap=cfg.heterogeneity.group_overlap,
            gamma_similarity=cfg.heterogeneity.gamma_similarity,
            dataset_mode=cfg.dataset.mode,
            nb_writers_limit=cfg.dataset.nb_writers_limit,
        )
    )

    workers = _init_workers(cfg, train_dict, local_test_dict, device)
    bw_cfg = _build_worker_config(cfg, device)
    byz_workers = [
        ByzantineWorker(i, workers[0].model_size, bw_cfg)
        for i in range(cfg.nb_honests, cfg.topology.nodes)
    ]
    byz_by_id = {byz.worker_id: byz for byz in byz_workers}

    with ResultTracker(cfg, result_dir, test_loader) as tracker:
        tracker.save_audit(
            {
                "partition": {
                    "method": cfg.heterogeneity.method,
                    "seed": cfg.evaluation.split_seed,
                    "requested_clusters": cfg.heterogeneity.clusters,
                    "resolved_clusters": cfg.resolved_clusters,
                    "alpha": cfg.heterogeneity.alpha,
                    "classes_per_group": cfg.heterogeneity.classes_per_group,
                    "group_overlap": cfg.heterogeneity.group_overlap,
                    "gamma_similarity": cfg.heterogeneity.gamma_similarity,
                },
                "distribution": dist_stats,
                "participants": {
                    "total": cfg.topology.nodes,
                    "honest": cfg.nb_honests,
                    "byzantine": cfg.adversary.byzcount,
                },
            }
        )
        cum_arm_r, cum_alg_r = (
            np.zeros((cfg.nb_honests, cfg.topology.nodes)),
            np.zeros(cfg.nb_honests),
        )

        for step in range(cfg.effective_rounds + 1):
            tracker.evaluate_step(step, workers, "dynamic")
            if step < cfg.effective_rounds:
                for w in workers:
                    w.train()
                tracker.record_gradient_norms(workers)
                tracker.record_probabilities(step, workers)
                h_weights = [w.pull(None) for w in workers]
                _step_dynamic(
                    step,
                    cfg,
                    workers,
                    byz_workers,
                    byz_by_id,
                    h_weights,
                    cum_arm_r,
                    cum_alg_r,
                    tracker,
                )
                _raise_if_nonfinite_weights(workers, step)
                if step % 10 == 0:
                    tracker.save_snapshot()
        tracker.finalize(workers)
