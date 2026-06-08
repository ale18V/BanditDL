from __future__ import annotations

import logging
import os
import pathlib
import random
from dataclasses import replace

import numpy as np
import numpy.lib.format
import torch
from hydra.utils import instantiate

from banditdl.core.sampling import (
    SamplerContext,
    make_neighbor_sampler,
    make_reward_strategy,
)
from banditdl.core.worker.byzantine import ByzantineWorker
from banditdl.core.worker.config import WorkerConfig
from banditdl.core.worker.dynamic import DynamicWorker
from banditdl.data import DatasetBuildConfig, build_dataset_bundle
from banditdl.experiments.config_schema import BanditDLConfig
from banditdl.utils.math_utils import consensus_drift, neighbor_disagreement

logger = logging.getLogger(__name__)


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
    logger.info(
        "starting run: "
        f"dataset={cfg.dataset.dataset}, model={cfg.dataset.model}, nodes={cfg.topology.nodes}, "
        f"honest={cfg.nb_honests}, byzantine={cfg.adversary.byzcount}, "
        f"rounds={cfg.effective_rounds}, seed={cfg.seed}, device={cfg.device}"
    )
    logger.info(f"results: {result_dir}")


def _log_progress(
    current_step: int,
    cfg: BanditDLConfig,
    accuracy=None,
    validation_loss=None,
    train_loss=None,
) -> None:
    message = f"round {current_step}/{cfg.effective_rounds}"
    if accuracy is not None:
        message += f" | mean_accuracy={accuracy:.4f}"
    if validation_loss is not None:
        message += f" | val_loss={validation_loss:.4f}"
    if train_loss is not None:
        message += f" | train_loss={train_loss:.4f}"
    logger.info(message)


def _log_done() -> None:
    logger.info("finished run")


def _raise_if_nonfinite_weights(honest_workers, current_step: int) -> None:
    for worker in honest_workers:
        weights = worker.pull(None)
        if not torch.isfinite(weights).all():
            raise FloatingPointError(
                f"produced non-finite weights at round {current_step} for worker {worker.worker_id}"
            )


class ResultTracker:
    """Consolidates metrics tracking, evaluation, and saving results."""

    def __init__(self, cfg: BanditDLConfig, result_dir: pathlib.Path, test_loader=None, test_loader_sub=None):
        self.cfg, self.result_dir, self.test_loader, self.test_loader_sub = cfg, result_dir, test_loader, test_loader_sub
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.validation_steps = []

        # Progressive saving for all metrics
        self.mmaps = {}
        delta = cfg.evaluation.evaluation_delta
        nb_evals = (
            (cfg.effective_rounds // delta)
            + 1
            + int(cfg.effective_rounds % delta != 0)
            if delta > 0
            else 1
        )

        mmap_configs = {
            "evaluation_steps.npy": (nb_evals,),
            "validation_accuracy.npy": (nb_evals, cfg.nb_honests),
            "validation_loss.npy": (nb_evals, cfg.nb_honests),
            "global_accuracy.npy": (nb_evals, cfg.nb_honests),  # Periodic subsampled eval
            "train_loss.npy": (cfg.effective_rounds + 1, cfg.nb_honests),
            "neighbor_disagreement.npy": (cfg.effective_rounds, cfg.nb_honests),
            "consensus_drift.npy": (cfg.effective_rounds, cfg.nb_honests),
            "gradient_norms.npy": (cfg.effective_rounds, cfg.nb_honests),
            "sampler_weights.npy": (
                cfg.effective_rounds,
                cfg.nb_honests,
                cfg.topology.nodes,
            ),
            "sampler_probabilities.npy": (
                cfg.effective_rounds,
                cfg.nb_honests,
                cfg.topology.nodes,
            ),
        }

        for name, shape in mmap_configs.items():
            path = result_dir / name
            mmap = numpy.lib.format.open_memmap(path, dtype="float32", mode="w+", shape=shape)
            mmap[:] = np.nan
            self.mmaps[name] = mmap

        self.algorithm_reward_history, self.oracle_reward_history = [], []
        self.selected_neighbor_history, self.oracle_neighbor_history = [], []
        self.reward_min_history, self.reward_max_history = [], []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mmap in self.mmaps.values():
            mmap.flush()

    def save_audit(self, audit_data: dict):
        import json
        with (self.result_dir / "audit.json").open("w") as f:
            json.dump(audit_data, f, indent=2)

    def evaluate_step(self, step, honest_workers):
        mean_acc, mean_v = None, None
        delta = self.cfg.evaluation.evaluation_delta
        should_evaluate = delta > 0 and (
            step % delta == 0 or step == self.cfg.effective_rounds
        )
        if should_evaluate and step not in self.validation_steps:
            eval_idx = len(self.validation_steps)
            accs = [w.compute_validation_accuracy() for w in honest_workers]
            v_losses = [w.compute_validation_loss() for w in honest_workers]

            self.mmaps["evaluation_steps.npy"][eval_idx] = step
            self.mmaps["validation_accuracy.npy"][eval_idx] = np.array(accs, dtype="float32")
            self.mmaps["validation_loss.npy"][eval_idx] = np.array(v_losses, dtype="float32")

            # Periodic Global Generalization (Subsampled)
            if self.test_loader_sub:
                g_accs = [w.compute_accuracy_on_loader(self.test_loader_sub) for w in honest_workers]
                self.mmaps["global_accuracy.npy"][eval_idx] = np.array(g_accs, dtype="float32")
                logger.info(f"Step {step} | Mean Local Acc: {sum(accs)/len(accs):.4f} | Mean Global Acc (sub): {sum(g_accs)/len(g_accs):.4f}")

            mean_acc, mean_v = sum(accs) / len(accs), sum(v_losses) / len(v_losses)
            self.validation_steps.append(step)

            if eval_idx % 5 == 0:
                for name in ["evaluation_steps.npy", "validation_accuracy.npy", "validation_loss.npy", "global_accuracy.npy"]:
                    self.mmaps[name].flush()

        if _should_log_step(step, self.cfg.effective_rounds):
            _log_progress(step, self.cfg, mean_acc, mean_v)
        return mean_acc, mean_v

    def record_train_loss(self, step, honest_workers):
        if step <= self.cfg.effective_rounds:
            losses = [w.compute_train_loss() for w in honest_workers]
            self.mmaps["train_loss.npy"][step] = np.array(losses, dtype="float32")
            if step % 10 == 0 or step == self.cfg.effective_rounds:
                self.mmaps["train_loss.npy"].flush()
            return sum(losses) / len(losses)
        return None

    def record_gradient_norms(self, step, honest_workers):
        """Record the gradient norm for each worker."""
        if step < self.cfg.effective_rounds:
            norms = [w.last_gradient_norm for w in honest_workers]
            self.mmaps["gradient_norms.npy"][step] = np.array(norms, dtype="float32")

    def record_drift(self, step, disagreement, consensus):
        if step < self.cfg.effective_rounds:
            dis_val = disagreement.cpu().numpy().astype("float32")
            con_val = consensus.cpu().numpy().astype("float32")
            self.mmaps["neighbor_disagreement.npy"][step] = dis_val
            self.mmaps["consensus_drift.npy"][step] = con_val

    def record_sampler_diagnostics(self, step, weights, probabilities):
        if step >= self.cfg.effective_rounds:
            return
        self.mmaps["sampler_weights.npy"][step] = weights
        self.mmaps["sampler_probabilities.npy"][step] = probabilities

    def record_rewards(self, alg, ora, sel, ora_n, r_min, r_max):
        self.algorithm_reward_history.append(alg.copy())
        self.oracle_reward_history.append(np.array(ora))
        self.selected_neighbor_history.append(sel.copy())
        self.oracle_neighbor_history.append(np.stack(ora_n))
        self.reward_min_history.append(r_min.copy())
        self.reward_max_history.append(r_max.copy())

    def save_snapshot(self):
        """Flush all mmaps and save dynamic reward histories."""
        for mmap in self.mmaps.values():
            mmap.flush()

        d = self.result_dir
        if self.algorithm_reward_history:
            _atomic_save(d / "reward_algorithm.npy", self.algorithm_reward_history)
            _atomic_save(d / "reward_oracle.npy", self.oracle_reward_history)
            _atomic_save(d / "reward_selected_min.npy", self.reward_min_history)
            _atomic_save(d / "reward_selected_max.npy", self.reward_max_history)
            _atomic_save(d / "selected_neighbors.npy", self.selected_neighbor_history, dtype=int)
            _atomic_save(d / "oracle_neighbors.npy", self.oracle_neighbor_history, dtype=int)

            regret = np.array(self.oracle_reward_history) - np.array(self.algorithm_reward_history)
            _atomic_save(d / "regret.npy", regret)

    def finalize(self, honest_workers):
        if not self.validation_steps or self.validation_steps[-1] != self.cfg.effective_rounds:
            self.evaluate_step(self.cfg.effective_rounds, honest_workers)

        if len(self.validation_steps) > 0:
            eval_idx = len(self.validation_steps) - 1
            last_accs = self.mmaps["validation_accuracy.npy"][eval_idx]
            # Replace NaNs with infinity for min finding
            last_accs_clean = np.where(np.isnan(last_accs), np.inf, last_accs)
            worst_idx = np.argmin(last_accs_clean)
            logger.info(f"Final Worst Local Client Accuracy: {last_accs[worst_idx]:.4f}")

        if self.cfg.evaluation.evaluate_test and self.test_loader:
            accs = [w.compute_accuracy_on_loader(self.test_loader) for w in honest_workers]
            global_acc_arr = np.array(accs, dtype="float32")
            np.save(self.result_dir / "test_accuracy.npy", global_acc_arr)
            logger.info(f"Final Mean Global Accuracy: {np.mean(global_acc_arr):.4f}")

        self.save_snapshot()
        with torch.no_grad():
            weights = torch.stack([w.pull(None) for w in honest_workers])
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
            max(
                1,
                min(
                    cfg.topology.nodes - 1,
                    round((cfg.topology.nodes - 1) * cfg.topology.sampling),
                ),
            ),
            cfg.effective_rounds,
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


def _dynamic_candidate_weights(w, honest_weights, byz_by_id):
    weights = {i: weight for i, weight in enumerate(honest_weights) if i != w.worker_id}
    for byz_id, byz in byz_by_id.items():
        weight = byz.pull()
        if weight is not None:
            weights[byz_id] = weight
    return weights


def _full_sampler_diagnostics(worker, nb_total: int) -> tuple[np.ndarray, np.ndarray]:
    population = [i for i in range(nb_total) if i != worker.worker_id]
    diagnostics = worker.neighbor_sampler.diagnostics(population, worker.nb_neighbors)
    weights = np.zeros(nb_total, dtype=float)
    probabilities = np.zeros(nb_total, dtype=float)
    for arm in population:
        weights[arm] = diagnostics.weights[arm]
        probabilities[arm] = diagnostics.probabilities[arm]
    return weights, probabilities


def _step_dynamic(
    step, cfg, honest_workers, byz_by_id, h_weights, cum_arm_r, cum_alg_r, tracker
):
    selected_round = np.full((cfg.nb_honests, honest_workers[0].nb_neighbors), -1, dtype=int)
    r_min_round, r_max_round = np.full(cfg.nb_honests, np.nan), np.full(cfg.nb_honests, np.nan)
    weight_rows = np.zeros((cfg.nb_honests, cfg.topology.nodes), dtype=float)
    probability_rows = np.zeros_like(weight_rows)

    for w in honest_workers:
        neighbor_indices = w._sample_neighbors()
        weight_rows[w.worker_id], probability_rows[w.worker_id] = (
            _full_sampler_diagnostics(w, cfg.topology.nodes)
        )
        c_weights = _dynamic_candidate_weights(w, h_weights, byz_by_id)
        sel_ids = [i for i in neighbor_indices if i in c_weights]

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
        w.observe_neighbors(sel_ids, s_rewards)
        w.aggregate(n_weights)

    tracker.record_sampler_diagnostics(step, weight_rows, probability_rows)
    ora_n_round, ora_r_round = [], []
    for w in honest_workers:
        oids, oreward = _best_fixed_subset(cum_arm_r[w.worker_id], w.worker_id, w.nb_neighbors)
        ora_n_round.append(oids)
        ora_r_round.append(oreward)

    tracker.record_rewards(
        cum_alg_r, ora_r_round, selected_round, ora_n_round, r_min_round, r_max_round
    )
    with torch.no_grad():
        updated = [w.pull(None) for w in honest_workers]
        n_matrix = selected_round.copy()
        n_matrix[n_matrix >= cfg.nb_honests] = -1
        tracker.record_drift(
            step,
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
    data = build_dataset_bundle(
        instantiate(cfg.dataset.provider),
        instantiate(cfg.partitioner_config),
        DatasetBuildConfig(
            nodes=cfg.nb_honests,
            train_batch=cfg.optimization.batch_size,
            test_batch=100,
            global_test_ratio=cfg.evaluation.global_test_ratio,
            local_test_ratio=cfg.evaluation.local_test_ratio,
            seed=cfg.evaluation.split_seed,
        ),
    )

    honest_workers = _init_workers(cfg, data.train, data.local_test, device)
    bw_cfg = _build_worker_config(cfg, device)
    byz_workers = [
        ByzantineWorker(i, honest_workers[0].model_size, bw_cfg)
        for i in range(cfg.nb_honests, cfg.total_nodes)
    ]
    byz_by_id = {byz.worker_id: byz for byz in byz_workers}

    with ResultTracker(cfg, result_dir, data.global_test, data.tracking_test) as tracker:
        tracker.save_audit(
            {
                **data.audit,
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
            tracker.evaluate_step(step, honest_workers)
            tracker.record_train_loss(step, honest_workers)
            if step < cfg.effective_rounds:
                for w in honest_workers:
                    w.train()
                tracker.record_gradient_norms(step, honest_workers)
                h_weights = [w.pull(None) for w in honest_workers]

                # Inform Byzantines once per round
                for byz in byz_workers:
                    byz.inform(h_weights, step)

                _step_dynamic(
                    step,
                    cfg,
                    honest_workers,
                    byz_by_id,
                    h_weights,
                    cum_arm_r,
                    cum_alg_r,
                    tracker,
                )
                _raise_if_nonfinite_weights(honest_workers, step)
                if step % 10 == 0:
                    tracker.save_snapshot()
        tracker.finalize(honest_workers)
