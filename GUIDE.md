# BanditDL Experiment Guide

This guide explains how to run, manage, and analyze decentralized learning experiments using the modernized BanditDL framework.

---

## 1. Core Architecture
The framework uses **Hydra Structured Configs**. Everything is type-safe and validated before the experiment starts.
- **Config Root:** `conf/config.yaml`
- **Results:** Automatically saved in `outputs/YYYY-MM-DD/HH-MM-SS/` (managed by Hydra).
- **Telemetry:** High-resolution data is saved as `.npy` files using `open_memmap` for zero-loss on crash.

---

## 2. Running a Single Experiment
Use `python -m banditdl.experiments.hydra_run` to start a standard run.

### Basic Command
```bash
python -m banditdl.experiments.hydra_run dataset=mnist topology.nodes=20
```

### Useful Overrides
- **Device:** `device=cuda` or `device=cpu` (defaults to auto-detect).
- **Byzantine Nodes:** `adversary.byzcount=5 adversary.attack=ALIE`.
- **Heterogeneity:** `heterogeneity/dirichlet_alpha0.5` or `heterogeneity=pathological`.
- **Optimization:** `optimization.rounds=5000 optimization.learning_rate=0.1`.

---

## 3. Running Parameter Sweeps
Sweeps use **Optuna** for intelligent searching or grid search.
- **Sweep Config:** `conf/optuna/sweep.yaml`
- **Launch Command:**
```bash
python -m banditdl.experiments.sweep optuna=customsweep
```

### Multi-Seed Averaging
To ensure statistical significance, the engine runs $N$ seeds per trial.
- Set `num_seeds=5` in your command or config.
- Results will be organized by seed index under the trial directory.

---

## 4. Configuration Schema
Key sections in the `BanditDLConfig`:

| Section | Key Parameters | Description |
| :--- | :--- | :--- |
| **`dataset`** | `dataset`, `model` | Support for MNIST, CIFAR10, FEMNIST. |
| **`topology`** | `nodes`, `sampling` | Total participants and neighbor sampling ratio. |
| **`heterogeneity`** | `method`, `clusters`, `alpha` | Control Non-IIDness (Dirichlet or Pathological). |
| **`adversary`** | `byzcount`, `attack` | Byzantine settings (ALIE, SF, mimic, etc.). |
| **`sampler`** | `name`, `params.epsilon` | Neighbor selection strategy (uniform, bandit, exp3). |

---

## 5. Analyzing Results (Personalization vs. Generalization)
All metrics are stored as `(Evaluations x Nodes)` tensors.

### Primary Files
- **`local_accuracy.npy`**: Performance of every node on its **local partitioned data**.
- **`global_accuracy.npy`**: Performance of every node on the **shared global (IID) set**.
- **`sampler_probabilities.npy`**: Full $T \times N \times N$ record of who each node wanted to talk to.
- **`audit.json`**: A manifest of the experiment's initial state (who got which labels).

### Logic for Personalization
You have successful personalization if:
`mean(local_accuracy.npy[-1]) > mean(global_accuracy.npy[-1])`

---

## 6. Plotting
The centralized plotting tool handles individual runs and sweeps.

### Plot a Sweep
```bash
python scripts/plot_sweep.py --dir outputs/2026-06-06/my_sweep --x alpha --y local_accuracy
```

### Key Plotting Scripts
- `scripts/plot_results.py`: Standard training curves (Accuracy/Loss vs. Time).
- `scripts/analyze_clustering.py`: Visualize the learned topology from `sampler_probabilities.npy`.

---

## 7. Zero-Loss Engineering
- **Logs:** Hydra automatically captures all `logger.info` calls into `hydra.log` in the run directory. No manual file writing required.
- **Crash Safety:** All `.npy` files are flushed periodically. If the OS kills the process, you can still load the partial results using `np.load()`.
