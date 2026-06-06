# BanditDL Codebase Investigation Report

## Executive Summary
The investigation of the BanditDL codebase identified one critical architectural bug (P0) regarding Byzantine worker state, a major performance bottleneck (P1) due to redundant deep copies, and several opportunities for hardening and modernization (P2). While the recent refactoring has significantly improved the structure and reliability of the engine, some legacy patterns and "overly defensive" logic remain.

---

## 1. Findings & Discrepancies

### [P0] Critical Bug: Byzantine State Corruption
**Location:** `banditdl/experiments/engine.py` & `banditdl/core/robustness/attacks.py`
**Issue:** Stateful attacks (e.g., `mimic`) update internal heuristic state (like `mu_mimic`, `z_mimic`) every time `pull()` is called. The current training loop in `_step_dynamic` calls `pull()` multiple times per round for each Byzantine worker:
1. First, inside `_dynamic_candidate_weights` to build the set of available weights for every honest node.
2. Second, inside the `sel_ids` loop if the node is selected.
**Impact:** This corrupts the attack vectors and invalidates research results for stateful attacks.

### [P1] Performance Bottleneck: Redundant Deepcopies
**Location:** `banditdl/experiments/engine.py` -> `_dynamic_candidate_weights`
**Issue:** The engine performs a `copy.deepcopy(byz_worker)` for every honest worker for every Byzantine worker in every round. 
**Impact:** In a 100-node run with 20% Byzantine nodes, this results in ~1,600 deepcopies per round. This causes massive slowdowns and is unnecessary.

### [P1] Partial Data Loss: Intermittent Numpy Saving
**Location:** `banditdl/experiments/engine.py` -> `ResultTracker.save_snapshot`
**Issue:** While raw probabilities use `open_memmap` (progressive), other metrics like accuracies and losses are stored in lists and saved only every 10 rounds.
**Impact:** A crash at round 99 results in the loss of 9 rounds of data. 

### [P2] Legacy Hyphen Logic
**Location:** `banditdl/experiments/config_adapter.py` -> `_run_name`
**Issue:** Run names still generate strings like `nb-local` and `sampling-ratio`, contradicting the "zero legacy / fuck legacy" directive to use underscores exclusively.

### [P2] Overly Defensive & Legacy Code
**Location:** `banditdl/experiments/engine.py`
**Issue:**
- Uses `getattr(getattr(cfg, "evaluation", cfg), "evaluation_delta")` and `hasattr(cfg, "effective_rounds")`. These are redundant since we use strict `BanditDLConfig` dataclasses.
- `banditdl/core/sampling.py` still maintains aliases for legacy parameters (e.g., `bandit_epsilon`).

---

## 2. Proposed Changes

### Immediate Fixes (Required for Main)
1. **Byzantine Vector Caching:**
   - Modify `ByzantineWorker` to have an `update_attack(honest_weights)` method that computes and caches the attack vector once.
   - Update `engine.py` to call `update_attack` once per round.
   - Change `pull()` to simply return the cached vector.

2. **Remove Deepcopies:**
   - Since Byzantine workers will now use cached vectors, `_dynamic_candidate_weights` no longer needs to copy the worker objects.

3. **Progressive Telemetry:**
   - Convert all lists in `ResultTracker` to `open_memmap` arrays or ensure they are saved atomically every round.

4. **Schema Cleanup:**
   - Remove all `getattr`/`hasattr` checks in `engine.py` and replace with direct attribute access.
   - Standardize `config_adapter.py` to use underscores in run tokens.

### Long-term Refactoring
- **RoundStrategy Abstraction:** The training loop logic (dynamic vs fixed) is still quite bulky. Moving the neighbor-weight collection and aggregation logic into a strategy pattern would further simplify the engine.
- **Label Wrapping Logic:** Move the modulo-based label wrapping in `dataset_utils.py` into a shared utility for any partitioning scheme that might exceed label counts.

---

## 3. Verification Plan
- **Unit Tests:** Add a test case to `tests/test_robust_aggregation.py` that verifies a stateful attack's internal state remains consistent within a single training round.
- **Performance Benchmarking:** Measure round time before/after removing deepcopies.
- **Crash Recovery:** Manually terminate an experiment and verify that `validation_accuracies.npy` contains all data up to the last completed step.
