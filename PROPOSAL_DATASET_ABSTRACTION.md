# Proposal: Unified Dataset Abstraction (UDA)

## 1. Problem Statement
The current dataset orchestration logic in `banditdl/data/dataset.py` suffers from **Violated Abstraction**. Specifically:
- **Hardcoded Conditionals:** Multiple `if _is_femnist(dataset_name):` checks force the high-level logic to know about dataset-specific implementation details.
- **Config Leakage:** The `DatasetConfig` dataclass contains parameters like `nb_writers_limit` which are irrelevant for non-FEMNIST datasets, leading to "sparse" and confusing configurations.
- **Rigid Partitioning:** The system differentiates between "legacy" and "hierarchical" partitioning via strings rather than polymorphic strategies, making it difficult to add "Natural" partitioning (e.g., by user, by device) for new datasets.

---

## 2. Proposed Architecture: The "Provider-Strategy" Pattern

### A. The `DatasetProvider` Abstraction
Every dataset (MNIST, CIFAR, FEMNIST) will be encapsulated in a class that implements a common interface.

```python
class DatasetProvider(ABC):
    @abstractmethod
    def get_full_pool(self) -> SamplePool:
        """Return a SamplePool containing data, targets, and optional metadata."""
```

The `SamplePool` is a thin wrapper that contains the indices and labels, but can also contain dataset-specific metadata like **Owner IDs** (Writers).

### B. Polymorphic Partitioning
We move the mathematical logic into **Partition Strategies**.

- **SyntheticStrategy:** Used for MNIST/CIFAR where we force a distribution (Dirichlet/Pathological).
- **NaturalStrategy:** Used for FEMNIST where we simply "group by Owner ID."

### C. Hydra Recursive Instantiation
Instead of one giant `DatasetConfig`, we use Hydra to instantiate the correct provider dynamically.

**Current Config:**
```yaml
dataset:
  dataset: femnist
  nb_writers_limit: 100
```

**Proposed Config:**
```yaml
dataset:
  _target_: banditdl.data.providers.FemnistProvider
  nb_writers_limit: 100
```
This ensures that `nb_writers_limit` only exists when FEMNIST is used.

---

## 3. Benefits

1.  **Zero-Change Scalability:** Adding a new dataset (e.g., CelebA) requires zero changes to `engine.py`. You just implement a new `CelebAProvider`.
2.  **Encapsulation:** The concept of "Writers" is hidden inside the `FemnistProvider`. The rest of the system only sees "Indices" and "Labels."
3.  **Cleaner Orchestration:** `make_train_validation_test_datasets` becomes a pure 5-step pipeline without a single `if` statement.
4.  **Testability:** You can now unit test the `DirichletStrategy` in isolation by passing it a mock `SamplePool`.

---

## 4. Implementation Roadmap

1.  **Define Core Interfaces:** Create `SamplePool` and `DatasetProvider` base classes.
2.  **Port Existing Datasets:** 
    - Create `MnistProvider` (simple indexing).
    - Create `FemnistProvider` (writer-aware logic).
3.  **Refactor Orchestration:** Rewrite `dataset.py` to accept a `DatasetProvider` object.
4.  **Modernize Config:** Update `config_schema.py` and `conf/dataset/*.yaml` to use recursive instantiation.
